from fastapi import FastAPI, UploadFile, File, HTTPException, status, BackgroundTasks, WebSocket, WebSocketDisconnect, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from typing import Annotated, AsyncGenerator
from contextlib import asynccontextmanager
from dotenv import load_dotenv
import os
import asyncio
import json as json_module
from sqlalchemy.orm import Session

from app.services.vision_service import VisionService
from app.services.rag_service import RAGService
from app.services.pageindex_service import PageIndexService
from app.services.knowledge_graph_service import KnowledgeGraphService
from app.services.retrieval_orchestrator import RetrievalOrchestrator
from app.services.websocket_manager import ws_manager
from app.services.agent_service import AgentService
from app.services.insights_service import InsightsService
from app.services.study_guide_service import StudyGuideService
from app.services.contradiction_detector_service import ContradictionDetectorService
from app.models.retrieval_schemas import QueryRequest, AnswerResponse, PipelineSummary, CitationModel

# Database and Authentication
from app.database.base import get_db, init_db
from app.database.dependencies import get_current_user, get_optional_user
from app.models.user import User
from app.models.document import Document, DocumentStatus
from app.models.conversation import Conversation
from app.models.auth_schemas import UserRegister, UserLogin, Token, UserResponse, ErrorResponse
from app.services.auth_service import auth_service, ACCESS_TOKEN_EXPIRE_MINUTES
from app.services.input_guardrail_service import input_guardrail
from app.services.output_guardrail_service import output_guardrail
from datetime import timedelta, datetime
import traceback
import time

# Load environment variables from .env file
load_dotenv()

# Initialize Gemini API client
try:
    import google.generativeai as genai
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if gemini_api_key:
        genai.configure(api_key=gemini_api_key)
        # Use the fast, production-ready model
        gemini_model = genai.GenerativeModel('models/gemini-2.0-flash')
        print("✅ Gemini API initialized successfully with gemini-2.0-flash")
    else:
        gemini_model = None
        print("⚠️ GEMINI_API_KEY not found in .env")
except Exception as e:
    gemini_model = None
    print(f"⚠️ Gemini initialization failed: {e}")

# Initialize Together AI client as fallback
together_client = None
try:
    from together import Together
    together_api_key = os.getenv("TOGETHER_API_KEY")
    if together_api_key:
        together_client = Together(api_key=together_api_key)
        print("✅ Together AI initialized successfully as fallback")
    else:
        print("⚠️ TOGETHER_API_KEY not found in .env")
except Exception as e:
    print(f"⚠️ Together AI initialization failed: {e}")

# Instantiate our services
vision_service = VisionService()
rag_service = RAGService()
pageindex_service = PageIndexService()

# Initialize ICDI-X components
knowledge_graph = KnowledgeGraphService()
retrieval_orchestrator = RetrievalOrchestrator(rag_service, knowledge_graph)

# Initialize Multi-Agent AI Team
agent_service = AgentService()

# Initialize novel ICDI-X features (beyond Perplexity / NotebookLM)
insights_service = InsightsService()
study_guide_service = StudyGuideService()
contradiction_detector = ContradictionDetectorService()

# Lifespan event: Runs once when the server starts
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Initializing Database...")
    init_db()
    print("Initializing Qdrant Vector Database...")
    await rag_service.initialize()
    yield
    print("Shutting down system...")

app = FastAPI(
    title="Multi-Modal Document Intelligence System",
    description="API for processing mixed text and image documents.",
    lifespan=lifespan
)

# Add CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def health_check():
    return {"status": "System is healthy and ready."}


# ===== AUTHENTICATION ENDPOINTS =====

@app.post("/auth/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register_user(
    user_data: UserRegister,
    db: Session = Depends(get_db)
):
    """
    Register a new user account
    
    - **email**: Valid email address
    - **password**: Minimum 8 characters
    - **full_name**: Optional full name
    """
    try:
        user = auth_service.create_user(
            db=db,
            email=user_data.email,
            password=user_data.password,
            full_name=user_data.full_name
        )
        return UserResponse.from_orm(user)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@app.post("/auth/login", response_model=Token)
async def login(
    credentials: UserLogin,
    db: Session = Depends(get_db)
):
    """
    Login with email and password to get JWT access token
    
    - **email**: User's email
    - **password**: User's password
    
    Returns JWT token for authentication
    """
    user = auth_service.authenticate_user(db, credentials.email, credentials.password)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Create access token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = auth_service.create_access_token(
        data={"sub": user.id, "email": user.email},
        expires_delta=access_token_expires
    )
    
    return Token(
        access_token=access_token,
        token_type="bearer",
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60  # Convert to seconds
    )


@app.get("/auth/me", response_model=UserResponse)
async def get_current_user_info(
    current_user: User = Depends(get_current_user)
):
    """
    Get current authenticated user information
    
    Requires valid JWT token in Authorization header:
    `Authorization: Bearer <your_token>`
    """
    return UserResponse.from_orm(current_user)


# ===== DOCUMENT ENDPOINTS =====

@app.post("/upload")
async def upload_document(
    file: Annotated[UploadFile, File(description="A PDF or Image")],
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_optional_user),
    db: Session = Depends(get_db)
):
    """Upload and process a document (Authentication optional)"""
    allowed_types = ["application/pdf", "image/jpeg", "image/png"]
    if file.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail="Invalid file type.")
    
    # Use user_id if authenticated, otherwise use 'anonymous'
    user_id = current_user.id if current_user else None
    file_bytes = await file.read()
    
    # Save document metadata to database
    file_path = f"uploads/{user_id or 'anonymous'}/{file.filename}"
    doc_record = Document(
        user_id=user_id,
        filename=file.filename,
        file_path=file_path,
        file_size=len(file_bytes),
        file_type=file.content_type,
        status=DocumentStatus.PROCESSING
    )
    db.add(doc_record)
    db.commit()
    db.refresh(doc_record)
    
    try:
        # 1. Run Computer Vision & OCR (Phase 2)
        pages = vision_service.process_document(file_bytes, file.content_type)
        
        full_document_text = ""
        detected_elements = []
        
        # Process each page
        yolo_descriptions = []
        for page in pages:
            elements = vision_service.detect_layouts(page)
            detected_elements.extend(elements)
            
            # Aggregate all extracted text from the page
            for el in elements:
                if el.extracted_text:
                    full_document_text += el.extracted_text + " "
                    # Collect YOLO object descriptions separately
                    if el.element_type == "figure" and el.extracted_text.startswith("Detected "):
                        yolo_descriptions.append(el.extracted_text)

        # For images with minimal OCR text, prepend a visual summary from YOLO
        if yolo_descriptions and file.content_type.startswith("image/"):
            visual_summary = "This image contains the following detected objects: " + ", ".join(
                set(d.replace("Detected ", "").split(":")[0] for d in yolo_descriptions)
            ) + ". " + " ".join(yolo_descriptions)
            full_document_text = visual_summary + " " + full_document_text
        
        # Update document record with processing results
        doc_record.total_pages = len(pages)
        doc_record.total_elements_detected = len(detected_elements)
        doc_record.total_text_length = len(full_document_text)
        doc_record.status = DocumentStatus.COMPLETED
        doc_record.processed_at = datetime.utcnow()
        db.commit()
        
        # Generate document summary immediately
        summary = ""
        if full_document_text.strip():
            try:
                # Use first 3000 chars for summary
                preview_text = full_document_text[:3000]
                # For images, give a visual-oriented prompt
                is_image = file.content_type.startswith("image/")
                prompt = (
                    f"Describe what is shown in this image based on these detected elements and any text:\n\n{preview_text}"
                    if is_image else
                    f"Summarize this document in 3-4 sentences:\n\n{preview_text}"
                )
                if together_client:
                    response = together_client.chat.completions.create(
                        model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=200,
                        temperature=0.3
                    )
                    summary = response.choices[0].message.content.strip()
            except Exception as e:
                print(f"Summary generation failed: {e}")
                summary = f"Document uploaded successfully with {len(pages)} pages and {len(full_document_text)} characters."
                    
        # 2. Trigger Background Tasks for Hybrid RAG (Phase 3)
        if full_document_text.strip():
            # Traditional RAG: Chunk text, embed with Jina, and store in Qdrant with document_id
            background_tasks.add_task(
                rag_service.embed_and_store, 
                full_document_text, 
                "text_paragraph",
                str(doc_record.id)  # Pass document_id for filtering
            )
            
            # Reasoning RAG: Build hierarchical JSON tree with Gemini (PageIndex)
            background_tasks.add_task(
                pageindex_service.build_document_tree,
                full_document_text,
                file.filename
            )
            
            # ICDI-X: Build knowledge graph from document
            background_tasks.add_task(
                knowledge_graph.build_graph_from_document,
                full_document_text,
                file.filename
            )
        
        return {
            "document_id": str(doc_record.id),
            "filename": file.filename,
            "elements_detected": len(detected_elements),
            "pages": len(pages),
            "text_length": len(full_document_text),
            "summary": summary,
            "message": "Document successfully uploaded! Vision analysis complete. Text is being embedded into Qdrant, PageIndex tree, and Knowledge Graph are being built in the background."
        }
    
    except Exception as e:
        # Update document status to failed
        doc_record.status = DocumentStatus.FAILED
        doc_record.processing_error = str(e)
        db.commit()
        raise HTTPException(status_code=500, detail=f"Document processing failed: {str(e)}")


@app.post("/query", response_model=AnswerResponse)
async def query_documents(
    request: QueryRequest,
    current_user: User = Depends(get_optional_user),
    db: Session = Depends(get_db)
):
    """
    ICDI-X Query Endpoint (Requires Authentication)
    
    Advanced document QA using:
    - Input guardrails (prompt injection, toxicity detection)
    - Adaptive query planning
    - Multi-armed bandit retrieval selection
    - Graph reasoning for multi-hop queries
    - Quantum-inspired similarity (optional)
    - Information bottleneck filtering
    - Evidence verification
    - Output guardrails (hallucination detection, safety)
    """
    
    start_time = time.time()
    
    try:
        # Step 1: Input Guardrails - Validate query
        is_valid, reason = input_guardrail.validate(request.query, check_toxicity=True)
        if not is_valid:
            # Log flagged input to database
            flagged_conv = Conversation(
                user_id=current_user.id if current_user else None,
                query=request.query,
                response="[Query blocked by guardrails]",
                input_flagged=reason,
                confidence_score=0.0
            )
            db.add(flagged_conv)
            db.commit()
            
            raise HTTPException(
                status_code=400,
                detail=f"Query validation failed: {reason}"
            )
        
        # ── Multi-turn memory: fetch prior conversation context ──────────────
        conversation_history = ""
        if request.conversation_id:
            prior_convs = (
                db.query(Conversation)
                .filter(
                    Conversation.document_id == int(request.document_id) if request.document_id else True,
                    Conversation.user_id == (current_user.id if current_user else None)
                )
                .order_by(Conversation.created_at.desc())
                .limit(5)
                .all()
            )
            if prior_convs:
                history_lines = []
                for c in reversed(prior_convs):
                    history_lines.append(f"User: {c.query}\nAssistant: {c.response[:300]}...")
                conversation_history = "\n\n".join(history_lines)

        # Step 2: Execute ICDI-X retrieval pipeline
        retrieval_result = await retrieval_orchestrator.retrieve(
            query=request.query,
            document_id=request.document_id,
            use_graph_reasoning=request.use_graph_reasoning,
            use_ib_filtering=request.use_ib_filtering,
            use_mab=request.use_mab,
            use_quantum=request.use_quantum,
        )

        # ── Build citations from raw retrieval results ───────────────────────
        raw_results = retrieval_result.get("results", [])
        citations = [
            CitationModel(
                chunk_index=r.get("chunk_index", i),
                text_preview=r.get("text", "")[:150],
                source_type=r.get("source_type", "text_paragraph"),
                document_id=r.get("document_id", request.document_id),
                relevance_score=round(r.get("score", 0.0), 4),
            )
            for i, r in enumerate(raw_results[:5])
        ]

        # Check if we got any context — if not, still try to answer or explain
        if not retrieval_result.get("context") or not retrieval_result.get("context").strip():
            return AnswerResponse(
                query=request.query,
                answer="I could not find relevant content in the uploaded document for this query. This may happen if: (1) the document is still being indexed in the background — please wait 10-15 seconds and try again, or (2) the document doesn't contain information related to your question.",
                context="",
                method="no_context_fallback",
                citations=[],
                metadata={"warning": "No context found", "query": request.query}
            )

        # Step 2: Generate answer using Gemini with Together AI fallback
        history_block = (
            f"\n\nConversation history (last 5 turns):\n{conversation_history}\n"
            if conversation_history else ""
        )
        answer_prompt = f"""You are a helpful AI assistant analyzing a document. Answer the following question based on the provided context.{history_block}
Question: {request.query}

Context from the document:
{retrieval_result['context']}

Instructions:
1. Provide a clear, comprehensive answer in 2-3 paragraphs
2. Synthesize information from the context, don't just quote it
3. If the context doesn't fully answer the question, say what information is available
4. Use specific details and examples from the context
5. Be accurate and cite specific facts when possible
6. If there is conversation history, consider the prior context for follow-up questions

Answer:"""

        
        answer = None
        generation_error = None
        
        # Try Gemini API first
        if gemini_model:
            try:
                response = gemini_model.generate_content(answer_prompt)
                answer = response.text
                print("✅ Gemini generated answer successfully")
            except Exception as e:
                generation_error = str(e)
                print(f"⚠️ Gemini generation failed: {e}")
        
        # Fallback to Together AI if Gemini fails
        if not answer and together_client:
            try:
                print("🔄 Trying Together AI fallback...")
                
                response = together_client.chat.completions.create(
                    model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a precise assistant. Answer questions using only the provided context. Write 2-3 clear paragraphs."
                        },
                        {
                            "role": "user",
                            "content": f"Question: {request.query}\n\nContext: {retrieval_result['context']}\n\nProvide a clear answer:"
                        }
                    ],
                    max_tokens=400,
                    temperature=0.3
                )
                answer = response.choices[0].message.content.strip()
                print("✅ Together AI generated answer successfully")
            except Exception as e:
                print(f"⚠️ Together AI also failed: {e}")
                generation_error = f"Both Gemini and Together AI failed. Last error: {str(e)}"
        
        # Final fallback: Create a better structured answer from context
        if not answer:
            context_sentences = retrieval_result['context'].split('. ')
            # Take first 5 meaningful sentences
            meaningful_sentences = [s.strip() for s in context_sentences[:5] if len(s.strip()) > 20]
            answer = f"Based on the document content:\n\n" + '. '.join(meaningful_sentences) + '.'
            # Don't show technical errors to users - just use fallback answer
        
        # Step 3: Verify answer support
        from app.services.evidence_verifier_service import EvidenceVerifierService, Evidence
        verifier = EvidenceVerifierService()
        
        evidence_list = [Evidence(retrieval_result['context'], "retrieved_context")]
        verification = verifier.verify_answer_support(answer, evidence_list)
        
        # Extract confidence score
        confidence_score = verification.get('confidence_score', 0.5) * 100
        
        # Step 4: Output Guardrails - Validate answer
        is_valid, reason, adjusted_confidence, guardrail_metadata = output_guardrail.validate(
            answer=answer,
            evidence=[retrieval_result['context']],
            confidence_score=confidence_score,
            check_hallucination=True
        )
        
        if not is_valid:
            # Log blocked output to database
            blocked_conv = Conversation(
                user_id=current_user.id if current_user else None,
                query=request.query,
                response=answer,
                output_flagged=reason,
                confidence_score=adjusted_confidence,
                retrieval_method=retrieval_result['method'],
                processing_time_ms=int((time.time() - start_time) * 1000)
            )
            db.add(blocked_conv)
            db.commit()
            
            raise HTTPException(
                status_code=422,
                detail=f"Answer validation failed: {reason}. Confidence adjusted to {adjusted_confidence:.1f}%"
            )
        
        # Update confidence with guardrail adjustment
        confidence_score = adjusted_confidence
        
        # Step 5: Save conversation to database
        processing_time = int((time.time() - start_time) * 1000)
        
        conversation = Conversation(
            user_id=current_user.id if current_user else None,
            document_id=request.document_id,
            query=request.query,
            response=answer,
            confidence_score=confidence_score,
            retrieval_method=retrieval_result['method'],
            evidence_sources=retrieval_result.get('reasoning_paths', []),
            processing_time_ms=processing_time
        )
        db.add(conversation)
        db.commit()
        db.refresh(conversation)
        
        # Step 6: Return comprehensive response
        return AnswerResponse(
            query=request.query,
            answer=answer,
            context=retrieval_result['context'],
            method=retrieval_result['method'],
            query_plan=retrieval_result.get('query_plan'),
            reasoning_paths=retrieval_result.get('reasoning_paths', []),
            evidence_verification=verification,
            mab_statistics=retrieval_result.get('mab_stats'),
            citations=citations,
            metadata={
                "num_results": retrieval_result.get('num_results', 0),
                "num_paths": retrieval_result.get('num_paths', 0),
                "conversation_id": conversation.id,
                "confidence_score": confidence_score,
                "processing_time_ms": processing_time,
                "guardrails_passed": guardrail_metadata.get('checks_passed', [])
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        error_details = traceback.format_exc()
        print(f"Query error: {error_details}")  # Log to console
        raise HTTPException(
            status_code=500,
            detail=f"Query processing failed: {str(e)}"
        )


@app.get("/pipeline/summary", response_model=PipelineSummary)
async def get_pipeline_summary(current_user: User = Depends(get_current_user)):
    """Get ICDI-X pipeline status and statistics (Requires Authentication)"""
    summary = retrieval_orchestrator.get_pipeline_summary()
    
    kg_stats = knowledge_graph.get_graph_summary()
    
    return PipelineSummary(
        components=summary["components"],
        mab_statistics=summary["mab_statistics"],
        knowledge_graph_stats=kg_stats
    )


@app.get("/knowledge-graph/export")
async def export_knowledge_graph(current_user: User = Depends(get_current_user)):
    """Export the current knowledge graph in JSON format (Requires Authentication)"""
    graph_data = knowledge_graph.export_graph()
    
    return {
        "knowledge_graph": graph_data,
        "summary": knowledge_graph.get_graph_summary()
    }


# ===== CONVERSATION HISTORY ENDPOINTS =====

@app.get("/conversations/history")
async def get_conversation_history(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    limit: int = 50,
    offset: int = 0
):
    """
    Get user's conversation history
    
    - **limit**: Number of conversations to return (default 50)
    - **offset**: Pagination offset (default 0)
    """
    conversations = db.query(Conversation).filter(
        Conversation.user_id == current_user.id
    ).order_by(
        Conversation.created_at.desc()
    ).limit(limit).offset(offset).all()
    
    return {
        "conversations": [
            {
                "id": conv.id,
                "query": conv.query,
                "response": conv.response[:200] + "..." if len(conv.response) > 200 else conv.response,
                "confidence_score": conv.confidence_score,
                "retrieval_method": conv.retrieval_method,
                "created_at": conv.created_at,
                "processing_time_ms": conv.processing_time_ms
            }
            for conv in conversations
        ],
        "total": db.query(Conversation).filter(Conversation.user_id == current_user.id).count()
    }


@app.get("/conversations/{conversation_id}")
async def get_conversation(
    conversation_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get a specific conversation by ID"""
    conversation = db.query(Conversation).filter(
        Conversation.id == conversation_id,
        Conversation.user_id == current_user.id
    ).first()
    
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    return {
        "id": conversation.id,
        "query": conversation.query,
        "response": conversation.response,
        "confidence_score": conversation.confidence_score,
        "retrieval_method": conversation.retrieval_method,
        "vision_analysis": conversation.vision_analysis,
        "text_analysis": conversation.text_analysis,
        "fusion_analysis": conversation.fusion_analysis,
        "evidence_sources": conversation.evidence_sources,
        "created_at": conversation.created_at,
        "processing_time_ms": conversation.processing_time_ms,
        "input_flagged": conversation.input_flagged,
        "output_flagged": conversation.output_flagged
    }


@app.delete("/conversations/{conversation_id}")
async def delete_conversation(
    conversation_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Delete a conversation"""
    conversation = db.query(Conversation).filter(
        Conversation.id == conversation_id,
        Conversation.user_id == current_user.id
    ).first()
    
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    db.delete(conversation)
    db.commit()
    
    return {"message": "Conversation deleted successfully"}


# ===== MULTI-AGENT WEBSOCKET =====

@app.websocket("/chat")
async def websocket_chat_endpoint(websocket: WebSocket):
    """
    Multi-Agent Chat WebSocket
    
    Note: WebSocket authentication is handled via query parameter
    Usage: ws://localhost:8000/chat?token=YOUR_JWT_TOKEN
    """
    await ws_manager.connect(websocket)
    
    try:
        # Extract token from query params (optional authentication)
        query_params = dict(websocket.query_params)
        token = query_params.get("token")
        
        user = None
        if token:
            # Verify token
            from app.database.base import SessionLocal
            db = SessionLocal()
            try:
                user = auth_service.get_current_user(db, token)
                if user:
                    await websocket.send_text(f"✅ Authenticated as {user.email}")
                else:
                    await websocket.send_text("⚠️ Invalid token - continuing as guest")
            finally:
                db.close()
        else:
            await websocket.send_text("ℹ️ Connected as guest (no authentication)")
        
        while True:
            # Receive query from client
            query = await websocket.receive_text()
            
            # Input guardrails
            is_valid, reason = input_guardrail.validate(query, check_toxicity=False)
            if not is_valid:
                await websocket.send_text(f"❌ Query blocked: {reason}")
                continue
            
            # Process with multi-agent system
            await agent_service.process_query_live(query, websocket)
            
    except WebSocketDisconnect:
        await ws_manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await ws_manager.disconnect(websocket)


# ═══════════════════════════════════════════════════════════════════════════════
# ===== NOVEL ICDI-X FEATURES (beyond Perplexity & NotebookLM) =================
# ═══════════════════════════════════════════════════════════════════════════════


# ── 1. PROACTIVE INSIGHTS ENGINE ──────────────────────────────────────────────

@app.get("/insights/{doc_id}")
async def get_document_insights(
    doc_id: str,
    current_user: User = Depends(get_optional_user),
):
    """
    🔬 Proactive Insights Engine (ICDI-X novel feature)

    Automatically analyses the uploaded document and returns:
    - 5-7 key insights (IB-selected, most information-dense content)
    - 5 suggested questions (clickable in the frontend)
    - Key entities and themes

    Novel: Uses Information Bottleneck scoring to select the most
    information-dense chunks BEFORE the user asks anything.
    Neither Perplexity nor NotebookLM offers this.
    """
    try:
        # Retrieve broad document coverage (up to 30 chunks)
        chunks = await rag_service.retrieve_document_chunks(doc_id, limit=30)
        result = await insights_service.generate_insights(
            doc_id=doc_id,
            chunks=chunks,
            filename=f"document_{doc_id}",
        )
        return result
    except Exception as e:
        logger.error(f"Insights error for doc {doc_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Insights generation failed: {str(e)}")


# ── 2. STUDY GUIDE GENERATOR ──────────────────────────────────────────────────

@app.get("/studyguide/{doc_id}")
async def get_study_guide(
    doc_id: str,
    current_user: User = Depends(get_optional_user),
):
    """
    📚 Bloom's Taxonomy Study Guide Generator (ICDI-X novel feature)

    Converts any document into a complete study guide with:
    - 6-level Bloom's Taxonomy questions (Remember → Create)
    - Key vocabulary with plain-language definitions
    - Concept map (entity relationships)
    - Estimated study time

    Novel: No commercial document AI product generates Bloom's
    taxonomy-classified questions from uploaded content.
    """
    try:
        chunks = await rag_service.retrieve_document_chunks(doc_id, limit=30)
        result = await study_guide_service.generate_study_guide(
            doc_id=doc_id,
            chunks=chunks,
            filename=f"document_{doc_id}",
        )
        return result
    except Exception as e:
        logger.error(f"Study guide error for doc {doc_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Study guide generation failed: {str(e)}")


# ── 3. CROSS-DOCUMENT CONTRADICTION DETECTOR ──────────────────────────────────

@app.post("/contradictions")
async def detect_contradictions(
    doc_a_id: str,
    doc_b_id: str,
    topic: str = "",
    current_user: User = Depends(get_optional_user),
):
    """
    ⚡ Cross-Document Contradiction Detector (ICDI-X novel feature)

    Compares two uploaded documents and identifies:
    - Direct factual contradictions (with severity: high/medium/low)
    - Implied incompatibilities
    - Scope/methodology differences
    - Shared agreements

    Novel: No existing RAG product (Perplexity, NotebookLM, ChatPDF)
    offers structured cross-document contradiction detection.

    Use cases:
    - Research: compare two papers on the same hypothesis
    - Legal: detect contract vs regulation conflicts
    - Medical: flag conflicting clinical guidelines
    """
    try:
        query_topic = topic if topic else "main claims findings methodology conclusions"

        # Retrieve representative chunks from each document
        chunks_a = await rag_service.retrieve(query=query_topic, top_k=8, document_id=doc_a_id)
        chunks_b = await rag_service.retrieve(query=query_topic, top_k=8, document_id=doc_b_id)

        texts_a = [c["text"] for c in chunks_a]
        texts_b = [c["text"] for c in chunks_b]

        result = await contradiction_detector.detect_contradictions(
            doc_a_chunks=texts_a,
            doc_b_chunks=texts_b,
            doc_a_name=f"Document {doc_a_id}",
            doc_b_name=f"Document {doc_b_id}",
            topic=topic,
        )
        return result
    except Exception as e:
        logger.error(f"Contradiction detection error: {e}")
        raise HTTPException(status_code=500, detail=f"Contradiction detection failed: {str(e)}")


# ── 4. MULTI-DOCUMENT CROSS-SYNTHESIS ─────────────────────────────────────────

@app.post("/query_multi")
async def query_multiple_documents(
    request: QueryRequest,
    current_user: User = Depends(get_optional_user),
    db: Session = Depends(get_db),
):
    """
    🌐 Multi-Document Cross-Synthesis (ICDI-X novel feature)

    Query across multiple uploaded documents simultaneously.
    Retrieves relevant chunks from each doc_id in request.document_ids,
    merges by relevance score, and synthesises a unified answer.

    Novel: NotebookLM supports multi-doc but doesn't use IB+MAB+knowledge-graph
    for cross-document retrieval. ICDI-X applies the full pipeline per document
    then performs cross-document synthesis.

    Usage: POST /query_multi with document_ids: ["id1", "id2", "id3"]
    """
    if not request.document_ids or len(request.document_ids) < 2:
        raise HTTPException(
            status_code=400,
            detail="Provide at least 2 document IDs in 'document_ids' for cross-document synthesis.",
        )

    try:
        all_chunks = []
        doc_contexts = {}

        # Retrieve chunks from each document
        for doc_id in request.document_ids[:5]:  # cap at 5 docs
            doc_chunks = await rag_service.retrieve(
                query=request.query,
                top_k=6,
                document_id=doc_id,
            )
            doc_contexts[doc_id] = "\n\n".join(c["text"] for c in doc_chunks)
            all_chunks.extend(doc_chunks)

        # Sort merged chunks by relevance score
        all_chunks.sort(key=lambda x: x.get("score", 0), reverse=True)
        merged_context = "\n\n---\n\n".join(c["text"] for c in all_chunks[:15])

        if not merged_context.strip():
            raise HTTPException(status_code=404, detail="No content found in the specified documents.")

        # Build per-document summary for context
        doc_summary = "\n".join(
            f"[Doc {i+1} ({did})]: {ctx[:400]}..."
            for i, (did, ctx) in enumerate(doc_contexts.items())
        )

        synthesis_prompt = f"""You are an expert research analyst. A user has uploaded {len(request.document_ids)} documents and is asking a cross-document question.

QUESTION: {request.query}

RETRIEVED CONTENT ACROSS ALL DOCUMENTS:
{merged_context[:5000]}

DOCUMENT BREAKDOWN:
{doc_summary[:2000]}

Instructions:
1. Synthesise information from ALL documents to answer the question comprehensively
2. Note where documents agree and where they differ
3. Clearly attribute key points to specific documents when relevant (e.g., "Document 1 states...", "Document 2 argues...")
4. Provide a unified, coherent answer in 3-4 paragraphs

Answer:"""

        answer = None
        if gemini_model:
            try:
                resp = gemini_model.generate_content(synthesis_prompt)
                answer = resp.text
            except Exception as e:
                logger.warning(f"Gemini failed in query_multi: {e}")

        if not answer and together_client:
            try:
                resp = together_client.chat.completions.create(
                    model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
                    messages=[{"role": "user", "content": synthesis_prompt}],
                    max_tokens=1200,
                    temperature=0.3,
                )
                answer = resp.choices[0].message.content
            except Exception as e:
                logger.error(f"Together AI failed in query_multi: {e}")

        if not answer:
            answer = f"Cross-document synthesis: {merged_context[:500]}..."

        citations = [
            CitationModel(
                chunk_index=i,
                text_preview=c.get("text", "")[:150],
                source_type=c.get("source_type", "text_paragraph"),
                document_id=c.get("document_id"),
                relevance_score=round(c.get("score", 0.0), 4),
            )
            for i, c in enumerate(all_chunks[:8])
        ]

        return AnswerResponse(
            query=request.query,
            answer=answer,
            context=merged_context[:2000],
            method="multi_document_synthesis",
            citations=citations,
            metadata={
                "documents_queried": request.document_ids,
                "total_chunks_retrieved": len(all_chunks),
                "chunks_used": min(15, len(all_chunks)),
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Query multi error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Multi-document query failed: {str(e)}")


# ── 5. STREAMING SSE QUERY ────────────────────────────────────────────────────

@app.post("/query/stream")
async def query_stream(
    request: QueryRequest,
    current_user: User = Depends(get_optional_user),
):
    """
    ⚡ Streaming Query via Server-Sent Events (Perplexity-style)

    Streams the LLM response token-by-token so the UI feels instantaneous.
    Uses the same ICDI-X retrieval pipeline as /query, then streams
    the generation.

    Frontend: consume with EventSource or fetch + ReadableStream.
    Format: each SSE event is JSON: {"token": "...", "done": false}
            final event: {"token": "", "done": true, "method": "...", "citations": [...]}
    """

    async def generate() -> AsyncGenerator[str, None]:
        try:
            # Input guardrail
            is_valid, reason = input_guardrail.validate(request.query, check_toxicity=True)
            if not is_valid:
                yield f"data: {json_module.dumps({'error': reason, 'done': True})}\n\n"
                return

            # Retrieve context
            retrieval_result = await retrieval_orchestrator.retrieve(
                query=request.query,
                document_id=request.document_id,
                use_graph_reasoning=request.use_graph_reasoning,
                use_ib_filtering=request.use_ib_filtering,
                use_mab=request.use_mab,
                use_quantum=request.use_quantum,
            )

            context = retrieval_result.get("context", "")
            method = retrieval_result.get("method", "dense_vector")

            # Build citations
            raw_results = retrieval_result.get("results", [])
            citations_data = [
                {
                    "chunk_index": i,
                    "text_preview": r.get("text", "")[:120],
                    "source_type": r.get("source_type", "text_paragraph"),
                    "document_id": r.get("document_id", request.document_id),
                    "relevance_score": round(r.get("score", 0.0), 4),
                }
                for i, r in enumerate(raw_results[:5])
            ]

            if not context.strip():
                yield f"data: {json_module.dumps({'token': 'No relevant content found in the document. The document may still be indexing — please wait and retry.', 'done': True})}\n\n"
                return

            answer_prompt = f"""You are a helpful AI assistant. Answer the question based on the document context.

Question: {request.query}

Context:
{context[:3000]}

Answer concisely and accurately:"""

            # Try Gemini streaming
            streamed = False
            try:
                from google import genai as google_genai
                g_client = google_genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
                for chunk in g_client.models.generate_content_stream(
                    model="gemini-2.0-flash",
                    contents=answer_prompt,
                ):
                    if chunk.text:
                        yield f"data: {json_module.dumps({'token': chunk.text, 'done': False})}\n\n"
                        await asyncio.sleep(0)  # yield control
                streamed = True
            except Exception as e:
                logger.warning(f"Gemini stream failed: {e}")

            # Fallback: Together AI (non-streaming, send all at once)
            if not streamed and together_client:
                try:
                    resp = together_client.chat.completions.create(
                        model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
                        messages=[{"role": "user", "content": answer_prompt}],
                        max_tokens=800,
                        temperature=0.3,
                    )
                    text = resp.choices[0].message.content
                    # Simulate streaming by sending in chunks of 8 words
                    words = text.split()
                    for i in range(0, len(words), 8):
                        chunk_text = " ".join(words[i:i+8]) + " "
                        yield f"data: {json_module.dumps({'token': chunk_text, 'done': False})}\n\n"
                        await asyncio.sleep(0.03)
                except Exception as e:
                    logger.error(f"Together AI stream fallback failed: {e}")

            # Final event with metadata
            yield f"data: {json_module.dumps({'token': '', 'done': True, 'method': method, 'citations': citations_data})}\n\n"

        except Exception as e:
            logger.error(f"Stream error: {e}")
            yield f"data: {json_module.dumps({'error': str(e), 'done': True})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ── 6. KNOWLEDGE GRAPH D3 EXPORT ──────────────────────────────────────────────

@app.get("/knowledge-graph/d3")
async def export_knowledge_graph_d3(
    current_user: User = Depends(get_optional_user),
):
    """
    🕸️ Knowledge Graph — D3.js-compatible export (ICDI-X novel feature)

    Returns the knowledge graph as a D3 force-directed graph JSON:
    {
      "nodes": [{"id": str, "label": str, "group": str (entity_type)}, ...],
      "links": [{"source": str, "target": str, "relation": str, "value": float}, ...]
    }

    Frontend: render with D3 forceSimulation for an interactive graph.
    Novel: No commercial document AI tool exposes an interactive knowledge graph.
    """
    try:
        raw = knowledge_graph.export_graph()
        stats = knowledge_graph.get_graph_summary()

        # Convert to D3 format
        nodes = []
        links = []
        seen_nodes = set()

        for entity_key, entity_data in raw.get("entities", {}).items():
            node_id = entity_data.get("name", entity_key)
            if node_id not in seen_nodes:
                nodes.append({
                    "id": node_id,
                    "label": node_id,
                    "group": entity_data.get("type", "CONCEPT"),
                })
                seen_nodes.add(node_id)

        for rel in raw.get("relations", []):
            src = rel.get("source", "")
            tgt = rel.get("target", "")
            if src and tgt:
                links.append({
                    "source": src,
                    "target": tgt,
                    "relation": rel.get("relation_type", "RELATED_TO"),
                    "value": rel.get("confidence", 1.0),
                })

        return {
            "nodes": nodes,
            "links": links,
            "stats": stats,
        }
    except Exception as e:
        logger.error(f"KG D3 export error: {e}")
        raise HTTPException(status_code=500, detail=f"Knowledge graph export failed: {str(e)}")

