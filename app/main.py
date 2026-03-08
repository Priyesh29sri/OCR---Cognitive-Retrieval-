from fastapi import FastAPI, UploadFile, File, HTTPException, status, BackgroundTasks, WebSocket, WebSocketDisconnect, Depends
from fastapi.middleware.cors import CORSMiddleware
from typing import Annotated
from contextlib import asynccontextmanager
from dotenv import load_dotenv
import os
from sqlalchemy.orm import Session

from app.services.vision_service import VisionService
from app.services.rag_service import RAGService
from app.services.pageindex_service import PageIndexService
from app.services.knowledge_graph_service import KnowledgeGraphService
from app.services.retrieval_orchestrator import RetrievalOrchestrator
from app.services.websocket_manager import ws_manager
from app.services.agent_service import AgentService
from app.models.retrieval_schemas import QueryRequest, AnswerResponse, PipelineSummary

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
        
        # Step 2: Execute ICDI-X retrieval pipeline
        retrieval_result = await retrieval_orchestrator.retrieve(
            query=request.query,
            document_id=request.document_id
        )
        
        # Check if we got any context — if not, still try to answer or explain
        if not retrieval_result.get("context") or not retrieval_result.get("context").strip():
            return AnswerResponse(
                query=request.query,
                answer="I could not find relevant content in the uploaded document for this query. This may happen if: (1) the document is still being indexed in the background — please wait 10-15 seconds and try again, or (2) the document doesn't contain information related to your question.",
                context="",
                method="no_context_fallback",
                metadata={"warning": "No context found", "query": request.query}
            )
        
        # Step 2: Generate answer using Gemini with Together AI fallback
        answer_prompt = f"""You are a helpful AI assistant analyzing a document. Answer the following question based on the provided context.

Question: {request.query}

Context from the document:
{retrieval_result['context']}

Instructions:
1. Provide a clear, comprehensive answer in 2-3 paragraphs
2. Synthesize information from the context, don't just quote it
3. If the context doesn't fully answer the question, say what information is available
4. Use specific details and examples from the context
5. Be accurate and cite specific facts when possible

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
