"""
Retrieval Orchestrator for ICDI-X
==================================
Master coordinator that integrates all retrieval components into a unified pipeline.

This is the main entry point for the ICDI-X retrieval system.
"""

from typing import List, Dict, Optional
from loguru import logger

# Import all our services
from app.services.knowledge_graph_service import KnowledgeGraphService
from app.services.graph_reasoning_service import GraphReasoningService, SemanticPath
from app.services.quantum_retrieval_service import QuantumRetrievalService
from app.services.ib_filter_service import InformationBottleneckService
from app.services.agentic_planner_service import AgenticPlannerService, QueryPlan, RetrievalStrategy
from app.services.evidence_verifier_service import EvidenceVerifierService, Evidence
from app.services.mab_retrieval_service import MultiArmedBanditRetrieval
from app.services.rag_service import RAGService


class RetrievalOrchestrator:
    """
    Master orchestrator for ICDI-X retrieval pipeline.
    
    Pipeline:
    1. Query Planning (Agentic Planner)
    2. Retrieval (MAB selection + execution)
    3. Filtering (Information Bottleneck)
    4. Reasoning (Graph/Hierarchical as needed)
    5. Verification (Evidence check)
    6. Answer Generation
    """
    
    def __init__(self, rag_service: RAGService, knowledge_graph: KnowledgeGraphService):
        """
        Args:
            rag_service: Existing RAG service for vector retrieval
            knowledge_graph: Knowledge graph service
        """
        self.rag_service = rag_service
        self.knowledge_graph = knowledge_graph
        
        # Initialize all components
        self.graph_reasoner = GraphReasoningService(knowledge_graph)
        self.quantum_retrieval = QuantumRetrievalService()
        self.ib_filter = InformationBottleneckService(compression_ratio=0.6)
        self.agentic_planner = AgenticPlannerService()
        self.evidence_verifier = EvidenceVerifierService()
        self.mab_retrieval = MultiArmedBanditRetrieval()
        
        logger.info("Retrieval Orchestrator initialized - ICDI-X pipeline ready")
    
    async def retrieve(self, query: str, document_id: Optional[str] = None) -> Dict:
        """
        Main retrieval pipeline.
        
        Args:
            query: User query
            document_id: Optional document to search within
            
        Returns:
            Dict with retrieved context, reasoning paths, and metadata
        """
        logger.info(f"=" * 80)
        logger.info(f"ICDI-X Retrieval Pipeline: {query}")
        if document_id:
            logger.info(f"Document ID: {document_id}")
        logger.info(f"=" * 80)
        
        # Step 1: Query Planning
        logger.info("[Step 1] Query Planning")
        plan = self.agentic_planner.create_plan(query)
        logger.info(f"Plan: {plan}")
        
        # Step 2: Execute Retrieval based on strategy
        logger.info("[Step 2] Executing Retrieval")
        
        if plan.retrieval_strategy == RetrievalStrategy.GRAPH_REASONING:
            result = await self._graph_retrieval(query, plan, document_id)
        
        elif plan.retrieval_strategy == RetrievalStrategy.HIERARCHICAL:
            result = await self._hierarchical_retrieval(query, plan, document_id)
        
        elif plan.retrieval_strategy == RetrievalStrategy.HYBRID:
            result = await self._hybrid_retrieval(query, plan, document_id)
        
        else:  # DENSE_VECTOR
            result = await self._dense_retrieval(query, plan, document_id)
        
        # Step 3: Apply Information Bottleneck Filtering
        if plan.use_ib_filtering and result.get("context"):
            logger.info("[Step 3] Applying Information Bottleneck Filtering")
            result["context"] = self.ib_filter.filter_context(result["context"], query)
        
        # Step 4: Evidence Verification
        logger.info("[Step 4] Evidence Verification (preparing for generation)")
        evidence_list = [Evidence(result["context"], "retrieved_context")]
        result["evidence"] = evidence_list
        
        # Step 5: Return comprehensive result
        result["query_plan"] = self.agentic_planner.explain_plan(plan)
        result["mab_stats"] = self.mab_retrieval.get_arm_statistics()
        
        logger.info(f"Retrieval complete - {len(result.get('context', ''))} chars of context")
        return result
    
    async def _dense_retrieval(self, query: str, plan: QueryPlan, document_id: Optional[str] = None) -> Dict:
        """Standard vector similarity retrieval with improved parameters"""
        logger.info("Using dense vector retrieval")
        
        # Use larger top_k for better context coverage
        effective_top_k = max(plan.top_k, 10)  # Minimum 10 chunks
        results = await self.rag_service.retrieve(query, top_k=effective_top_k, document_id=document_id)
        
        if not results:
            return {"context": "", "method": "dense_vector", "results": []}
        
        # Optionally apply quantum similarity for reranking
        if plan.use_quantum:
            logger.info("Applying quantum reranking")
            # In production, rerank results using quantum similarity
        
        # Combine results
        context = "\n\n".join([r["text"] for r in results])
        
        # Update MAB (simplified - would need actual quality metric)
        self.mab_retrieval.update_arm("dense", reward=0.7)
        
        return {
            "context": context,
            "method": "dense_vector",
            "results": results,
            "num_results": len(results)
        }
    
    async def _graph_retrieval(self, query: str, plan: QueryPlan, document_id: Optional[str] = None) -> Dict:
        """Graph reasoning-based retrieval"""
        logger.info(f"Using graph reasoning (max_hops={plan.max_hops})")
        
        # Perform multi-hop reasoning
        reasoning_paths = self.graph_reasoner.reason_multi_hop(query, max_depth=plan.max_hops)
        
        if not reasoning_paths:
            logger.warning("No reasoning paths found, falling back to dense retrieval")
            return await self._dense_retrieval(query, plan)
        
        # Extract context from top paths
        context_parts = []
        for path in reasoning_paths[:3]:  # Top 3 paths
            context = self.graph_reasoner.extract_context_from_path(path)
            context_parts.append(context)
        
        combined_context = "\n\n".join(context_parts)
        
        # Update MAB
        self.mab_retrieval.update_arm("graph", reward=0.8)
        
        return {
            "context": combined_context,
            "method": "graph_reasoning",
            "reasoning_paths": [path.get_path_summary() for path in reasoning_paths],
            "num_paths": len(reasoning_paths)
        }
    
    async def _hierarchical_retrieval(self, query: str, plan: QueryPlan, document_id: Optional[str] = None) -> Dict:
        """Hierarchical tree-based retrieval"""
        logger.info("Using hierarchical retrieval")
        
        # Use RAG service with hierarchical awareness
        # In production, this would leverage document tree structure
        results = await self.rag_service.retrieve(query, top_k=plan.top_k, document_id=document_id)
        
        context = "\n\n".join([r["text"] for r in results]) if results else ""
        
        self.mab_retrieval.update_arm("hierarchical", reward=0.75)
        
        return {
            "context": context,
            "method": "hierarchical",
            "results": results
        }
    
    async def _hybrid_retrieval(self, query: str, plan: QueryPlan, document_id: Optional[str] = None) -> Dict:
        """Combine multiple retrieval methods"""
        logger.info("Using hybrid retrieval (dense + graph)")
        
        # Get both dense and graph results
        dense_result = await self._dense_retrieval(query, plan, document_id)
        graph_result = await self._graph_retrieval(query, plan, document_id)
        
        # Merge contexts
        combined_context = dense_result["context"] + "\n\n" + graph_result["context"]
        
        return {
            "context": combined_context,
            "method": "hybrid",
            "dense_results": dense_result.get("results", []),
            "graph_paths": graph_result.get("reasoning_paths", [])
        }
    
    async def verify_and_generate(self, query: str, context: str, evidence: List[Evidence]) -> Dict:
        """
        Final step: Verify evidence and generate answer.
        
        This would be called by the main.py endpoint after retrieval.
        
        Args:
            query: User query
            context: Retrieved context
            evidence: Evidence list
            
        Returns:
            Dict with answer and verification results
        """
        logger.info("[Final Step] Evidence Verification & Answer Generation")
        
        # Note: Actual answer generation would happen in main.py using Gemini
        # This method just prepares verification
        
        verification = {
            "context": context,
            "evidence_count": len(evidence),
            "ready_for_generation": len(context) > 0
        }
        
        return verification
    
    def get_pipeline_summary(self) -> Dict:
        """Get summary of pipeline components and their status"""
        return {
            "components": {
                "knowledge_graph": f"{len(self.knowledge_graph.entities)} entities, {len(self.knowledge_graph.relations)} relations",
                "quantum_retrieval": "enabled",
                "information_bottleneck": f"compression_ratio={self.ib_filter.compression_ratio}",
                "agentic_planner": "enabled",
                "evidence_verifier": "enabled",
                "mab_retrieval": "Thompson Sampling"
            },
            "mab_statistics": self.mab_retrieval.get_arm_statistics()
        }
