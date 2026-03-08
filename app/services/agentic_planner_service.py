"""
Agentic Query Planner for ICDI-X
=================================
Adaptive query routing that classifies queries and selects optimal retrieval strategy.

Different queries require different reasoning approaches:
- Simple queries: Direct vector retrieval
- Multi-hop queries: Graph reasoning
- Global queries: Hierarchical summarization

Research basis: Adaptive-RAG, FAIR-RAG, GARLIC
"""

import os
from typing import Dict, List, Optional
from enum import Enum
from loguru import logger
from google import genai


class QueryType(Enum):
    """Query complexity classification"""
    SIMPLE = "simple"              # Direct fact lookup
    MULTI_HOP = "multi_hop"        # Requires reasoning across entities
    COMPARISON = "comparison"       # Comparing multiple items
    AGGREGATION = "aggregation"     # Summary or statistics
    DEFINITIONAL = "definitional"   # "What is X?"
    PROCEDURAL = "procedural"       # "How to X?"


class RetrievalStrategy(Enum):
    """Retrieval strategy to use"""
    DENSE_VECTOR = "dense_vector"          # Standard vector similarity
    GRAPH_REASONING = "graph_reasoning"     # Multi-hop graph traversal
    HIERARCHICAL = "hierarchical"           # Tree-based retrieval
    HYBRID = "hybrid"                       # Combine multiple methods


class QueryPlan:
    """Represents an execution plan for a query"""
    def __init__(self, query: str):
        self.query = query
        self.query_type: Optional[QueryType] = None
        self.retrieval_strategy: Optional[RetrievalStrategy] = None
        self.max_hops: int = 1
        self.top_k: int = 5
        self.use_reranking: bool = False
        self.use_ib_filtering: bool = True
        self.use_quantum: bool = False
        self.confidence: float = 1.0
        self.reasoning: str = ""
    
    def __repr__(self):
        return (f"QueryPlan(type={self.query_type.value if self.query_type else 'unknown'}, "
                f"strategy={self.retrieval_strategy.value if self.retrieval_strategy else 'unknown'})")


class AgenticPlannerService:
    """
    Adaptive query planner using LLM-based classification.
    
    Architecture:
    1. Query Analysis: Classify query type and complexity
    2. Strategy Selection: Choose optimal retrieval approach
    3. Parameter Tuning: Set retrieval parameters
    4. Plan Generation: Create execution plan
    
    Inspired by Adaptive-RAG and FAIR-RAG frameworks.
    """
    
    def __init__(self):
        # Initialize Gemini for query analysis
        api_key = os.getenv("GEMINI_API_KEY")
        self.client = genai.Client(api_key=api_key) if api_key else None
        
        logger.info("Agentic Planner Service initialized")
    
    def analyze_query_complexity(self, query: str) -> QueryType:
        """
        Classify query into type categories.
        
        Args:
            query: User query
            
        Returns:
            QueryType classification
        """
        if not self.client:
            return self._rule_based_classification(query)
        
        try:
            prompt = f"""Classify this query into ONE of these categories:

Query: {query}

Categories:
1. SIMPLE - Direct fact lookup (e.g., "What is X?", "Who created Y?")
2. MULTI_HOP - Requires connecting multiple pieces of information
3. COMPARISON - Comparing multiple items (e.g., "X vs Y", "differences between...")
4. AGGREGATION - Summary or statistics (e.g., "summarize", "how many", "list all")
5. DEFINITIONAL - Asking for definition or explanation
6. PROCEDURAL - How-to questions (e.g., "how to", "steps to")

Respond with ONLY the category name (e.g., "MULTI_HOP").
"""
            
            response = self.client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )
            
            category = response.text.strip().upper()
            
            # Map to QueryType
            type_map = {
                "SIMPLE": QueryType.SIMPLE,
                "MULTI_HOP": QueryType.MULTI_HOP,
                "COMPARISON": QueryType.COMPARISON,
                "AGGREGATION": QueryType.AGGREGATION,
                "DEFINITIONAL": QueryType.DEFINITIONAL,
                "PROCEDURAL": QueryType.PROCEDURAL
            }
            
            return type_map.get(category, QueryType.SIMPLE)
            
        except Exception as e:
            logger.warning(f"LLM classification failed: {e}, using rule-based fallback")
            return self._rule_based_classification(query)
    
    def _rule_based_classification(self, query: str) -> QueryType:
        """Fallback rule-based classification"""
        query_lower = query.lower()
        
        # Multi-hop indicators
        multi_hop_keywords = ["what dataset", "which method", "how did", "relationship between"]
        if any(kw in query_lower for kw in multi_hop_keywords):
            return QueryType.MULTI_HOP
        
        # Comparison indicators
        if any(word in query_lower for word in ["vs", "versus", "compare", "difference"]):
            return QueryType.COMPARISON
        
        # Aggregation indicators
        if any(word in query_lower for word in ["summarize", "list", "all", "how many"]):
            return QueryType.AGGREGATION
        
        # Definitional
        if query_lower.startswith("what is") or query_lower.startswith("define"):
            return QueryType.DEFINITIONAL
        
        # Procedural
        if query_lower.startswith("how to") or "steps" in query_lower:
            return QueryType.PROCEDURAL
        
        return QueryType.SIMPLE
    
    def select_retrieval_strategy(self, query_type: QueryType) -> RetrievalStrategy:
        """
        Select optimal retrieval strategy based on query type.
        
        Args:
            query_type: Classified query type
            
        Returns:
            Recommended retrieval strategy
        """
        strategy_map = {
            QueryType.SIMPLE: RetrievalStrategy.DENSE_VECTOR,
            QueryType.MULTI_HOP: RetrievalStrategy.GRAPH_REASONING,
            QueryType.COMPARISON: RetrievalStrategy.HYBRID,
            QueryType.AGGREGATION: RetrievalStrategy.HIERARCHICAL,
            QueryType.DEFINITIONAL: RetrievalStrategy.DENSE_VECTOR,
            QueryType.PROCEDURAL: RetrievalStrategy.HIERARCHICAL
        }
        
        return strategy_map.get(query_type, RetrievalStrategy.DENSE_VECTOR)
    
    def estimate_computational_cost(self, plan: QueryPlan) -> Dict:
        """
        Estimate computational cost of the plan.
        
        Args:
            plan: Query plan
            
        Returns:
            Dict with cost estimates
        """
        # Base costs (relative units)
        strategy_costs = {
            RetrievalStrategy.DENSE_VECTOR: 1.0,
            RetrievalStrategy.GRAPH_REASONING: 3.0,
            RetrievalStrategy.HIERARCHICAL: 2.0,
            RetrievalStrategy.HYBRID: 2.5
        }
        
        base_cost = strategy_costs.get(plan.retrieval_strategy, 1.0)
        
        # Multiply by hops for graph reasoning
        if plan.retrieval_strategy == RetrievalStrategy.GRAPH_REASONING:
            base_cost *= plan.max_hops
        
        # Add costs for optional components
        if plan.use_quantum:
            base_cost *= 1.3
        
        if plan.use_reranking:
            base_cost *= 1.2
        
        # Estimate tokens
        estimated_tokens = plan.top_k * 200  # Assume 200 tokens per retrieved chunk
        
        return {
            "relative_compute_cost": base_cost,
            "estimated_latency_ms": base_cost * 500,  # Rough estimate
            "estimated_tokens": estimated_tokens
        }
    
    def create_plan(self, query: str) -> QueryPlan:
        """
        Create comprehensive execution plan for query.
        
        This is the main entry point for query planning.
        
        Args:
            query: User query
            
        Returns:
            Complete QueryPlan
        """
        logger.info(f"Creating query plan for: {query}")
        
        plan = QueryPlan(query)
        
        # Step 1: Analyze query complexity
        plan.query_type = self.analyze_query_complexity(query)
        logger.info(f"Query classified as: {plan.query_type.value}")
        
        # Step 2: Select retrieval strategy
        plan.retrieval_strategy = self.select_retrieval_strategy(plan.query_type)
        logger.info(f"Selected strategy: {plan.retrieval_strategy.value}")
        
        # Step 3: Set parameters based on query type
        if plan.query_type == QueryType.MULTI_HOP:
            plan.max_hops = 3
            plan.top_k = 10
            plan.use_reranking = True
            plan.reasoning = "Multi-hop reasoning requires graph traversal with reranking"
        
        elif plan.query_type == QueryType.COMPARISON:
            plan.top_k = 8
            plan.use_quantum = True  # Quantum helps with subtle semantic differences
            plan.reasoning = "Comparison queries benefit from quantum similarity"
        
        elif plan.query_type == QueryType.AGGREGATION:
            plan.top_k = 15
            plan.use_ib_filtering = True
            plan.reasoning = "Aggregation requires broader retrieval with IB compression"
        
        elif plan.query_type == QueryType.SIMPLE:
            plan.top_k = 3
            plan.use_ib_filtering = False
            plan.reasoning = "Simple lookup requires minimal retrieval"
        
        else:
            plan.top_k = 5
            plan.reasoning = "Standard retrieval parameters"
        
        # Step 4: Estimate cost
        cost_estimate = self.estimate_computational_cost(plan)
        plan.confidence = 0.9  # High confidence in classification
        
        logger.info(f"Plan created: {plan}")
        logger.info(f"Estimated cost: {cost_estimate['relative_compute_cost']:.1f}x baseline")
        
        return plan
    
    def adaptive_replanning(self, original_plan: QueryPlan, 
                           retrieval_results: List, 
                           quality_threshold: float = 0.5) -> Optional[QueryPlan]:
        """
        Adaptively replan if initial results are poor.
        
        This implements FAIR-RAG's iterative refinement concept.
        
        Args:
            original_plan: Original query plan
            retrieval_results: Results from initial retrieval
            quality_threshold: Minimum quality to accept results
            
        Returns:
            New plan if replanning needed, None otherwise
        """
        if not retrieval_results:
            logger.info("No results returned - replanning with broader strategy")
            
            # Create more aggressive plan
            new_plan = QueryPlan(original_plan.query)
            new_plan.query_type = original_plan.query_type
            new_plan.retrieval_strategy = RetrievalStrategy.HYBRID
            new_plan.top_k = original_plan.top_k * 2
            new_plan.use_quantum = True
            new_plan.reasoning = "Broadening search due to no initial results"
            
            return new_plan
        
        # Check result quality (simplified - in production, use LLM evaluation)
        avg_score = sum(r.get('score', 0) for r in retrieval_results) / len(retrieval_results)
        
        if avg_score < quality_threshold:
            logger.info(f"Low quality results (avg={avg_score:.2f}) - replanning")
            
            # Upgrade strategy
            strategy_upgrades = {
                RetrievalStrategy.DENSE_VECTOR: RetrievalStrategy.HYBRID,
                RetrievalStrategy.HIERARCHICAL: RetrievalStrategy.GRAPH_REASONING,
                RetrievalStrategy.GRAPH_REASONING: RetrievalStrategy.HYBRID
            }
            
            new_plan = QueryPlan(original_plan.query)
            new_plan.query_type = original_plan.query_type
            new_plan.retrieval_strategy = strategy_upgrades.get(
                original_plan.retrieval_strategy,
                RetrievalStrategy.HYBRID
            )
            new_plan.top_k = min(original_plan.top_k * 1.5, 20)
            new_plan.reasoning = f"Upgraded strategy due to low quality (score={avg_score:.2f})"
            
            return new_plan
        
        logger.info(f"Results quality acceptable (avg={avg_score:.2f})")
        return None
    
    def explain_plan(self, plan: QueryPlan) -> Dict:
        """
        Generate human-readable explanation of the plan.
        
        Args:
            plan: Query plan
            
        Returns:
            Dictionary with explanation
        """
        cost = self.estimate_computational_cost(plan)
        
        return {
            "query": plan.query,
            "classification": plan.query_type.value if plan.query_type else "unknown",
            "strategy": plan.retrieval_strategy.value if plan.retrieval_strategy else "unknown",
            "parameters": {
                "top_k": plan.top_k,
                "max_hops": plan.max_hops,
                "use_reranking": plan.use_reranking,
                "use_ib_filtering": plan.use_ib_filtering,
                "use_quantum": plan.use_quantum
            },
            "reasoning": plan.reasoning,
            "cost_estimate": cost,
            "confidence": plan.confidence
        }
