"""
Pydantic Models for ICDI-X Retrieval Components
===============================================
Data models for all retrieval services.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from enum import Enum


# Enums
class QueryTypeEnum(str, Enum):
    """Query complexity types"""
    SIMPLE = "simple"
    MULTI_HOP = "multi_hop"
    COMPARISON = "comparison"
    AGGREGATION = "aggregation"
    DEFINITIONAL = "definitional"
    PROCEDURAL = "procedural"


class RetrievalStrategyEnum(str, Enum):
    """Retrieval strategies"""
    DENSE_VECTOR = "dense_vector"
    GRAPH_REASONING = "graph_reasoning"
    HIERARCHICAL = "hierarchical"
    HYBRID = "hybrid"


# Request/Response Models
class QueryRequest(BaseModel):
    """Query request model"""
    query: str = Field(..., description="User query string")
    document_id: Optional[str] = Field(None, description="Optional document ID to search within")
    use_graph_reasoning: bool = Field(True, description="Enable graph reasoning")
    use_quantum: bool = Field(False, description="Enable quantum retrieval")
    use_ib_filtering: bool = Field(True, description="Enable information bottleneck filtering")
    use_mab: bool = Field(True, description="Enable Multi-Armed Bandit retrieval selection")
    top_k: Optional[int] = Field(5, description="Number of results to retrieve")


class EvidenceModel(BaseModel):
    """Evidence piece model"""
    text: str
    source: str
    confidence: float = 1.0


class QueryPlanModel(BaseModel):
    """Query execution plan"""
    query: str
    query_type: Optional[QueryTypeEnum]
    retrieval_strategy: Optional[RetrievalStrategyEnum]
    max_hops: int = 1
    top_k: int = 5
    use_reranking: bool = False
    use_ib_filtering: bool = True
    use_quantum: bool = False
    confidence: float = 1.0
    reasoning: str = ""


class ReasoningPathModel(BaseModel):
    """Reasoning path through knowledge graph"""
    path_summary: str
    confidence: float
    entities: List[str]
    relations: List[str]


class RetrievalResultModel(BaseModel):
    """Retrieval result"""
    context: str
    method: str
    num_results: Optional[int] = 0
    reasoning_paths: Optional[List[str]] = []
    query_plan: Optional[Dict] = None
    evidence: Optional[List[EvidenceModel]] = []


class AnswerResponse(BaseModel):
    """Final answer response"""
    query: str
    answer: str
    context: str
    method: str
    query_plan: Optional[Dict] = None  # Changed from QueryPlanModel to Dict for flexibility
    reasoning_paths: Optional[List[str]] = []
    evidence_verification: Optional[Dict] = None
    mab_statistics: Optional[Dict] = None
    metadata: Optional[Dict] = None


class KnowledgeGraphStats(BaseModel):
    """Knowledge graph statistics"""
    num_entities: int
    num_relations: int
    entity_types: List[str]
    relation_types: List[str]


class PipelineSummary(BaseModel):
    """Pipeline component summary"""
    components: Dict[str, str]
    mab_statistics: Dict
    knowledge_graph_stats: Optional[KnowledgeGraphStats] = None
