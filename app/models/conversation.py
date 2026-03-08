"""
Conversation Model
Stores query-response pairs with confidence scores
"""
from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, Float, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.database.base import Base


class Conversation(Base):
    __tablename__ = "conversations"
    
    # Primary Key
    id = Column(Integer, primary_key=True, index=True)
    
    # User Reference (nullable for anonymous users)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True, index=True)
    
    # Document Reference (optional - query may not be document-specific)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=True, index=True)
    
    # Query & Response
    query = Column(Text, nullable=False)
    response = Column(Text, nullable=False)
    
    # Confidence & Metadata
    confidence_score = Column(Float, nullable=True)  # 0.0 to 100.0
    retrieval_method = Column(String(50), nullable=True)  # dense, graph, hierarchical, quantum
    
    # Agent Information (for multi-agent system)
    vision_analysis = Column(Text, nullable=True)
    text_analysis = Column(Text, nullable=True)
    fusion_analysis = Column(Text, nullable=True)
    
    # Performance Metrics
    processing_time_ms = Column(Integer, nullable=True)  # Response time in milliseconds
    tokens_used = Column(Integer, nullable=True)  # LLM tokens consumed
    
    # Evidence & Sources
    evidence_sources = Column(JSON, nullable=True)  # List of source chunks used
    
    # Guardrail Flags
    input_flagged = Column(String(255), nullable=True)  # Reason if input was flagged
    output_flagged = Column(String(255), nullable=True)  # Reason if output was flagged
    
    # Timestamp
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    user = relationship("User", back_populates="conversations")
    document = relationship("Document", back_populates="conversations")
    
    def __repr__(self):
        return f"<Conversation(id={self.id}, query='{self.query[:50]}...', confidence={self.confidence_score})>"
