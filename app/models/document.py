"""
Document Model
Stores metadata about uploaded documents
"""
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text, Enum as SQLEnum
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.database.base import Base
import enum


class DocumentStatus(str, enum.Enum):
    """Document processing status"""
    UPLOADED = "uploaded"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class Document(Base):
    __tablename__ = "documents"
    
    # Primary Key
    id = Column(Integer, primary_key=True, index=True)
    
    # Owner Reference (nullable for anonymous uploads)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True, index=True)
    
    # Document Metadata
    filename = Column(String(255), nullable=False)
    file_path = Column(String(500), nullable=False)  # Path where file is stored
    file_size = Column(Integer, nullable=True)  # Size in bytes
    file_type = Column(String(50), nullable=True)  # PDF, JPG, PNG
    
    # Processing Status
    status = Column(SQLEnum(DocumentStatus), default=DocumentStatus.UPLOADED)
    processing_error = Column(Text, nullable=True)
    
    # Processing Results
    total_pages = Column(Integer, nullable=True)
    total_text_length = Column(Integer, nullable=True)
    total_elements_detected = Column(Integer, nullable=True)  # YOLO detections
    
    # Timestamps
    uploaded_at = Column(DateTime(timezone=True), server_default=func.now())
    processed_at = Column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    owner = relationship("User", back_populates="documents")
    conversations = relationship("Conversation", back_populates="document", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Document(id={self.id}, filename='{self.filename}', status='{self.status}')>"
