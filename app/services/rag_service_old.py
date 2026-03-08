import os
from app.repositories.vector_repository import VectorRepository
from sentence_transformers import SentenceTransformer
import torch

class RAGService:
    def __init__(self):
        self.vector_repo = VectorRepository()
        self._embedder = None
    
    @property
    def embedder(self):
        """Lazy initialization of embeddings model"""
        if self._embedder is None:
            # Using sentence-transformers for stable, production-ready embeddings
            # all-MiniLM-L6-v2 produces 384-dimensional vectors
            self._embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        return self._embedder

    async def initialize(self):
        """Creates the Qdrant collection on startup."""
        # all-MiniLM-L6-v2 embeddings have a vector size of 384
        await self.vector_repo.create_collection(size=384)

    def chunk_text(self, text: str, chunk_size: int = 500) -> list[str]:
        """A simple chunking strategy."""
        words = text.split(" ")
        chunks = []
        current_chunk = []
        current_length = 0
        
        for word in words:
            current_length += len(word) + 1
            if current_length > chunk_size:
                chunks.append(" ".join(current_chunk))
                current_chunk = [word]
                current_length = len(word)
            else:
                current_chunk.append(word)
                
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        return chunks

    async def embed_and_store(self, text: str, element_type: str = "text_paragraph"):
        """Converts text to numbers using the local Jina model and saves to Qdrant."""
        chunks = self.chunk_text(text)
        
        for chunk in chunks:
            # 1. Use the local Jina model to create the embedding vector
            # We use .tolist() to convert the PyTorch tensor to standard Python floats
            embedding_vector = self.embedder.encode(chunk).tolist()
            
            # 2. Save to Qdrant Database
            await self.vector_repo.store_chunk(
                embedding_vector=embedding_vector,
                original_text=chunk,
                source_type=element_type
            )