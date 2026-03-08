from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models
import uuid

class VectorRepository:
    def __init__(self):
        # Use persistent storage on disk instead of in-memory
        # This ensures data persists across requests and server restarts
        self.db_client = AsyncQdrantClient(path="./qdrant_data")
        self.collection_name = "document_knowledgebase"

    async def create_collection(self, size: int = 1536) -> None:
        """Initializes the database table (collection) for OpenAI embeddings (size 1536)"""
        # We tell Qdrant to use Cosine Similarity to compare our vectors
        vectors_config = models.VectorParams(
            size=size, 
            distance=models.Distance.COSINE
        )
        
        # Check if collection exists, if not, create it
        if not await self.db_client.collection_exists(self.collection_name):
            await self.db_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=vectors_config,
            )

    async def store_chunk(self, embedding_vector: list[float], original_text: str, source_type: str, document_id: str = None) -> None:
        """Stores an embedding vector and its original text into Qdrant."""
        point_id = str(uuid.uuid4())
        
        payload = {
            "source_type": source_type,  # e.g., "text", "table", "figure"
            "original_text": original_text
        }
        
        # Add document_id if provided
        if document_id:
            payload["document_id"] = document_id
        
        await self.db_client.upsert(
            collection_name=self.collection_name,
            points=[
                models.PointStruct(
                    id=point_id,
                    vector=embedding_vector,
                    payload=payload,
                )
            ],
        )

    async def search(self, query_vector: list[float], limit: int = 3, document_id: str = None) -> list:
        """Performs a Cosine Similarity semantic search to find the closest matches."""
        query_filter = None
        
        # Filter by document_id if provided
        if document_id:
            query_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="document_id",
                        match=models.MatchValue(value=document_id)
                    )
                ]
            )
        
        response = await self.db_client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=limit,
            query_filter=query_filter,
        )
        return response.points
