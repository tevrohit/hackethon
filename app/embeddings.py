"""
FastAPI endpoints for embeddings with Ollama and Qdrant integration.
Handles document chunking, embedding generation, and vector search.
"""

import asyncio
import logging
import os
from typing import List, Dict, Any, Optional
import httpx
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import ResponseHandlingException
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "bge-large")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
COLLECTION_NAME = "course_kb"

# Pydantic models
class UpsertRequest(BaseModel):
    doc_id: str = Field(..., description="Unique document identifier")
    text: str = Field(..., description="Text content to embed")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

class UpsertResponse(BaseModel):
    success: bool
    doc_id: str
    chunks_processed: int
    message: str

class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query text")
    k: int = Field(default=5, ge=1, le=100, description="Number of results to return")

class SearchResult(BaseModel):
    id: str
    score: float
    payload: Dict[str, Any]

class SearchResponse(BaseModel):
    results: List[SearchResult]
    query: str
    total_found: int

# Initialize FastAPI app
app = FastAPI(title="Embeddings API", version="1.0.0")

# Initialize Qdrant client
try:
    qdrant_client = QdrantClient(url=QDRANT_URL)
    logger.info(f"Connected to Qdrant at {QDRANT_URL}")
except Exception as e:
    logger.error(f"Failed to connect to Qdrant: {e}")
    qdrant_client = None

class EmbeddingService:
    """Service class for handling embeddings and vector operations."""
    
    def __init__(self):
        self.ollama_url = OLLAMA_BASE_URL
        self.embedding_model = EMBEDDING_MODEL
        self.chunk_size = CHUNK_SIZE
        self.chunk_overlap = CHUNK_OVERLAP
        
    async def _ensure_collection_exists(self) -> None:
        """Ensure the Qdrant collection exists with proper configuration."""
        if not qdrant_client:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Qdrant client not available"
            )
        
        try:
            # Check if collection exists
            collections = qdrant_client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if COLLECTION_NAME not in collection_names:
                logger.info(f"Creating collection: {COLLECTION_NAME}")
                # Get embedding dimension by creating a test embedding
                test_embedding = await self._get_embedding("test")
                dimension = len(test_embedding)
                
                qdrant_client.create_collection(
                    collection_name=COLLECTION_NAME,
                    vectors_config=models.VectorParams(
                        size=dimension,
                        distance=models.Distance.COSINE
                    )
                )
                logger.info(f"Collection {COLLECTION_NAME} created with dimension {dimension}")
            else:
                logger.info(f"Collection {COLLECTION_NAME} already exists")
                
        except ResponseHandlingException as e:
            logger.error(f"Qdrant error: {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Qdrant service error: {str(e)}"
            )
    
    async def _get_embedding(self, text: str) -> List[float]:
        """Get embedding vector from Ollama API."""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.ollama_url}/api/embeddings",
                    json={
                        "model": self.embedding_model,
                        "prompt": text
                    }
                )
                
                if response.status_code != 200:
                    logger.error(f"Ollama API error: {response.status_code} - {response.text}")
                    raise HTTPException(
                        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                        detail=f"Ollama embedding service error: {response.status_code}"
                    )
                
                result = response.json()
                embedding = result.get("embedding")
                
                if not embedding:
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail="No embedding returned from Ollama"
                    )
                
                return embedding
                
        except httpx.RequestError as e:
            logger.error(f"HTTP request error: {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Failed to connect to Ollama service: {str(e)}"
            )
        except Exception as e:
            logger.error(f"Unexpected error getting embedding: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Embedding generation failed: {str(e)}"
            )
    
    def _chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks."""
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # Try to break at sentence boundaries
            if end < len(text):
                # Look for sentence endings within the last 100 characters
                sentence_end = text.rfind('.', start, end)
                if sentence_end == -1:
                    sentence_end = text.rfind('!', start, end)
                if sentence_end == -1:
                    sentence_end = text.rfind('?', start, end)
                
                if sentence_end > start + self.chunk_size // 2:
                    end = sentence_end + 1
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move start position with overlap
            start = end - self.chunk_overlap
            if start >= len(text):
                break
        
        return chunks
    
    async def upsert_document(self, doc_id: str, text: str, metadata: Dict[str, Any]) -> UpsertResponse:
        """Process document: chunk, embed, and upsert to Qdrant."""
        try:
            # Ensure collection exists
            await self._ensure_collection_exists()
            
            # Chunk the text
            chunks = self._chunk_text(text)
            logger.info(f"Split document {doc_id} into {len(chunks)} chunks")
            
            # Generate embeddings for all chunks
            points = []
            for i, chunk in enumerate(chunks):
                try:
                    embedding = await self._get_embedding(chunk)
                    
                    # Create point payload
                    payload = {
                        "doc_id": doc_id,
                        "chunk_id": f"{doc_id}_chunk_{i}",
                        "text": chunk,
                        "chunk_index": i,
                        **metadata
                    }
                    
                    point = models.PointStruct(
                        id=f"{doc_id}_chunk_{i}",
                        vector=embedding,
                        payload=payload
                    )
                    points.append(point)
                    
                except Exception as e:
                    logger.error(f"Failed to process chunk {i} of document {doc_id}: {e}")
                    continue
            
            if not points:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="No chunks could be processed successfully"
                )
            
            # Upsert points to Qdrant
            qdrant_client.upsert(
                collection_name=COLLECTION_NAME,
                points=points
            )
            
            logger.info(f"Successfully upserted {len(points)} chunks for document {doc_id}")
            
            return UpsertResponse(
                success=True,
                doc_id=doc_id,
                chunks_processed=len(points),
                message=f"Document {doc_id} processed and stored successfully"
            )
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error upserting document {doc_id}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to process document: {str(e)}"
            )
    
    async def search_documents(self, query: str, k: int) -> SearchResponse:
        """Search for similar documents using vector similarity."""
        try:
            # Ensure collection exists
            await self._ensure_collection_exists()
            
            # Get query embedding
            query_embedding = await self._get_embedding(query)
            
            # Search in Qdrant
            search_results = qdrant_client.search(
                collection_name=COLLECTION_NAME,
                query_vector=query_embedding,
                limit=k,
                with_payload=True
            )
            
            # Format results
            results = []
            for result in search_results:
                results.append(SearchResult(
                    id=str(result.id),
                    score=result.score,
                    payload=result.payload or {}
                ))
            
            logger.info(f"Search for '{query}' returned {len(results)} results")
            
            return SearchResponse(
                results=results,
                query=query,
                total_found=len(results)
            )
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Search failed: {str(e)}"
            )

# Initialize service
embedding_service = EmbeddingService()

# API Endpoints
@app.post("/api/embeddings/upsert", response_model=UpsertResponse)
async def upsert_embeddings(request: UpsertRequest):
    """
    Upsert document embeddings.
    
    Chunks the input text, generates embeddings using Ollama,
    and stores them in Qdrant vector database.
    """
    logger.info(f"Upserting document: {request.doc_id}")
    
    if not request.text.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Text content cannot be empty"
        )
    
    return await embedding_service.upsert_document(
        doc_id=request.doc_id,
        text=request.text,
        metadata=request.metadata
    )

@app.post("/api/embeddings/search", response_model=SearchResponse)
async def search_embeddings(request: SearchRequest):
    """
    Search for similar documents using vector similarity.
    
    Generates embedding for the query and searches for similar
    vectors in the Qdrant collection.
    """
    logger.info(f"Searching for: '{request.query}' (k={request.k})")
    
    if not request.query.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query cannot be empty"
        )
    
    return await embedding_service.search_documents(
        query=request.query,
        k=request.k
    )

@app.get("/api/embeddings/health")
async def health_check():
    """Health check endpoint for the embeddings service."""
    try:
        # Check Ollama connection
        async with httpx.AsyncClient(timeout=5.0) as client:
            ollama_response = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
            ollama_status = ollama_response.status_code == 200
        
        # Check Qdrant connection
        qdrant_status = False
        if qdrant_client:
            try:
                qdrant_client.get_collections()
                qdrant_status = True
            except Exception:
                pass
        
        return {
            "status": "healthy" if (ollama_status and qdrant_status) else "degraded",
            "services": {
                "ollama": "up" if ollama_status else "down",
                "qdrant": "up" if qdrant_status else "down"
            },
            "config": {
                "embedding_model": EMBEDDING_MODEL,
                "chunk_size": CHUNK_SIZE,
                "chunk_overlap": CHUNK_OVERLAP,
                "collection_name": COLLECTION_NAME
            }
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }

@app.get("/api/embeddings/collection/info")
async def collection_info():
    """Get information about the Qdrant collection."""
    try:
        if not qdrant_client:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Qdrant client not available"
            )
        
        collection_info = qdrant_client.get_collection(COLLECTION_NAME)
        return {
            "collection_name": COLLECTION_NAME,
            "points_count": collection_info.points_count,
            "vectors_count": collection_info.vectors_count,
            "status": collection_info.status
        }
        
    except ResponseHandlingException as e:
        if "Not found" in str(e):
            return {
                "collection_name": COLLECTION_NAME,
                "exists": False,
                "message": "Collection does not exist yet"
            }
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Qdrant error: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Error getting collection info: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get collection info: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "embeddings:app",
        host="0.0.0.0",
        port=8001,
        reload=True
    )
