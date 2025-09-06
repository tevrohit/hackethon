"""
Tests for the embeddings API endpoints.
Tests both happy path and failure scenarios for /api/embeddings/upsert.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient
from fastapi import status
import json

# Test data
SAMPLE_UPSERT_REQUEST = {
    "doc_id": "test_doc_123",
    "text": "This is a sample document for testing embeddings. It contains multiple sentences to test chunking functionality.",
    "metadata": {
        "source": "test_document.pdf",
        "page": 1,
        "section": "introduction"
    }
}

SAMPLE_SEARCH_REQUEST = {
    "query": "sample document testing",
    "k": 5
}

class TestEmbeddingsUpsert:
    """Test cases for /api/embeddings/upsert endpoint."""
    
    @patch('app.embeddings.qdrant_client')
    @patch('app.embeddings.httpx.AsyncClient')
    def test_upsert_success(self, mock_httpx, mock_qdrant, client):
        """Test successful document upsert."""
        # Mock Ollama embedding response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "embedding": [0.1] * 384  # Mock 384-dimensional embedding
        }
        
        mock_httpx_instance = AsyncMock()
        mock_httpx_instance.post.return_value = mock_response
        mock_httpx.return_value.__aenter__.return_value = mock_httpx_instance
        
        # Mock Qdrant operations
        mock_qdrant.get_collections.return_value = Mock(
            collections=[Mock(name="course_kb")]
        )
        mock_qdrant.upsert.return_value = Mock(
            operation_id=12345,
            status="completed"
        )
        
        # Make request
        response = client.post("/api/embeddings/upsert", json=SAMPLE_UPSERT_REQUEST)
        
        # Assertions
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["success"] is True
        assert data["doc_id"] == SAMPLE_UPSERT_REQUEST["doc_id"]
        assert data["chunks_processed"] >= 1
        assert "successfully" in data["message"].lower()
        
        # Verify Qdrant was called
        mock_qdrant.upsert.assert_called_once()
    
    @patch('app.embeddings.qdrant_client')
    def test_upsert_empty_text_failure(self, mock_qdrant, client):
        """Test upsert failure with empty text."""
        request_data = SAMPLE_UPSERT_REQUEST.copy()
        request_data["text"] = ""
        
        response = client.post("/api/embeddings/upsert", json=request_data)
        
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        data = response.json()
        assert "empty" in data["detail"].lower()
    
    @patch('app.embeddings.qdrant_client')
    def test_upsert_whitespace_only_text_failure(self, mock_qdrant, client):
        """Test upsert failure with whitespace-only text."""
        request_data = SAMPLE_UPSERT_REQUEST.copy()
        request_data["text"] = "   \n\t   "
        
        response = client.post("/api/embeddings/upsert", json=request_data)
        
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        data = response.json()
        assert "empty" in data["detail"].lower()
    
    @patch('app.embeddings.qdrant_client', None)
    def test_upsert_qdrant_unavailable_failure(self, client):
        """Test upsert failure when Qdrant is unavailable."""
        response = client.post("/api/embeddings/upsert", json=SAMPLE_UPSERT_REQUEST)
        
        assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
        data = response.json()
        assert "qdrant" in data["detail"].lower()
    
    @patch('app.embeddings.qdrant_client')
    @patch('app.embeddings.httpx.AsyncClient')
    def test_upsert_ollama_failure(self, mock_httpx, mock_qdrant, client):
        """Test upsert failure when Ollama service fails."""
        # Mock Ollama failure
        mock_httpx_instance = AsyncMock()
        mock_httpx_instance.post.side_effect = Exception("Ollama connection failed")
        mock_httpx.return_value.__aenter__.return_value = mock_httpx_instance
        
        # Mock Qdrant collection exists
        mock_qdrant.get_collections.return_value = Mock(
            collections=[Mock(name="course_kb")]
        )
        
        response = client.post("/api/embeddings/upsert", json=SAMPLE_UPSERT_REQUEST)
        
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        data = response.json()
        assert "error" in data["detail"].lower()
    
    @patch('app.embeddings.qdrant_client')
    @patch('app.embeddings.httpx.AsyncClient')
    def test_upsert_qdrant_upsert_failure(self, mock_httpx, mock_qdrant, client):
        """Test upsert failure when Qdrant upsert operation fails."""
        # Mock successful Ollama response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "embedding": [0.1] * 384
        }
        
        mock_httpx_instance = AsyncMock()
        mock_httpx_instance.post.return_value = mock_response
        mock_httpx.return_value.__aenter__.return_value = mock_httpx_instance
        
        # Mock Qdrant collection exists but upsert fails
        mock_qdrant.get_collections.return_value = Mock(
            collections=[Mock(name="course_kb")]
        )
        mock_qdrant.upsert.side_effect = Exception("Qdrant upsert failed")
        
        response = client.post("/api/embeddings/upsert", json=SAMPLE_UPSERT_REQUEST)
        
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        data = response.json()
        assert "error" in data["detail"].lower()
    
    def test_upsert_invalid_request_format(self, client):
        """Test upsert failure with invalid request format."""
        invalid_request = {
            "doc_id": "test_doc",
            # Missing required "text" field
            "metadata": {}
        }
        
        response = client.post("/api/embeddings/upsert", json=invalid_request)
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        data = response.json()
        assert "validation error" in data["detail"][0]["type"]
    
    @patch('app.embeddings.qdrant_client')
    @patch('app.embeddings.httpx.AsyncClient')
    def test_upsert_large_document_chunking(self, mock_httpx, mock_qdrant, client):
        """Test upsert with large document that requires chunking."""
        # Create a large document
        large_text = "This is a test sentence. " * 200  # ~5000 characters
        request_data = SAMPLE_UPSERT_REQUEST.copy()
        request_data["text"] = large_text
        
        # Mock Ollama embedding response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "embedding": [0.1] * 384
        }
        
        mock_httpx_instance = AsyncMock()
        mock_httpx_instance.post.return_value = mock_response
        mock_httpx.return_value.__aenter__.return_value = mock_httpx_instance
        
        # Mock Qdrant operations
        mock_qdrant.get_collections.return_value = Mock(
            collections=[Mock(name="course_kb")]
        )
        mock_qdrant.upsert.return_value = Mock(
            operation_id=12345,
            status="completed"
        )
        
        response = client.post("/api/embeddings/upsert", json=request_data)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["success"] is True
        assert data["chunks_processed"] > 1  # Should be chunked into multiple pieces
    
    @patch('app.embeddings.qdrant_client')
    @patch('app.embeddings.httpx.AsyncClient')
    def test_upsert_with_metadata(self, mock_httpx, mock_qdrant, client):
        """Test upsert with rich metadata."""
        request_data = SAMPLE_UPSERT_REQUEST.copy()
        request_data["metadata"] = {
            "source": "advanced_ml_textbook.pdf",
            "page": 42,
            "chapter": "Deep Learning",
            "section": "Neural Networks",
            "author": "Dr. Smith",
            "tags": ["machine learning", "neural networks", "deep learning"]
        }
        
        # Mock successful responses
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "embedding": [0.1] * 384
        }
        
        mock_httpx_instance = AsyncMock()
        mock_httpx_instance.post.return_value = mock_response
        mock_httpx.return_value.__aenter__.return_value = mock_httpx_instance
        
        mock_qdrant.get_collections.return_value = Mock(
            collections=[Mock(name="course_kb")]
        )
        mock_qdrant.upsert.return_value = Mock(
            operation_id=12345,
            status="completed"
        )
        
        response = client.post("/api/embeddings/upsert", json=request_data)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["success"] is True
        
        # Verify that upsert was called with metadata
        mock_qdrant.upsert.assert_called_once()
        call_args = mock_qdrant.upsert.call_args
        points = call_args[1]["points"]
        assert len(points) >= 1
        assert "author" in points[0].payload
        assert points[0].payload["author"] == "Dr. Smith"

class TestEmbeddingsSearch:
    """Test cases for /api/embeddings/search endpoint."""
    
    @patch('app.embeddings.qdrant_client')
    @patch('app.embeddings.httpx.AsyncClient')
    def test_search_success(self, mock_httpx, mock_qdrant, client):
        """Test successful document search."""
        # Mock Ollama embedding response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "embedding": [0.1] * 384
        }
        
        mock_httpx_instance = AsyncMock()
        mock_httpx_instance.post.return_value = mock_response
        mock_httpx.return_value.__aenter__.return_value = mock_httpx_instance
        
        # Mock Qdrant search results
        mock_qdrant.search.return_value = [
            Mock(
                id="doc1_chunk1",
                score=0.95,
                payload={
                    "content": "Sample document content about testing",
                    "doc_id": "doc1",
                    "metadata": {"source": "test1.pdf"}
                }
            ),
            Mock(
                id="doc2_chunk1",
                score=0.87,
                payload={
                    "content": "Another document with relevant content",
                    "doc_id": "doc2", 
                    "metadata": {"source": "test2.pdf"}
                }
            )
        ]
        
        response = client.post("/api/embeddings/search", json=SAMPLE_SEARCH_REQUEST)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["query"] == SAMPLE_SEARCH_REQUEST["query"]
        assert data["total_found"] == 2
        assert len(data["results"]) == 2
        assert data["results"][0]["score"] == 0.95
        assert "testing" in data["results"][0]["payload"]["content"]
    
    def test_search_empty_query_failure(self, client):
        """Test search failure with empty query."""
        request_data = SAMPLE_SEARCH_REQUEST.copy()
        request_data["query"] = ""
        
        response = client.post("/api/embeddings/search", json=request_data)
        
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        data = response.json()
        assert "empty" in data["detail"].lower()

class TestEmbeddingsHealth:
    """Test cases for embeddings health endpoint."""
    
    @patch('app.embeddings.qdrant_client')
    @patch('app.embeddings.httpx.AsyncClient')
    def test_health_check_success(self, mock_httpx, mock_qdrant, client):
        """Test successful health check."""
        # Mock Ollama health
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "ok"}
        
        mock_httpx_instance = AsyncMock()
        mock_httpx_instance.get.return_value = mock_response
        mock_httpx.return_value.__aenter__.return_value = mock_httpx_instance
        
        # Mock Qdrant health
        mock_qdrant.get_collections.return_value = Mock(
            collections=[Mock(name="course_kb")]
        )
        
        response = client.get("/api/embeddings/health")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["status"] == "healthy"
        assert "ollama" in data["services"]
        assert "qdrant" in data["services"]
    
    @patch('app.embeddings.qdrant_client', None)
    def test_health_check_qdrant_unavailable(self, client):
        """Test health check when Qdrant is unavailable."""
        response = client.get("/api/embeddings/health")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["status"] == "degraded"
        assert data["services"]["qdrant"] == "unavailable"
