"""
Tests for the AI ask API endpoint with mocked Qdrant and Ollama responses.
Tests both successful responses and various failure scenarios.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient
from fastapi import status
import json

# Test data
SAMPLE_AI_REQUEST = {
    "user_id_hash": "test_user_hash_123",
    "course_id": "CS101",
    "module_id": "deep_learning_fundamentals",
    "question": "Can you explain what neural networks are and how they work?",
    "lang": "English"
}

SAMPLE_CONTEXT_CHUNKS = [
    {
        "id": "doc1_chunk1",
        "score": 0.95,
        "payload": {
            "content": "Neural networks are computational models inspired by biological neural networks. They consist of interconnected nodes (neurons) that process information.",
            "doc_id": "neural_networks_intro",
            "chunk_id": "chunk_1",
            "metadata": {"source": "ml_textbook.pdf", "page": 42}
        }
    },
    {
        "id": "doc2_chunk1", 
        "score": 0.87,
        "payload": {
            "content": "Deep learning uses multiple layers of neural networks to learn complex patterns in data. Each layer transforms the input data progressively.",
            "doc_id": "deep_learning_basics",
            "chunk_id": "chunk_1",
            "metadata": {"source": "dl_guide.pdf", "page": 15}
        }
    }
]

SAMPLE_OLLAMA_RESPONSE = {
    "message": {
        "content": """Neural networks are computational models inspired by the human brain. They consist of interconnected nodes called neurons that process information through weighted connections.

Here's a simple example:
- Input layer: receives data (e.g., pixel values of an image)
- Hidden layers: process and transform the data
- Output layer: produces the final result (e.g., image classification)

**Mini Quiz:**
1. What are the basic components of a neural network?
2. How do neural networks learn from data?

**Sources:** DOC:neural_networks_intro:chunk_1, DOC:deep_learning_basics:chunk_1

{"answer": "Neural networks are computational models inspired by the human brain that process information through interconnected nodes called neurons. They learn by adjusting connection weights based on training data.", "sources": [{"doc": "neural_networks_intro", "chunk": "chunk_1", "score": 0.95}, {"doc": "deep_learning_basics", "chunk": "chunk_1", "score": 0.87}], "followups": ["Would you like to learn about different types of neural network architectures?", "Are you interested in understanding how neural networks are trained?"], "escalate": false}"""
    },
    "done": True,
    "total_duration": 2500000000,
    "load_duration": 200000000,
    "prompt_eval_count": 150,
    "eval_count": 75
}

SAMPLE_ESCALATION_RESPONSE = {
    "message": {
        "content": """I don't have enough information about payment policies in my knowledge base. Creating internal ticket.

{"answer": "I don't have enough information about payment policies. I've created a support ticket for you.", "sources": [], "followups": ["Is this regarding a specific payment issue?", "Would you like me to connect you with our billing team?"], "escalate": true}"""
    },
    "done": True,
    "total_duration": 1500000000
}

class TestAIAsk:
    """Test cases for /api/ai/ask endpoint."""
    
    @patch('app.ai.httpx.AsyncClient')
    def test_ai_ask_success(self, mock_httpx, client):
        """Test successful AI ask with context retrieval and response generation."""
        # Mock embeddings search response
        embeddings_response = Mock()
        embeddings_response.status_code = 200
        embeddings_response.json.return_value = {
            "results": SAMPLE_CONTEXT_CHUNKS,
            "query": SAMPLE_AI_REQUEST["question"],
            "total_found": 2
        }
        
        # Mock Ollama chat response
        ollama_response = Mock()
        ollama_response.status_code = 200
        ollama_response.json.return_value = SAMPLE_OLLAMA_RESPONSE
        
        # Configure mock client
        mock_client_instance = AsyncMock()
        mock_client_instance.post.side_effect = [embeddings_response, ollama_response]
        mock_httpx.return_value.__aenter__.return_value = mock_client_instance
        
        # Make request
        response = client.post("/api/ai/ask", json=SAMPLE_AI_REQUEST)
        
        # Assertions
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert "neural networks" in data["answer"].lower()
        assert len(data["sources"]) == 2
        assert data["sources"][0]["doc"] == "neural_networks_intro"
        assert data["sources"][0]["chunk"] == "chunk_1"
        assert data["sources"][0]["score"] == 0.95
        assert len(data["followups"]) == 2
        assert data["escalate"] is False
        assert data["ticket_id"] is None
        
        # Verify API calls were made
        assert mock_client_instance.post.call_count == 2
    
    @patch('app.ai.httpx.AsyncClient')
    def test_ai_ask_escalation_with_ticket_creation(self, mock_httpx, client):
        """Test AI ask that requires escalation and ticket creation."""
        # Mock embeddings search response (empty results)
        embeddings_response = Mock()
        embeddings_response.status_code = 200
        embeddings_response.json.return_value = {
            "results": [],
            "query": "What is your refund policy?",
            "total_found": 0
        }
        
        # Mock Ollama escalation response
        ollama_response = Mock()
        ollama_response.status_code = 200
        ollama_response.json.return_value = SAMPLE_ESCALATION_RESPONSE
        
        # Mock ticket creation response
        ticket_response = Mock()
        ticket_response.status_code = 200
        ticket_response.json.return_value = {
            "id": "ticket_12345",
            "title": "AI Escalation: Payment policy inquiry",
            "status": "open",
            "priority": "medium"
        }
        
        # Configure mock client
        mock_client_instance = AsyncMock()
        mock_client_instance.post.side_effect = [embeddings_response, ollama_response, ticket_response]
        mock_httpx.return_value.__aenter__.return_value = mock_client_instance
        
        # Make request with escalation-triggering question
        escalation_request = SAMPLE_AI_REQUEST.copy()
        escalation_request["question"] = "What is your refund policy?"
        
        response = client.post("/api/ai/ask", json=escalation_request)
        
        # Assertions
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert data["escalate"] is True
        assert data["ticket_id"] == "ticket_12345"
        assert "support ticket" in data["answer"].lower()
        assert len(data["sources"]) == 0
        
        # Verify all API calls were made (embeddings, ollama, tickets)
        assert mock_client_instance.post.call_count == 3
    
    def test_ai_ask_empty_question_failure(self, client):
        """Test AI ask failure with empty question."""
        request_data = SAMPLE_AI_REQUEST.copy()
        request_data["question"] = ""
        
        response = client.post("/api/ai/ask", json=request_data)
        
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        data = response.json()
        assert "empty" in data["detail"].lower()
    
    def test_ai_ask_whitespace_only_question_failure(self, client):
        """Test AI ask failure with whitespace-only question."""
        request_data = SAMPLE_AI_REQUEST.copy()
        request_data["question"] = "   \n\t   "
        
        response = client.post("/api/ai/ask", json=request_data)
        
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        data = response.json()
        assert "empty" in data["detail"].lower()
    
    @patch('app.ai.httpx.AsyncClient')
    def test_ai_ask_embeddings_service_failure(self, mock_httpx, client):
        """Test AI ask when embeddings service fails."""
        # Mock embeddings service failure
        mock_client_instance = AsyncMock()
        mock_client_instance.post.side_effect = Exception("Embeddings service unavailable")
        mock_httpx.return_value.__aenter__.return_value = mock_client_instance
        
        response = client.post("/api/ai/ask", json=SAMPLE_AI_REQUEST)
        
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        data = response.json()
        assert "error" in data["detail"].lower()
    
    @patch('app.ai.httpx.AsyncClient')
    def test_ai_ask_ollama_service_failure(self, mock_httpx, client):
        """Test AI ask when Ollama service fails."""
        # Mock successful embeddings response
        embeddings_response = Mock()
        embeddings_response.status_code = 200
        embeddings_response.json.return_value = {
            "results": SAMPLE_CONTEXT_CHUNKS,
            "query": SAMPLE_AI_REQUEST["question"],
            "total_found": 2
        }
        
        # Configure mock client - embeddings succeeds, Ollama fails
        mock_client_instance = AsyncMock()
        mock_client_instance.post.side_effect = [
            embeddings_response,
            Exception("Ollama service unavailable")
        ]
        mock_httpx.return_value.__aenter__.return_value = mock_client_instance
        
        response = client.post("/api/ai/ask", json=SAMPLE_AI_REQUEST)
        
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        data = response.json()
        assert "error" in data["detail"].lower()
    
    @patch('app.ai.httpx.AsyncClient')
    def test_ai_ask_invalid_ollama_response(self, mock_httpx, client):
        """Test AI ask with invalid JSON response from Ollama."""
        # Mock embeddings response
        embeddings_response = Mock()
        embeddings_response.status_code = 200
        embeddings_response.json.return_value = {
            "results": SAMPLE_CONTEXT_CHUNKS,
            "query": SAMPLE_AI_REQUEST["question"],
            "total_found": 2
        }
        
        # Mock Ollama response with invalid JSON
        invalid_ollama_response = Mock()
        invalid_ollama_response.status_code = 200
        invalid_ollama_response.json.return_value = {
            "message": {
                "content": "This is a response without valid JSON at the end."
            },
            "done": True
        }
        
        mock_client_instance = AsyncMock()
        mock_client_instance.post.side_effect = [embeddings_response, invalid_ollama_response]
        mock_httpx.return_value.__aenter__.return_value = mock_client_instance
        
        response = client.post("/api/ai/ask", json=SAMPLE_AI_REQUEST)
        
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        data = response.json()
        assert "parsing" in data["detail"].lower() or "json" in data["detail"].lower()
    
    def test_ai_ask_invalid_request_format(self, client):
        """Test AI ask with invalid request format."""
        invalid_request = {
            "user_id_hash": "test_user",
            "course_id": "CS101",
            # Missing required fields: module_id, question
        }
        
        response = client.post("/api/ai/ask", json=invalid_request)
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        data = response.json()
        assert "validation error" in data["detail"][0]["type"]
    
    @patch('app.ai.httpx.AsyncClient')
    def test_ai_ask_with_different_languages(self, mock_httpx, client):
        """Test AI ask with different language preferences."""
        # Mock responses
        embeddings_response = Mock()
        embeddings_response.status_code = 200
        embeddings_response.json.return_value = {
            "results": SAMPLE_CONTEXT_CHUNKS,
            "query": "न्यूरल नेटवर्क क्या है?",
            "total_found": 2
        }
        
        hindi_ollama_response = Mock()
        hindi_ollama_response.status_code = 200
        hindi_ollama_response.json.return_value = {
            "message": {
                "content": """न्यूरल नेटवर्क एक कम्प्यूटेशनल मॉडल है जो मानव मस्तिष्क से प्रेरित है।

{"answer": "न्यूरल नेटवर्क एक कम्प्यूटेशनल मॉडल है जो मानव मस्तिष्क से प्रेरित है।", "sources": [{"doc": "neural_networks_intro", "chunk": "chunk_1", "score": 0.95}], "followups": ["क्या आप विभिन्न प्रकार के न्यूरल नेटवर्क के बारे में जानना चाहते हैं?", "क्या आप न्यूरल नेटवर्क की ट्रेनिंग प्रक्रिया समझना चाहते हैं?"], "escalate": false}"""
            },
            "done": True
        }
        
        mock_client_instance = AsyncMock()
        mock_client_instance.post.side_effect = [embeddings_response, hindi_ollama_response]
        mock_httpx.return_value.__aenter__.return_value = mock_client_instance
        
        # Make request in Hindi
        hindi_request = SAMPLE_AI_REQUEST.copy()
        hindi_request["question"] = "न्यूरल नेटवर्क क्या है?"
        hindi_request["lang"] = "Hindi"
        
        response = client.post("/api/ai/ask", json=hindi_request)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "न्यूरल नेटवर्क" in data["answer"]
        assert data["escalate"] is False
    
    @patch('app.ai.httpx.AsyncClient')
    def test_ai_ask_ticket_creation_failure(self, mock_httpx, client):
        """Test AI ask when escalation is needed but ticket creation fails."""
        # Mock embeddings response (empty)
        embeddings_response = Mock()
        embeddings_response.status_code = 200
        embeddings_response.json.return_value = {
            "results": [],
            "query": "billing question",
            "total_found": 0
        }
        
        # Mock Ollama escalation response
        ollama_response = Mock()
        ollama_response.status_code = 200
        ollama_response.json.return_value = SAMPLE_ESCALATION_RESPONSE
        
        # Mock ticket creation failure
        mock_client_instance = AsyncMock()
        mock_client_instance.post.side_effect = [
            embeddings_response,
            ollama_response,
            Exception("Ticket service unavailable")
        ]
        mock_httpx.return_value.__aenter__.return_value = mock_client_instance
        
        escalation_request = SAMPLE_AI_REQUEST.copy()
        escalation_request["question"] = "I need help with billing"
        
        response = client.post("/api/ai/ask", json=escalation_request)
        
        # Should still return successful response even if ticket creation fails
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["escalate"] is True
        assert data["ticket_id"] is None  # No ticket created due to failure
    
    @patch('app.ai.httpx.AsyncClient')
    def test_ai_ask_with_observability_logging(self, mock_httpx, client):
        """Test that AI ask properly logs to observability system."""
        # Mock successful responses
        embeddings_response = Mock()
        embeddings_response.status_code = 200
        embeddings_response.json.return_value = {
            "results": SAMPLE_CONTEXT_CHUNKS,
            "query": SAMPLE_AI_REQUEST["question"],
            "total_found": 2
        }
        
        ollama_response = Mock()
        ollama_response.status_code = 200
        ollama_response.json.return_value = SAMPLE_OLLAMA_RESPONSE
        
        mock_client_instance = AsyncMock()
        mock_client_instance.post.side_effect = [embeddings_response, ollama_response]
        mock_httpx.return_value.__aenter__.return_value = mock_client_instance
        
        # Mock observability logging
        with patch('app.ai.log_prompt_call') as mock_log:
            response = client.post("/api/ai/ask", json=SAMPLE_AI_REQUEST)
            
            assert response.status_code == status.HTTP_200_OK
            
            # Verify observability logging was called
            mock_log.assert_called_once()
            call_args = mock_log.call_args[1]
            assert call_args["user_hash"] == SAMPLE_AI_REQUEST["user_id_hash"]
            assert call_args["model"] == "mistral:7b"
            assert call_args["status"] == "success"
            assert len(call_args["retrieved_doc_ids"]) == 2

class TestAIHealth:
    """Test cases for AI service health endpoint."""
    
    @patch('app.ai.httpx.AsyncClient')
    def test_ai_health_check_success(self, mock_httpx, client):
        """Test successful AI health check."""
        # Mock Ollama health response
        ollama_response = Mock()
        ollama_response.status_code = 200
        ollama_response.json.return_value = {"models": []}
        
        # Mock embeddings health response
        embeddings_response = Mock()
        embeddings_response.status_code = 200
        embeddings_response.json.return_value = {"status": "healthy"}
        
        mock_client_instance = AsyncMock()
        mock_client_instance.get.side_effect = [ollama_response, embeddings_response]
        mock_httpx.return_value.__aenter__.return_value = mock_client_instance
        
        response = client.get("/api/ai/health")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["status"] == "healthy"
        assert data["services"]["ollama"] == "up"
        assert data["services"]["embeddings"] == "up"
    
    @patch('app.ai.httpx.AsyncClient')
    def test_ai_health_check_degraded(self, mock_httpx, client):
        """Test AI health check when services are partially down."""
        # Mock Ollama success, embeddings failure
        ollama_response = Mock()
        ollama_response.status_code = 200
        
        mock_client_instance = AsyncMock()
        mock_client_instance.get.side_effect = [
            ollama_response,
            Exception("Embeddings service down")
        ]
        mock_httpx.return_value.__aenter__.return_value = mock_client_instance
        
        response = client.get("/api/ai/health")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["status"] == "degraded"
        assert data["services"]["ollama"] == "up"
        assert data["services"]["embeddings"] == "down"
