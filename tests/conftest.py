"""
Pytest configuration and fixtures for the hackathon backend tests.
"""

import pytest
import asyncio
from typing import Generator, AsyncGenerator
from fastapi.testclient import TestClient
from unittest.mock import Mock, AsyncMock, patch
import sys
import os

# Add the backend directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import app

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def client() -> Generator[TestClient, None, None]:
    """Create a test client for the FastAPI app."""
    with TestClient(app) as test_client:
        yield test_client

@pytest.fixture
def mock_qdrant_client():
    """Mock Qdrant client for testing."""
    mock_client = Mock()
    
    # Mock successful upsert
    mock_client.upsert.return_value = Mock(
        operation_id=12345,
        status="completed"
    )
    
    # Mock successful search
    mock_client.search.return_value = [
        Mock(
            id="doc1",
            score=0.95,
            payload={"content": "Sample document content 1", "metadata": {"source": "test1.pdf"}}
        ),
        Mock(
            id="doc2", 
            score=0.87,
            payload={"content": "Sample document content 2", "metadata": {"source": "test2.pdf"}}
        )
    ]
    
    # Mock collection info
    mock_client.get_collection.return_value = Mock(
        config=Mock(params=Mock(vectors=Mock(size=384))),
        points_count=100
    )
    
    return mock_client

@pytest.fixture
def mock_ollama_client():
    """Mock Ollama client for testing."""
    mock_client = AsyncMock()
    
    # Mock successful chat completion
    mock_client.chat.return_value = {
        "message": {
            "content": "This is a mocked response from Ollama. The user asked about deep learning fundamentals."
        },
        "done": True,
        "total_duration": 1500000000,  # 1.5 seconds in nanoseconds
        "load_duration": 100000000,
        "prompt_eval_count": 50,
        "eval_count": 25
    }
    
    # Mock model list
    mock_client.list.return_value = {
        "models": [
            {"name": "llama2:latest", "size": 3800000000},
            {"name": "codellama:latest", "size": 3800000000}
        ]
    }
    
    return mock_client

@pytest.fixture
def mock_database():
    """Mock database connection for testing."""
    mock_db = Mock()
    
    # Mock ticket operations
    mock_db.execute.return_value = Mock(
        fetchone=Mock(return_value={
            "id": 1,
            "title": "Test Ticket",
            "description": "Test Description",
            "status": "open",
            "priority": "medium",
            "created_by": "test@example.com",
            "created_at": "2024-01-01T12:00:00Z"
        }),
        fetchall=Mock(return_value=[
            {
                "id": 1,
                "title": "Test Ticket 1",
                "status": "open",
                "priority": "high"
            },
            {
                "id": 2,
                "title": "Test Ticket 2", 
                "status": "escalated",
                "priority": "urgent"
            }
        ])
    )
    
    return mock_db

@pytest.fixture
def sample_embedding_data():
    """Sample data for embedding tests."""
    return {
        "text": "This is a sample text for embedding generation.",
        "metadata": {
            "source": "test_document.pdf",
            "page": 1,
            "section": "introduction"
        }
    }

@pytest.fixture
def sample_ticket_data():
    """Sample data for ticket tests."""
    return {
        "user_hash": "test_user_hash_123",
        "course_id": "CS101",
        "module_id": "module_1",
        "title": "Cannot access course materials",
        "description": "I'm having trouble accessing the course materials for module 1. The links are not working.",
        "priority": "medium",
        "language": "English"
    }

@pytest.fixture
def sample_ai_request():
    """Sample AI request data."""
    return {
        "message": "Can you explain deep learning fundamentals?",
        "context": "The user is asking about neural networks and machine learning concepts.",
        "user_hash": "test_user_123",
        "course_id": "CS101"
    }

@pytest.fixture(autouse=True)
def reset_observability_state():
    """Reset observability state before each test."""
    try:
        from app.observability import obs_state, ObservabilityState
        # Reset the global state
        obs_state.__dict__.update(ObservabilityState().__dict__)
    except ImportError:
        # Observability module might not be available in all tests
        pass
    yield

@pytest.fixture
def mock_file_system(tmp_path):
    """Mock file system for testing file operations."""
    # Create temporary directories and files
    test_dir = tmp_path / "test_content"
    test_dir.mkdir()
    
    # Create sample files
    (test_dir / "sample.txt").write_text("Sample content for testing")
    (test_dir / "course_notes.md").write_text("# Course Notes\nThis is a test document.")
    
    return test_dir
