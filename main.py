from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
from dotenv import load_dotenv

# Import risk router
from app.risk import router as risk_router

# Load environment variables
load_dotenv()

app = FastAPI(title="Hackathon API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(risk_router)

# Pydantic models
class AIAskRequest(BaseModel):
    message: str
    context: Optional[str] = None

class AIAskResponse(BaseModel):
    response: str
    confidence: Optional[float] = None

class Ticket(BaseModel):
    id: Optional[int] = None
    title: str
    description: str
    status: str = "open"
    priority: str = "medium"
    created_by: str

class EmbeddingRequest(BaseModel):
    text: str

class EmbeddingResponse(BaseModel):
    embedding: List[float]
    dimension: int

class NudgeRequest(BaseModel):
    user_id: str
    message: str
    type: str = "reminder"

# Routes
@app.get("/")
async def root():
    return {"message": "Hackathon API is running!"}

@app.post("/api/ai/ask", response_model=AIAskResponse)
async def ai_ask(request: AIAskRequest):
    """
    AI chat endpoint - integrates with Ollama for local LLM inference
    """
    try:
        # TODO: Implement Ollama integration
        # For now, return a placeholder response
        return AIAskResponse(
            response=f"I received your message: '{request.message}'. This is a placeholder response. Ollama integration pending.",
            confidence=0.8
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI service error: {str(e)}")

@app.get("/api/tickets")
async def get_tickets():
    """
    Get all support tickets from PostgreSQL
    """
    try:
        # TODO: Implement PostgreSQL integration
        # Placeholder response
        return [
            {
                "id": 1,
                "title": "Sample Ticket",
                "description": "This is a sample ticket",
                "status": "open",
                "priority": "medium",
                "created_by": "user@example.com"
            }
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.post("/api/tickets")
async def create_ticket(ticket: Ticket):
    """
    Create a new support ticket in PostgreSQL
    """
    try:
        # TODO: Implement PostgreSQL integration
        # Placeholder response
        return {
            "id": 1,
            "title": ticket.title,
            "description": ticket.description,
            "status": ticket.status,
            "priority": ticket.priority,
            "created_by": ticket.created_by,
            "message": "Ticket created successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.post("/api/embeddings", response_model=EmbeddingResponse)
async def create_embeddings(request: EmbeddingRequest):
    """
    Generate embeddings using Qdrant vector database
    """
    try:
        # TODO: Implement Qdrant integration for embeddings
        # Placeholder response with fake embedding
        fake_embedding = [0.1] * 384  # Typical embedding dimension
        return EmbeddingResponse(
            embedding=fake_embedding,
            dimension=len(fake_embedding)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding service error: {str(e)}")

@app.post("/api/nudge")
async def send_nudge(request: NudgeRequest):
    """
    Send nudge/notification to user
    """
    try:
        # TODO: Implement notification system
        return {
            "message": f"Nudge sent to user {request.user_id}",
            "type": request.type,
            "content": request.message,
            "status": "sent"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Nudge service error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
