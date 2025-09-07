from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
from dotenv import load_dotenv

# Import all routers
from app.risk import router as risk_router
from app.observability import observability_router, health_router

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
app.include_router(observability_router)
app.include_router(health_router)

# Pydantic models for basic endpoints
class AIAskRequest(BaseModel):
    user_id_hash: str
    course_id: str
    module_id: str
    question: str
    lang: str = "English"

class AIAskResponse(BaseModel):
    answer: str
    sources: List[dict] = []
    followups: List[str] = []
    escalate: bool = False
    ticket_id: Optional[str] = None

class TicketCreate(BaseModel):
    user_hash: str
    course_id: str
    module_id: str
    title: str
    description: str
    priority: str = "medium"
    language: str = "English"
    attachments: List[str] = []

class TicketResponse(BaseModel):
    id: str
    title: str
    description: str
    status: str
    priority: str
    user_hash: str
    course_id: str
    created_at: str
    sla_status: str
    sla_due_date: str

class EmbeddingRequest(BaseModel):
    doc_id: str
    text: str
    metadata: dict = {}

class EmbeddingResponse(BaseModel):
    success: bool
    doc_id: str
    chunks_processed: int
    message: str

class SearchRequest(BaseModel):
    query: str
    k: int = 5

class SearchResponse(BaseModel):
    results: List[dict]
    query: str
    total_found: int

class NudgeRequest(BaseModel):
    user_id: str
    message: str
    type: str = "reminder"

# Basic Routes
@app.get("/")
async def root():
    return {"message": "Hackathon API is running!"}

# AI Chat endpoints
@app.post("/api/ai/ask", response_model=AIAskResponse)
async def ai_ask(request: AIAskRequest):
    """AI chat endpoint with RAG"""
    try:
        # Placeholder implementation
        return AIAskResponse(
            answer=f"I received your question: '{request.question}'. This is a placeholder response for {request.course_id}.",
            sources=[],
            followups=["Would you like more details?", "Any other questions?"],
            escalate=False
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI service error: {str(e)}")

@app.get("/api/ai/health")
async def ai_health():
    """AI service health check"""
    return {
        "status": "healthy",
        "services": {
            "ollama": "up",
            "embeddings": "up"
        }
    }

# Embeddings endpoints
@app.post("/api/embeddings/upsert", response_model=EmbeddingResponse)
async def upsert_embeddings(request: EmbeddingRequest):
    """Upsert document embeddings"""
    try:
        # Placeholder implementation
        return EmbeddingResponse(
            success=True,
            doc_id=request.doc_id,
            chunks_processed=1,
            message=f"Document {request.doc_id} processed successfully"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding service error: {str(e)}")

@app.post("/api/embeddings/search", response_model=SearchResponse)
async def search_embeddings(request: SearchRequest):
    """Search similar documents"""
    try:
        # Placeholder implementation
        return SearchResponse(
            results=[
                {
                    "id": "doc1_chunk1",
                    "score": 0.95,
                    "payload": {
                        "content": f"Sample content related to: {request.query}",
                        "metadata": {"source": "sample.pdf"}
                    }
                }
            ],
            query=request.query,
            total_found=1
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")

@app.get("/api/embeddings/health")
async def embeddings_health():
    """Embeddings service health check"""
    return {
        "status": "healthy",
        "services": {
            "ollama": "up",
            "qdrant": "up"
        }
    }

# Tickets endpoints
@app.post("/api/tickets", response_model=TicketResponse)
async def create_ticket(ticket: TicketCreate):
    """Create a new support ticket"""
    try:
        from datetime import datetime
        now = datetime.utcnow()
        ticket_id = f"TICKET_{hash(ticket.title) % 10000:04d}"
        
        return TicketResponse(
            id=ticket_id,
            title=ticket.title,
            description=ticket.description,
            status="open",
            priority=ticket.priority,
            user_hash=ticket.user_hash,
            course_id=ticket.course_id,
            created_at=now.isoformat(),
            sla_status="on_time",
            sla_due_date=(now.replace(hour=now.hour + 24)).isoformat()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ticket creation error: {str(e)}")

@app.get("/api/tickets")
async def get_tickets():
    """Get all support tickets"""
    try:
        return [
            {
                "id": "TICKET_0001",
                "title": "Sample Ticket",
                "description": "This is a sample ticket",
                "status": "open",
                "priority": "medium",
                "user_hash": "user_123_hash",
                "course_id": "CS101",
                "created_at": "2024-01-01T12:00:00",
                "sla_status": "on_time"
            }
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.get("/api/tickets/{ticket_id}")
async def get_ticket(ticket_id: str):
    """Get a specific ticket"""
    try:
        return {
            "id": ticket_id,
            "title": "Sample Ticket",
            "description": "This is a sample ticket",
            "status": "open",
            "priority": "medium",
            "user_hash": "user_123_hash",
            "course_id": "CS101",
            "created_at": "2024-01-01T12:00:00",
            "sla_status": "on_time"
        }
    except Exception as e:
        raise HTTPException(status_code=404, detail="Ticket not found")

@app.patch("/api/tickets/{ticket_id}/assign")
async def assign_ticket(ticket_id: str, assignment: dict):
    """Assign a ticket"""
    return {
        "id": ticket_id,
        "status": "in_progress",
        "assigned_to": assignment.get("assigned_to"),
        "message": "Ticket assigned successfully"
    }

@app.patch("/api/tickets/{ticket_id}/resolve")
async def resolve_ticket(ticket_id: str, resolution: dict):
    """Resolve a ticket"""
    return {
        "id": ticket_id,
        "status": "resolved",
        "resolution_notes": resolution.get("resolution_notes"),
        "message": "Ticket resolved successfully"
    }

@app.post("/api/tickets/{ticket_id}/comment")
async def add_comment(ticket_id: str, comment: dict):
    """Add comment to ticket"""
    return {
        "id": f"comment_{hash(comment.get('content', '')) % 1000:03d}",
        "ticket_id": ticket_id,
        "content": comment.get("content"),
        "author": comment.get("author"),
        "created_at": "2024-01-01T12:00:00"
    }

@app.get("/api/tickets/{ticket_id}/comments")
async def get_ticket_comments(ticket_id: str):
    """Get ticket comments"""
    return [
        {
            "id": "comment_001",
            "ticket_id": ticket_id,
            "content": "Sample comment",
            "author": "support_agent",
            "created_at": "2024-01-01T12:00:00"
        }
    ]

@app.get("/api/tickets/health")
async def tickets_health():
    """Tickets service health check"""
    return {
        "status": "healthy",
        "total_tickets": 1,
        "total_comments": 1
    }

# Student profile endpoints
@app.get("/api/students/{user_hash}/profile")
async def get_student_profile(user_hash: str, course_id: str):
    """Get student profile"""
    return {
        "user_hash": user_hash,
        "course_id": course_id,
        "total_tickets": 1,
        "risk_score": 0.3,
        "last_activity": "2024-01-01T12:00:00"
    }

# Nudge endpoint
@app.post("/api/nudge")
async def send_nudge(request: NudgeRequest):
    """Send nudge/notification to user"""
    try:
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
        "main_unified:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
