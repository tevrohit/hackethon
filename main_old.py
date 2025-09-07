from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
from dotenv import load_dotenv

# Import all routers and sub-apps
from app.risk import router as risk_router
from app.observability import observability_router, health_router

# Import sub-applications
try:
    from app.ai import app as ai_app
    from app.embeddings import app as embeddings_app
    from app.tickets import app as tickets_app
except ImportError as e:
    print(f"Warning: Could not import sub-apps: {e}")
    ai_app = embeddings_app = tickets_app = None

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

# Include the sub-applications as routers (convert endpoints to avoid path conflicts)
# For now, keep the basic nudge endpoint here and let sub-apps handle their own routes

class NudgeRequest(BaseModel):
    user_id: str
    message: str
    type: str = "reminder"

# Routes
@app.get("/")
async def root():
    return {"message": "Hackathon API is running!"}

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

# Mount sub-applications to handle their respective API routes
if ai_app:
    app.mount("/ai_service", ai_app)
if embeddings_app:
    app.mount("/embeddings_service", embeddings_app)  
if tickets_app:
    app.mount("/tickets_service", tickets_app)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
