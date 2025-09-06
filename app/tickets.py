"""
Ticket management API endpoints for support system.
Handles ticket creation, listing, assignment, resolution, and comments.
"""

import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from enum import Enum
from fastapi import FastAPI, HTTPException, status, Query, Path, UploadFile, File
from pydantic import BaseModel, Field
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Enums for ticket management
class TicketStatus(str, Enum):
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    CLOSED = "closed"
    ESCALATED = "escalated"

class TicketPriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"

class SLAStatus(str, Enum):
    ON_TIME = "on_time"
    AT_RISK = "at_risk"
    OVERDUE = "overdue"

# Pydantic models
class TicketCreate(BaseModel):
    user_hash: str = Field(..., description="Hashed user identifier")
    course_id: str = Field(..., description="Course identifier")
    module_id: str = Field(..., description="Module identifier")
    title: str = Field(..., min_length=1, max_length=200, description="Ticket title")
    description: str = Field(..., min_length=1, description="Detailed description")
    attachments: List[str] = Field(default=[], description="List of attachment URLs/paths")
    priority: TicketPriority = Field(default=TicketPriority.MEDIUM, description="Ticket priority")
    language: str = Field(default="English", description="User's preferred language")

class TicketUpdate(BaseModel):
    status: Optional[TicketStatus] = None
    priority: Optional[TicketPriority] = None
    assigned_to: Optional[str] = None
    resolution_notes: Optional[str] = None

class TicketComment(BaseModel):
    comment: str = Field(..., min_length=1, description="Comment text")
    author: str = Field(..., description="Comment author")
    is_internal: bool = Field(default=False, description="Internal comment not visible to student")

class TicketAssignment(BaseModel):
    assigned_to: str = Field(..., description="Mentor/agent to assign ticket to")
    notes: Optional[str] = Field(None, description="Assignment notes")

class TicketResolution(BaseModel):
    resolution_notes: str = Field(..., min_length=1, description="Resolution details")
    resolved_by: str = Field(..., description="Who resolved the ticket")

class TicketResponse(BaseModel):
    id: str
    user_hash: str
    course_id: str
    module_id: str
    title: str
    description: str
    status: TicketStatus
    priority: TicketPriority
    language: str
    attachments: List[str]
    assigned_to: Optional[str]
    created_at: datetime
    updated_at: datetime
    resolved_at: Optional[datetime]
    sla_status: SLAStatus
    sla_due_date: datetime
    resolution_notes: Optional[str]
    comments_count: int
    student_risk_score: int

class CommentResponse(BaseModel):
    id: str
    ticket_id: str
    comment: str
    author: str
    is_internal: bool
    created_at: datetime

class StudentProfile(BaseModel):
    user_hash: str
    course_id: str
    risk_score: int
    engagement_level: str
    last_activity: datetime
    total_tickets: int
    resolved_tickets: int
    avg_resolution_time: float
    preferred_language: str
    learning_progress: float

# Initialize FastAPI app
app = FastAPI(title="Tickets API", version="1.0.0")

# In-memory storage (replace with database in production)
tickets_db: Dict[str, Dict] = {}
comments_db: Dict[str, List[Dict]] = {}
student_profiles: Dict[str, Dict] = {}

class TicketService:
    """Service class for handling ticket operations."""
    
    def __init__(self):
        self.sla_hours = {
            TicketPriority.URGENT: 2,
            TicketPriority.HIGH: 8,
            TicketPriority.MEDIUM: 24,
            TicketPriority.LOW: 72
        }
    
    def _generate_ticket_id(self) -> str:
        """Generate unique ticket ID."""
        import uuid
        return f"TKT-{uuid.uuid4().hex[:8].upper()}"
    
    def _calculate_sla_status(self, created_at: datetime, priority: TicketPriority, status: TicketStatus) -> tuple[SLAStatus, datetime]:
        """Calculate SLA status and due date."""
        if status in [TicketStatus.RESOLVED, TicketStatus.CLOSED]:
            sla_due = created_at + timedelta(hours=self.sla_hours[priority])
            return SLAStatus.ON_TIME, sla_due
        
        sla_due = created_at + timedelta(hours=self.sla_hours[priority])
        now = datetime.utcnow()
        
        if now > sla_due:
            return SLAStatus.OVERDUE, sla_due
        elif now > sla_due - timedelta(hours=2):  # At risk if within 2 hours of deadline
            return SLAStatus.AT_RISK, sla_due
        else:
            return SLAStatus.ON_TIME, sla_due
    
    def _get_student_risk_score(self, user_hash: str, course_id: str) -> int:
        """Calculate student risk score based on various factors."""
        profile = student_profiles.get(f"{user_hash}_{course_id}")
        if not profile:
            # Create default profile
            profile = {
                "user_hash": user_hash,
                "course_id": course_id,
                "risk_score": 50,  # Default medium risk
                "engagement_level": "medium",
                "last_activity": datetime.utcnow(),
                "total_tickets": 0,
                "resolved_tickets": 0,
                "avg_resolution_time": 24.0,
                "preferred_language": "English",
                "learning_progress": 0.5
            }
            student_profiles[f"{user_hash}_{course_id}"] = profile
        
        # Calculate risk score based on factors
        risk_score = 50  # Base score
        
        # Ticket history factor
        total_tickets = profile["total_tickets"]
        if total_tickets > 10:
            risk_score += 20
        elif total_tickets > 5:
            risk_score += 10
        
        # Resolution rate factor
        if total_tickets > 0:
            resolution_rate = profile["resolved_tickets"] / total_tickets
            if resolution_rate < 0.5:
                risk_score += 15
        
        # Learning progress factor
        if profile["learning_progress"] < 0.3:
            risk_score += 20
        elif profile["learning_progress"] < 0.6:
            risk_score += 10
        
        # Engagement factor
        if profile["engagement_level"] == "low":
            risk_score += 15
        elif profile["engagement_level"] == "high":
            risk_score -= 10
        
        # Cap the score between 0 and 100
        risk_score = max(0, min(100, risk_score))
        
        # Update profile
        profile["risk_score"] = risk_score
        student_profiles[f"{user_hash}_{course_id}"] = profile
        
        return risk_score
    
    def create_ticket(self, ticket_data: TicketCreate) -> TicketResponse:
        """Create a new support ticket."""
        ticket_id = self._generate_ticket_id()
        now = datetime.utcnow()
        
        # Calculate SLA
        sla_status, sla_due = self._calculate_sla_status(now, ticket_data.priority, TicketStatus.OPEN)
        
        # Get student risk score
        risk_score = self._get_student_risk_score(ticket_data.user_hash, ticket_data.course_id)
        
        ticket = {
            "id": ticket_id,
            "user_hash": ticket_data.user_hash,
            "course_id": ticket_data.course_id,
            "module_id": ticket_data.module_id,
            "title": ticket_data.title,
            "description": ticket_data.description,
            "status": TicketStatus.OPEN,
            "priority": ticket_data.priority,
            "language": ticket_data.language,
            "attachments": ticket_data.attachments,
            "assigned_to": None,
            "created_at": now,
            "updated_at": now,
            "resolved_at": None,
            "sla_status": sla_status,
            "sla_due_date": sla_due,
            "resolution_notes": None,
            "comments_count": 0,
            "student_risk_score": risk_score
        }
        
        tickets_db[ticket_id] = ticket
        comments_db[ticket_id] = []
        
        # Update student profile
        profile_key = f"{ticket_data.user_hash}_{ticket_data.course_id}"
        if profile_key in student_profiles:
            student_profiles[profile_key]["total_tickets"] += 1
        
        logger.info(f"Created ticket {ticket_id} for user {ticket_data.user_hash}")
        return TicketResponse(**ticket)
    
    def get_tickets(self, 
                   sla_filter: Optional[str] = None,
                   language_filter: Optional[str] = None,
                   status_filter: Optional[str] = None,
                   priority_filter: Optional[str] = None,
                   assigned_filter: Optional[str] = None,
                   limit: int = 50,
                   offset: int = 0) -> List[TicketResponse]:
        """Get tickets with filtering options."""
        tickets = list(tickets_db.values())
        
        # Apply filters
        if sla_filter:
            tickets = [t for t in tickets if t["sla_status"] == sla_filter]
        
        if language_filter:
            tickets = [t for t in tickets if t["language"].lower() == language_filter.lower()]
        
        if status_filter:
            tickets = [t for t in tickets if t["status"] == status_filter]
        
        if priority_filter:
            tickets = [t for t in tickets if t["priority"] == priority_filter]
        
        if assigned_filter:
            if assigned_filter == "unassigned":
                tickets = [t for t in tickets if t["assigned_to"] is None]
            else:
                tickets = [t for t in tickets if t["assigned_to"] == assigned_filter]
        
        # Sort by priority and creation date
        priority_order = {TicketPriority.URGENT: 4, TicketPriority.HIGH: 3, TicketPriority.MEDIUM: 2, TicketPriority.LOW: 1}
        tickets.sort(key=lambda x: (priority_order.get(x["priority"], 0), x["created_at"]), reverse=True)
        
        # Apply pagination
        total_tickets = tickets[offset:offset + limit]
        
        # Update SLA status for each ticket
        for ticket in total_tickets:
            sla_status, _ = self._calculate_sla_status(
                ticket["created_at"], 
                ticket["priority"], 
                ticket["status"]
            )
            ticket["sla_status"] = sla_status
        
        return [TicketResponse(**ticket) for ticket in total_tickets]
    
    def get_ticket(self, ticket_id: str) -> TicketResponse:
        """Get a specific ticket by ID."""
        if ticket_id not in tickets_db:
            raise HTTPException(status_code=404, detail="Ticket not found")
        
        ticket = tickets_db[ticket_id]
        
        # Update SLA status
        sla_status, _ = self._calculate_sla_status(
            ticket["created_at"], 
            ticket["priority"], 
            ticket["status"]
        )
        ticket["sla_status"] = sla_status
        
        return TicketResponse(**ticket)
    
    def assign_ticket(self, ticket_id: str, assignment: TicketAssignment) -> TicketResponse:
        """Assign ticket to a mentor/agent."""
        if ticket_id not in tickets_db:
            raise HTTPException(status_code=404, detail="Ticket not found")
        
        ticket = tickets_db[ticket_id]
        ticket["assigned_to"] = assignment.assigned_to
        ticket["updated_at"] = datetime.utcnow()
        
        if ticket["status"] == TicketStatus.OPEN:
            ticket["status"] = TicketStatus.IN_PROGRESS
        
        # Add assignment comment
        if assignment.notes:
            self.add_comment(ticket_id, TicketComment(
                comment=f"Ticket assigned to {assignment.assigned_to}. Notes: {assignment.notes}",
                author="system",
                is_internal=True
            ))
        
        logger.info(f"Assigned ticket {ticket_id} to {assignment.assigned_to}")
        return TicketResponse(**ticket)
    
    def resolve_ticket(self, ticket_id: str, resolution: TicketResolution) -> TicketResponse:
        """Resolve a ticket."""
        if ticket_id not in tickets_db:
            raise HTTPException(status_code=404, detail="Ticket not found")
        
        ticket = tickets_db[ticket_id]
        ticket["status"] = TicketStatus.RESOLVED
        ticket["resolved_at"] = datetime.utcnow()
        ticket["updated_at"] = datetime.utcnow()
        ticket["resolution_notes"] = resolution.resolution_notes
        
        # Add resolution comment
        self.add_comment(ticket_id, TicketComment(
            comment=f"Ticket resolved by {resolution.resolved_by}. Resolution: {resolution.resolution_notes}",
            author=resolution.resolved_by,
            is_internal=False
        ))
        
        # Update student profile
        profile_key = f"{ticket['user_hash']}_{ticket['course_id']}"
        if profile_key in student_profiles:
            student_profiles[profile_key]["resolved_tickets"] += 1
        
        logger.info(f"Resolved ticket {ticket_id}")
        return TicketResponse(**ticket)
    
    def add_comment(self, ticket_id: str, comment: TicketComment) -> CommentResponse:
        """Add a comment to a ticket."""
        if ticket_id not in tickets_db:
            raise HTTPException(status_code=404, detail="Ticket not found")
        
        comment_id = f"CMT-{len(comments_db[ticket_id]) + 1:04d}"
        comment_data = {
            "id": comment_id,
            "ticket_id": ticket_id,
            "comment": comment.comment,
            "author": comment.author,
            "is_internal": comment.is_internal,
            "created_at": datetime.utcnow()
        }
        
        comments_db[ticket_id].append(comment_data)
        tickets_db[ticket_id]["comments_count"] += 1
        tickets_db[ticket_id]["updated_at"] = datetime.utcnow()
        
        return CommentResponse(**comment_data)
    
    def get_ticket_comments(self, ticket_id: str) -> List[CommentResponse]:
        """Get all comments for a ticket."""
        if ticket_id not in tickets_db:
            raise HTTPException(status_code=404, detail="Ticket not found")
        
        comments = comments_db.get(ticket_id, [])
        return [CommentResponse(**comment) for comment in comments]
    
    def get_student_profile(self, user_hash: str, course_id: str) -> StudentProfile:
        """Get student 360 profile."""
        profile_key = f"{user_hash}_{course_id}"
        profile = student_profiles.get(profile_key)
        
        if not profile:
            # Create default profile
            risk_score = self._get_student_risk_score(user_hash, course_id)
            profile = student_profiles[profile_key]
        
        return StudentProfile(**profile)

# Initialize service
ticket_service = TicketService()

# API Endpoints
@app.post("/api/tickets", response_model=TicketResponse)
async def create_ticket(ticket: TicketCreate):
    """Create a new support ticket."""
    logger.info(f"Creating ticket for user {ticket.user_hash}")
    return ticket_service.create_ticket(ticket)

@app.get("/api/tickets", response_model=List[TicketResponse])
async def get_tickets(
    sla_filter: Optional[str] = Query(None, description="Filter by SLA status"),
    language: Optional[str] = Query(None, description="Filter by language"),
    status: Optional[str] = Query(None, description="Filter by status"),
    priority: Optional[str] = Query(None, description="Filter by priority"),
    assigned_to: Optional[str] = Query(None, description="Filter by assignee"),
    limit: int = Query(50, ge=1, le=100, description="Number of tickets to return"),
    offset: int = Query(0, ge=0, description="Number of tickets to skip")
):
    """Get tickets with filtering options."""
    return ticket_service.get_tickets(
        sla_filter=sla_filter,
        language_filter=language,
        status_filter=status,
        priority_filter=priority,
        assigned_filter=assigned_to,
        limit=limit,
        offset=offset
    )

@app.get("/api/tickets/{ticket_id}", response_model=TicketResponse)
async def get_ticket(ticket_id: str = Path(..., description="Ticket ID")):
    """Get a specific ticket by ID."""
    return ticket_service.get_ticket(ticket_id)

@app.patch("/api/tickets/{ticket_id}/assign", response_model=TicketResponse)
async def assign_ticket(
    ticket_id: str = Path(..., description="Ticket ID"),
    assignment: TicketAssignment = ...
):
    """Assign a ticket to a mentor/agent."""
    return ticket_service.assign_ticket(ticket_id, assignment)

@app.patch("/api/tickets/{ticket_id}/resolve", response_model=TicketResponse)
async def resolve_ticket(
    ticket_id: str = Path(..., description="Ticket ID"),
    resolution: TicketResolution = ...
):
    """Resolve a ticket."""
    return ticket_service.resolve_ticket(ticket_id, resolution)

@app.post("/api/tickets/{ticket_id}/comment", response_model=CommentResponse)
async def add_comment(
    ticket_id: str = Path(..., description="Ticket ID"),
    comment: TicketComment = ...
):
    """Add a comment to a ticket."""
    return ticket_service.add_comment(ticket_id, comment)

@app.get("/api/tickets/{ticket_id}/comments", response_model=List[CommentResponse])
async def get_ticket_comments(ticket_id: str = Path(..., description="Ticket ID")):
    """Get all comments for a ticket."""
    return ticket_service.get_ticket_comments(ticket_id)

@app.get("/api/students/{user_hash}/profile", response_model=StudentProfile)
async def get_student_profile(
    user_hash: str = Path(..., description="User hash"),
    course_id: str = Query(..., description="Course ID")
):
    """Get student 360 profile."""
    return ticket_service.get_student_profile(user_hash, course_id)

@app.get("/api/tickets/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "total_tickets": len(tickets_db),
        "total_comments": sum(len(comments) for comments in comments_db.values()),
        "total_students": len(student_profiles)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "tickets:app",
        host="0.0.0.0",
        port=8003,
        reload=True
    )
