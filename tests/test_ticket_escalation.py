"""
Tests for ticket escalation flow and ticket management functionality.
Tests ticket creation, assignment, resolution, and escalation scenarios.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient
from fastapi import status
import json
from datetime import datetime, timedelta

# Test data
SAMPLE_TICKET_DATA = {
    "user_hash": "test_user_hash_123",
    "course_id": "CS101",
    "module_id": "deep_learning_fundamentals",
    "title": "Cannot access course materials",
    "description": "I'm having trouble accessing the course materials for module 1. The links are not working and I get a 404 error.",
    "priority": "medium",
    "language": "English",
    "attachments": ["screenshot1.png", "error_log.txt"]
}

HIGH_PRIORITY_TICKET_DATA = {
    "user_hash": "urgent_user_456",
    "course_id": "CS102",
    "module_id": "neural_networks",
    "title": "Assignment submission deadline issue",
    "description": "The assignment submission portal is not working and the deadline is in 2 hours. This is urgent!",
    "priority": "urgent",
    "language": "English",
    "attachments": []
}

ESCALATION_TICKET_DATA = {
    "user_hash": "escalation_user_789",
    "course_id": "CS103",
    "module_id": "machine_learning",
    "title": "Billing and refund inquiry",
    "description": "I need help with billing issues and want to request a refund for my course enrollment.",
    "priority": "high",
    "language": "English",
    "attachments": []
}

SAMPLE_ASSIGNMENT_DATA = {
    "assigned_to": "mentor_john_doe",
    "notes": "Assigning to John for technical support"
}

SAMPLE_RESOLUTION_DATA = {
    "resolution_notes": "Fixed the course material links. Issue was with the CDN configuration.",
    "resolved_by": "mentor_john_doe"
}

SAMPLE_COMMENT_DATA = {
    "content": "I've looked into this issue and it seems to be a server-side problem. Working on a fix.",
    "author": "mentor_john_doe",
    "is_internal": False
}

class TestTicketCreation:
    """Test cases for ticket creation and initial processing."""
    
    def test_create_ticket_success(self, client):
        """Test successful ticket creation."""
        response = client.post("/api/tickets", json=SAMPLE_TICKET_DATA)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert data["title"] == SAMPLE_TICKET_DATA["title"]
        assert data["description"] == SAMPLE_TICKET_DATA["description"]
        assert data["status"] == "open"
        assert data["priority"] == SAMPLE_TICKET_DATA["priority"]
        assert data["user_hash"] == SAMPLE_TICKET_DATA["user_hash"]
        assert data["course_id"] == SAMPLE_TICKET_DATA["course_id"]
        assert data["sla_status"] in ["on_time", "at_risk", "overdue"]
        assert data["sla_due_date"] is not None
        assert "id" in data
        assert data["assigned_to"] is None
        assert data["resolved_at"] is None
    
    def test_create_urgent_ticket_priority_handling(self, client):
        """Test that urgent tickets get proper priority handling."""
        response = client.post("/api/tickets", json=HIGH_PRIORITY_TICKET_DATA)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert data["priority"] == "urgent"
        assert data["status"] == "open"
        # Urgent tickets should have shorter SLA
        assert data["sla_due_date"] is not None
    
    def test_create_ticket_with_risk_scoring(self, client):
        """Test that tickets include student risk scoring."""
        response = client.post("/api/tickets", json=SAMPLE_TICKET_DATA)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert "student_risk_score" in data
        assert isinstance(data["student_risk_score"], (int, float))
        assert 0 <= data["student_risk_score"] <= 1
    
    def test_create_ticket_invalid_data(self, client):
        """Test ticket creation with invalid data."""
        invalid_data = SAMPLE_TICKET_DATA.copy()
        del invalid_data["title"]  # Remove required field
        
        response = client.post("/api/tickets", json=invalid_data)
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        data = response.json()
        assert "validation error" in data["detail"][0]["type"]
    
    def test_create_ticket_empty_title(self, client):
        """Test ticket creation with empty title."""
        invalid_data = SAMPLE_TICKET_DATA.copy()
        invalid_data["title"] = ""
        
        response = client.post("/api/tickets", json=invalid_data)
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

class TestTicketEscalationFlow:
    """Test cases for ticket escalation scenarios."""
    
    def test_ticket_escalation_scenario(self, client):
        """Test complete ticket escalation flow."""
        # Step 1: Create a ticket
        response = client.post("/api/tickets", json=ESCALATION_TICKET_DATA)
        assert response.status_code == status.HTTP_200_OK
        ticket_data = response.json()
        ticket_id = ticket_data["id"]
        
        # Step 2: Assign ticket to mentor
        assignment_response = client.patch(
            f"/api/tickets/{ticket_id}/assign",
            json=SAMPLE_ASSIGNMENT_DATA
        )
        assert assignment_response.status_code == status.HTTP_200_OK
        assigned_ticket = assignment_response.json()
        assert assigned_ticket["assigned_to"] == SAMPLE_ASSIGNMENT_DATA["assigned_to"]
        assert assigned_ticket["status"] == "in_progress"
        
        # Step 3: Add comment indicating escalation need
        escalation_comment = {
            "content": "This ticket requires escalation to billing team due to refund request.",
            "author": "mentor_john_doe",
            "is_internal": True
        }
        comment_response = client.post(
            f"/api/tickets/{ticket_id}/comment",
            json=escalation_comment
        )
        assert comment_response.status_code == status.HTTP_200_OK
        
        # Step 4: Verify ticket can be retrieved and has proper escalation indicators
        get_response = client.get(f"/api/tickets/{ticket_id}")
        assert get_response.status_code == status.HTTP_200_OK
        current_ticket = get_response.json()
        
        # Check if ticket has escalation indicators
        assert current_ticket["priority"] == "high"
        assert "billing" in current_ticket["description"].lower()
        assert current_ticket["assigned_to"] is not None
    
    def test_sla_breach_escalation(self, client):
        """Test automatic escalation due to SLA breach."""
        # Create a high priority ticket
        response = client.post("/api/tickets", json=HIGH_PRIORITY_TICKET_DATA)
        assert response.status_code == status.HTTP_200_OK
        ticket_data = response.json()
        ticket_id = ticket_data["id"]
        
        # Get the ticket to check SLA status
        get_response = client.get(f"/api/tickets/{ticket_id}")
        assert get_response.status_code == status.HTTP_200_OK
        ticket = get_response.json()
        
        # Verify SLA tracking is in place
        assert ticket["sla_status"] in ["on_time", "at_risk", "overdue"]
        assert ticket["sla_due_date"] is not None
        
        # For urgent tickets, SLA should be tight
        if ticket["priority"] == "urgent":
            sla_due = datetime.fromisoformat(ticket["sla_due_date"].replace('Z', '+00:00'))
            created_at = datetime.fromisoformat(ticket["created_at"].replace('Z', '+00:00'))
            sla_duration = sla_due - created_at
            # Urgent tickets should have SLA <= 4 hours
            assert sla_duration <= timedelta(hours=4)
    
    def test_high_risk_student_escalation(self, client):
        """Test escalation for high-risk students."""
        # Create multiple tickets for the same user to simulate high-risk scenario
        high_risk_data = SAMPLE_TICKET_DATA.copy()
        high_risk_data["user_hash"] = "high_risk_student_999"
        
        # Create first ticket
        response1 = client.post("/api/tickets", json=high_risk_data)
        assert response1.status_code == status.HTTP_200_OK
        
        # Create second ticket (should increase risk score)
        high_risk_data["title"] = "Another issue with course access"
        high_risk_data["description"] = "Still having problems with course materials"
        response2 = client.post("/api/tickets", json=high_risk_data)
        assert response2.status_code == status.HTTP_200_OK
        
        # Check student profile for risk escalation
        profile_response = client.get(
            f"/api/students/{high_risk_data['user_hash']}/profile",
            params={"course_id": high_risk_data["course_id"]}
        )
        assert profile_response.status_code == status.HTTP_200_OK
        profile = profile_response.json()
        
        # High-risk students should have elevated risk scores
        assert profile["total_tickets"] >= 2
        assert profile["risk_score"] > 0.3  # Threshold for escalation consideration
    
    def test_language_based_escalation(self, client):
        """Test escalation based on language requirements."""
        # Create ticket in Hindi
        hindi_ticket = SAMPLE_TICKET_DATA.copy()
        hindi_ticket["language"] = "Hindi"
        hindi_ticket["title"] = "कोर्स मैटेरियल एक्सेस नहीं हो रहा"
        hindi_ticket["description"] = "मुझे कोर्स मैटेरियल एक्सेस करने में समस्या हो रही है।"
        
        response = client.post("/api/tickets", json=hindi_ticket)
        assert response.status_code == status.HTTP_200_OK
        ticket_data = response.json()
        
        assert ticket_data["language"] == "Hindi"
        # Non-English tickets might need specialized handling
        
        # Test filtering by language
        filter_response = client.get("/api/tickets", params={"language_filter": "Hindi"})
        assert filter_response.status_code == status.HTTP_200_OK
        filtered_tickets = filter_response.json()
        
        # Should find the Hindi ticket
        hindi_tickets = [t for t in filtered_tickets if t["language"] == "Hindi"]
        assert len(hindi_tickets) >= 1

class TestTicketAssignmentAndResolution:
    """Test cases for ticket assignment and resolution flow."""
    
    def test_ticket_assignment_flow(self, client):
        """Test complete ticket assignment flow."""
        # Create ticket
        response = client.post("/api/tickets", json=SAMPLE_TICKET_DATA)
        assert response.status_code == status.HTTP_200_OK
        ticket_id = response.json()["id"]
        
        # Assign ticket
        assignment_response = client.patch(
            f"/api/tickets/{ticket_id}/assign",
            json=SAMPLE_ASSIGNMENT_DATA
        )
        assert assignment_response.status_code == status.HTTP_200_OK
        assigned_ticket = assignment_response.json()
        
        assert assigned_ticket["assigned_to"] == SAMPLE_ASSIGNMENT_DATA["assigned_to"]
        assert assigned_ticket["status"] == "in_progress"
        assert assigned_ticket["updated_at"] != assigned_ticket["created_at"]
    
    def test_ticket_resolution_flow(self, client):
        """Test complete ticket resolution flow."""
        # Create and assign ticket
        response = client.post("/api/tickets", json=SAMPLE_TICKET_DATA)
        ticket_id = response.json()["id"]
        
        client.patch(f"/api/tickets/{ticket_id}/assign", json=SAMPLE_ASSIGNMENT_DATA)
        
        # Resolve ticket
        resolution_response = client.patch(
            f"/api/tickets/{ticket_id}/resolve",
            json=SAMPLE_RESOLUTION_DATA
        )
        assert resolution_response.status_code == status.HTTP_200_OK
        resolved_ticket = resolution_response.json()
        
        assert resolved_ticket["status"] == "resolved"
        assert resolved_ticket["resolution_notes"] == SAMPLE_RESOLUTION_DATA["resolution_notes"]
        assert resolved_ticket["resolved_at"] is not None
    
    def test_ticket_comment_system(self, client):
        """Test ticket comment system."""
        # Create ticket
        response = client.post("/api/tickets", json=SAMPLE_TICKET_DATA)
        ticket_id = response.json()["id"]
        
        # Add comment
        comment_response = client.post(
            f"/api/tickets/{ticket_id}/comment",
            json=SAMPLE_COMMENT_DATA
        )
        assert comment_response.status_code == status.HTTP_200_OK
        comment = comment_response.json()
        
        assert comment["content"] == SAMPLE_COMMENT_DATA["content"]
        assert comment["author"] == SAMPLE_COMMENT_DATA["author"]
        assert comment["ticket_id"] == ticket_id
        
        # Get comments
        comments_response = client.get(f"/api/tickets/{ticket_id}/comments")
        assert comments_response.status_code == status.HTTP_200_OK
        comments = comments_response.json()
        
        assert len(comments) >= 1
        assert comments[0]["content"] == SAMPLE_COMMENT_DATA["content"]

class TestTicketFiltering:
    """Test cases for ticket filtering and querying."""
    
    def test_ticket_filtering_by_status(self, client):
        """Test filtering tickets by status."""
        # Create multiple tickets with different statuses
        client.post("/api/tickets", json=SAMPLE_TICKET_DATA)
        
        # Get open tickets
        response = client.get("/api/tickets", params={"status_filter": "open"})
        assert response.status_code == status.HTTP_200_OK
        tickets = response.json()
        
        # All returned tickets should be open
        for ticket in tickets:
            assert ticket["status"] == "open"
    
    def test_ticket_filtering_by_priority(self, client):
        """Test filtering tickets by priority."""
        # Create urgent ticket
        client.post("/api/tickets", json=HIGH_PRIORITY_TICKET_DATA)
        
        # Filter by urgent priority
        response = client.get("/api/tickets", params={"priority_filter": "urgent"})
        assert response.status_code == status.HTTP_200_OK
        tickets = response.json()
        
        # All returned tickets should be urgent
        for ticket in tickets:
            assert ticket["priority"] == "urgent"
    
    def test_ticket_filtering_by_sla_status(self, client):
        """Test filtering tickets by SLA status."""
        # Create ticket
        client.post("/api/tickets", json=SAMPLE_TICKET_DATA)
        
        # Filter by SLA status
        response = client.get("/api/tickets", params={"sla_filter": "on_time"})
        assert response.status_code == status.HTTP_200_OK
        tickets = response.json()
        
        # Verify SLA filtering works
        for ticket in tickets:
            assert ticket["sla_status"] == "on_time"
    
    def test_ticket_filtering_by_assignment(self, client):
        """Test filtering tickets by assignment status."""
        # Create and assign a ticket
        response = client.post("/api/tickets", json=SAMPLE_TICKET_DATA)
        ticket_id = response.json()["id"]
        client.patch(f"/api/tickets/{ticket_id}/assign", json=SAMPLE_ASSIGNMENT_DATA)
        
        # Filter unassigned tickets
        unassigned_response = client.get("/api/tickets", params={"assigned_filter": "unassigned"})
        assert unassigned_response.status_code == status.HTTP_200_OK
        unassigned_tickets = unassigned_response.json()
        
        # Filter assigned tickets
        assigned_response = client.get("/api/tickets", params={"assigned_filter": SAMPLE_ASSIGNMENT_DATA["assigned_to"]})
        assert assigned_response.status_code == status.HTTP_200_OK
        assigned_tickets = assigned_response.json()
        
        # Verify filtering works
        for ticket in assigned_tickets:
            assert ticket["assigned_to"] == SAMPLE_ASSIGNMENT_DATA["assigned_to"]

class TestTicketErrorHandling:
    """Test cases for ticket error handling and edge cases."""
    
    def test_get_nonexistent_ticket(self, client):
        """Test getting a ticket that doesn't exist."""
        response = client.get("/api/tickets/nonexistent_ticket_id")
        assert response.status_code == status.HTTP_404_NOT_FOUND
    
    def test_assign_nonexistent_ticket(self, client):
        """Test assigning a ticket that doesn't exist."""
        response = client.patch(
            "/api/tickets/nonexistent_ticket_id/assign",
            json=SAMPLE_ASSIGNMENT_DATA
        )
        assert response.status_code == status.HTTP_404_NOT_FOUND
    
    def test_resolve_nonexistent_ticket(self, client):
        """Test resolving a ticket that doesn't exist."""
        response = client.patch(
            "/api/tickets/nonexistent_ticket_id/resolve",
            json=SAMPLE_RESOLUTION_DATA
        )
        assert response.status_code == status.HTTP_404_NOT_FOUND
    
    def test_comment_on_nonexistent_ticket(self, client):
        """Test adding comment to a ticket that doesn't exist."""
        response = client.post(
            "/api/tickets/nonexistent_ticket_id/comment",
            json=SAMPLE_COMMENT_DATA
        )
        assert response.status_code == status.HTTP_404_NOT_FOUND
    
    def test_resolve_unassigned_ticket(self, client):
        """Test resolving a ticket that hasn't been assigned."""
        # Create ticket but don't assign it
        response = client.post("/api/tickets", json=SAMPLE_TICKET_DATA)
        ticket_id = response.json()["id"]
        
        # Try to resolve without assignment
        resolution_response = client.patch(
            f"/api/tickets/{ticket_id}/resolve",
            json=SAMPLE_RESOLUTION_DATA
        )
        # This might be allowed or not depending on business rules
        # For now, let's assume it's allowed
        assert resolution_response.status_code in [status.HTTP_200_OK, status.HTTP_400_BAD_REQUEST]

class TestStudentProfile:
    """Test cases for student profile and risk assessment."""
    
    def test_student_profile_creation(self, client):
        """Test that student profiles are created and updated with tickets."""
        # Create ticket for new student
        response = client.post("/api/tickets", json=SAMPLE_TICKET_DATA)
        assert response.status_code == status.HTTP_200_OK
        
        # Get student profile
        profile_response = client.get(
            f"/api/students/{SAMPLE_TICKET_DATA['user_hash']}/profile",
            params={"course_id": SAMPLE_TICKET_DATA["course_id"]}
        )
        assert profile_response.status_code == status.HTTP_200_OK
        profile = profile_response.json()
        
        assert profile["user_hash"] == SAMPLE_TICKET_DATA["user_hash"]
        assert profile["course_id"] == SAMPLE_TICKET_DATA["course_id"]
        assert profile["total_tickets"] >= 1
        assert "risk_score" in profile
        assert "last_activity" in profile
    
    def test_student_risk_score_calculation(self, client):
        """Test student risk score calculation with multiple tickets."""
        user_hash = "risk_test_user_123"
        ticket_data = SAMPLE_TICKET_DATA.copy()
        ticket_data["user_hash"] = user_hash
        
        # Create multiple tickets to increase risk score
        for i in range(3):
            ticket_data["title"] = f"Issue #{i+1}: Course access problem"
            response = client.post("/api/tickets", json=ticket_data)
            assert response.status_code == status.HTTP_200_OK
        
        # Get updated profile
        profile_response = client.get(
            f"/api/students/{user_hash}/profile",
            params={"course_id": ticket_data["course_id"]}
        )
        assert profile_response.status_code == status.HTTP_200_OK
        profile = profile_response.json()
        
        # Risk score should increase with more tickets
        assert profile["total_tickets"] == 3
        assert profile["risk_score"] > 0.2  # Should be elevated due to multiple tickets

class TestTicketHealth:
    """Test cases for ticket service health monitoring."""
    
    def test_ticket_health_endpoint(self, client):
        """Test ticket service health endpoint."""
        response = client.get("/api/tickets/health")
        assert response.status_code == status.HTTP_200_OK
        
        health_data = response.json()
        assert health_data["status"] == "healthy"
        assert "total_tickets" in health_data
        assert "total_comments" in health_data
        assert "total_students" in health_data
        assert isinstance(health_data["total_tickets"], int)
        assert isinstance(health_data["total_comments"], int)
        assert isinstance(health_data["total_students"], int)
