"""
Observability module for monitoring application health, performance, and SLA compliance.
Provides logging functions, health checks, and SLA breach tracking.
"""

import logging
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from enum import Enum
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Enums
class CallStatus(str, Enum):
    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"
    RATE_LIMITED = "rate_limited"

class HealthStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"

# Data models
@dataclass
class PromptCall:
    """Data structure for prompt call logging"""
    timestamp: datetime
    user_hash: str
    model: str
    prompt_len: int
    retrieved_doc_ids: List[str]
    duration_ms: float
    status: CallStatus
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data

class HealthCheckResponse(BaseModel):
    """Health check response model"""
    status: HealthStatus = Field(..., description="Overall health status")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Health check timestamp")
    uptime_seconds: float = Field(..., description="Application uptime in seconds")
    version: str = Field(default="1.0.0", description="Application version")
    checks: Dict[str, Any] = Field(default_factory=dict, description="Individual health checks")
    metrics: Dict[str, Any] = Field(default_factory=dict, description="Performance metrics")
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "timestamp": "2024-01-01T12:00:00Z",
                "uptime_seconds": 3600.0,
                "version": "1.0.0",
                "checks": {
                    "database": "healthy",
                    "ai_service": "healthy",
                    "vector_db": "healthy"
                },
                "metrics": {
                    "total_requests": 1500,
                    "avg_response_time_ms": 250.5,
                    "error_rate": 0.02,
                    "sla_breaches": 3
                }
            }
        }

class SLAMetrics(BaseModel):
    """SLA metrics response model"""
    total_calls: int = Field(..., description="Total number of calls")
    successful_calls: int = Field(..., description="Number of successful calls")
    failed_calls: int = Field(..., description="Number of failed calls")
    avg_response_time_ms: float = Field(..., description="Average response time in milliseconds")
    p95_response_time_ms: float = Field(..., description="95th percentile response time")
    sla_breaches: int = Field(..., description="Number of SLA breaches")
    error_rate: float = Field(..., description="Error rate (0-1)")
    uptime_percentage: float = Field(..., description="Uptime percentage")

# Global state for observability
class ObservabilityState:
    """Global state for tracking metrics and health"""
    
    def __init__(self):
        self.start_time = datetime.utcnow()
        self.prompt_calls: deque = deque(maxlen=10000)  # Keep last 10k calls
        self.sla_breaches = 0
        self.total_requests = 0
        self.failed_requests = 0
        self.response_times: deque = deque(maxlen=1000)  # Keep last 1k response times
        self.health_checks = {
            "database": True,
            "ai_service": True,
            "vector_db": True,
            "file_system": True
        }
        
        # SLA thresholds
        self.sla_response_time_ms = 5000  # 5 seconds
        self.sla_error_rate_threshold = 0.05  # 5%
        
    def add_prompt_call(self, call: PromptCall):
        """Add a prompt call to the tracking system"""
        self.prompt_calls.append(call)
        self.total_requests += 1
        self.response_times.append(call.duration_ms)
        
        if call.status != CallStatus.SUCCESS:
            self.failed_requests += 1
        
        # Check for SLA breaches
        if call.duration_ms > self.sla_response_time_ms:
            self.sla_breaches += 1
            logger.warning(f"SLA breach: Response time {call.duration_ms}ms exceeds threshold {self.sla_response_time_ms}ms")
        
        # Check error rate SLA
        if self.total_requests > 0:
            error_rate = self.failed_requests / self.total_requests
            if error_rate > self.sla_error_rate_threshold:
                logger.warning(f"SLA breach: Error rate {error_rate:.3f} exceeds threshold {self.sla_error_rate_threshold}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics"""
        uptime = (datetime.utcnow() - self.start_time).total_seconds()
        
        # Calculate response time percentiles
        response_times_list = list(self.response_times)
        avg_response_time = sum(response_times_list) / len(response_times_list) if response_times_list else 0
        
        p95_response_time = 0
        if response_times_list:
            sorted_times = sorted(response_times_list)
            p95_index = int(0.95 * len(sorted_times))
            p95_response_time = sorted_times[p95_index] if p95_index < len(sorted_times) else sorted_times[-1]
        
        error_rate = self.failed_requests / self.total_requests if self.total_requests > 0 else 0
        uptime_percentage = 100.0  # Simplified - would be calculated based on downtime tracking
        
        return {
            "uptime_seconds": uptime,
            "total_requests": self.total_requests,
            "successful_calls": self.total_requests - self.failed_requests,
            "failed_calls": self.failed_requests,
            "avg_response_time_ms": round(avg_response_time, 2),
            "p95_response_time_ms": round(p95_response_time, 2),
            "sla_breaches": self.sla_breaches,
            "error_rate": round(error_rate, 4),
            "uptime_percentage": uptime_percentage,
            "recent_calls": len(self.prompt_calls)
        }
    
    def get_health_status(self) -> HealthStatus:
        """Determine overall health status"""
        metrics = self.get_metrics()
        
        # Check for unhealthy conditions
        if (metrics["error_rate"] > 0.1 or  # 10% error rate
            metrics["sla_breaches"] > 10 or
            not all(self.health_checks.values())):
            return HealthStatus.UNHEALTHY
        
        # Check for degraded conditions
        if (metrics["error_rate"] > 0.05 or  # 5% error rate
            metrics["avg_response_time_ms"] > 3000 or  # 3 second average
            metrics["sla_breaches"] > 5):
            return HealthStatus.DEGRADED
        
        return HealthStatus.HEALTHY

# Global observability state
obs_state = ObservabilityState()

# Core logging functions
def log_prompt_call(
    user_hash: str,
    model: str,
    prompt_len: int,
    retrieved_doc_ids: List[str],
    duration_ms: float,
    status: str
) -> None:
    """
    Log a prompt call with all relevant metrics.
    
    Args:
        user_hash: Hashed user identifier for privacy
        model: Model name used for the call
        prompt_len: Length of the prompt in characters
        retrieved_doc_ids: List of document IDs retrieved for context
        duration_ms: Duration of the call in milliseconds
        status: Call status (success, error, timeout, rate_limited)
    """
    try:
        # Validate status
        call_status = CallStatus(status.lower())
        
        # Create prompt call record
        call = PromptCall(
            timestamp=datetime.utcnow(),
            user_hash=user_hash,
            model=model,
            prompt_len=prompt_len,
            retrieved_doc_ids=retrieved_doc_ids or [],
            duration_ms=duration_ms,
            status=call_status
        )
        
        # Add to tracking
        obs_state.add_prompt_call(call)
        
        # Log the call
        logger.info(
            f"Prompt call logged: user={user_hash[:8]}..., model={model}, "
            f"prompt_len={prompt_len}, docs={len(retrieved_doc_ids)}, "
            f"duration={duration_ms}ms, status={status}"
        )
        
        # Log as structured JSON for external log aggregation
        structured_log = {
            "event": "prompt_call",
            "user_hash": user_hash,
            "model": model,
            "prompt_length": prompt_len,
            "retrieved_documents": len(retrieved_doc_ids),
            "document_ids": retrieved_doc_ids,
            "duration_ms": duration_ms,
            "status": status,
            "timestamp": call.timestamp.isoformat()
        }
        
        # Use a separate logger for structured logs
        structured_logger = logging.getLogger("structured")
        structured_logger.info(json.dumps(structured_log))
        
    except ValueError as e:
        logger.error(f"Invalid status value '{status}': {e}")
        raise
    except Exception as e:
        logger.error(f"Error logging prompt call: {e}")
        raise

def log_error(error_type: str, error_message: str, user_hash: Optional[str] = None, **kwargs):
    """Log application errors with context"""
    error_data = {
        "event": "error",
        "error_type": error_type,
        "error_message": error_message,
        "timestamp": datetime.utcnow().isoformat(),
        **kwargs
    }
    
    if user_hash:
        error_data["user_hash"] = user_hash
    
    logger.error(f"Application error: {error_type} - {error_message}")
    
    # Structured error logging
    structured_logger = logging.getLogger("structured")
    structured_logger.error(json.dumps(error_data))

def log_performance_metric(metric_name: str, value: float, tags: Optional[Dict[str, str]] = None):
    """Log performance metrics"""
    metric_data = {
        "event": "performance_metric",
        "metric_name": metric_name,
        "value": value,
        "timestamp": datetime.utcnow().isoformat(),
        "tags": tags or {}
    }
    
    logger.info(f"Performance metric: {metric_name}={value}")
    
    # Structured metric logging
    structured_logger = logging.getLogger("structured")
    structured_logger.info(json.dumps(metric_data))

# Health check functions
def check_database_health() -> bool:
    """Check database connectivity"""
    try:
        # TODO: Implement actual database health check
        # For now, return True as placeholder
        return True
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return False

def check_ai_service_health() -> bool:
    """Check AI service connectivity"""
    try:
        # TODO: Implement actual AI service health check
        # For now, return True as placeholder
        return True
    except Exception as e:
        logger.error(f"AI service health check failed: {e}")
        return False

def check_vector_db_health() -> bool:
    """Check vector database connectivity"""
    try:
        # TODO: Implement actual vector DB health check
        # For now, return True as placeholder
        return True
    except Exception as e:
        logger.error(f"Vector DB health check failed: {e}")
        return False

def check_file_system_health() -> bool:
    """Check file system accessibility"""
    try:
        import tempfile
        import os
        
        # Try to write and read a temporary file
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(b"health_check")
            tmp_path = tmp.name
        
        with open(tmp_path, 'rb') as tmp:
            content = tmp.read()
        
        os.unlink(tmp_path)
        return content == b"health_check"
        
    except Exception as e:
        logger.error(f"File system health check failed: {e}")
        return False

def update_health_checks():
    """Update all health check statuses"""
    obs_state.health_checks.update({
        "database": check_database_health(),
        "ai_service": check_ai_service_health(),
        "vector_db": check_vector_db_health(),
        "file_system": check_file_system_health()
    })

# FastAPI router for health endpoints
def create_observability_router():
    """Create FastAPI router for observability endpoints"""
    router = APIRouter(prefix="/api/observability", tags=["observability"])
    
    @router.get("/health", response_model=HealthCheckResponse)
    async def health_check():
        """
        Comprehensive health check endpoint.
        
        Returns overall system health status, uptime, and individual component checks.
        """
        try:
            # Update health checks
            update_health_checks()
            
            # Get current metrics
            metrics = obs_state.get_metrics()
            
            # Determine overall health
            health_status = obs_state.get_health_status()
            
            # Prepare individual checks
            checks = {}
            for service, is_healthy in obs_state.health_checks.items():
                checks[service] = "healthy" if is_healthy else "unhealthy"
            
            response = HealthCheckResponse(
                status=health_status,
                uptime_seconds=metrics["uptime_seconds"],
                checks=checks,
                metrics=metrics
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Health check failed: {str(e)}"
            )
    
    @router.get("/metrics", response_model=SLAMetrics)
    async def get_metrics():
        """
        Get detailed SLA and performance metrics.
        """
        try:
            metrics = obs_state.get_metrics()
            
            return SLAMetrics(
                total_calls=metrics["total_requests"],
                successful_calls=metrics["successful_calls"],
                failed_calls=metrics["failed_calls"],
                avg_response_time_ms=metrics["avg_response_time_ms"],
                p95_response_time_ms=metrics["p95_response_time_ms"],
                sla_breaches=metrics["sla_breaches"],
                error_rate=metrics["error_rate"],
                uptime_percentage=metrics["uptime_percentage"]
            )
            
        except Exception as e:
            logger.error(f"Metrics retrieval failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Metrics retrieval failed: {str(e)}"
            )
    
    @router.get("/sla-breaches")
    async def get_sla_breaches():
        """
        Get current SLA breach count and details.
        """
        try:
            return {
                "sla_breaches": obs_state.sla_breaches,
                "sla_response_time_threshold_ms": obs_state.sla_response_time_ms,
                "sla_error_rate_threshold": obs_state.sla_error_rate_threshold,
                "current_error_rate": obs_state.failed_requests / obs_state.total_requests if obs_state.total_requests > 0 else 0,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"SLA breach retrieval failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"SLA breach retrieval failed: {str(e)}"
            )
    
    @router.post("/reset-metrics")
    async def reset_metrics():
        """
        Reset all metrics (for testing/maintenance purposes).
        """
        try:
            global obs_state
            obs_state = ObservabilityState()
            
            logger.info("Metrics reset successfully")
            return {"message": "Metrics reset successfully", "timestamp": datetime.utcnow().isoformat()}
            
        except Exception as e:
            logger.error(f"Metrics reset failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Metrics reset failed: {str(e)}"
            )
    
    return router

# Convenience function for health endpoint (standalone)
def create_health_router():
    """Create a simple health router for the main /health endpoint"""
    router = APIRouter(tags=["health"])
    
    @router.get("/health", response_model=HealthCheckResponse)
    async def health():
        """Simple health check endpoint at root level"""
        # Update health checks
        update_health_checks()
        
        # Get current metrics
        metrics = obs_state.get_metrics()
        
        # Determine overall health
        health_status = obs_state.get_health_status()
        
        # Prepare individual checks
        checks = {}
        for service, is_healthy in obs_state.health_checks.items():
            checks[service] = "healthy" if is_healthy else "unhealthy"
        
        return HealthCheckResponse(
            status=health_status,
            uptime_seconds=metrics["uptime_seconds"],
            checks=checks,
            metrics=metrics
        )
    
    return router

# Export routers
observability_router = create_observability_router()
health_router = create_health_router()

# Utility functions for external use
def get_current_sla_breaches() -> int:
    """Get current SLA breach count"""
    return obs_state.sla_breaches

def get_current_metrics() -> Dict[str, Any]:
    """Get current performance metrics"""
    return obs_state.get_metrics()

def increment_sla_breaches(reason: str = "manual"):
    """Manually increment SLA breach counter"""
    obs_state.sla_breaches += 1
    logger.warning(f"SLA breach incremented manually: {reason}")

# Initialize logging
logger.info("Observability module initialized")
logger.info(f"SLA thresholds: response_time={obs_state.sla_response_time_ms}ms, error_rate={obs_state.sla_error_rate_threshold}")
