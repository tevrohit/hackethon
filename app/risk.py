"""
Risk scoring API endpoint for student risk assessment.
Loads ML model from disk and provides risk scores with feature contributions.
"""

import logging
import os
import pickle
import joblib
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from enum import Enum
from fastapi import FastAPI, HTTPException, status, Query, Path
from pydantic import BaseModel, Field
import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Risk band enumeration
class RiskBand(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

# Pydantic models
class UserEventFeatures(BaseModel):
    """User event features for risk scoring"""
    last_login_ts: Optional[float] = Field(None, description="Unix timestamp of last login")
    quizzes_failed_last_3: Optional[int] = Field(0, description="Number of quizzes failed in last 3 attempts")
    avg_watch_time: Optional[float] = Field(0.0, description="Average video watch time in minutes")
    assignments_missed: Optional[int] = Field(0, description="Number of assignments missed")
    ticket_sentiment: Optional[float] = Field(0.0, description="Average sentiment score of support tickets (-1 to 1)")
    
    # Additional computed features
    days_since_last_login: Optional[float] = Field(None, description="Days since last login")
    engagement_score: Optional[float] = Field(None, description="Computed engagement score")

class RiskScoreRequest(BaseModel):
    """Request model for risk scoring"""
    user_id: Optional[str] = Field(None, description="User ID to fetch events for")
    features: Optional[UserEventFeatures] = Field(None, description="Direct feature input")
    
    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "user123",
                "features": {
                    "last_login_ts": 1704067200.0,
                    "quizzes_failed_last_3": 2,
                    "avg_watch_time": 15.5,
                    "assignments_missed": 1,
                    "ticket_sentiment": -0.2
                }
            }
        }

class FeatureContribution(BaseModel):
    """Feature contribution to risk score"""
    feature: str = Field(..., description="Feature name")
    contribution: float = Field(..., description="Contribution to risk score (-1 to 1)")
    value: Union[float, int, str] = Field(..., description="Feature value")

class RiskScoreResponse(BaseModel):
    """Response model for risk scoring"""
    score: float = Field(..., description="Risk score between 0 and 1", ge=0, le=1)
    band: RiskBand = Field(..., description="Risk band classification")
    drivers: List[FeatureContribution] = Field(..., description="Top feature contributions")
    user_id: Optional[str] = Field(None, description="User ID if provided")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Scoring timestamp")
    
    class Config:
        json_schema_extra = {
            "example": {
                "score": 0.75,
                "band": "high",
                "drivers": [
                    {"feature": "days_since_last_login", "contribution": 0.35, "value": 14.5},
                    {"feature": "quizzes_failed_last_3", "contribution": 0.25, "value": 3},
                    {"feature": "assignments_missed", "contribution": 0.15, "value": 2}
                ],
                "user_id": "user123",
                "timestamp": "2024-01-01T12:00:00Z"
            }
        }

class RiskModel:
    """Risk scoring model wrapper"""
    
    def __init__(self, model_path: str = "models/risk_model.joblib"):
        self.model_path = model_path
        self.model = None
        self.feature_names = [
            'last_login_ts', 'quizzes_failed_last_3', 'avg_watch_time', 
            'assignments_missed', 'ticket_sentiment', 'days_since_last_login', 
            'engagement_score'
        ]
        self.load_model()
    
    def load_model(self):
        """Load the trained model from disk"""
        try:
            # Try multiple possible paths
            possible_paths = [
                self.model_path,
                os.path.join("backend", self.model_path),
                os.path.join("app", "models", "risk_model.joblib"),
                os.path.join("models", "risk_model.joblib"),
                "risk_model.joblib"
            ]
            
            model_loaded = False
            for path in possible_paths:
                if os.path.exists(path):
                    try:
                        self.model = joblib.load(path)
                        logger.info(f"Risk model loaded successfully from {path}")
                        model_loaded = True
                        break
                    except Exception as e:
                        logger.warning(f"Failed to load model from {path}: {e}")
                        continue
            
            if not model_loaded:
                logger.warning("No trained model found. Creating mock model for demonstration.")
                self.model = self._create_mock_model()
                
        except Exception as e:
            logger.error(f"Error loading risk model: {e}")
            logger.info("Creating mock model for demonstration")
            self.model = self._create_mock_model()
    
    def _create_mock_model(self):
        """Create a mock model for demonstration purposes"""
        class MockModel:
            def predict_proba(self, X):
                # Simple rule-based mock scoring
                scores = []
                for row in X:
                    # Extract features
                    days_since_login = row[5] if len(row) > 5 and row[5] is not None else 0
                    quizzes_failed = row[1] if len(row) > 1 and row[1] is not None else 0
                    assignments_missed = row[3] if len(row) > 3 and row[3] is not None else 0
                    avg_watch_time = row[2] if len(row) > 2 and row[2] is not None else 0
                    ticket_sentiment = row[4] if len(row) > 4 and row[4] is not None else 0
                    
                    # Simple risk calculation
                    risk_score = 0.0
                    
                    # Days since last login (higher = more risk)
                    if days_since_login > 7:
                        risk_score += min(0.4, days_since_login / 30)
                    
                    # Failed quizzes (more failures = more risk)
                    risk_score += min(0.3, quizzes_failed * 0.1)
                    
                    # Missed assignments (more missed = more risk)
                    risk_score += min(0.2, assignments_missed * 0.05)
                    
                    # Low watch time (less engagement = more risk)
                    if avg_watch_time < 10:
                        risk_score += 0.1
                    
                    # Negative sentiment (more negative = more risk)
                    if ticket_sentiment < 0:
                        risk_score += min(0.1, abs(ticket_sentiment) * 0.1)
                    
                    # Ensure score is between 0 and 1
                    risk_score = max(0.0, min(1.0, risk_score))
                    
                    # Return as probability array [low_risk, high_risk]
                    scores.append([1 - risk_score, risk_score])
                
                return np.array(scores)
            
            def feature_importances_(self):
                # Mock feature importances
                return np.array([0.15, 0.25, 0.10, 0.20, 0.10, 0.35, 0.15])
        
        return MockModel()
    
    def engineer_features(self, features: UserEventFeatures) -> Dict[str, float]:
        """Engineer additional features from raw inputs"""
        engineered = {}
        
        # Copy base features
        engineered['last_login_ts'] = features.last_login_ts or 0
        engineered['quizzes_failed_last_3'] = features.quizzes_failed_last_3 or 0
        engineered['avg_watch_time'] = features.avg_watch_time or 0
        engineered['assignments_missed'] = features.assignments_missed or 0
        engineered['ticket_sentiment'] = features.ticket_sentiment or 0
        
        # Calculate days since last login
        if features.last_login_ts:
            current_ts = datetime.utcnow().timestamp()
            days_diff = (current_ts - features.last_login_ts) / (24 * 3600)
            engineered['days_since_last_login'] = max(0, days_diff)
        else:
            engineered['days_since_last_login'] = 30  # Default to 30 days if no login
        
        # Calculate engagement score
        watch_time_score = min(1.0, (features.avg_watch_time or 0) / 30)  # Normalize to 30 min
        quiz_penalty = min(1.0, (features.quizzes_failed_last_3 or 0) * 0.2)
        assignment_penalty = min(1.0, (features.assignments_missed or 0) * 0.1)
        
        engineered['engagement_score'] = max(0, watch_time_score - quiz_penalty - assignment_penalty)
        
        return engineered
    
    def get_feature_contributions(self, features: Dict[str, float], risk_score: float) -> List[FeatureContribution]:
        """Calculate feature contributions to the risk score"""
        contributions = []
        
        try:
            # Get feature importances (mock if using mock model)
            if hasattr(self.model, 'feature_importances_'):
                importances = self.model.feature_importances_()
            else:
                # Default importances for mock model
                importances = np.array([0.15, 0.25, 0.10, 0.20, 0.10, 0.35, 0.15])
            
            # Calculate contributions based on feature values and importances
            for i, feature_name in enumerate(self.feature_names):
                if feature_name in features:
                    value = features[feature_name]
                    importance = importances[i] if i < len(importances) else 0.1
                    
                    # Normalize contribution based on feature type
                    if feature_name == 'days_since_last_login':
                        # Higher days = higher risk contribution
                        normalized_value = min(1.0, value / 30)  # Normalize to 30 days
                        contribution = importance * normalized_value * risk_score
                    elif feature_name in ['quizzes_failed_last_3', 'assignments_missed']:
                        # Higher counts = higher risk contribution
                        normalized_value = min(1.0, value / 5)  # Normalize to 5 max
                        contribution = importance * normalized_value * risk_score
                    elif feature_name == 'avg_watch_time':
                        # Lower watch time = higher risk contribution
                        normalized_value = max(0, 1 - min(1.0, value / 30))  # Invert and normalize
                        contribution = importance * normalized_value * risk_score
                    elif feature_name == 'ticket_sentiment':
                        # More negative sentiment = higher risk contribution
                        normalized_value = max(0, -value) if value < 0 else 0
                        contribution = importance * normalized_value * risk_score
                    elif feature_name == 'engagement_score':
                        # Lower engagement = higher risk contribution
                        normalized_value = max(0, 1 - value)
                        contribution = importance * normalized_value * risk_score
                    else:
                        contribution = importance * 0.5 * risk_score  # Default
                    
                    contributions.append(FeatureContribution(
                        feature=feature_name,
                        contribution=round(contribution, 3),
                        value=value
                    ))
        
        except Exception as e:
            logger.error(f"Error calculating feature contributions: {e}")
            # Fallback contributions
            for feature_name in self.feature_names:
                if feature_name in features:
                    contributions.append(FeatureContribution(
                        feature=feature_name,
                        contribution=0.1,
                        value=features[feature_name]
                    ))
        
        # Sort by absolute contribution and return top contributors
        contributions.sort(key=lambda x: abs(x.contribution), reverse=True)
        return contributions[:5]  # Return top 5 contributors
    
    def predict_risk(self, features: UserEventFeatures) -> tuple[float, List[FeatureContribution]]:
        """Predict risk score and feature contributions"""
        try:
            # Engineer features
            engineered_features = self.engineer_features(features)
            
            # Prepare feature vector
            feature_vector = []
            for feature_name in self.feature_names:
                feature_vector.append(engineered_features.get(feature_name, 0))
            
            # Make prediction
            X = np.array([feature_vector])
            prediction = self.model.predict_proba(X)
            
            # Extract risk score (probability of high risk)
            risk_score = float(prediction[0][1])  # Second column is high risk probability
            
            # Get feature contributions
            contributions = self.get_feature_contributions(engineered_features, risk_score)
            
            return risk_score, contributions
            
        except Exception as e:
            logger.error(f"Error predicting risk: {e}")
            # Return default values
            return 0.5, []

# Global model instance
risk_model = RiskModel()

def classify_risk_band(score: float) -> RiskBand:
    """Classify risk score into bands"""
    if score < 0.3:
        return RiskBand.LOW
    elif score < 0.7:
        return RiskBand.MEDIUM
    else:
        return RiskBand.HIGH

def fetch_user_events(user_id: str) -> UserEventFeatures:
    """Fetch user events and compute features (mock implementation)"""
    # This would typically query a database
    # For now, return mock data based on user_id
    
    logger.info(f"Fetching events for user: {user_id}")
    
    # Mock data generation based on user_id hash
    import hashlib
    user_hash = int(hashlib.md5(user_id.encode()).hexdigest()[:8], 16)
    
    # Generate deterministic mock features
    np.random.seed(user_hash % 1000)
    
    current_time = datetime.utcnow().timestamp()
    days_ago = np.random.randint(0, 30)
    last_login = current_time - (days_ago * 24 * 3600)
    
    return UserEventFeatures(
        last_login_ts=last_login,
        quizzes_failed_last_3=np.random.randint(0, 5),
        avg_watch_time=np.random.uniform(5, 45),
        assignments_missed=np.random.randint(0, 3),
        ticket_sentiment=np.random.uniform(-1, 1)
    )

# FastAPI endpoint
def create_risk_router():
    """Create FastAPI router for risk endpoints"""
    from fastapi import APIRouter
    
    router = APIRouter(prefix="/api/risk", tags=["risk"])
    
    @router.post("/score", response_model=RiskScoreResponse)
    async def score_risk(request: RiskScoreRequest):
        """
        Calculate risk score for a user based on event features or user_id.
        
        - **user_id**: User identifier to fetch events for
        - **features**: Direct feature input (alternative to user_id)
        
        Returns risk score (0-1), risk band (low/medium/high), and top feature drivers.
        """
        try:
            # Validate input
            if not request.user_id and not request.features:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Either user_id or features must be provided"
                )
            
            # Get features
            if request.features:
                features = request.features
                user_id = request.user_id
            else:
                features = fetch_user_events(request.user_id)
                user_id = request.user_id
            
            # Predict risk
            risk_score, contributions = risk_model.predict_risk(features)
            
            # Classify risk band
            risk_band = classify_risk_band(risk_score)
            
            # Create response
            response = RiskScoreResponse(
                score=round(risk_score, 3),
                band=risk_band,
                drivers=contributions,
                user_id=user_id
            )
            
            logger.info(f"Risk score calculated for user {user_id}: {risk_score:.3f} ({risk_band})")
            
            return response
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error in risk scoring: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Internal server error: {str(e)}"
            )
    
    return router

# For direct usage
router = create_risk_router()
