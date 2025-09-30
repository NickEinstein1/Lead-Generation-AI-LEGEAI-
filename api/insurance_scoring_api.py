from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import logging
from models.insurance_lead_scoring.inference import InsuranceLeadScorer
import asyncio

app = FastAPI(title="Insurance Lead Scoring API", version="1.0.0")
logger = logging.getLogger(__name__)

# Initialize scorer
scorer = InsuranceLeadScorer()

class LeadData(BaseModel):
    lead_id: str
    age: int = Field(..., ge=18, le=100)
    income: float = Field(..., ge=0)
    policy_type: str = Field(..., regex="^(auto|home|life)$")
    quote_requests_30d: int = Field(..., ge=0)
    social_engagement_score: float = Field(..., ge=0, le=10)
    location_risk_score: float = Field(..., ge=0, le=10)
    previous_insurance: str = Field(..., regex="^(yes|no)$")
    credit_score_proxy: Optional[int] = Field(None, ge=300, le=850)
    consent_given: bool
    consent_timestamp: Optional[str] = None

class ScoringResponse(BaseModel):
    lead_id: str
    score: float
    confidence: Optional[float] = None
    timestamp: str
    model_version: str
    error: Optional[str] = None

@app.post("/score-lead", response_model=ScoringResponse)
async def score_single_lead(lead: LeadData):
    """Score a single insurance lead"""
    try:
        result = scorer.score_lead(lead.dict())
        return ScoringResponse(**result)
    except Exception as e:
        logger.error(f"Error scoring lead: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/score-leads", response_model=List[ScoringResponse])
async def score_multiple_leads(leads: List[LeadData]):
    """Score multiple insurance leads"""
    try:
        lead_dicts = [lead.dict() for lead in leads]
        results = scorer.batch_score(lead_dicts)
        return [ScoringResponse(**result) for result in results]
    except Exception as e:
        logger.error(f"Error scoring leads: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": scorer.model is not None}

@app.get("/model-info")
async def model_info():
    """Get model information"""
    return {
        "model_type": "XGBoost Regressor",
        "version": "1.0",
        "features": [
            "age", "income", "policy_type", "quote_requests_30d",
            "social_engagement_score", "location_risk_score",
            "previous_insurance", "credit_score_proxy"
        ],
        "compliance": ["GDPR", "CCPA", "TCPA"]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)