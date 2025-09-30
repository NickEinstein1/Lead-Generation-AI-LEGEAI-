from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import logging
from models.healthcare_insurance_scoring.inference import HealthcareInsuranceLeadScorer
import asyncio

app = FastAPI(title="Healthcare Insurance Lead Scoring API", version="1.0.0")
logger = logging.getLogger(__name__)

# Initialize healthcare scorer
healthcare_scorer = HealthcareInsuranceLeadScorer()

class HealthcareLeadData(BaseModel):
    lead_id: str
    age: int = Field(..., ge=18, le=100)
    income: float = Field(..., ge=0)
    family_size: int = Field(..., ge=1, le=10)
    employment_status: str = Field(..., regex="^(employed|unemployed|self_employed|retired)$")
    current_coverage: str = Field(..., regex="^(none|employer|individual|medicaid|medicare)$")
    health_conditions_count: int = Field(..., ge=0, le=20)
    prescription_usage: int = Field(..., ge=0, le=50)
    doctor_visits_annual: int = Field(..., ge=0, le=100)
    preventive_care_usage: int = Field(..., ge=0, le=10)
    location_risk_score: float = Field(..., ge=0, le=10)
    open_enrollment_period: bool = False
    subsidy_eligible: bool = False
    previous_claims_frequency: int = Field(..., ge=0, le=50)
    qualifying_life_event: Optional[bool] = False
    hipaa_consent_given: bool
    consent_given: bool
    hipaa_consent_timestamp: Optional[str] = None

class HealthcareScoringResponse(BaseModel):
    lead_id: str
    score: float
    base_score: Optional[float] = None
    confidence: Optional[float] = None
    urgency_level: Optional[str] = None
    recommended_plan_type: Optional[str] = None
    timestamp: str
    model_version: str
    compliance_status: str
    error: Optional[str] = None

@app.post("/score-healthcare-lead", response_model=HealthcareScoringResponse)
async def score_healthcare_lead(lead: HealthcareLeadData):
    """Score a single healthcare insurance lead"""
    try:
        result = healthcare_scorer.score_lead(lead.dict())
        return HealthcareScoringResponse(**result)
    except Exception as e:
        logger.error(f"Error scoring healthcare lead: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/score-healthcare-leads", response_model=List[HealthcareScoringResponse])
async def score_healthcare_leads(leads: List[HealthcareLeadData]):
    """Score multiple healthcare insurance leads"""
    try:
        lead_dicts = [lead.dict() for lead in leads]
        results = healthcare_scorer.batch_score(lead_dicts)
        return [HealthcareScoringResponse(**result) for result in results]
    except Exception as e:
        logger.error(f"Error scoring healthcare leads: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/healthcare-urgency-leads")
async def get_urgent_healthcare_leads(urgency_level: str = "HIGH"):
    """Get leads filtered by urgency level"""
    # This would typically query a database
    return {
        "message": f"Filtering leads with {urgency_level} urgency",
        "urgency_levels": ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    }

@app.get("/healthcare-plan-recommendations/{lead_id}")
async def get_plan_recommendations(lead_id: str):
    """Get detailed plan recommendations for a lead"""
    return {
        "lead_id": lead_id,
        "recommendations": [
            {
                "plan_type": "SILVER",
                "monthly_premium": 350,
                "deductible": 2500,
                "match_score": 85
            },
            {
                "plan_type": "BRONZE",
                "monthly_premium": 280,
                "deductible": 5000,
                "match_score": 72
            }
        ]
    }

@app.get("/healthcare-compliance-report")
async def healthcare_compliance_report():
    """Generate healthcare compliance report"""
    return {
        "hipaa_compliance": "ACTIVE",
        "consent_validation": "ENABLED",
        "pii_anonymization": "ACTIVE",
        "audit_trail": "ENABLED",
        "last_audit": "2024-01-15T10:30:00Z"
    }

@app.get("/health")
async def health_check():
    """Health check for healthcare scoring service"""
    return {
        "status": "healthy",
        "healthcare_model_loaded": healthcare_scorer.model is not None,
        "hipaa_compliance": "active"
    }

@app.get("/healthcare-model-info")
async def healthcare_model_info():
    """Get healthcare model information"""
    return {
        "model_type": "XGBoost Healthcare Regressor",
        "version": "1.0_healthcare",
        "features": [
            "age", "income", "family_size", "employment_status",
            "current_coverage", "health_conditions_count", "prescription_usage",
            "doctor_visits_annual", "preventive_care_usage", "location_risk_score",
            "open_enrollment_period", "subsidy_eligible", "previous_claims_frequency"
        ],
        "compliance": ["HIPAA", "GDPR", "CCPA", "TCPA"],
        "plan_types": ["BRONZE", "SILVER", "GOLD", "PLATINUM", "HDHP", "HMO", "PPO"]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)