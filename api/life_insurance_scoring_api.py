from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import logging
from models.life_insurance_scoring.inference import LifeInsuranceLeadScorer
import asyncio

app = FastAPI(title="Life Insurance Lead Scoring API", version="1.0.0")
logger = logging.getLogger(__name__)

# Initialize life insurance scorer
life_scorer = LifeInsuranceLeadScorer()

class LifeInsuranceLeadData(BaseModel):
    lead_id: str
    age: int = Field(..., ge=18, le=85)
    income: float = Field(..., ge=0)
    marital_status: str = Field(..., regex="^(single|married|divorced|widowed|partnered)$")
    dependents_count: int = Field(..., ge=0, le=10)
    employment_status: str = Field(..., regex="^(employed|unemployed|self_employed|retired)$")
    health_status: str = Field(..., regex="^(excellent|good|fair|poor)$")
    smoking_status: str = Field(..., regex="^(smoker|non_smoker|former_smoker)$")
    coverage_amount_requested: float = Field(..., ge=10000, le=10000000)
    policy_term: Optional[int] = Field(None, ge=5, le=40)
    existing_life_insurance: bool = False
    beneficiary_count: int = Field(..., ge=1, le=10)
    debt_obligations: float = Field(..., ge=0)
    mortgage_balance: float = Field(..., ge=0)
    education_level: str = Field(..., regex="^(high_school|associates|bachelors|masters|doctorate)$")
    occupation_risk_level: str = Field(..., regex="^(low|medium|high|very_high)$")
    financial_dependents: int = Field(..., ge=0, le=10)
    estate_planning_needs: int = Field(..., ge=0, le=10)
    consent_given: bool
    consent_timestamp: Optional[str] = None

class LifeInsuranceScoringResponse(BaseModel):
    lead_id: str
    score: float
    base_score: Optional[float] = None
    confidence: Optional[float] = None
    life_stage: Optional[str] = None
    mortality_risk_score: Optional[float] = None
    recommended_coverage: Optional[float] = None
    coverage_adequacy: Optional[str] = None
    coverage_gap: Optional[float] = None
    recommended_policy_type: Optional[str] = None
    urgency_level: Optional[str] = None
    timestamp: str
    model_version: str
    compliance_status: str
    error: Optional[str] = None

@app.post("/score-life-insurance-lead", response_model=LifeInsuranceScoringResponse)
async def score_life_insurance_lead(lead: LifeInsuranceLeadData):
    """Score a single life insurance lead"""
    try:
        result = life_scorer.score_lead(lead.dict())
        return LifeInsuranceScoringResponse(**result)
    except Exception as e:
        logger.error(f"Error scoring life insurance lead: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/score-life-insurance-leads", response_model=List[LifeInsuranceScoringResponse])
async def score_life_insurance_leads(leads: List[LifeInsuranceLeadData]):
    """Score multiple life insurance leads"""
    try:
        lead_dicts = [lead.dict() for lead in leads]
        results = life_scorer.batch_score(lead_dicts)
        return [LifeInsuranceScoringResponse(**result) for result in results]
    except Exception as e:
        logger.error(f"Error scoring life insurance leads: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/life-insurance-coverage-calculator")
async def calculate_coverage_needs(
    income: float,
    dependents: int = 0,
    mortgage: float = 0,
    debt: float = 0,
    years_to_retirement: int = 20
):
    """Calculate recommended life insurance coverage"""
    education_cost = dependents * 50000  # Estimated per child
    income_replacement = income * years_to_retirement * 0.7  # 70% income replacement
    final_expenses = 15000  # Funeral and final expenses
    
    total_coverage = income_replacement + mortgage + debt + education_cost + final_expenses
    
    return {
        "recommended_coverage": total_coverage,
        "breakdown": {
            "income_replacement": income_replacement,
            "mortgage_payoff": mortgage,
            "debt_payoff": debt,
            "education_costs": education_cost,
            "final_expenses": final_expenses
        },
        "coverage_multiples": {
            "conservative": income * 8,
            "moderate": income * 10,
            "comprehensive": income * 12
        }
    }

@app.get("/life-insurance-policy-recommendations/{lead_id}")
async def get_policy_recommendations(lead_id: str, coverage_amount: float):
    """Get detailed policy recommendations for a lead"""
    return {
        "lead_id": lead_id,
        "coverage_amount": coverage_amount,
        "policy_recommendations": [
            {
                "policy_type": "TERM_LIFE",
                "term": "20_YEAR",
                "estimated_premium": coverage_amount * 0.0008,
                "pros": ["Lower cost", "Temporary coverage", "Convertible"],
                "cons": ["No cash value", "Temporary coverage"],
                "best_for": "Young families with temporary needs"
            },
            {
                "policy_type": "WHOLE_LIFE",
                "estimated_premium": coverage_amount * 0.012,
                "pros": ["Permanent coverage", "Cash value", "Guaranteed premiums"],
                "cons": ["Higher cost", "Lower returns"],
                "best_for": "Estate planning and permanent needs"
            },
            {
                "policy_type": "UNIVERSAL_LIFE",
                "estimated_premium": coverage_amount * 0.008,
                "pros": ["Flexible premiums", "Cash value", "Investment options"],
                "cons": ["Variable returns", "Complex"],
                "best_for": "Flexible premium needs with investment growth"
            }
        ]
    }

@app.get("/life-insurance-urgency-leads")
async def get_urgent_life_insurance_leads(urgency_level: str = "HIGH"):
    """Get life insurance leads filtered by urgency level"""
    return {
        "message": f"Filtering life insurance leads with {urgency_level} urgency",
        "urgency_criteria": {
            "CRITICAL": "Family building stage with dependents",
            "HIGH": "Age 55+ or multiple dependents",
            "MEDIUM": "Wealth accumulation or estate planning stage",
            "LOW": "Young professionals without dependents"
        }
    }

@app.get("/life-insurance-mortality-risk/{lead_id}")
async def assess_mortality_risk(
    age: int,
    smoking_status: str,
    health_status: str,
    occupation_risk: str
):
    """Assess mortality risk for underwriting purposes"""
    # This would integrate with actual mortality tables
    base_risk = age / 100
    
    smoking_multiplier = {"smoker": 2.0, "former_smoker": 1.3, "non_smoker": 1.0}
    health_multiplier = {"excellent": 0.8, "good": 1.0, "fair": 1.4, "poor": 2.0}
    occupation_multiplier = {"low": 1.0, "medium": 1.2, "high": 1.5, "very_high": 2.0}
    
    total_risk = (base_risk * 
                  smoking_multiplier.get(smoking_status, 1.0) * 
                  health_multiplier.get(health_status, 1.0) * 
                  occupation_multiplier.get(occupation_risk, 1.0))
    
    return {
        "mortality_risk_score": round(total_risk, 3),
        "risk_category": "low" if total_risk < 0.5 else "medium" if total_risk < 1.0 else "high",
        "underwriting_recommendation": "standard" if total_risk < 1.0 else "substandard",
        "factors": {
            "age_factor": base_risk,
            "smoking_impact": smoking_multiplier.get(smoking_status, 1.0),
            "health_impact": health_multiplier.get(health_status, 1.0),
            "occupation_impact": occupation_multiplier.get(occupation_risk, 1.0)
        }
    }

@app.get("/health")
async def health_check():
    """Health check for life insurance scoring service"""
    return {
        "status": "healthy",
        "life_insurance_model_loaded": life_scorer.model is not None,
        "compliance_status": "active"
    }

@app.get("/life-insurance-model-info")
async def life_insurance_model_info():
    """Get life insurance model information"""
    return {
        "model_type": "XGBoost Life Insurance Regressor",
        "version": "1.0_life_insurance",
        "features": [
            "age", "income", "marital_status", "dependents_count", "employment_status",
            "health_status", "smoking_status", "coverage_amount_requested", "policy_term",
            "existing_life_insurance", "beneficiary_count", "debt_obligations",
            "mortgage_balance", "education_level", "occupation_risk_level",
            "life_stage", "financial_dependents", "estate_planning_needs"
        ],
        "life_stages": ["young_professional", "family_building", "wealth_accumulation", "estate_planning"],
        "policy_types": ["TERM_LIFE", "WHOLE_LIFE", "UNIVERSAL_LIFE", "VARIABLE_LIFE"],
        "risk_factors": ["age", "health", "smoking", "occupation"],
        "compliance": ["GDPR", "CCPA", "TCPA", "State Insurance Regulations"]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)