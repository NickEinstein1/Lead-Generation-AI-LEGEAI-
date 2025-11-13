from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import logging
from backend.models.life_insurance_scoring.inference import LifeInsuranceLeadScorer
from backend.models.insurance_products import (
    LifeInsurancePolicyType,
    get_all_life_insurance_categories,
    get_policy_display_name,
    LIFE_INSURANCE_PRODUCTS
)
import asyncio

router = APIRouter(prefix="/life-insurance", tags=["Life Insurance"])
logger = logging.getLogger(__name__)

# Initialize life insurance scorer (lazy loading)
life_scorer = None

def get_life_scorer():
    """Lazy initialization of life insurance scorer"""
    global life_scorer
    if life_scorer is None:
        try:
            life_scorer = LifeInsuranceLeadScorer()
        except Exception as e:
            logger.warning(f"Life insurance scorer initialization failed: {e}. Using fallback mode.")
            life_scorer = None
    return life_scorer

class LifeInsuranceLeadData(BaseModel):
    lead_id: str
    age: int = Field(..., ge=18, le=85)
    income: float = Field(..., ge=0)
    marital_status: str = Field(..., pattern="^(single|married|divorced|widowed|partnered)$")
    dependents_count: int = Field(..., ge=0, le=10)
    employment_status: str = Field(..., pattern="^(employed|unemployed|self_employed|retired)$")
    health_status: str = Field(..., pattern="^(excellent|good|fair|poor)$")
    smoking_status: str = Field(..., pattern="^(smoker|non_smoker|former_smoker)$")
    coverage_amount_requested: float = Field(..., ge=10000, le=10000000)
    policy_term: Optional[int] = Field(None, ge=5, le=40)
    existing_life_insurance: bool = False
    beneficiary_count: int = Field(..., ge=1, le=10)
    debt_obligations: float = Field(..., ge=0)
    mortgage_balance: float = Field(..., ge=0)
    education_level: str = Field(..., pattern="^(high_school|associates|bachelors|masters|doctorate)$")
    occupation_risk_level: str = Field(..., pattern="^(low|medium|high|very_high)$")
    financial_dependents: int = Field(..., ge=0, le=10)
    estate_planning_needs: int = Field(..., ge=0, le=10)
    primary_goal: Optional[str] = Field(None, pattern="^(income_replacement|estate_planning|retirement_income|wealth_accumulation|final_expense)$")
    consent_given: bool
    consent_timestamp: Optional[str] = None

class PolicyRecommendation(BaseModel):
    """Detailed policy recommendation"""
    policy_type: str
    display_name: str
    category: str
    description: str
    estimated_monthly_premium: float
    estimated_annual_premium: float
    cash_value: bool
    investment_component: bool
    best_for: List[str]
    key_features: List[str]
    underwriting_complexity: str

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
    policy_recommendations: Optional[List[PolicyRecommendation]] = None
    urgency_level: Optional[str] = None
    timestamp: str
    model_version: str
    compliance_status: str
    error: Optional[str] = None

@router.post("/score-lead", response_model=LifeInsuranceScoringResponse)
async def score_life_insurance_lead(lead: LifeInsuranceLeadData):
    """Score a single life insurance lead"""
    scorer = get_life_scorer()
    if scorer is None:
        raise HTTPException(status_code=503, detail="Life insurance scoring model not available")
    try:
        result = scorer.score_lead(lead.dict())
        return LifeInsuranceScoringResponse(**result)
    except Exception as e:
        logger.error(f"Error scoring life insurance lead: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/score-leads", response_model=List[LifeInsuranceScoringResponse])
async def score_life_insurance_leads(leads: List[LifeInsuranceLeadData]):
    """Score multiple life insurance leads"""
    scorer = get_life_scorer()
    if scorer is None:
        raise HTTPException(status_code=503, detail="Life insurance scoring model not available")
    try:
        lead_dicts = [lead.dict() for lead in leads]
        results = scorer.batch_score(lead_dicts)
        return [LifeInsuranceScoringResponse(**result) for result in results]
    except Exception as e:
        logger.error(f"Error scoring life insurance leads: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/coverage-calculator")
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

@router.get("/policy-recommendations/{lead_id}")
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

@router.get("/mortality-risk/{lead_id}")
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

@router.get("/health")
async def health_check():
    """Health check for life insurance scoring service"""
    scorer = get_life_scorer()
    return {
        "status": "healthy",
        "life_insurance_model_loaded": scorer is not None and scorer.model is not None if scorer else False,
        "compliance_status": "active"
    }

@router.get("/model-info")
async def life_insurance_model_info():
    """Get life insurance model information"""
    # Get all policy types from the comprehensive catalog
    all_policy_types = [policy_type.value for policy_type in LifeInsurancePolicyType]

    return {
        "model_type": "XGBoost Life Insurance Regressor",
        "version": "1.0_life_insurance",
        "features": [
            "age", "income", "marital_status", "dependents_count", "employment_status",
            "health_status", "smoking_status", "coverage_amount_requested", "policy_term",
            "existing_life_insurance", "beneficiary_count", "debt_obligations",
            "mortgage_balance", "education_level", "occupation_risk_level",
            "life_stage", "financial_dependents", "estate_planning_needs", "primary_goal"
        ],
        "life_stages": ["young_professional", "family_building", "wealth_accumulation", "estate_planning"],
        "policy_types": all_policy_types,
        "policy_categories": get_all_life_insurance_categories(),
        "risk_factors": ["age", "health", "smoking", "occupation"],
        "compliance": ["GDPR", "CCPA", "TCPA", "State Insurance Regulations"]
    }

@router.get("/policy-types")
async def get_policy_types():
    """
    Get all available life insurance policy types organized by category

    Returns comprehensive list of all life insurance products including:
    - Term Life (various types)
    - Permanent Life (Whole Life, Universal Life, Variable Life)
    - Annuities (Fixed, Variable, Indexed)
    - Specialty Products (Final Expense, etc.)
    """
    categories = {}

    for policy_type, product_info in LIFE_INSURANCE_PRODUCTS.items():
        category = product_info.category

        if category not in categories:
            categories[category] = {
                "category_name": category.replace("_", " ").title(),
                "products": []
            }

        categories[category]["products"].append({
            "policy_type": policy_type.value,
            "display_name": product_info.display_name,
            "description": product_info.description,
            "age_range": {
                "min": product_info.typical_age_range[0],
                "max": product_info.typical_age_range[1]
            },
            "coverage_range": {
                "min": product_info.typical_coverage_range[0],
                "max": product_info.typical_coverage_range[1]
            },
            "features": {
                "cash_value": product_info.cash_value,
                "investment_component": product_info.investment_component,
                "premium_flexibility": product_info.premium_flexibility
            },
            "best_for": product_info.best_for,
            "key_features": product_info.key_features,
            "underwriting_complexity": product_info.underwriting_complexity
        })

    return {
        "total_products": len(LIFE_INSURANCE_PRODUCTS),
        "categories": categories,
        "category_summary": {
            "term": "Temporary coverage for specific period",
            "permanent": "Lifetime coverage with cash value",
            "annuity": "Retirement income products",
            "specialty": "Specialized coverage for specific needs",
            "hybrid": "Combined benefits products"
        }
    }

@router.get("/policy-types/{category}")
async def get_policy_types_by_category(category: str):
    """
    Get policy types for a specific category

    Categories:
    - term: Term life insurance products
    - permanent: Whole life, universal life, variable life
    - annuity: Fixed, variable, and indexed annuities
    - specialty: Final expense, burial, guaranteed issue
    - hybrid: Products with combined benefits
    """
    products = []

    for policy_type, product_info in LIFE_INSURANCE_PRODUCTS.items():
        if product_info.category == category:
            products.append({
                "policy_type": policy_type.value,
                "display_name": product_info.display_name,
                "description": product_info.description,
                "age_range": {
                    "min": product_info.typical_age_range[0],
                    "max": product_info.typical_age_range[1]
                },
                "coverage_range": {
                    "min": product_info.typical_coverage_range[0],
                    "max": product_info.typical_coverage_range[1]
                },
                "features": {
                    "cash_value": product_info.cash_value,
                    "investment_component": product_info.investment_component,
                    "premium_flexibility": product_info.premium_flexibility
                },
                "best_for": product_info.best_for,
                "key_features": product_info.key_features,
                "underwriting_complexity": product_info.underwriting_complexity
            })

    if not products:
        raise HTTPException(status_code=404, detail=f"Category '{category}' not found")

    return {
        "category": category,
        "total_products": len(products),
        "products": products
    }

# Standalone mode for testing
if __name__ == "__main__":
    import uvicorn
    from fastapi import FastAPI

    app = FastAPI(title="Life Insurance Lead Scoring API", version="1.0.0")
    app.include_router(router)
    uvicorn.run(app, host="0.0.0.0", port=8002)
