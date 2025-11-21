from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import logging
from backend.models.insurance_lead_scoring.inference import InsuranceLeadScorer
from backend.models.auto_insurance_scoring.ensemble_scorer import EnsembleAutoInsuranceScorer
import os
import random
from datetime import datetime, timedelta

router = APIRouter(prefix="/auto-insurance", tags=["Auto Insurance"])
logger = logging.getLogger(__name__)

# Initialize scorers
try:
    ensemble_scorer = EnsembleAutoInsuranceScorer(
        xgboost_weight=0.4,
        deep_learning_weight=0.6,
        use_adaptive_weighting=True
    )
    logger.info("Auto insurance ensemble scorer initialized successfully")
except Exception as e:
    logger.warning(f"Ensemble scorer failed to initialize: {e}. Falling back to XGBoost only.")
    ensemble_scorer = None

# Fallback to XGBoost only
try:
    xgboost_scorer = InsuranceLeadScorer()
    logger.info("XGBoost scorer initialized successfully")
except Exception as e:
    logger.error(f"XGBoost scorer failed to initialize: {e}")
    xgboost_scorer = None

def get_auto_scorer():
    """Get the best available auto insurance scorer (ensemble preferred)"""
    if ensemble_scorer is not None:
        return ensemble_scorer
    elif xgboost_scorer is not None:
        return xgboost_scorer
    else:
        return None

class AutoInsuranceLeadData(BaseModel):
    lead_id: str
    age: int = Field(..., ge=18, le=100)
    income: float = Field(..., ge=0)
    policy_type: str = Field(..., pattern="^(auto|home|life)$")
    quote_requests_30d: int = Field(..., ge=0)
    social_engagement_score: float = Field(..., ge=0, le=10)
    location_risk_score: float = Field(..., ge=0, le=10)
    previous_insurance: str = Field(..., pattern="^(yes|no)$")
    credit_score_proxy: Optional[int] = Field(None, ge=300, le=850)
    consent_given: bool
    consent_timestamp: Optional[str] = None

class AutoInsuranceScoringResponse(BaseModel):
    lead_id: str
    score: float
    confidence: Optional[float] = None
    timestamp: str
    model_version: Optional[str] = None
    model_type: Optional[str] = None  # 'xgboost', 'deep_learning', or 'ensemble'
    error: Optional[str] = None
    # Ensemble-specific fields
    models_used: Optional[List[str]] = None
    xgboost_score: Optional[float] = None
    xgboost_confidence: Optional[float] = None
    deep_learning_score: Optional[float] = None
    deep_learning_confidence: Optional[float] = None
    ensemble_weights: Optional[Dict] = None

@router.post("/score-lead", response_model=AutoInsuranceScoringResponse)
async def score_auto_insurance_lead(lead: AutoInsuranceLeadData):
    """Score a single auto insurance lead using ensemble model"""
    scorer = get_auto_scorer()
    if scorer is None:
        raise HTTPException(status_code=503, detail="Auto insurance scoring model not available")
    try:
        result = scorer.score_lead(lead.dict())
        return AutoInsuranceScoringResponse(**result)
    except Exception as e:
        logger.error(f"Error scoring auto insurance lead: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/score-leads", response_model=List[AutoInsuranceScoringResponse])
async def score_auto_insurance_leads(leads: List[AutoInsuranceLeadData]):
    """Score multiple auto insurance leads"""
    scorer = get_auto_scorer()
    if scorer is None:
        raise HTTPException(status_code=503, detail="Auto insurance scoring model not available")
    try:
        lead_dicts = [lead.dict() for lead in leads]
        results = scorer.batch_score(lead_dicts)
        return [AutoInsuranceScoringResponse(**result) for result in results]
    except Exception as e:
        logger.error(f"Error scoring auto insurance leads: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/compare-models")
async def compare_scoring_models(lead: AutoInsuranceLeadData):
    """
    Compare scores from XGBoost, Deep Learning, and Ensemble models
    
    This endpoint is useful for:
    - Model performance analysis
    - Understanding model agreement/disagreement
    - Debugging and validation
    """
    lead_dict = lead.dict()
    results = {
        'lead_id': lead.lead_id,
        'models': {}
    }
    
    # Try XGBoost
    if xgboost_scorer:
        try:
            xgb_result = xgboost_scorer.score_lead(lead_dict)
            results['models']['xgboost'] = {
                'score': xgb_result.get('score', 0),
                'confidence': xgb_result.get('confidence', 0),
                'available': True
            }
        except Exception as e:
            results['models']['xgboost'] = {
                'available': False,
                'error': str(e)
            }
    else:
        results['models']['xgboost'] = {'available': False}
    
    # Try Deep Learning (if ensemble is available, it has DL)
    if ensemble_scorer and hasattr(ensemble_scorer, 'deep_learning_scorer'):
        try:
            dl_result = ensemble_scorer.deep_learning_scorer.score_lead(lead_dict)
            results['models']['deep_learning'] = {
                'score': dl_result.get('score', 0),
                'confidence': dl_result.get('confidence', 0),
                'available': True
            }
        except Exception as e:
            results['models']['deep_learning'] = {
                'available': False,
                'error': str(e)
            }
    else:
        results['models']['deep_learning'] = {'available': False}
    
    # Try Ensemble
    if ensemble_scorer:
        try:
            ensemble_result = ensemble_scorer.score_lead(lead_dict)
            results['models']['ensemble'] = {
                'score': ensemble_result.get('score', 0),
                'confidence': ensemble_result.get('confidence', 0),
                'ensemble_weights': ensemble_result.get('ensemble_weights', {}),
                'available': True
            }
            results['recommended_score'] = ensemble_result.get('score', 0)
            results['recommended_model'] = 'ensemble'
        except Exception as e:
            results['models']['ensemble'] = {
                'available': False,
                'error': str(e)
            }
    else:
        results['models']['ensemble'] = {'available': False}

    # Set recommended score if ensemble not available
    if 'recommended_score' not in results:
        if results['models']['xgboost'].get('available'):
            results['recommended_score'] = results['models']['xgboost']['score']
            results['recommended_model'] = 'xgboost'
        else:
            results['recommended_score'] = 50.0
            results['recommended_model'] = 'none'

    return results

@router.get("/health")
async def health_check():
    """Health check for auto insurance scoring service"""
    scorer = get_auto_scorer()
    return {
        "status": "healthy",
        "auto_insurance_model_loaded": scorer is not None,
        "ensemble_available": ensemble_scorer is not None,
        "xgboost_available": xgboost_scorer is not None
    }

@router.get("/model-info")
async def auto_insurance_model_info():
    """Get auto insurance model information"""
    return {
        "model_type": "Ensemble (XGBoost + Deep Learning)",
        "version": "1.0_auto_insurance",
        "features": [
            "age", "income", "policy_type", "quote_requests_30d",
            "social_engagement_score", "location_risk_score",
            "previous_insurance", "credit_score_proxy"
        ],
        "ensemble_weights": {
            "xgboost": 0.4,
            "deep_learning": 0.6
        },
        "adaptive_weighting": True,
        "compliance": ["GDPR", "CCPA", "TCPA"]
    }

@router.get("/risk-assessment/{lead_id}")
async def assess_auto_insurance_risk(
    age: int,
    location_risk_score: float,
    previous_insurance: str,
    credit_score_proxy: int
):
    """Assess risk factors for auto insurance underwriting"""
    # Age-based risk
    if age < 25:
        age_risk = "high"
    elif age < 65:
        age_risk = "low"
    else:
        age_risk = "medium"

    # Location risk
    if location_risk_score < 3:
        location_risk = "low"
    elif location_risk_score < 7:
        location_risk = "medium"
    else:
        location_risk = "high"

    # Previous insurance risk
    insurance_risk = "low" if previous_insurance == "yes" else "high"

    # Credit score risk
    if credit_score_proxy >= 720:
        credit_risk = "low"
    elif credit_score_proxy >= 650:
        credit_risk = "medium"
    else:
        credit_risk = "high"

    # Overall risk calculation
    risk_scores = {
        "low": 1,
        "medium": 2,
        "high": 3
    }

    total_risk = (
        risk_scores[age_risk] +
        risk_scores[location_risk] +
        risk_scores[insurance_risk] +
        risk_scores[credit_risk]
    ) / 4

    if total_risk < 1.5:
        overall_risk = "low"
    elif total_risk < 2.5:
        overall_risk = "medium"
    else:
        overall_risk = "high"

    return {
        "risk_factors": {
            "age_risk": age_risk,
            "location_risk": location_risk,
            "insurance_history_risk": insurance_risk,
            "credit_risk": credit_risk
        },
        "overall_risk": overall_risk,
        "underwriting_recommendation": "standard" if overall_risk != "high" else "substandard",
        "estimated_premium_multiplier": 1.0 if overall_risk == "low" else 1.3 if overall_risk == "medium" else 1.8
    }


@router.get("/analytics")
async def get_auto_insurance_analytics(
    date_range: Optional[str] = Query(None, description="Date range filter"),
    score_filter: Optional[str] = Query(None, description="Score filter")
):
    """Get analytics data for auto insurance leads"""
    try:
        # Generate mock analytics data
        # In production, this would query a database
        analytics_data = {
            "total_leads": random.randint(1100, 1300),
            "avg_score": round(random.uniform(0.65, 0.75), 2),
            "high_quality_leads": random.randint(300, 400),
            "model_accuracy": "74.2%",
            "score_distribution": {
                "high": round(random.uniform(25, 32), 1),
                "medium": round(random.uniform(42, 48), 1),
                "low": round(random.uniform(22, 28), 1)
            },
            "vehicle_type_distribution": {
                "sedan": 38.2,
                "suv": 29.4,
                "truck": 18.7,
                "sports_car": 13.7
            },
            "coverage_type_distribution": {
                "liability": 42.3,
                "comprehensive": 31.8,
                "collision": 25.9
            },
            "trend_data": [
                {"date": (datetime.now() - timedelta(days=6)).strftime("%Y-%m-%d"), "leads": random.randint(150, 200), "avg_score": round(random.uniform(0.6, 0.75), 2)},
                {"date": (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d"), "leads": random.randint(150, 200), "avg_score": round(random.uniform(0.6, 0.75), 2)},
                {"date": (datetime.now() - timedelta(days=4)).strftime("%Y-%m-%d"), "leads": random.randint(150, 200), "avg_score": round(random.uniform(0.6, 0.75), 2)},
                {"date": (datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d"), "leads": random.randint(150, 200), "avg_score": round(random.uniform(0.6, 0.75), 2)},
                {"date": (datetime.now() - timedelta(days=2)).strftime("%Y-%m-%d"), "leads": random.randint(150, 200), "avg_score": round(random.uniform(0.6, 0.75), 2)},
                {"date": (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d"), "leads": random.randint(150, 200), "avg_score": round(random.uniform(0.6, 0.75), 2)},
                {"date": datetime.now().strftime("%Y-%m-%d"), "leads": random.randint(150, 200), "avg_score": round(random.uniform(0.6, 0.75), 2)}
            ],
            "filters_applied": {
                "date_range": date_range or "all",
                "score_filter": score_filter or "all"
            }
        }

        return {
            "status": "success",
            "data": analytics_data,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error fetching auto insurance analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Standalone mode for testing
if __name__ == "__main__":
    import uvicorn
    from fastapi import FastAPI

    app = FastAPI(title="Auto Insurance Lead Scoring API", version="1.0.0")
    app.include_router(router)
    uvicorn.run(app, host="0.0.0.0", port=8003)

