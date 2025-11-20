"""
Health Insurance Scoring API
FastAPI router for health insurance lead scoring with ensemble models
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import logging
from datetime import datetime, timezone
from backend.models.insurance_lead_scoring.inference import InsuranceLeadScorer
from backend.models.healthcare_insurance_scoring.ensemble_scorer import EnsembleHealthInsuranceScorer

router = APIRouter(prefix="/health-insurance", tags=["Health Insurance"])
logger = logging.getLogger(__name__)

# Initialize scorers
try:
    ensemble_scorer = EnsembleHealthInsuranceScorer()
except Exception as e:
    logger.error(f"Could not initialize health insurance ensemble scorer: {e}")
    ensemble_scorer = None

try:
    xgboost_scorer = InsuranceLeadScorer()
except Exception as e:
    logger.error(f"Could not initialize XGBoost scorer: {e}")
    xgboost_scorer = None


def get_health_scorer():
    """Get the best available health insurance scorer (ensemble preferred)"""
    if ensemble_scorer is not None:
        return ensemble_scorer
    elif xgboost_scorer is not None:
        return xgboost_scorer
    else:
        return None


class HealthInsuranceLeadData(BaseModel):
    lead_id: str
    age: int = Field(..., ge=18, le=100)
    income: float = Field(..., ge=0)
    policy_type: str = Field(..., pattern="^(auto|home|life|health)$")
    quote_requests_30d: int = Field(..., ge=0)
    social_engagement_score: float = Field(..., ge=0, le=10)
    location_risk_score: float = Field(..., ge=0, le=10)
    previous_insurance: str = Field(..., pattern="^(yes|no)$")
    credit_score_proxy: Optional[int] = Field(None, ge=300, le=850)
    consent_given: bool
    consent_timestamp: Optional[str] = None


class HealthInsuranceScoringResponse(BaseModel):
    lead_id: str
    score: float
    confidence: Optional[float] = None
    timestamp: str
    model_version: Optional[str] = None
    model_type: Optional[str] = None
    error: Optional[str] = None
    models_used: Optional[List[str]] = None
    xgboost_score: Optional[float] = None
    xgboost_confidence: Optional[float] = None
    deep_learning_score: Optional[float] = None
    deep_learning_confidence: Optional[float] = None
    ensemble_weights: Optional[Dict] = None


@router.post("/score-lead", response_model=HealthInsuranceScoringResponse)
async def score_health_insurance_lead(lead: HealthInsuranceLeadData):
    """Score a single health insurance lead using ensemble"""
    scorer = get_health_scorer()
    if scorer is None:
        raise HTTPException(status_code=503, detail="Health insurance scoring model not available")

    try:
        result = scorer.score_lead(lead.dict())
        return HealthInsuranceScoringResponse(**result)
    except Exception as e:
        logger.error(f"Error scoring health insurance lead: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/score-leads", response_model=List[HealthInsuranceScoringResponse])
async def score_health_insurance_leads(leads: List[HealthInsuranceLeadData]):
    """Score multiple health insurance leads"""
    scorer = get_health_scorer()
    if scorer is None:
        raise HTTPException(status_code=503, detail="Health insurance scoring model not available")

    results = []
    for lead in leads:
        try:
            result = scorer.score_lead(lead.dict())
            results.append(HealthInsuranceScoringResponse(**result))
        except Exception as e:
            logger.error(f"Error scoring lead {lead.lead_id}: {e}")
            results.append(HealthInsuranceScoringResponse(
                lead_id=lead.lead_id,
                score=0,
                error=str(e),
                timestamp=datetime.now(timezone.utc).isoformat()
            ))

    return results


@router.post("/compare-models")
async def compare_health_insurance_models(lead: HealthInsuranceLeadData):
    """Compare all available models for a health insurance lead"""
    results = {
        'lead_id': lead.lead_id,
        'models': {}
    }
    
    lead_dict = lead.dict()
    
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
            logger.error(f"XGBoost scoring failed: {e}")
            results['models']['xgboost'] = {'available': False, 'error': str(e)}
    
    # Try Deep Learning (via ensemble)
    if ensemble_scorer and ensemble_scorer.deep_learning_scorer:
        try:
            dl_result = ensemble_scorer.deep_learning_scorer.score_lead(lead_dict)
            results['models']['deep_learning'] = {
                'score': dl_result.get('score', 0),
                'confidence': dl_result.get('confidence', 0),
                'available': True
            }
        except Exception as e:
            logger.error(f"Deep Learning scoring failed: {e}")
            results['models']['deep_learning'] = {'available': False, 'error': str(e)}
    
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
            logger.error(f"Ensemble scoring failed: {e}")
            results['models']['ensemble'] = {'available': False, 'error': str(e)}
    
    return results


@router.get("/health")
async def health_check():
    """Health check for health insurance scoring service"""
    scorer = get_health_scorer()
    return {
        "status": "healthy",
        "health_insurance_model_loaded": scorer is not None,
        "ensemble_available": ensemble_scorer is not None,
        "xgboost_available": xgboost_scorer is not None
    }


@router.get("/model-info")
async def model_info():
    """Get health insurance model information"""
    return {
        "model_type": "Ensemble (XGBoost + Deep Learning)",
        "version": "1.0",
        "ensemble_weights": {
            "xgboost_default": 0.4,
            "deep_learning_default": 0.6,
            "adaptive": True
        },
        "features": [
            "age", "income", "policy_type", "quote_requests_30d",
            "social_engagement_score", "location_risk_score",
            "previous_insurance", "credit_score_proxy"
        ],
        "compliance": ["GDPR", "CCPA", "TCPA"]
    }

