from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict, Any
from pydantic import BaseModel
import logging

from models.meta_lead_generation.inference import MetaLeadGenerationInference

router = APIRouter(prefix="/meta-lead-generation", tags=["Meta Lead Generation"])

# Initialize the meta model
meta_scorer = MetaLeadGenerationInference()

class LeadData(BaseModel):
    lead_id: str
    age: int
    income: float
    employment_status: str
    location: str
    consent_given: bool
    # Optional fields for specialized models
    family_size: int = None
    health_conditions: List[str] = None
    marital_status: str = None
    dependents_count: int = None
    smoking_status: str = None

class BatchLeadRequest(BaseModel):
    leads: List[LeadData]

@router.post("/score-lead")
async def score_lead(lead_data: LeadData):
    """
    Score a single lead across all insurance products using the meta model
    """
    try:
        lead_dict = lead_data.dict()
        result = meta_scorer.score_lead(lead_dict)
        
        return {
            "status": "success",
            "data": result
        }
        
    except Exception as e:
        logging.error(f"Error scoring lead {lead_data.lead_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error scoring lead: {str(e)}")

@router.post("/batch-score-leads")
async def batch_score_leads(request: BatchLeadRequest):
    """
    Score multiple leads in batch using the meta model
    """
    try:
        leads_data = [lead.dict() for lead in request.leads]
        results = meta_scorer.batch_score_leads(leads_data)
        
        return {
            "status": "success",
            "total_leads": len(results),
            "data": results
        }
        
    except Exception as e:
        logging.error(f"Error in batch scoring: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in batch scoring: {str(e)}")

@router.post("/lead-recommendations")
async def get_lead_recommendations(lead_data: LeadData):
    """
    Get detailed recommendations and action plan for a lead
    """
    try:
        lead_dict = lead_data.dict()
        recommendations = meta_scorer.get_lead_recommendations(lead_dict)
        
        return {
            "status": "success",
            "data": recommendations
        }
        
    except Exception as e:
        logging.error(f"Error getting recommendations for lead {lead_data.lead_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting recommendations: {str(e)}")

@router.get("/lead-insights/{lead_id}")
async def get_lead_insights(lead_id: str, lead_data: Dict[str, Any]):
    """
    Get comprehensive insights about a specific lead
    """
    try:
        lead_data['lead_id'] = lead_id
        insights = meta_scorer.model.get_lead_insights(lead_data)
        
        return {
            "status": "success",
            "lead_id": lead_id,
            "insights": insights
        }
        
    except Exception as e:
        logging.error(f"Error getting insights for lead {lead_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting insights: {str(e)}")

@router.get("/model-health")
async def check_model_health():
    """
    Check the health of all underlying models
    """
    try:
        # Test with sample data
        test_lead = {
            'lead_id': 'HEALTH_CHECK',
            'age': 35,
            'income': 75000,
            'employment_status': 'employed',
            'consent_given': True
        }
        
        result = meta_scorer.score_lead(test_lead)
        
        return {
            "status": "healthy",
            "meta_model": "operational",
            "underlying_models": {
                "base_insurance": "operational" if result['product_scores'].get('base', 0) > 0 else "error",
                "healthcare": "operational" if result['product_scores'].get('healthcare', 0) > 0 else "error",
                "life_insurance": "operational" if result['product_scores'].get('life', 0) > 0 else "error"
            },
            "test_score": result['overall_score']
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }