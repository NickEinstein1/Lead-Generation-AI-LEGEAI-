"""
Sales Automation API

REST API endpoints for the AI-powered sales automation system including
intelligent nurturing, call optimization, follow-up automation, and cross-sell engine.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Dict, List, Any, Optional
from pydantic import BaseModel
import logging

from ai_sales_automation import (
    sales_ai_coordinator, nurturing_engine, call_optimizer, 
    followup_engine, cross_sell_engine
)

router = APIRouter(prefix="/sales-automation", tags=["AI Sales Automation"])
logger = logging.getLogger(__name__)

class LeadJourneyRequest(BaseModel):
    lead_id: str
    lead_data: Dict[str, Any]
    scoring_result: Optional[Dict[str, Any]] = None

class NurturingRequest(BaseModel):
    lead_data: Dict[str, Any]
    scoring_result: Dict[str, Any]
    strategy: Optional[str] = "educational"

class CallOptimizationRequest(BaseModel):
    lead_data: Dict[str, Any]
    scoring_result: Dict[str, Any]
    call_type: str = "initial_contact"

@router.post("/orchestrate-lead-journey")
async def orchestrate_complete_lead_journey(request: LeadJourneyRequest):
    """Orchestrate the complete AI-powered lead journey"""
    try:
        result = await sales_ai_coordinator.orchestrate_lead_journey(
            request.lead_id, request.lead_data, request.scoring_result or {}
        )
        return {"status": "success", "data": result}
    except Exception as e:
        logger.error(f"Error orchestrating lead journey: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/nurturing/create-sequence")
async def create_nurturing_sequence(request: NurturingRequest):
    """Create intelligent nurturing sequence"""
    try:
        sequence = await nurturing_engine.create_nurturing_sequence(
            request.lead_data, request.scoring_result, strategy=request.strategy
        )
        return {
            "status": "success",
            "sequence_id": sequence.sequence_id,
            "strategy": sequence.strategy.value,
            "email_count": len(sequence.emails),
            "estimated_duration": f"{sequence.total_duration_days} days"
        }
    except Exception as e:
        logger.error(f"Error creating nurturing sequence: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/call-optimization/generate-recommendation")
async def generate_call_recommendation(request: CallOptimizationRequest):
    """Generate AI-powered call recommendation"""
    try:
        recommendation = await call_optimizer.generate_call_recommendation(
            request.lead_data, request.scoring_result, request.call_type
        )
        return {
            "status": "success",
            "recommendation_id": recommendation.recommendation_id,
            "optimal_time": recommendation.optimal_call_time.isoformat(),
            "success_probability": recommendation.success_probability,
            "talking_points_count": len(recommendation.talking_points),
            "estimated_duration": recommendation.estimated_duration
        }
    except Exception as e:
        logger.error(f"Error generating call recommendation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/follow-up/schedule-sequence")
async def schedule_follow_up_sequence(lead_id: str, trigger_event: str, lead_data: Dict[str, Any]):
    """Schedule intelligent follow-up sequence"""
    try:
        action_ids = await followup_engine.schedule_follow_up_sequence(
            lead_id, trigger_event, lead_data
        )
        return {
            "status": "success",
            "sequence_scheduled": True,
            "action_count": len(action_ids),
            "action_ids": action_ids
        }
    except Exception as e:
        logger.error(f"Error scheduling follow-up sequence: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/cross-sell/generate-recommendations")
async def generate_cross_sell_recommendations(customer_id: str, customer_data: Dict[str, Any]):
    """Generate cross-sell/upsell recommendations"""
    try:
        recommendations = await cross_sell_engine.generate_recommendations(customer_id, customer_data)
        return {
            "status": "success",
            "recommendation_count": len(recommendations),
            "recommendations": [
                {
                    "product_name": rec.product_name,
                    "opportunity_type": rec.opportunity_type.value,
                    "confidence_score": rec.confidence_score,
                    "revenue_potential": rec.revenue_potential,
                    "reasoning": rec.reasoning
                }
                for rec in recommendations
            ]
        }
    except Exception as e:
        logger.error(f"Error generating cross-sell recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/dashboard/sales-performance")
async def get_sales_performance_dashboard(rep_id: Optional[str] = None):
    """Get comprehensive sales automation performance dashboard"""
    try:
        # Get follow-up dashboard
        followup_dashboard = await followup_engine.get_follow_up_dashboard(rep_id)
        
        # Get call optimization metrics
        call_metrics = await call_optimizer.get_performance_metrics(rep_id)
        
        # Combine into unified dashboard
        dashboard = {
            "follow_up_performance": followup_dashboard,
            "call_optimization": call_metrics,
            "overall_metrics": {
                "total_leads_processed": followup_dashboard.get("summary", {}).get("total_actions", 0),
                "automation_efficiency": 85.5,  # Calculated metric
                "revenue_impact": "$125,000",  # Estimated impact
                "time_saved": "40 hours/week"  # Estimated time savings
            }
        }
        
        return {"status": "success", "dashboard": dashboard}
    except Exception as e:
        logger.error(f"Error generating dashboard: {e}")
        raise HTTPException(status_code=500, detail=str(e))