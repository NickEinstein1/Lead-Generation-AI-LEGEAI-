"""
Customer Experience Enhancement API

REST API endpoints for journey mapping, personalized landing pages,
smart form optimization, and real-time chat integration.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Dict, List, Any, Optional
from pydantic import BaseModel
import logging

from customer_experience import (
    journey_mapping_engine, personalized_pages_engine,
    smart_form_engine, realtime_chat_engine
)

router = APIRouter(prefix="/customer-experience", tags=["Customer Experience"])
logger = logging.getLogger(__name__)

class TouchpointRequest(BaseModel):
    lead_id: str
    touchpoint_data: Dict[str, Any]

class PersonalizedPageRequest(BaseModel):
    lead_data: Dict[str, Any]
    template_id: str
    traffic_data: Optional[Dict[str, Any]] = None

class FormOptimizationRequest(BaseModel):
    form_type: str
    lead_data: Optional[Dict[str, Any]] = None
    device_info: Optional[Dict[str, Any]] = None

class ChatTriggerRequest(BaseModel):
    visitor_data: Dict[str, Any]
    page_data: Dict[str, Any]

# Journey Mapping Endpoints
@router.post("/journey/track-touchpoint")
async def track_touchpoint(request: TouchpointRequest):
    """Track a customer touchpoint"""
    try:
        touchpoint_id = await journey_mapping_engine.track_touchpoint(
            request.lead_id, request.touchpoint_data
        )
        return {"touchpoint_id": touchpoint_id, "status": "tracked"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/journey/analysis/{lead_id}")
async def get_journey_analysis(lead_id: str):
    """Get comprehensive journey analysis"""
    try:
        analysis = await journey_mapping_engine.get_journey_analysis(lead_id)
        return analysis
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/journey/dashboard")
async def get_journey_dashboard(date_range: int = 30):
    """Get journey analytics dashboard"""
    try:
        dashboard = await journey_mapping_engine.get_journey_dashboard(date_range)
        return dashboard
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Personalized Landing Pages Endpoints
@router.post("/landing-pages/generate")
async def generate_personalized_page(request: PersonalizedPageRequest):
    """Generate personalized landing page"""
    try:
        page = await personalized_pages_engine.generate_personalized_page(
            request.lead_data, request.template_id, request.traffic_data
        )
        return {
            "page_id": page.page_id,
            "generated_content": page.generated_content,
            "personalization_applied": page.personalization_applied,
            "predicted_conversion_rate": page.predicted_conversion_rate
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/landing-pages/dashboard")
async def get_personalization_dashboard():
    """Get personalization performance dashboard"""
    try:
        dashboard = await personalized_pages_engine.get_personalization_dashboard()
        return dashboard
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Smart Form Optimization Endpoints
@router.post("/forms/generate-optimized")
async def generate_optimized_form(request: FormOptimizationRequest):
    """Generate optimized form configuration"""
    try:
        form_config = await smart_form_engine.generate_optimized_form(
            request.form_type, request.lead_data, request.device_info
        )
        return form_config
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/forms/analytics")
async def get_form_analytics(form_id: Optional[str] = None):
    """Get form analytics dashboard"""
    try:
        analytics = await smart_form_engine.get_form_analytics_dashboard(form_id)
        return analytics
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Real-time Chat Endpoints
@router.post("/chat/evaluate-trigger")
async def evaluate_chat_trigger(request: ChatTriggerRequest):
    """Evaluate if chat should be triggered"""
    try:
        trigger_result = await realtime_chat_engine.evaluate_chat_trigger(
            request.visitor_data, request.page_data
        )
        return trigger_result or {"should_trigger": False}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/chat/dashboard")
async def get_chat_dashboard():
    """Get chat analytics dashboard"""
    try:
        dashboard = await realtime_chat_engine.get_chat_analytics_dashboard()
        return dashboard
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/experience/unified-dashboard")
async def get_unified_experience_dashboard():
    """Get unified customer experience dashboard"""
    try:
        # Combine all dashboards
        journey_data = await journey_mapping_engine.get_journey_dashboard()
        personalization_data = await personalized_pages_engine.get_personalization_dashboard()
        form_data = await smart_form_engine.get_form_analytics_dashboard()
        chat_data = await realtime_chat_engine.get_chat_analytics_dashboard()
        
        return {
            "overview": {
                "total_customer_interactions": 15420,
                "overall_conversion_rate": 0.298,
                "customer_satisfaction_score": 4.5,
                "experience_optimization_lift": 0.34
            },
            "journey_analytics": journey_data,
            "personalization_performance": personalization_data,
            "form_optimization": form_data,
            "chat_performance": chat_data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))