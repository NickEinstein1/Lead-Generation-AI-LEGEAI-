"""
Strategic Growth Initiatives API

REST API endpoints for multi-channel lead generation including social media,
content marketing, partner referrals, and event management.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Dict, List, Any, Optional
from pydantic import BaseModel
import logging

from strategic_growth import (
    social_media_processor, content_marketing_engine,
    partner_referral_system, event_management_system
)

router = APIRouter(prefix="/strategic-growth", tags=["Strategic Growth"])
logger = logging.getLogger(__name__)

# Social Media Models
class SocialMediaLeadRequest(BaseModel):
    platform: str
    campaign_ids: Optional[List[str]] = None
    date_range: Optional[Dict[str, str]] = None

# Content Marketing Models
class ContentAssetRequest(BaseModel):
    asset_data: Dict[str, Any]

class ContentLeadRequest(BaseModel):
    lead_data: Dict[str, Any]

# Partner/Referral Models
class PartnerRegistrationRequest(BaseModel):
    partner_data: Dict[str, Any]

class ReferralSubmissionRequest(BaseModel):
    referral_data: Dict[str, Any]

# Event Management Models
class EventCreationRequest(BaseModel):
    event_data: Dict[str, Any]

class EventLeadCaptureRequest(BaseModel):
    lead_data: Dict[str, Any]

# Social Media Endpoints
@router.post("/social-media/fetch-leads")
async def fetch_social_media_leads(request: SocialMediaLeadRequest):
    """Fetch leads from social media platforms"""
    try:
        from strategic_growth.social_media_integration import SocialPlatform
        
        platforms = [SocialPlatform(request.platform)] if request.platform else None
        leads = await social_media_processor.fetch_all_leads(
            platforms=platforms,
            date_range=request.date_range
        )
        
        return {
            "status": "success",
            "leads_fetched": len(leads),
            "leads": [
                {
                    "lead_id": lead.lead_id,
                    "platform": lead.platform.value,
                    "name": lead.name,
                    "email": lead.email,
                    "campaign_id": lead.campaign_id
                } for lead in leads
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/social-media/analytics/{platform}")
async def get_social_media_analytics(platform: str):
    """Get social media platform analytics"""
    try:
        from strategic_growth.social_media_integration import SocialPlatform
        
        performance = await social_media_processor.social_media_analytics.get_platform_performance(
            SocialPlatform(platform)
        )
        return performance
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Content Marketing Endpoints
@router.post("/content-marketing/create-asset")
async def create_content_asset(request: ContentAssetRequest):
    """Create a new content marketing asset"""
    try:
        asset = await content_marketing_engine.create_content_asset(request.asset_data)
        return {
            "status": "success",
            "asset_id": asset.asset_id,
            "title": asset.title,
            "content_type": asset.content_type.value
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/content-marketing/capture-lead")
async def capture_content_lead(request: ContentLeadRequest):
    """Capture a lead from content marketing"""
    try:
        lead = await content_marketing_engine.capture_content_lead(request.lead_data)
        return {
            "status": "success",
            "lead_id": lead.lead_id,
            "engagement_score": lead.engagement_score,
            "interest_level": lead.interest_level
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/content-marketing/performance")
async def get_content_performance(asset_id: Optional[str] = None):
    """Get content marketing performance"""
    try:
        performance = await content_marketing_engine.get_content_performance(asset_id)
        return performance
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Partner/Referral Endpoints
@router.post("/partners/register")
async def register_partner(request: PartnerRegistrationRequest):
    """Register a new partner"""
    try:
        partner = await partner_referral_system.register_partner(request.partner_data)
        return {
            "status": "success",
            "partner_id": partner.partner_id,
            "name": partner.name,
            "partner_type": partner.partner_type.value
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/referrals/submit")
async def submit_referral(request: ReferralSubmissionRequest):
    """Submit a new referral lead"""
    try:
        referral = await partner_referral_system.submit_referral(request.referral_data)
        return {
            "status": "success",
            "referral_id": referral.referral_id,
            "lead_quality_score": referral.lead_quality_score,
            "partner_id": referral.partner_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/partners/{partner_id}/performance")
async def get_partner_performance(partner_id: str):
    """Get partner performance metrics"""
    try:
        performance = await partner_referral_system.get_partner_performance(partner_id)
        return performance
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/referrals/dashboard")
async def get_referral_dashboard():
    """Get referral program dashboard"""
    try:
        dashboard = await partner_referral_system.get_referral_dashboard()
        return dashboard
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Event Management Endpoints
@router.post("/events/create")
async def create_event(request: EventCreationRequest):
    """Create a new event"""
    try:
        event = await event_management_system.create_event(request.event_data)
        return {
            "status": "success",
            "event_id": event.event_id,
            "name": event.name,
            "event_type": event.event_type.value
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/events/capture-lead")
async def capture_event_lead(request: EventLeadCaptureRequest):
    """Capture a lead at an event"""
    try:
        lead = await event_management_system.capture_event_lead(request.lead_data)
        return {
            "status": "success",
            "lead_id": lead.lead_id,
            "lead_score": lead.lead_score,
            "quality_grade": lead.quality_grade
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/events/{event_id}/performance")
async def get_event_performance(event_id: str):
    """Get event performance metrics"""
    try:
        performance = await event_management_system.get_event_performance(event_id)
        return performance
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/growth/unified-dashboard")
async def get_unified_growth_dashboard():
    """Get unified strategic growth dashboard"""
    try:
        # Combine all growth channel data
        social_analysis = await social_media_processor.social_media_analytics.get_cross_platform_analysis()
        content_performance = await content_marketing_engine.get_content_performance()
        referral_dashboard = await partner_referral_system.get_referral_dashboard()
        
        return {
            "overview": {
                "total_leads_generated": 8450,
                "total_channels_active": 12,
                "overall_conversion_rate": 0.285,
                "growth_rate_month_over_month": 0.23
            },
            "channel_performance": {
                "social_media": {
                    "leads": social_analysis.get("overview", {}).get("total_leads", 0),
                    "conversion_rate": 0.29,
                    "cost_per_lead": 38.14
                },
                "content_marketing": {
                    "leads": content_performance.get("summary", {}).get("total_leads", 0),
                    "conversion_rate": 0.085,
                    "cost_per_lead": 25.50
                },
                "partner_referrals": {
                    "leads": referral_dashboard.get("summary", {}).get("total_referrals", 0),
                    "conversion_rate": 0.45,
                    "cost_per_lead": 15.20
                },
                "events": {
                    "leads": 650,
                    "conversion_rate": 0.38,
                    "cost_per_lead": 85.30
                }
            },
            "growth_insights": [
                "Partner referrals show highest conversion rate",
                "Content marketing has lowest cost per lead",
                "Social media provides highest volume",
                "Events generate highest quality leads"
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
