"""
Marketing Automation API
"""
from fastapi import APIRouter, HTTPException, Depends, Query
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime
import logging
import enum

logger = logging.getLogger(__name__)

# Enums
class CampaignType(str, enum.Enum):
    EMAIL = "email"
    SMS = "sms"
    MULTI_CHANNEL = "multi_channel"
    DRIP = "drip"

class CampaignStatus(str, enum.Enum):
    DRAFT = "draft"
    SCHEDULED = "scheduled"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    ARCHIVED = "archived"

class TriggerType(str, enum.Enum):
    TIME_BASED = "time_based"
    EVENT_BASED = "event_based"
    BEHAVIOR_BASED = "behavior_based"
    SCORE_BASED = "score_based"

class SegmentOperator(str, enum.Enum):
    AND = "and"
    OR = "or"

# In-memory storage (replace with database later)
CAMPAIGNS_DB = {}
SEGMENTS_DB = {}
TEMPLATES_DB = {}
TRIGGERS_DB = {}
ANALYTICS_DB = {}
SENDS_DB = {}

CAMPAIGN_ID_COUNTER = 1
SEGMENT_ID_COUNTER = 1
TEMPLATE_ID_COUNTER = 1
TRIGGER_ID_COUNTER = 1
ANALYTICS_ID_COUNTER = 1
SEND_ID_COUNTER = 1

router = APIRouter(prefix="/marketing", tags=["Marketing Automation"])


# ==================== Pydantic Schemas ====================

class CampaignCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    campaign_type: CampaignType
    segment_id: Optional[int] = None
    template_id: Optional[int] = None
    subject_line: Optional[str] = None
    preview_text: Optional[str] = None
    scheduled_at: Optional[datetime] = None
    is_ab_test: bool = False
    ab_test_config: Optional[dict] = None
    is_automated: bool = False
    automation_trigger_id: Optional[int] = None
    created_by: str


class CampaignUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    status: Optional[CampaignStatus] = None
    segment_id: Optional[int] = None
    template_id: Optional[int] = None
    subject_line: Optional[str] = None
    preview_text: Optional[str] = None
    scheduled_at: Optional[datetime] = None


class CampaignResponse(BaseModel):
    id: int
    name: str
    description: Optional[str]
    campaign_type: CampaignType
    status: CampaignStatus
    segment_id: Optional[int]
    template_id: Optional[int]
    target_count: int
    subject_line: Optional[str]
    scheduled_at: Optional[datetime]
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    is_ab_test: bool
    is_automated: bool
    created_by: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class SegmentCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    criteria: dict = Field(..., description="Segment criteria as JSON")
    operator: SegmentOperator = SegmentOperator.AND
    created_by: str


class SegmentUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    criteria: Optional[dict] = None
    operator: Optional[SegmentOperator] = None
    is_active: Optional[bool] = None


class SegmentResponse(BaseModel):
    id: int
    name: str
    description: Optional[str]
    criteria: dict
    operator: SegmentOperator
    estimated_size: int
    last_calculated_at: Optional[datetime]
    created_by: str
    created_at: datetime
    is_active: bool

    class Config:
        from_attributes = True


class TemplateCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    template_type: CampaignType
    subject_line: Optional[str] = None
    html_content: Optional[str] = None
    text_content: Optional[str] = None
    available_tokens: Optional[List[str]] = None
    created_by: str


class TemplateUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    subject_line: Optional[str] = None
    html_content: Optional[str] = None
    text_content: Optional[str] = None
    available_tokens: Optional[List[str]] = None
    is_active: Optional[bool] = None


class TemplateResponse(BaseModel):
    id: int
    name: str
    description: Optional[str]
    template_type: CampaignType
    subject_line: Optional[str]
    html_content: Optional[str]
    text_content: Optional[str]
    available_tokens: Optional[dict]
    created_by: str
    created_at: datetime
    is_active: bool

    class Config:
        from_attributes = True


class TriggerCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    trigger_type: TriggerType
    trigger_config: dict = Field(..., description="Trigger configuration as JSON")
    created_by: str


class TriggerUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    trigger_config: Optional[dict] = None
    is_active: Optional[bool] = None


class TriggerResponse(BaseModel):
    id: int
    name: str
    description: Optional[str]
    trigger_type: TriggerType
    trigger_config: dict
    created_by: str
    created_at: datetime
    is_active: bool

    class Config:
        from_attributes = True


class AnalyticsResponse(BaseModel):
    id: int
    campaign_id: int
    total_sent: int
    total_delivered: int
    total_bounced: int
    total_opened: int
    unique_opened: int
    total_clicked: int
    unique_clicked: int
    total_conversions: int
    total_revenue: float
    delivery_rate: float
    open_rate: float
    click_rate: float
    click_to_open_rate: float
    conversion_rate: float
    roi: float
    last_updated: datetime

    class Config:
        from_attributes = True


# ==================== Campaign Endpoints ====================

def serialize_datetime(obj):
    """Convert datetime objects to ISO format strings"""
    if isinstance(obj, datetime):
        return obj.isoformat()
    return obj

def serialize_campaign(campaign: dict) -> dict:
    """Serialize campaign data for JSON response"""
    return {
        **campaign,
        "campaign_type": campaign.get("campaign_type") if isinstance(campaign.get("campaign_type"), str) else campaign.get("campaign_type").value if campaign.get("campaign_type") else None,
        "status": campaign.get("status") if isinstance(campaign.get("status"), str) else campaign.get("status").value if campaign.get("status") else None,
        "created_at": serialize_datetime(campaign.get("created_at")),
        "updated_at": serialize_datetime(campaign.get("updated_at")),
        "scheduled_at": serialize_datetime(campaign.get("scheduled_at")),
        "started_at": serialize_datetime(campaign.get("started_at")),
        "completed_at": serialize_datetime(campaign.get("completed_at")),
    }

@router.post("/campaigns", status_code=201)
def create_campaign(campaign: CampaignCreate):
    """Create a new marketing campaign"""
    global CAMPAIGN_ID_COUNTER
    try:
        # Validate segment exists if provided
        if campaign.segment_id and campaign.segment_id not in SEGMENTS_DB:
            raise HTTPException(status_code=404, detail=f"Segment {campaign.segment_id} not found")

        # Validate template exists if provided
        if campaign.template_id and campaign.template_id not in TEMPLATES_DB:
            raise HTTPException(status_code=404, detail=f"Template {campaign.template_id} not found")

        # Create campaign
        campaign_id = CAMPAIGN_ID_COUNTER
        CAMPAIGN_ID_COUNTER += 1

        now = datetime.utcnow()
        campaign_data = campaign.dict()
        # Convert enum to string
        if hasattr(campaign_data.get("campaign_type"), "value"):
            campaign_data["campaign_type"] = campaign_data["campaign_type"].value

        db_campaign = {
            "id": campaign_id,
            **campaign_data,
            "status": CampaignStatus.DRAFT.value,
            "target_count": 0,
            "started_at": None,
            "completed_at": None,
            "created_at": now,
            "updated_at": now,
        }
        CAMPAIGNS_DB[campaign_id] = db_campaign

        # Create analytics record
        global ANALYTICS_ID_COUNTER
        analytics_id = ANALYTICS_ID_COUNTER
        ANALYTICS_ID_COUNTER += 1
        ANALYTICS_DB[campaign_id] = {
            "id": analytics_id,
            "campaign_id": campaign_id,
            "total_sent": 0,
            "total_delivered": 0,
            "total_bounced": 0,
            "total_failed": 0,
            "total_opened": 0,
            "unique_opened": 0,
            "total_clicked": 0,
            "unique_clicked": 0,
            "total_conversions": 0,
            "total_revenue": 0.0,
            "delivery_rate": 0.0,
            "open_rate": 0.0,
            "click_rate": 0.0,
            "conversion_rate": 0.0,
            "roi": 0.0,
            "last_updated": now,
        }

        return serialize_campaign(db_campaign)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating campaign: {str(e)}")


@router.get("/campaigns")
def get_campaigns(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    status: Optional[str] = None,
    campaign_type: Optional[str] = None
):
    """Get all campaigns with optional filtering"""
    campaigns = list(CAMPAIGNS_DB.values())

    # Filter by status
    if status:
        campaigns = [c for c in campaigns if c.get("status") == status]

    # Filter by campaign_type
    if campaign_type:
        campaigns = [c for c in campaigns if c.get("campaign_type") == campaign_type]

    # Sort by created_at descending
    campaigns.sort(key=lambda x: x.get("created_at", datetime.min), reverse=True)

    # Pagination
    paginated = campaigns[skip:skip + limit]
    return [serialize_campaign(c) for c in paginated]


@router.get("/campaigns/{campaign_id}")
def get_campaign(campaign_id: int):
    """Get a specific campaign by ID"""
    campaign = CAMPAIGNS_DB.get(campaign_id)
    if not campaign:
        raise HTTPException(status_code=404, detail=f"Campaign {campaign_id} not found")
    return serialize_campaign(campaign)


@router.put("/campaigns/{campaign_id}")
def update_campaign(campaign_id: int, campaign_update: CampaignUpdate):
    """Update a campaign"""
    campaign = CAMPAIGNS_DB.get(campaign_id)
    if not campaign:
        raise HTTPException(status_code=404, detail=f"Campaign {campaign_id} not found")

    try:
        update_data = campaign_update.dict(exclude_unset=True)
        for key, value in update_data.items():
            # Convert enum to string
            if hasattr(value, "value"):
                campaign[key] = value.value
            else:
                campaign[key] = value

        campaign["updated_at"] = datetime.utcnow()
        return serialize_campaign(campaign)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating campaign: {str(e)}")


@router.delete("/campaigns/{campaign_id}", status_code=204)
def delete_campaign(campaign_id: int):
    """Delete a campaign"""
    if campaign_id not in CAMPAIGNS_DB:
        raise HTTPException(status_code=404, detail=f"Campaign {campaign_id} not found")

    try:
        del CAMPAIGNS_DB[campaign_id]
        if campaign_id in ANALYTICS_DB:
            del ANALYTICS_DB[campaign_id]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting campaign: {str(e)}")


@router.post("/campaigns/{campaign_id}/launch")
def launch_campaign(campaign_id: int):
    """Launch a campaign (change status to ACTIVE)"""
    campaign = CAMPAIGNS_DB.get(campaign_id)
    if not campaign:
        raise HTTPException(status_code=404, detail=f"Campaign {campaign_id} not found")

    if campaign["status"] not in [CampaignStatus.DRAFT.value, CampaignStatus.SCHEDULED.value, CampaignStatus.PAUSED.value]:
        raise HTTPException(status_code=400, detail=f"Cannot launch campaign with status {campaign['status']}")

    try:
        campaign["status"] = CampaignStatus.ACTIVE.value
        campaign["started_at"] = datetime.utcnow()
        campaign["updated_at"] = datetime.utcnow()
        return serialize_campaign(campaign)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error launching campaign: {str(e)}")


@router.post("/campaigns/{campaign_id}/pause")
def pause_campaign(campaign_id: int):
    """Pause an active campaign"""
    campaign = CAMPAIGNS_DB.get(campaign_id)
    if not campaign:
        raise HTTPException(status_code=404, detail=f"Campaign {campaign_id} not found")

    if campaign["status"] != CampaignStatus.ACTIVE.value:
        raise HTTPException(status_code=400, detail=f"Cannot pause campaign with status {campaign['status']}")

    try:
        campaign["status"] = CampaignStatus.PAUSED.value
        campaign["updated_at"] = datetime.utcnow()
        return serialize_campaign(campaign)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error pausing campaign: {str(e)}")


@router.get("/campaigns/{campaign_id}/analytics")
def get_campaign_analytics(campaign_id: int):
    """Get analytics for a specific campaign"""
    analytics = ANALYTICS_DB.get(campaign_id)
    if not analytics:
        raise HTTPException(status_code=404, detail=f"Analytics for campaign {campaign_id} not found")

    # Serialize datetime
    return {
        **analytics,
        "last_updated": serialize_datetime(analytics.get("last_updated"))
    }


# ==================== Audience Segment Endpoints ====================

def serialize_segment(segment: dict) -> dict:
    """Serialize segment data for JSON response"""
    return {
        **segment,
        "operator": segment.get("operator") if isinstance(segment.get("operator"), str) else segment.get("operator").value if segment.get("operator") else None,
        "created_at": serialize_datetime(segment.get("created_at")),
        "last_calculated_at": serialize_datetime(segment.get("last_calculated_at")),
    }

@router.post("/segments", status_code=201)
def create_segment(segment: SegmentCreate):
    """Create a new audience segment"""
    global SEGMENT_ID_COUNTER
    try:
        segment_id = SEGMENT_ID_COUNTER
        SEGMENT_ID_COUNTER += 1

        now = datetime.utcnow()
        segment_data = segment.dict()
        # Convert enum to string
        if hasattr(segment_data.get("operator"), "value"):
            segment_data["operator"] = segment_data["operator"].value

        db_segment = {
            "id": segment_id,
            **segment_data,
            "estimated_size": 0,
            "last_calculated_at": None,
            "is_active": True,
            "created_at": now,
        }
        SEGMENTS_DB[segment_id] = db_segment
        return serialize_segment(db_segment)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating segment: {str(e)}")


@router.get("/segments")
def get_segments(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    is_active: Optional[bool] = None
):
    """Get all audience segments"""
    segments = list(SEGMENTS_DB.values())

    if is_active is not None:
        segments = [s for s in segments if s.get("is_active") == is_active]

    segments.sort(key=lambda x: x.get("created_at", datetime.min), reverse=True)
    paginated = segments[skip:skip + limit]
    return [serialize_segment(s) for s in paginated]


@router.get("/segments/{segment_id}")
def get_segment(segment_id: int):
    """Get a specific segment by ID"""
    segment = SEGMENTS_DB.get(segment_id)
    if not segment:
        raise HTTPException(status_code=404, detail=f"Segment {segment_id} not found")
    return serialize_segment(segment)


@router.put("/segments/{segment_id}")
def update_segment(segment_id: int, segment_update: SegmentUpdate):
    """Update a segment"""
    segment = SEGMENTS_DB.get(segment_id)
    if not segment:
        raise HTTPException(status_code=404, detail=f"Segment {segment_id} not found")

    try:
        update_data = segment_update.dict(exclude_unset=True)
        for key, value in update_data.items():
            # Convert enum to string
            if hasattr(value, "value"):
                segment[key] = value.value
            else:
                segment[key] = value
        return serialize_segment(segment)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating segment: {str(e)}")


@router.delete("/segments/{segment_id}", status_code=204)
def delete_segment(segment_id: int):
    """Delete a segment"""
    if segment_id not in SEGMENTS_DB:
        raise HTTPException(status_code=404, detail=f"Segment {segment_id} not found")

    try:
        del SEGMENTS_DB[segment_id]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting segment: {str(e)}")


# ==================== Template Endpoints ====================

def serialize_template(template: dict) -> dict:
    """Serialize template data for JSON response"""
    return {
        **template,
        "template_type": template.get("template_type") if isinstance(template.get("template_type"), str) else template.get("template_type").value if template.get("template_type") else None,
        "created_at": serialize_datetime(template.get("created_at")),
    }

@router.post("/templates", status_code=201)
def create_template(template: TemplateCreate):
    """Create a new marketing template"""
    global TEMPLATE_ID_COUNTER
    try:
        template_id = TEMPLATE_ID_COUNTER
        TEMPLATE_ID_COUNTER += 1

        now = datetime.utcnow()
        template_data = template.dict()
        # Convert enum to string
        if hasattr(template_data.get("template_type"), "value"):
            template_data["template_type"] = template_data["template_type"].value

        db_template = {
            "id": template_id,
            **template_data,
            "available_tokens": template_data.get("available_tokens", []),
            "is_active": True,
            "created_at": now,
        }
        TEMPLATES_DB[template_id] = db_template
        return serialize_template(db_template)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating template: {str(e)}")


@router.get("/templates")
def get_templates(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    template_type: Optional[str] = None,
    is_active: Optional[bool] = None
):
    """Get all templates"""
    templates = list(TEMPLATES_DB.values())

    if template_type:
        templates = [t for t in templates if t.get("template_type") == template_type]
    if is_active is not None:
        templates = [t for t in templates if t.get("is_active") == is_active]

    templates.sort(key=lambda x: x.get("created_at", datetime.min), reverse=True)
    paginated = templates[skip:skip + limit]
    return [serialize_template(t) for t in paginated]


@router.get("/templates/{template_id}")
def get_template(template_id: int):
    """Get a specific template by ID"""
    template = TEMPLATES_DB.get(template_id)
    if not template:
        raise HTTPException(status_code=404, detail=f"Template {template_id} not found")
    return serialize_template(template)


@router.put("/templates/{template_id}")
def update_template(template_id: int, template_update: TemplateUpdate):
    """Update a template"""
    template = TEMPLATES_DB.get(template_id)
    if not template:
        raise HTTPException(status_code=404, detail=f"Template {template_id} not found")

    try:
        update_data = template_update.dict(exclude_unset=True)
        for key, value in update_data.items():
            template[key] = value
        return serialize_template(template)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating template: {str(e)}")


@router.delete("/templates/{template_id}", status_code=204)
def delete_template(template_id: int):
    """Delete a template"""
    if template_id not in TEMPLATES_DB:
        raise HTTPException(status_code=404, detail=f"Template {template_id} not found")

    try:
        del TEMPLATES_DB[template_id]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting template: {str(e)}")


# ==================== Automation Trigger Endpoints ====================

def serialize_trigger(trigger: dict) -> dict:
    """Serialize trigger data for JSON response"""
    return {
        **trigger,
        "trigger_type": trigger.get("trigger_type") if isinstance(trigger.get("trigger_type"), str) else trigger.get("trigger_type").value if trigger.get("trigger_type") else None,
        "created_at": serialize_datetime(trigger.get("created_at")),
    }

@router.post("/triggers", status_code=201)
def create_trigger(trigger: TriggerCreate):
    """Create a new automation trigger"""
    global TRIGGER_ID_COUNTER
    try:
        trigger_id = TRIGGER_ID_COUNTER
        TRIGGER_ID_COUNTER += 1

        now = datetime.utcnow()
        trigger_data = trigger.dict()
        # Convert enum to string
        if hasattr(trigger_data.get("trigger_type"), "value"):
            trigger_data["trigger_type"] = trigger_data["trigger_type"].value

        db_trigger = {
            "id": trigger_id,
            **trigger_data,
            "is_active": True,
            "created_at": now,
        }
        TRIGGERS_DB[trigger_id] = db_trigger
        return serialize_trigger(db_trigger)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating trigger: {str(e)}")


@router.get("/triggers")
def get_triggers(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    trigger_type: Optional[str] = None,
    is_active: Optional[bool] = None
):
    """Get all automation triggers"""
    triggers = list(TRIGGERS_DB.values())

    if trigger_type:
        triggers = [t for t in triggers if t.get("trigger_type") == trigger_type]
    if is_active is not None:
        triggers = [t for t in triggers if t.get("is_active") == is_active]

    triggers.sort(key=lambda x: x.get("created_at", datetime.min), reverse=True)
    paginated = triggers[skip:skip + limit]
    return [serialize_trigger(t) for t in paginated]


@router.get("/triggers/{trigger_id}")
def get_trigger(trigger_id: int):
    """Get a specific trigger by ID"""
    trigger = TRIGGERS_DB.get(trigger_id)
    if not trigger:
        raise HTTPException(status_code=404, detail=f"Trigger {trigger_id} not found")
    return serialize_trigger(trigger)


@router.put("/triggers/{trigger_id}")
def update_trigger(trigger_id: int, trigger_update: TriggerUpdate):
    """Update a trigger"""
    trigger = TRIGGERS_DB.get(trigger_id)
    if not trigger:
        raise HTTPException(status_code=404, detail=f"Trigger {trigger_id} not found")

    try:
        update_data = trigger_update.dict(exclude_unset=True)
        for key, value in update_data.items():
            trigger[key] = value
        return serialize_trigger(trigger)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating trigger: {str(e)}")


@router.delete("/triggers/{trigger_id}", status_code=204)
def delete_trigger(trigger_id: int):
    """Delete a trigger"""
    if trigger_id not in TRIGGERS_DB:
        raise HTTPException(status_code=404, detail=f"Trigger {trigger_id} not found")

    try:
        del TRIGGERS_DB[trigger_id]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting trigger: {str(e)}")


# ==================== Analytics & Dashboard ====================

@router.get("/analytics/overview")
def get_marketing_overview():
    """Get overall marketing automation statistics"""
    total_campaigns = len(CAMPAIGNS_DB)
    active_campaigns = len([c for c in CAMPAIGNS_DB.values() if c.get("status") == CampaignStatus.ACTIVE.value])
    total_segments = len([s for s in SEGMENTS_DB.values() if s.get("is_active") == True])
    total_templates = len([t for t in TEMPLATES_DB.values() if t.get("is_active") == True])

    # Aggregate analytics
    analytics = list(ANALYTICS_DB.values())
    total_sent = sum(a.get("total_sent", 0) for a in analytics)
    total_opened = sum(a.get("unique_opened", 0) for a in analytics)
    total_clicked = sum(a.get("unique_clicked", 0) for a in analytics)
    total_conversions = sum(a.get("total_conversions", 0) for a in analytics)
    total_revenue = sum(a.get("total_revenue", 0.0) for a in analytics)

    avg_open_rate = sum(a.get("open_rate", 0.0) for a in analytics) / len(analytics) if analytics else 0
    avg_click_rate = sum(a.get("click_rate", 0.0) for a in analytics) / len(analytics) if analytics else 0
    avg_conversion_rate = sum(a.get("conversion_rate", 0.0) for a in analytics) / len(analytics) if analytics else 0

    return {
        "total_campaigns": total_campaigns,
        "active_campaigns": active_campaigns,
        "total_segments": total_segments,
        "total_templates": total_templates,
        "total_sent": total_sent,
        "total_opened": total_opened,
        "total_clicked": total_clicked,
        "total_conversions": total_conversions,
        "total_revenue": total_revenue,
        "avg_open_rate": round(avg_open_rate, 2),
        "avg_click_rate": round(avg_click_rate, 2),
        "avg_conversion_rate": round(avg_conversion_rate, 2),
    }


