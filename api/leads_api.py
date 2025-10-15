from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from uuid import uuid4
from datetime import datetime

# Minimal in-memory stores for MVP (replace with DB layer later)
LEADS_DB: Dict[str, Dict[str, Any]] = {}
LEADS_BY_IDEMPOTENCY: Dict[str, str] = {}
SCORES_DB: Dict[str, List[Dict[str, Any]]] = {}

router = APIRouter(prefix="/leads", tags=["leads"])


class LeadContact(BaseModel):
    email: Optional[str] = None
    phone: Optional[str] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None


class LeadGeo(BaseModel):
    country: Optional[str] = None
    state: Optional[str] = None
    city: Optional[str] = None
    zip: Optional[str] = None


class LeadCreate(BaseModel):
    idempotency_key: str = Field(..., description="Client-provided idempotency key to dedupe submissions")
    source: Optional[str] = Field(None, description="Source system name")
    channel: Optional[str] = Field(None, description="acquisition channel: web, api, affiliate, dialer")
    product_interest: Optional[str] = Field(None, description="auto | home | life | health")
    contact: Optional[LeadContact] = None
    geo: Optional[LeadGeo] = None
    attributes: Dict[str, Any] = Field(default_factory=dict)
    consent: Dict[str, Any] = Field(default_factory=dict)


@router.post("", summary="Create/ingest a lead (idempotent)")
async def create_lead(payload: LeadCreate):
    # Idempotency check
    if payload.idempotency_key in LEADS_BY_IDEMPOTENCY:
        lead_id = LEADS_BY_IDEMPOTENCY[payload.idempotency_key]
        return {"lead_id": lead_id, "status": "duplicate", "idempotent": True}

    lead_id = str(uuid4())
    now = datetime.utcnow().isoformat()

    lead_record = {
        "id": lead_id,
        "created_at": now,
        "updated_at": now,
        "source": payload.source,
        "channel": payload.channel,
        "product_interest": payload.product_interest,
        "contact": payload.contact.dict() if payload.contact else {},
        "geo": payload.geo.dict() if payload.geo else {},
        "attributes": payload.attributes,
        "consent": payload.consent,
        "status": "new",
    }

    LEADS_DB[lead_id] = lead_record
    LEADS_BY_IDEMPOTENCY[payload.idempotency_key] = lead_id

    return {"lead_id": lead_id, "status": "created", "idempotent": False}


@router.get("", summary="List leads")
async def list_leads(limit: int = 50, offset: int = 0):
    items = list(LEADS_DB.values())
    total = len(items)
    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "items": items[offset: offset + limit],
    }


@router.get("/{lead_id}", summary="Get lead details")
async def get_lead(lead_id: str):
    lead = LEADS_DB.get(lead_id)
    if not lead:
        raise HTTPException(status_code=404, detail="Lead not found")
    return lead


# Scoring payload compatible with existing insurance model
class ScoreInput(BaseModel):
    # Minimal set; extend as models evolve
    age: Optional[int] = None
    income: Optional[float] = None
    policy_type: Optional[str] = None
    state: Optional[str] = None
    quote_requests_30d: Optional[int] = None
    credit_score_bucket: Optional[str] = None
    past_claims_count: Optional[int] = None
    # Free-form additional features map
    features: Dict[str, Any] = Field(default_factory=dict)


@router.post("/{lead_id}/score", summary="Score a lead using current model")
async def score_lead(lead_id: str, payload: ScoreInput):
    lead = LEADS_DB.get(lead_id)
    if not lead:
        raise HTTPException(status_code=404, detail="Lead not found")

    try:
        from models.insurance_lead_scoring.inference import InsuranceLeadScorer
        scorer = InsuranceLeadScorer()
        # Merge attributes from lead with payload.features for flexibility
        features = {
            **(lead.get("attributes") or {}),
            **payload.features,
        }
        # Map common fields if provided explicitly
        if payload.age is not None:
            features["age"] = payload.age
        if payload.income is not None:
            features["income"] = payload.income
        if payload.policy_type:
            features["policy_type"] = payload.policy_type
        if payload.state:
            features["state"] = payload.state
        if payload.quote_requests_30d is not None:
            features["quote_requests_30d"] = payload.quote_requests_30d
        if payload.credit_score_bucket:
            features["credit_score_bucket"] = payload.credit_score_bucket
        if payload.past_claims_count is not None:
            features["past_claims_count"] = payload.past_claims_count

        result = scorer.score_lead(features)
        score_entry = {
            "lead_id": lead_id,
            "score": result.get("score"),
            "band": result.get("band"),
            "explanation": result.get("explanation"),
            "model_version": result.get("model_version"),
            "scored_at": datetime.utcnow().isoformat(),
        }
        SCORES_DB.setdefault(lead_id, []).append(score_entry)
        return score_entry
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scoring failed: {e}")


@router.get("/{lead_id}/scores", summary="Get score history for a lead")
async def get_scores(lead_id: str):
    return SCORES_DB.get(lead_id, [])

