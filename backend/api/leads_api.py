from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from uuid import uuid4
from datetime import datetime, timezone
import os
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from backend.database.connection import session_dep
from backend.models.lead import Lead
from backend.models.score import Score
from backend.api.auth_dependencies import require_permission
from backend.security.authentication import Permission

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
async def create_lead(
    payload: LeadCreate,
    session: AsyncSession = Depends(session_dep),
    current_user: dict = Depends(require_permission(Permission.CREATE_LEADS)),
):
    use_db = os.getenv("USE_DB", "false").lower() == "true"

    if use_db:
        # Idempotency check in DB
        existing = (await session.execute(select(Lead).where(Lead.idempotency_key == payload.idempotency_key))).scalar_one_or_none()
        if existing:
            return {"lead_id": str(existing.id), "status": "duplicate", "idempotent": True}

        contact_info = payload.contact.dict() if payload.contact else {}
        metadata = {
            "geo": payload.geo.dict() if payload.geo else {},
            "attributes": payload.attributes,
        }
        new_lead = Lead(
            idempotency_key=payload.idempotency_key,
            channel=payload.channel,
            source=payload.source,
            product_interest=payload.product_interest,
            contact_info=contact_info,
            consent=payload.consent,
            lead_metadata=metadata,
        )
        session.add(new_lead)
        await session.flush()
        await session.commit()
        return {"lead_id": str(new_lead.id), "status": "created", "idempotent": False}

    # In-memory fallback
    if payload.idempotency_key in LEADS_BY_IDEMPOTENCY:
        lead_id = LEADS_BY_IDEMPOTENCY[payload.idempotency_key]
        return {"lead_id": lead_id, "status": "duplicate", "idempotent": True}

    lead_id = str(uuid4())
    now = datetime.now(timezone.utc).isoformat()

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
async def list_leads(
    limit: int = 50,
    offset: int = 0,
    session: AsyncSession = Depends(session_dep),
    current_user: dict = Depends(require_permission(Permission.VIEW_LEADS)),
):
    use_db = os.getenv("USE_DB", "false").lower() == "true"

    if use_db:
        result = await session.execute(select(Lead).offset(offset).limit(limit))
        rows = result.scalars().all()
        # Basic total count (could optimize)
        total = len((await session.execute(select(Lead))).scalars().all())
        items = [
            {
                "id": str(r.id),
                "source": r.source,
                "channel": r.channel,
                "product_interest": r.product_interest,
                "contact": r.contact_info or {},
                "attributes": (r.lead_metadata or {}).get("attributes", {}),
                "geo": (r.lead_metadata or {}).get("geo", {}),
                "consent": r.consent,
                "created_at": str(r.created_at),
                "updated_at": str(r.updated_at),
            }
            for r in rows
        ]
        return {"total": total, "limit": limit, "offset": offset, "items": items}

    items = list(LEADS_DB.values())
    total = len(items)
    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "items": items[offset: offset + limit],
    }


@router.get("/{lead_id}", summary="Get lead details")
async def get_lead(
    lead_id: str,
    session: AsyncSession = Depends(session_dep),
    current_user: dict = Depends(require_permission(Permission.VIEW_LEADS)),
):
    use_db = os.getenv("USE_DB", "false").lower() == "true"

    if use_db:
        row = (await session.execute(select(Lead).where(Lead.id == int(lead_id)))).scalar_one_or_none() if lead_id.isdigit() else None
        if not row:
            raise HTTPException(status_code=404, detail="Lead not found")
        return {
            "id": str(row.id),
            "source": row.source,
            "channel": row.channel,
            "product_interest": row.product_interest,
            "contact": row.contact_info or {},
            "attributes": (row.lead_metadata or {}).get("attributes", {}),
            "geo": (row.lead_metadata or {}).get("geo", {}),
            "consent": row.consent,
            "created_at": str(row.created_at),
            "updated_at": str(row.updated_at),
        }

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
async def score_lead(
    lead_id: str,
    payload: ScoreInput,
    session: AsyncSession = Depends(session_dep),
    current_user: dict = Depends(require_permission(Permission.GENERATE_SCORES)),
):
    use_db = os.getenv("USE_DB", "false").lower() == "true"

    try:
        from backend.models.insurance_lead_scoring.inference import InsuranceLeadScorer
        scorer = InsuranceLeadScorer()

        if use_db:
            row = (await session.execute(select(Lead).where(Lead.id == int(lead_id)))).scalar_one_or_none() if lead_id.isdigit() else None
            if not row:
                raise HTTPException(status_code=404, detail="Lead not found")
            # Merge metadata attributes with payload features
            features = {
                **(((row.lead_metadata or {}).get("attributes")) or {}),
                **payload.features,
            }
        else:
            lead = LEADS_DB.get(lead_id)
            if not lead:
                raise HTTPException(status_code=404, detail="Lead not found")
            features = {
                **(lead.get("attributes") or {}),
                **payload.features,
            }

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

        if use_db:
            db_score = Score(
                lead_id=row.id,
                score=result.get("score"),
                band=result.get("band"),
                explanation=result.get("explanation"),
                model_version=result.get("model_version"),
                features=features,
            )
            session.add(db_score)
            await session.commit()
            return {
                "lead_id": str(row.id),
                "score": db_score.score,
                "band": db_score.band,
                "explanation": db_score.explanation,
                "model_version": db_score.model_version,
                "scored_at": datetime.now(timezone.utc).isoformat(),
            }
        else:
            score_entry = {
                "lead_id": lead_id,
                "score": result.get("score"),
                "band": result.get("band"),
                "explanation": result.get("explanation"),
                "model_version": result.get("model_version"),
                "scored_at": datetime.now(timezone.utc).isoformat(),
            }
            SCORES_DB.setdefault(lead_id, []).append(score_entry)
            return score_entry
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scoring failed: {e}")


@router.get("/{lead_id}/scores", summary="Get score history for a lead")
async def get_scores(
    lead_id: str,
    session: AsyncSession = Depends(session_dep),
    current_user: dict = Depends(require_permission(Permission.VIEW_SCORES)),
):
    use_db = os.getenv("USE_DB", "false").lower() == "true"
    if use_db:
        rows = (await session.execute(select(Score).where(Score.lead_id == int(lead_id)).order_by(Score.scored_at.desc()))).scalars().all() if lead_id.isdigit() else []
        return [
            {
                "lead_id": str(r.lead_id),
                "score": r.score,
                "band": r.band,
                "explanation": r.explanation,
                "model_version": r.model_version,
                "scored_at": str(r.scored_at),
            }
            for r in rows
        ]
    return SCORES_DB.get(lead_id, [])


