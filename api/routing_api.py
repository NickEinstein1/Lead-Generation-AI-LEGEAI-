from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional

from api.leads_api import LEADS_DB
import os
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from database.connection import session_dep
from models.lead import Lead
from automation.lead_routing import lead_router

router = APIRouter(prefix="/leads", tags=["routing"])


class RouteInput(BaseModel):
    band: Optional[str] = None


@router.post("/{lead_id}/route", summary="Route a lead to an agent")
async def route_lead(lead_id: str, payload: RouteInput | None = None, session: AsyncSession = Depends(session_dep)):
    use_db = os.getenv("USE_DB", "false").lower() == "true"
    lead = None
    if use_db:
        row = (await session.execute(select(Lead).where(Lead.id == int(lead_id)))).scalar_one_or_none() if lead_id.isdigit() else None
        if row:
            lead = {
                "id": str(row.id),
                "source": row.source,
                "channel": row.channel,
                "product_interest": row.product_interest,
                "contact": row.contact_info or {},
                "attributes": (row.metadata or {}).get("attributes", {}),
                "geo": (row.metadata or {}).get("geo", {}),
                "consent": row.consent,
            }
    if not lead:
        lead = LEADS_DB.get(lead_id)
    if not lead:
        raise HTTPException(status_code=404, detail="Lead not found")
    band = payload.band if payload else None
    assignment = lead_router.route_lead(lead, band)
    return assignment

