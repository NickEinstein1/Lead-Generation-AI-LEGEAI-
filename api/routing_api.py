from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

from api.leads_api import LEADS_DB
from automation.lead_routing import lead_router

router = APIRouter(prefix="/leads", tags=["routing"])


class RouteInput(BaseModel):
    band: Optional[str] = None


@router.post("/{lead_id}/route", summary="Route a lead to an agent")
async def route_lead(lead_id: str, payload: RouteInput | None = None):
    lead = LEADS_DB.get(lead_id)
    if not lead:
        raise HTTPException(status_code=404, detail="Lead not found")
    band = payload.band if payload else None
    assignment = lead_router.route_lead(lead, band)
    return assignment

