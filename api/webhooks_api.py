from fastapi import APIRouter, HTTPException, Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import Dict, Any
import hashlib, hmac, os
from datetime import datetime

from database.connection import session_dep
from models.lead import Lead

router = APIRouter(prefix="/webhooks", tags=["webhooks"])


def _compute_idempotency_key(source: str, payload: Dict[str, Any]) -> str:
    candidates = [
        str(payload.get("id") or ""),
        str(payload.get("event_id") or ""),
        str(payload.get("timestamp") or ""),
    ]
    raw = f"{source}|" + "|".join(candidates)
    if raw.strip("|"):
        return raw
    # fallback stable hash of payload
    h = hashlib.sha256(repr(sorted(payload.items())).encode("utf-8")).hexdigest()[:32]
    return f"{source}|{h}"


# --- Per-source normalizers -------------------------------------------------

def _norm_facebook(payload: Dict[str, Any]) -> Dict[str, Any]:
    # Facebook Lead Ads often sends field_data: [{name: ..., values: [...] }]
    fd = {item.get("name"): (item.get("values") or [None])[0] for item in payload.get("field_data", [])}
    return {
        "contact": {
            "email": payload.get("email") or fd.get("email"),
            "phone": payload.get("phone_number") or fd.get("phone_number") or fd.get("phone"),
            "first_name": fd.get("first_name") or payload.get("first_name"),
            "last_name": fd.get("last_name") or payload.get("last_name"),
        },
        "product": payload.get("product") or fd.get("product") or fd.get("product_interest"),
        "consent": payload.get("consent") or {},
        "attributes": {k: v for k, v in fd.items() if k not in {"email","phone_number","phone","first_name","last_name","product","product_interest"}},
        "channel": "facebook",
    }


def _norm_zapier(payload: Dict[str, Any]) -> Dict[str, Any]:
    data = payload.get("data") or payload
    return {
        "contact": {
            "email": data.get("email"),
            "phone": data.get("phone"),
            "first_name": data.get("first_name") or data.get("firstName"),
            "last_name": data.get("last_name") or data.get("lastName"),
        },
        "product": data.get("product") or data.get("product_interest"),
        "consent": data.get("consent") or {},
        "attributes": data.get("attributes") or {},
        "channel": data.get("channel") or "zapier",
    }


def _norm_webflow(payload: Dict[str, Any]) -> Dict[str, Any]:
    data = payload.get("data") or {}
    return {
        "contact": {
            "email": data.get("Email") or data.get("email"),
            "phone": data.get("Phone") or data.get("phone"),
            "first_name": data.get("First Name") or data.get("first_name"),
            "last_name": data.get("Last Name") or data.get("last_name"),
        },
        "product": data.get("Product") or data.get("product"),
        "consent": {},
        "attributes": {k: v for k, v in data.items() if k not in {"Email","Phone","First Name","Last Name","Product"}},
        "channel": "webflow",
    }


def _normalize_payload(source: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    s = (source or "").lower()
    if "facebook" in s:
        return _norm_facebook(payload)
    if "webflow" in s:
        return _norm_webflow(payload)
    if "zapier" in s:
        return _norm_zapier(payload)
    # default passthrough
    return {
        "contact": payload.get("contact") or {
            "email": payload.get("email"),
            "phone": payload.get("phone"),
            "first_name": payload.get("first_name"),
            "last_name": payload.get("last_name"),
        },
        "product": payload.get("product") or payload.get("product_interest"),
        "consent": payload.get("consent") or {},
        "attributes": payload.get("attributes") or {},
        "channel": payload.get("channel") or "webhook",
    }

@router.post("/{source}")
async def receive_webhook(source: str, payload: Dict[str, Any], request: Request, session: AsyncSession = Depends(session_dep)):
    try:
        # Optional HMAC signature validation
        secret = os.getenv("WEBHOOK_SECRET")
        if secret:
            raw_body = await request.body()
            provided = request.headers.get("x-hub-signature-256") or request.headers.get("x-signature") or ""
            expected = "sha256=" + hmac.new(secret.encode("utf-8"), raw_body, hashlib.sha256).hexdigest()
            if not provided or not hmac.compare_digest(provided, expected):
                raise HTTPException(status_code=401, detail="Invalid webhook signature")

        idempotency_key = _compute_idempotency_key(source, payload)
        # Idempotency check
        existing = (await session.execute(select(Lead).where(Lead.idempotency_key == idempotency_key))).scalar_one_or_none()
        if existing:
            return {"status": "duplicate", "lead_id": str(existing.id), "idempotent": True}

        # Normalize
        norm = _normalize_payload(source, payload)
        contact = norm["contact"]
        consent = norm.get("consent") or {}
        attributes = norm.get("attributes") or {}
        channel = norm.get("channel") or "webhook"
        product = norm.get("product")

        metadata = {
            "attributes": attributes,
            "geo": payload.get("geo") or {},
            "raw": payload,
            "webhook": {
                "source": source,
                "received_at": datetime.utcnow().isoformat(),
                "ip": request.client.host if request and request.client else None,
                "user_agent": request.headers.get("user-agent", ""),
            },
        }

        new_lead = Lead(
            idempotency_key=idempotency_key,
            channel=channel,
            source=source,
            product_interest=product,
            contact_info=contact,
            consent=consent,
            metadata=metadata,
        )
        session.add(new_lead)
        await session.flush()
        await session.commit()
        return {"status": "accepted", "lead_id": str(new_lead.id), "idempotent": False}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Webhook processing failed: {e}")

