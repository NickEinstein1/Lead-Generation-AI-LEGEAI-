from fastapi import APIRouter, Request, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from datetime import datetime
from typing import Any, Dict

from backend.database.connection import session_dep
from backend.models.document import Document

router = APIRouter(prefix="/webhooks/docuseal", tags=["webhooks"])  # mounted under /v1


@router.post("")
async def receive_docuseal_webhook(payload: Dict[str, Any], request: Request, session: AsyncSession = Depends(session_dep)):
    try:
        event_type = payload.get("event_type") or payload.get("type")
        data = payload.get("data") or {}
        submission_id = data.get("id")
        # Only handle if DB is enabled (documents are stored in DB mode)
        if not submission_id:
            return {"status": "ignored", "reason": "missing submission id"}

        # Try to find by provider_request_id
        doc = (await session.execute(select(Document).where(Document.provider == "docuseal", Document.provider_request_id == str(submission_id)))).scalars().first()

        # Fallback: try by submitter slug present in event
        if not doc:
            submitters = data.get("submitters") or []
            slugs = {s.get("slug") for s in submitters if s.get("slug")}
            if slugs:
                # scan by metadata submitter slug if stored
                doc = (await session.execute(select(Document).where(Document.provider == "docuseal"))).scalars().first()
                # Note: for simplicity, we try to match any with metadata containing the slug
                # In production, add JSON path filter using DB JSON operators
                if doc and doc.doc_metadata:
                    m = doc.doc_metadata or {}
                    ds = (m.get("docuseal") or {})
                    if ds.get("submitter_slug") not in slugs:
                        doc = None

        if not doc:
            return {"status": "not_found"}

        now = datetime.now(datetime.UTC)
        if event_type == "submission.completed":
            doc.status = "signed"
            doc.signed_at = now
        elif event_type == "submission.declined":
            doc.status = "declined"
        else:
            # ignore other events
            return {"status": "ignored", "event_type": event_type}

        await session.commit()
        return {"status": "ok"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DocuSeal webhook error: {e}")


