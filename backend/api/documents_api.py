from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from backend.database.connection import session_dep
from backend.models.document import Document
from backend.models.lead import Lead
from datetime import datetime, timezone
import os

from backend.api.auth_dependencies import require_permission
from backend.security.authentication import Permission

from backend.api.docuseal_client import (
    create_submission,
    extract_first_submitter_slug,
    extract_submission_id,
    build_signing_url_from_slug,
    DocuSealError,
    get_default_template_id,
)

router = APIRouter(tags=["documents"])  # mounted under /v1

USE_DB = os.getenv("USE_DB", "false").lower() == "true"
INMEM_DOCS: List[Dict[str, Any]] = []


class CreateDocumentRequest(BaseModel):
    title: str = Field(default="Insurance Agreement", max_length=255)
    provider: Optional[str] = Field(default=None, max_length=64)
    template_id: Optional[str | int] = None
    signer_email: Optional[str] = None
    signer_name: Optional[str] = None
    role: Optional[str] = Field(default=None, max_length=64)


def _doc_to_dict(d: Document | Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(d, dict):
        return d
    return {
        "id": d.id,
        "lead_id": d.lead_id,
        "title": d.title,
        "status": d.status,
        "provider": d.provider,
        "provider_request_id": d.provider_request_id,
        "signing_url": d.signing_url,
        "metadata": d.doc_metadata,
        "created_at": getattr(d, "created_at", None),
        "updated_at": getattr(d, "updated_at", None),
        "signed_at": getattr(d, "signed_at", None),
    }


@router.get("/leads/{lead_id}/documents")
async def list_documents_for_lead(
    lead_id: int,
    session: AsyncSession = Depends(session_dep),
    current_user: dict = Depends(require_permission(Permission.VIEW_LEADS)),
):
    if USE_DB:
        res = await session.execute(select(Document).where(Document.lead_id == lead_id))
        items = [_doc_to_dict(d) for d in res.scalars().all()]
        return {"items": items}
    # in-memory fallback
    items = [d for d in INMEM_DOCS if d.get("lead_id") == lead_id]
    return {"items": items}


@router.post("/leads/{lead_id}/documents")
async def create_document_for_lead(
    lead_id: int,
    body: CreateDocumentRequest,
    session: AsyncSession = Depends(session_dep),
    current_user: dict = Depends(require_permission(Permission.CREATE_LEADS)),
):
    title = (body.title or "").strip() or "Insurance Agreement"

    # Determine provider: prefer explicit; else use DocuSeal if API key present; otherwise internal
    provider = (body.provider or ("docuseal" if os.getenv("DOCUSEAL_API_KEY") else "internal")).lower()

    if USE_DB:
        lead = (await session.execute(select(Lead).where(Lead.id == lead_id))).scalars().first()
        if not lead:
            raise HTTPException(status_code=404, detail="Lead not found")

        # If using DocuSeal, create submission and use returned signing URL
        signing_url: Optional[str] = None
        provider_request_id: Optional[str] = None
        if provider == "docuseal":
            template_id = body.template_id or os.getenv("DOCUSEAL_TEMPLATE_ID")
            if not template_id:
                try:
                    template_id = await get_default_template_id()
                except DocuSealError as e:
                    raise HTTPException(status_code=400, detail=f"DocuSeal template_id is required and auto-discovery failed: {e}")
            signer_email = body.signer_email or (lead.contact_info or {}).get("email")
            if not signer_email:
                raise HTTPException(status_code=400, detail="Signer email is required (lead contact email missing)")
            signer_name = body.signer_name or (
                f"{(lead.contact_info or {}).get('first_name') or ''} {(lead.contact_info or {}).get('last_name') or ''}"
            ).strip() or None
            try:
                data = await create_submission(
                    template_id=template_id,
                    email=signer_email,
                    name=signer_name,
                    role=body.role or "Signer",
                    metadata={"lead_id": str(lead_id), "title": title},
                    send_email=False,
                )
                slug = extract_first_submitter_slug(data)
                if not slug:
                    raise DocuSealError("Missing submitter slug in DocuSeal response")
                signing_url = build_signing_url_from_slug(slug)
                provider_request_id = extract_submission_id(data)
            except DocuSealError as e:
                raise HTTPException(status_code=502, detail=f"DocuSeal integration failed: {e}")

        doc = Document(
            lead_id=lead_id,
            title=title,
            status="pending",
            provider=provider,
            provider_request_id=provider_request_id,
            signing_url=signing_url or f"/v1/documents/{{}}/sign",  # placeholder if internal
            doc_metadata={"provider": provider},
        )
        session.add(doc)
        await session.flush()
        if not signing_url:
            doc.signing_url = f"/v1/documents/{doc.id}/sign"
        await session.commit()
        await session.refresh(doc)
        return _doc_to_dict(doc)

    # In-memory path
    signing_url: Optional[str] = None
    provider_request_id: Optional[str] = None
    if provider == "docuseal":
        template_id = body.template_id or os.getenv("DOCUSEAL_TEMPLATE_ID")
        if not template_id:
            try:
                template_id = await get_default_template_id()
            except DocuSealError as e:
                raise HTTPException(status_code=400, detail=f"DocuSeal template_id is required and auto-discovery failed: {e}")
        signer_email = body.signer_email
        if not signer_email:
            raise HTTPException(status_code=400, detail="Signer email is required for DocuSeal in in-memory mode")
        try:
            data = await create_submission(
                template_id=template_id,
                email=signer_email,
                name=body.signer_name,
                role=body.role or "Signer",
                metadata={"lead_id": str(lead_id), "title": title},
                send_email=False,
            )
            slug = extract_first_submitter_slug(data)
            if not slug:
                raise DocuSealError("Missing submitter slug in DocuSeal response")
            signing_url = build_signing_url_from_slug(slug)
            provider_request_id = extract_submission_id(data)
        except DocuSealError as e:
            raise HTTPException(status_code=502, detail=f"DocuSeal integration failed: {e}")

    new_id = (max([d["id"] for d in INMEM_DOCS], default=0) + 1)
    doc = {
        "id": new_id,
        "lead_id": lead_id,
        "title": title,
        "status": "pending",
        "provider": provider,
        "provider_request_id": provider_request_id,
        "signing_url": signing_url or f"/v1/documents/{new_id}/sign",
        "metadata": {"provider": provider},
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "signed_at": None,
    }
    INMEM_DOCS.append(doc)
    return doc


@router.get("/documents/{doc_id}")
async def get_document(
    doc_id: int,
    session: AsyncSession = Depends(session_dep),
    current_user: dict = Depends(require_permission(Permission.VIEW_LEADS)),
):
    if USE_DB:
        doc = (await session.execute(select(Document).where(Document.id == doc_id))).scalars().first()
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")
        return _doc_to_dict(doc)
    for d in INMEM_DOCS:
        if d.get("id") == doc_id:
            return d
    raise HTTPException(status_code=404, detail="Document not found")


@router.post("/documents/{doc_id}/simulate-sign")
async def simulate_sign(
    doc_id: int,
    session: AsyncSession = Depends(session_dep),
    current_user: dict = Depends(require_permission(Permission.UPDATE_LEADS)),
):
    now = datetime.now(timezone.utc)
    if USE_DB:
        doc = (await session.execute(select(Document).where(Document.id == doc_id))).scalars().first()
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")
        doc.status = "signed"
        doc.signed_at = now
        await session.commit()
        await session.refresh(doc)
        return _doc_to_dict(doc)
    # in-memory
    for d in INMEM_DOCS:
        if d.get("id") == doc_id:
            d["status"] = "signed"
            d["signed_at"] = now.isoformat()
            return d
    raise HTTPException(status_code=404, detail="Document not found")


