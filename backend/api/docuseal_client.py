import os
from typing import Any, Dict, Optional
import httpx


class DocuSealError(Exception):
    pass


def _get_api_key() -> str:
    api_key = os.getenv("DOCUSEAL_API_KEY")
    if not api_key:
        raise DocuSealError("DOCUSEAL_API_KEY is not configured")
    return api_key


async def create_submission(
    template_id: str | int,
    email: str,
    name: Optional[str] = None,
    role: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    send_email: bool = False,
    completed_redirect_url: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Create a DocuSeal submission and return the API response.
    Expected response contains `id` and `submitters` with per-signer `slug`.
    """
    base_url = os.getenv("DOCUSEAL_API_BASE", "https://api.docuseal.com")
    api_key = _get_api_key()

    payload: Dict[str, Any] = {
        "template_id": template_id,
        "send_email": send_email,
        "submitters": [
            {
                "email": email,
                **({"name": name} if name else {}),
                **({"role": role} if role else {}),
                **({"metadata": metadata} if metadata else {}),
            }
        ],
    }
    if completed_redirect_url:
        payload["completed_redirect_url"] = completed_redirect_url

    async with httpx.AsyncClient(base_url=base_url, timeout=30.0) as client:
        res = await client.post("/submissions", headers={"X-Auth-Token": api_key}, json=payload)
        if res.status_code >= 400:
            raise DocuSealError(f"DocuSeal error {res.status_code}: {res.text}")
        return res.json()


def extract_first_submitter_slug(data: Dict[str, Any]) -> Optional[str]:
    submitters = data.get("submitters") or data.get("data", {}).get("submitters")
    if isinstance(submitters, list) and submitters:
        slug = submitters[0].get("slug")
        return slug
    return None


def extract_submission_id(data: Dict[str, Any]) -> Optional[str]:
    sid = data.get("id") or (data.get("submission") or {}).get("id")
    return str(sid) if sid is not None else None


def build_signing_url_from_slug(slug: str) -> str:
    base = os.getenv("DOCUSEAL_FORM_BASE", "https://docuseal.com")
    return f"{base}/s/{slug}"



async def list_templates(limit: int = 10) -> dict:
    base_url = os.getenv("DOCUSEAL_API_BASE", "https://api.docuseal.com")
    api_key = _get_api_key()
    async with httpx.AsyncClient(base_url=base_url, timeout=30.0) as client:
        res = await client.get(f"/templates?limit={limit}", headers={"X-Auth-Token": api_key})
        if res.status_code >= 400:
            raise DocuSealError(f"DocuSeal error {res.status_code}: {res.text}")
        return res.json()


async def get_default_template_id() -> str | int:
    """
    Discover a default template_id by listing templates and returning the first available.
    """
    data = await list_templates(limit=1)
    # API may return either {"data": [ {"id": ...}, ... ]} or just a list
    items = data.get("data") if isinstance(data, dict) else data
    if isinstance(items, list) and items:
        tid = items[0].get("id") if isinstance(items[0], dict) else None
        if tid is not None:
            return tid
    raise DocuSealError("No DocuSeal templates available to auto-select")