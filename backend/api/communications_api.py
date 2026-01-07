"""Communications API - CRUD endpoints for communications management.

These endpoints are currently backed by in-memory storage and are intended for
demo/sandbox use only. They are guarded by the ENABLE_DEMO_MODE feature flag to
avoid accidentally relying on mock data in production.
"""

from fastapi import APIRouter, HTTPException, Query, Depends
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any, Iterable
import logging
from datetime import datetime, timedelta
import os
from backend.utils.validators import (
    validate_string_length,
    validate_choice,
    validate_date_string,
    ValidationError
)
from backend.utils.business_rules import (
    find_customer_by_name,
    can_send_marketing_to_customer,
    get_communication_priority
)
from backend.api.auth_dependencies import require_permission
from backend.security.authentication import Permission
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from backend.database.connection import session_dep
from backend.models.communication import Communication

router = APIRouter(prefix="/communications", tags=["Communications"])
logger = logging.getLogger(__name__)

ENABLE_DEMO_MODE = os.getenv("ENABLE_DEMO_MODE", "false").lower() == "true"
USE_DB = os.getenv("USE_DB", "false").lower() == "true"


def _ensure_demo_mode_enabled() -> None:
    """Guard demo-only, in-memory communication endpoints behind ENABLE_DEMO_MODE.

    When ENABLE_DEMO_MODE is false (e.g., production), these endpoints return
    501 to make it clear that they are mock-only.
    """

    if not ENABLE_DEMO_MODE:
        raise HTTPException(
            status_code=501,
            detail=(
                "Communications endpoints are mock-only and are available "
                "only when ENABLE_DEMO_MODE=true."
            ),
        )


# In-memory storage
COMMUNICATIONS_DB: Dict[str, Dict[str, Any]] = {}
COMMUNICATION_ID_COUNTER = 1

# Import customers database for business rules
from backend.api.customers_api import CUSTOMERS_DB


def _get_attr(obj: Any, name: str, default: Any = None) -> Any:
    """Helper to read attributes from either ORM objects or dicts."""

    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


class CommunicationStats(BaseModel):
    """Aggregate statistics for communications, used by dashboard KPI cards."""

    total_communications: int
    sent_last_30_days: int
    delivered_count: int
    read_count: int
    by_status: Dict[str, int]
    by_channel: Dict[str, int]
    by_type: Dict[str, int]


def _compute_communication_stats(comms: Iterable[Any]) -> CommunicationStats:
    comms_list = list(comms)
    total_communications = len(comms_list)

    thirty_days_ago = datetime.now() - timedelta(days=30)

    sent_last_30_days = 0
    delivered_count = 0
    read_count = 0
    by_status: Dict[str, int] = {}
    by_channel: Dict[str, int] = {}
    by_type: Dict[str, int] = {}

    for c in comms_list:
        status = str(_get_attr(c, "status", "")).lower()
        channel = str(_get_attr(c, "channel", "unknown")).lower()
        comm_type = str(_get_attr(c, "comm_type", "unknown")).lower()

        by_status[status] = by_status.get(status, 0) + 1
        by_channel[channel] = by_channel.get(channel, 0) + 1
        by_type[comm_type] = by_type.get(comm_type, 0) + 1

        if status in {"sent", "delivered", "completed", "read"}:
            comm_date_raw = _get_attr(c, "comm_date")
            if comm_date_raw:
                try:
                    dt = datetime.fromisoformat(comm_date_raw) if "T" in comm_date_raw else datetime.strptime(
                        comm_date_raw, "%Y-%m-%d"
                    )
                    if dt >= thirty_days_ago:
                        sent_last_30_days += 1
                except Exception:  # pragma: no cover - defensive
                    pass

        if status in {"delivered", "completed"}:
            delivered_count += 1
        if status == "read":
            read_count += 1

    return CommunicationStats(
        total_communications=total_communications,
        sent_last_30_days=sent_last_30_days,
        delivered_count=delivered_count,
        read_count=read_count,
        by_status=by_status,
        by_channel=by_channel,
        by_type=by_type,
    )

class CommunicationCreate(BaseModel):
    customer_name: str = Field(..., min_length=2, max_length=100, description="Customer name")
    comm_type: str = Field(..., description="Communication type (email, sms, call)")
    channel: str = Field(..., description="Communication channel")
    subject: str = Field(..., min_length=1, max_length=200, description="Subject/title")
    status: str = Field(default="sent", description="Communication status")
    comm_date: Optional[str] = Field(None, description="Communication date (ISO format)")
    content: Optional[str] = Field(None, max_length=5000, description="Message content")

    @validator('customer_name')
    def validate_customer_name(cls, v):
        """Validate customer name"""
        try:
            return validate_string_length(v, min_length=2, max_length=100, field_name="Customer name")
        except ValidationError as e:
            raise ValueError(str(e))

    @validator('comm_type')
    def validate_comm_type_field(cls, v):
        """Validate communication type"""
        allowed_types = ['email', 'sms', 'call', 'chat', 'letter']
        try:
            return validate_choice(v.lower(), allowed_types, field_name="Communication type")
        except ValidationError as e:
            raise ValueError(str(e))

    @validator('channel')
    def validate_channel_field(cls, v):
        """Validate channel"""
        allowed_channels = ['email', 'sms', 'phone', 'whatsapp', 'chat', 'mail']
        try:
            return validate_choice(v.lower(), allowed_channels, field_name="Channel")
        except ValidationError as e:
            raise ValueError(str(e))

    @validator('subject')
    def validate_subject(cls, v):
        """Validate subject"""
        try:
            return validate_string_length(v, min_length=1, max_length=200, field_name="Subject")
        except ValidationError as e:
            raise ValueError(str(e))

    @validator('status')
    def validate_status_field(cls, v):
        """Validate status"""
        allowed_statuses = ['sent', 'pending', 'failed', 'delivered', 'read']
        try:
            return validate_choice(v.lower(), allowed_statuses, field_name="Status")
        except ValidationError as e:
            raise ValueError(str(e))

    @validator('comm_date')
    def validate_comm_date(cls, v):
        """Validate communication date"""
        if v is None:
            return v
        try:
            return validate_date_string(v)
        except ValidationError as e:
            raise ValueError(str(e))

    @validator('content')
    def validate_content(cls, v):
        """Validate content"""
        if v is None:
            return v
        try:
            return validate_string_length(v, min_length=1, max_length=5000, field_name="Content")
        except ValidationError as e:
            raise ValueError(str(e))

class CommunicationUpdate(BaseModel):
    customer_name: Optional[str] = Field(None, min_length=2, max_length=100)
    comm_type: Optional[str] = None
    channel: Optional[str] = None
    subject: Optional[str] = Field(None, min_length=1, max_length=200)
    status: Optional[str] = None
    comm_date: Optional[str] = None
    content: Optional[str] = Field(None, max_length=5000)

    @validator('customer_name')
    def validate_customer_name(cls, v):
        """Validate customer name"""
        if v is None:
            return v
        try:
            return validate_string_length(v, min_length=2, max_length=100, field_name="Customer name")
        except ValidationError as e:
            raise ValueError(str(e))

    @validator('comm_type')
    def validate_comm_type_field(cls, v):
        """Validate communication type"""
        if v is None:
            return v
        allowed_types = ['email', 'sms', 'call', 'chat', 'letter']
        try:
            return validate_choice(v.lower(), allowed_types, field_name="Communication type")
        except ValidationError as e:
            raise ValueError(str(e))

    @validator('channel')
    def validate_channel_field(cls, v):
        """Validate channel"""
        if v is None:
            return v
        allowed_channels = ['email', 'sms', 'phone', 'whatsapp', 'chat', 'mail']
        try:
            return validate_choice(v.lower(), allowed_channels, field_name="Channel")
        except ValidationError as e:
            raise ValueError(str(e))

    @validator('status')
    def validate_status_field(cls, v):
        """Validate status"""
        if v is None:
            return v
        allowed_statuses = ['sent', 'pending', 'failed', 'delivered', 'read']
        try:
            return validate_choice(v.lower(), allowed_statuses, field_name="Status")
        except ValidationError as e:
            raise ValueError(str(e))

    @validator('comm_date')
    def validate_comm_date(cls, v):
        """Validate communication date"""
        if v is None:
            return v
        try:
            return validate_date_string(v)
        except ValidationError as e:
            raise ValueError(str(e))

class CommunicationResponse(BaseModel):
    id: str
    customer_name: str
    comm_type: str
    channel: str
    subject: str
    status: str
    comm_date: Optional[str]
    content: Optional[str]

class PaginatedCommunicationsResponse(BaseModel):
    communications: List[CommunicationResponse]
    total: int
    page: int
    page_size: int
    total_pages: int


@router.get("/stats", response_model=CommunicationStats)
async def get_communication_stats(
    session: AsyncSession = Depends(session_dep),
    current_user: Dict[str, Any] = Depends(require_permission(Permission.VIEW_MESSAGES)),
):
    """Get aggregate communication statistics for dashboards.

    When USE_DB=true this endpoint aggregates statistics from the real
    ``communications`` table. When USE_DB=false it falls back to the in-memory
    ``COMMUNICATIONS_DB`` demo store. In that case the values are mock-only and
    should not be relied on in production.
    """

    if USE_DB:
        result = await session.execute(select(Communication))
        comms = result.scalars().all()
        return _compute_communication_stats(comms)

    # Demo-only fallback using in-memory store
    return _compute_communication_stats(COMMUNICATIONS_DB.values())

@router.get("", response_model=PaginatedCommunicationsResponse)
async def get_communications(
    comm_type: Optional[str] = None,
    status: Optional[str] = None,
    page: int = Query(1, ge=1, description="Page number (starts at 1)"),
    page_size: int = Query(10, ge=1, le=100, description="Number of items per page"),
    current_user: Dict[str, Any] = Depends(require_permission(Permission.VIEW_MESSAGES)),
):
    """Get all communications with pagination, optionally filtered by type or status.

    Authentication: Requires VIEW_MESSAGES permission.
    """

    _ensure_demo_mode_enabled()

    communications = list(COMMUNICATIONS_DB.values())

    # Apply filters
    if comm_type:
        communications = [c for c in communications if c["comm_type"] == comm_type]
    if status:
        communications = [c for c in communications if c["status"] == status]

    # Calculate pagination
    total = len(communications)
    total_pages = (total + page_size - 1) // page_size if total > 0 else 1
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size

    # Get paginated results
    paginated_communications = communications[start_idx:end_idx]

    return {
        "communications": paginated_communications,
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": total_pages
    }

@router.get("/{communication_id}", response_model=CommunicationResponse)
async def get_communication(
    communication_id: str,
    current_user: Dict[str, Any] = Depends(require_permission(Permission.VIEW_MESSAGES)),
):
    """Get a specific communication by ID.

    Authentication: Requires VIEW_MESSAGES permission.
    """

    _ensure_demo_mode_enabled()

    if communication_id not in COMMUNICATIONS_DB:
        raise HTTPException(status_code=404, detail="Communication not found")
    return COMMUNICATIONS_DB[communication_id]

@router.post("", response_model=CommunicationResponse)
async def create_communication(
    communication: CommunicationCreate,
    current_user: dict = Depends(require_permission(Permission.SEND_MESSAGES)),
):
    """
    Create a new communication

    Authentication: Required
    Headers: X-Session-ID or X-API-Key

    Business Rules:
    - Customer must exist (warning if not found)
    - Cannot send marketing to inactive customers
    - Priority is auto-determined based on type and subject
    - Communication date defaults to now if not provided
    """
    global COMMUNICATION_ID_COUNTER

    _ensure_demo_mode_enabled()

    # Business Rule: Verify customer exists (warning only, not blocking)
    customer = find_customer_by_name(communication.customer_name, CUSTOMERS_DB)
    if not customer:
        logger.warning(
            f"Communication created for unknown customer: {communication.customer_name}"
        )

    # Business Rule: Check if marketing can be sent to this customer
    comm_dict = communication.dict()
    is_marketing = 'marketing' in communication.subject.lower() or \
                   'promotion' in communication.subject.lower() or \
                   'offer' in communication.subject.lower()

    if customer and is_marketing:
        if not can_send_marketing_to_customer(customer):
            raise HTTPException(
                status_code=400,
                detail=f"Cannot send marketing communications to customer '{communication.customer_name}'. "
                       f"Customer status: {customer.get('status')}, Policies: {customer.get('policies_count', 0)}"
            )

    # Business Rule: Auto-determine priority
    priority = get_communication_priority(communication.comm_type, communication.subject)
    comm_dict['priority'] = priority
    logger.info(f"Auto-set priority to '{priority}' based on type and subject")

    # Business Rule: Set default communication date
    if not comm_dict.get('comm_date'):
        comm_dict['comm_date'] = datetime.now().isoformat()

    # Generate communication ID
    communication_id = f"COMM-{str(COMMUNICATION_ID_COUNTER).zfill(3)}"
    COMMUNICATION_ID_COUNTER += 1

    communication_data = {
        "id": communication_id,
        **comm_dict
    }
    COMMUNICATIONS_DB[communication_id] = communication_data

    logger.info(
        f"User {current_user['username']} created communication {communication_id} "
        f"for {communication.customer_name} (type: {communication.comm_type}, priority: {priority})"
    )

    return communication_data

@router.put("/{communication_id}", response_model=CommunicationResponse)
async def update_communication(
    communication_id: str,
    communication: CommunicationUpdate,
    current_user: dict = Depends(require_permission(Permission.SEND_MESSAGES)),
):
    """
    Update an existing communication

    Authentication: Required
    Headers: X-Session-ID or X-API-Key

    Business Rules:
    - Status changes are logged
    """
    _ensure_demo_mode_enabled()

    if communication_id not in COMMUNICATIONS_DB:
        raise HTTPException(status_code=404, detail="Communication not found")

    existing_comm = COMMUNICATIONS_DB[communication_id]
    update_data = communication.dict(exclude_unset=True)

    # Business Rule: Log status changes
    if 'status' in update_data and update_data['status'] != existing_comm['status']:
        logger.info(
            f"Communication {communication_id} status changed: "
            f"{existing_comm['status']} -> {update_data['status']}"
        )

    # Apply updates
    COMMUNICATIONS_DB[communication_id].update(update_data)

    logger.info(f"User {current_user['username']} updated communication: {communication_id}")
    return COMMUNICATIONS_DB[communication_id]

@router.delete("/{communication_id}")
async def delete_communication(
    communication_id: str,
    current_user: dict = Depends(require_permission(Permission.SEND_MESSAGES)),
):
    """
    Delete a communication

    Authentication: Required
    Headers: X-Session-ID or X-API-Key
    """
    _ensure_demo_mode_enabled()

    if communication_id not in COMMUNICATIONS_DB:
        raise HTTPException(status_code=404, detail="Communication not found")

    del COMMUNICATIONS_DB[communication_id]

    logger.info(f"User {current_user['username']} deleted communication: {communication_id}")
    return {"message": "Communication deleted successfully", "id": communication_id}

