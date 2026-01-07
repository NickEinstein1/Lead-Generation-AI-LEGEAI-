"""
Claims API - CRUD endpoints for claims management
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
    find_policy_by_number,
    verify_policy_active,
    validate_claim_amount_vs_coverage,
    should_auto_approve_claim,
    # Advanced business rules
    should_escalate_claim,
    get_claim_priority,
)
from backend.api.auth_dependencies import (
    get_current_user_from_session,
    get_optional_user,
    require_permission,
)
from backend.security.authentication import Permission
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from backend.database.connection import session_dep
from backend.models.claim import Claim

router = APIRouter(prefix="/claims", tags=["Claims"])
logger = logging.getLogger(__name__)

ENABLE_DEMO_MODE = os.getenv("ENABLE_DEMO_MODE", "false").lower() == "true"
USE_DB = os.getenv("USE_DB", "false").lower() == "true"


def _ensure_demo_mode_enabled() -> None:
    """Guard demo-only, in-memory claims endpoints behind ENABLE_DEMO_MODE.

    These claims endpoints currently use in-memory storage only. When
    ENABLE_DEMO_MODE is false (e.g., production), they are disabled to avoid
    accidentally relying on mock data.
    """
    if not ENABLE_DEMO_MODE:
        raise HTTPException(
            status_code=501,
            detail=(
                "Claims endpoints are mock-only and are available "
                "only when ENABLE_DEMO_MODE=true."
            ),
        )

# In-memory storage
CLAIMS_DB: Dict[str, Dict[str, Any]] = {}
CLAIM_ID_COUNTER = 1


def _get_attr(obj: Any, name: str, default: Any = None) -> Any:
    """Helper to read attributes from either ORM objects or dicts."""

    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


class ClaimStats(BaseModel):
    """Aggregate statistics for claims, used by dashboard KPI cards."""

    total_claims: int
    pending_claims: int
    approved_claims: int
    rejected_claims: int
    total_payout: float
    by_status: Dict[str, Dict[str, float]]


def _parse_amount_to_float(amount: Optional[str]) -> float:
    """Parse amount string like "$5,000" into a float."""

    if not amount:
        return 0.0
    try:
        cleaned = amount.replace("$", "").replace(",", "").strip()
        return float(cleaned)
    except Exception:  # pragma: no cover - defensive
        return 0.0


def _compute_claim_stats(claims: Iterable[Any]) -> ClaimStats:
    claims_list = list(claims)
    total_claims = len(claims_list)

    pending = 0
    approved = 0
    rejected = 0
    total_payout = 0.0
    by_status: Dict[str, Dict[str, float]] = {}

    for c in claims_list:
        status = str(_get_attr(c, "status", "")).lower()
        amount_raw = _get_attr(c, "amount")
        amount_val = _parse_amount_to_float(amount_raw)

        if status == "pending":
            pending += 1
        elif status in {"approved", "paid"}:
            approved += 1
            total_payout += amount_val
        elif status in {"rejected", "denied"}:
            rejected += 1

        entry = by_status.setdefault(status or "unknown", {"count": 0, "total_amount": 0.0})
        entry["count"] += 1
        entry["total_amount"] += amount_val

    return ClaimStats(
        total_claims=total_claims,
        pending_claims=pending,
        approved_claims=approved,
        rejected_claims=rejected,
        total_payout=round(total_payout, 2),
        by_status={k: {"count": v["count"], "total_amount": round(v["total_amount"], 2)} for k, v in by_status.items()},
    )


# Import policies database for business rules
from backend.api.policies_api import POLICIES_DB

class ClaimCreate(BaseModel):
    policy_number: str = Field(..., min_length=3, max_length=50, description="Policy number")
    customer_name: str = Field(..., min_length=2, max_length=100, description="Customer name")
    claim_type: str = Field(..., description="Type of claim")
    amount: str = Field(..., description="Claim amount")
    status: str = Field(default="pending", description="Claim status")
    claim_date: Optional[str] = Field(None, description="Claim date (ISO format)")
    due_date: Optional[str] = Field(None, description="Due date (ISO format)")
    description: Optional[str] = Field(None, max_length=1000, description="Claim description")

    @validator('policy_number')
    def validate_policy_number(cls, v):
        """Validate policy number"""
        try:
            return validate_string_length(v, min_length=3, max_length=50, field_name="Policy number")
        except ValidationError as e:
            raise ValueError(str(e))

    @validator('customer_name')
    def validate_customer_name(cls, v):
        """Validate customer name"""
        try:
            return validate_string_length(v, min_length=2, max_length=100, field_name="Customer name")
        except ValidationError as e:
            raise ValueError(str(e))

    @validator('claim_type')
    def validate_claim_type_field(cls, v):
        """Validate claim type"""
        allowed_types = ['accident', 'theft', 'damage', 'medical', 'liability', 'other']
        try:
            return validate_choice(v.lower(), allowed_types, field_name="Claim type")
        except ValidationError as e:
            raise ValueError(str(e))

    @validator('status')
    def validate_status_field(cls, v):
        """Validate status"""
        allowed_statuses = ['pending', 'approved', 'rejected', 'processing', 'paid']
        try:
            return validate_choice(v.lower(), allowed_statuses, field_name="Status")
        except ValidationError as e:
            raise ValueError(str(e))

    @validator('claim_date', 'due_date')
    def validate_date_fields(cls, v):
        """Validate date format"""
        if v is None:
            return v
        try:
            return validate_date_string(v)
        except ValidationError as e:
            raise ValueError(str(e))

    @validator('description')
    def validate_description(cls, v):
        """Validate description"""
        if v is None:
            return v
        try:
            return validate_string_length(v, min_length=1, max_length=1000, field_name="Description")
        except ValidationError as e:
            raise ValueError(str(e))

class ClaimUpdate(BaseModel):
    policy_number: Optional[str] = Field(None, min_length=3, max_length=50)
    customer_name: Optional[str] = Field(None, min_length=2, max_length=100)
    claim_type: Optional[str] = None
    amount: Optional[str] = None
    status: Optional[str] = None
    claim_date: Optional[str] = None
    due_date: Optional[str] = None
    description: Optional[str] = Field(None, max_length=1000)

    @validator('policy_number')
    def validate_policy_number(cls, v):
        """Validate policy number"""
        if v is None:
            return v
        try:
            return validate_string_length(v, min_length=3, max_length=50, field_name="Policy number")
        except ValidationError as e:
            raise ValueError(str(e))

    @validator('customer_name')
    def validate_customer_name(cls, v):
        """Validate customer name"""
        if v is None:
            return v
        try:
            return validate_string_length(v, min_length=2, max_length=100, field_name="Customer name")
        except ValidationError as e:
            raise ValueError(str(e))

    @validator('claim_type')
    def validate_claim_type_field(cls, v):
        """Validate claim type"""
        if v is None:
            return v
        allowed_types = ['accident', 'theft', 'damage', 'medical', 'liability', 'other']
        try:
            return validate_choice(v.lower(), allowed_types, field_name="Claim type")
        except ValidationError as e:
            raise ValueError(str(e))

    @validator('status')
    def validate_status_field(cls, v):
        """Validate status"""
        if v is None:
            return v
        allowed_statuses = ['pending', 'approved', 'rejected', 'processing', 'paid']
        try:
            return validate_choice(v.lower(), allowed_statuses, field_name="Status")
        except ValidationError as e:
            raise ValueError(str(e))

    @validator('claim_date', 'due_date')
    def validate_date_fields(cls, v):
        """Validate date format"""
        if v is None:
            return v
        try:
            return validate_date_string(v)
        except ValidationError as e:
            raise ValueError(str(e))

class ClaimResponse(BaseModel):
    id: str
    claim_number: str
    policy_number: str
    customer_name: str
    claim_type: str
    amount: str
    status: str
    claim_date: Optional[str]
    due_date: Optional[str]
    description: Optional[str]

class PaginatedClaimsResponse(BaseModel):
    claims: List[ClaimResponse]
    total: int
    page: int
    page_size: int
    total_pages: int


@router.get("/stats", response_model=ClaimStats)
async def get_claim_stats(
    session: AsyncSession = Depends(session_dep),
    current_user: dict = Depends(require_permission(Permission.VIEW_LEADS)),
):
    """Get aggregate claim statistics for dashboards.

    When USE_DB=true this endpoint aggregates statistics from the real
    ``claims`` table. When USE_DB=false it falls back to the in-memory
    ``CLAIMS_DB`` demo store. In that case the values are mock-only and should
    not be relied on in production.
    """

    if USE_DB:
        result = await session.execute(select(Claim))
        claims = result.scalars().all()
        return _compute_claim_stats(claims)

    # Demo-only fallback using in-memory store
    return _compute_claim_stats(CLAIMS_DB.values())

@router.get("", response_model=PaginatedClaimsResponse)
async def get_claims(
    status: Optional[str] = None,
    claim_type: Optional[str] = None,
    page: int = Query(1, ge=1, description="Page number (starts at 1)"),
    page_size: int = Query(10, ge=1, le=100, description="Number of items per page"),
    current_user: dict = Depends(require_permission(Permission.VIEW_LEADS)),
):
    """Get all claims with pagination, optionally filtered by status or type.

    Authentication: Requires VIEW_LEADS permission.
    """
    _ensure_demo_mode_enabled()
    claims = list(CLAIMS_DB.values())

    # Apply filters
    if status:
        claims = [c for c in claims if c["status"] == status]
    if claim_type:
        claims = [c for c in claims if c["claim_type"].lower() == claim_type.lower()]

    # Calculate pagination
    total = len(claims)
    total_pages = (total + page_size - 1) // page_size if total > 0 else 1
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size

    # Get paginated results
    paginated_claims = claims[start_idx:end_idx]

    return {
        "claims": paginated_claims,
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": total_pages
    }

@router.get("/{claim_id}", response_model=ClaimResponse)
async def get_claim(
    claim_id: str,
    current_user: dict = Depends(require_permission(Permission.VIEW_LEADS)),
):
    """Get a specific claim by ID.

    Authentication: Requires VIEW_LEADS permission.
    """
    _ensure_demo_mode_enabled()
    if claim_id not in CLAIMS_DB:
        raise HTTPException(status_code=404, detail="Claim not found")
    return CLAIMS_DB[claim_id]

@router.post("", response_model=ClaimResponse)
async def create_claim(
    claim: ClaimCreate,
    current_user: dict = Depends(require_permission(Permission.CREATE_LEADS)),
):
    """
    Create a new claim.

    Authentication: Requires CREATE_LEADS permission.
    Headers: X-Session-ID or X-API-Key

    Business Rules:
    - Policy must exist
    - Policy must be active
    - Claim amount cannot exceed policy coverage
    - Claims under threshold are auto-approved
    - Claim date defaults to today if not provided
    - Due date is auto-set to 30 days from claim date
    """
    _ensure_demo_mode_enabled()
    global CLAIM_ID_COUNTER

    # Business Rule: Verify policy exists
    policy = find_policy_by_number(claim.policy_number, POLICIES_DB)
    if not policy:
        raise HTTPException(
            status_code=404,
            detail=f"Policy '{claim.policy_number}' not found. Please verify policy number."
        )

    # Business Rule: Verify policy is active
    try:
        verify_policy_active(policy)
    except HTTPException as e:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot file claim on inactive policy. {e.detail}"
        )

    # Business Rule: Validate claim amount doesn't exceed coverage
    validate_claim_amount_vs_coverage(claim.amount, policy)

    # Business Rule: Auto-approve small claims
    claim_dict = claim.dict()
    if should_auto_approve_claim(claim.amount, claim.claim_type):
        claim_dict['status'] = 'approved'
        logger.info(f"Claim auto-approved (amount: {claim.amount}, type: {claim.claim_type})")
    else:
        claim_dict['status'] = 'pending'
        logger.info(f"Claim requires manual review (amount: {claim.amount})")

    # Business Rule: Set default claim date to today if not provided
    if not claim_dict.get('claim_date'):
        claim_dict['claim_date'] = datetime.now().isoformat()

    # Business Rule: Auto-set due date to 30 days from claim date
    if not claim_dict.get('due_date'):
        claim_date = datetime.fromisoformat(claim_dict['claim_date'].replace('Z', '+00:00'))
        due_date = claim_date + timedelta(days=30)
        claim_dict['due_date'] = due_date.isoformat()

    # Generate claim ID and number
    claim_id = f"CLM-{str(CLAIM_ID_COUNTER).zfill(3)}"
    claim_number = f"CLM-{str(CLAIM_ID_COUNTER).zfill(6)}"
    CLAIM_ID_COUNTER += 1

    claim_data = {
        "id": claim_id,
        "claim_number": claim_number,
        **claim_dict
    }
    CLAIMS_DB[claim_id] = claim_data

    # Business Rule: Log notification (in real system, would send email/SMS)
    logger.info(
        f"User {current_user['username']} created claim {claim_id} for policy {claim.policy_number} "
        f"(amount: {claim.amount}, status: {claim_dict['status']})"
    )
    logger.info(f"NOTIFICATION: Claim {claim_number} filed for customer {claim.customer_name}")

    return claim_data

@router.put("/{claim_id}", response_model=ClaimResponse)
async def update_claim(
    claim_id: str,
    claim: ClaimUpdate,
    current_user: dict = Depends(require_permission(Permission.UPDATE_LEADS)),
):
    """
    Update an existing claim.

    Authentication: Requires UPDATE_LEADS permission.
    Headers: X-Session-ID or X-API-Key

    Business Rules:
    - Claim amount cannot exceed policy coverage if changed
    - Status changes are logged and notifications sent
    - Approved/Paid claims trigger notifications
    """
    _ensure_demo_mode_enabled()
    if claim_id not in CLAIMS_DB:
        raise HTTPException(status_code=404, detail="Claim not found")

    existing_claim = CLAIMS_DB[claim_id]
    update_data = claim.dict(exclude_unset=True)

    # Business Rule: Validate claim amount vs coverage if changing
    if 'amount' in update_data:
        policy = find_policy_by_number(existing_claim['policy_number'], POLICIES_DB)
        if policy:
            validate_claim_amount_vs_coverage(update_data['amount'], policy)

    # Business Rule: Log status changes and send notifications
    if 'status' in update_data and update_data['status'] != existing_claim['status']:
        old_status = existing_claim['status']
        new_status = update_data['status']

        logger.info(f"Claim {claim_id} status changed: {old_status} -> {new_status}")

        # Send notifications for important status changes
        if new_status in ['approved', 'paid']:
            logger.info(
                f"NOTIFICATION: Claim {existing_claim['claim_number']} has been {new_status} "
                f"for customer {existing_claim['customer_name']}"
            )
        elif new_status == 'rejected':
            logger.info(
                f"NOTIFICATION: Claim {existing_claim['claim_number']} has been rejected "
                f"for customer {existing_claim['customer_name']}"
            )

    # Apply updates
    CLAIMS_DB[claim_id].update(update_data)

    logger.info(f"User {current_user['username']} updated claim: {claim_id}")
    return CLAIMS_DB[claim_id]

@router.delete("/{claim_id}")
async def delete_claim(
    claim_id: str,
    current_user: dict = Depends(require_permission(Permission.DELETE_LEADS)),
):
    """
    Delete a claim.

    Authentication: Requires DELETE_LEADS permission.
    Headers: X-Session-ID or X-API-Key

    Business Rules:
    - Deletion is logged for audit purposes
    """
    _ensure_demo_mode_enabled()
    if claim_id not in CLAIMS_DB:
        raise HTTPException(status_code=404, detail="Claim not found")

    claim = CLAIMS_DB[claim_id]
    claim_number = claim['claim_number']

    # Delete the claim
    del CLAIMS_DB[claim_id]

    logger.info(
        f"User {current_user['username']} deleted claim: {claim_id} ({claim_number})"
    )
    return {"message": "Claim deleted successfully", "id": claim_id}


# ============================================================================
# ADVANCED CLAIM ENDPOINTS - Escalation, Priority, Analytics
# ============================================================================

@router.get("/{claim_id}/escalation-check")
async def check_claim_escalation(
    claim_id: str,
    current_user: dict = Depends(require_permission(Permission.VIEW_LEADS)),
):
    """
    Check if a claim should be escalated.

    Authentication: Requires VIEW_LEADS permission (read-only).

    Returns:
    - Escalation recommendation
    - Reasons for escalation
    - Priority level
    - Suggested actions
    """
    if claim_id not in CLAIMS_DB:
        raise HTTPException(status_code=404, detail="Claim not found")

    claim = CLAIMS_DB[claim_id]

    # Get policy for coverage comparison
    policy = find_policy_by_number(claim['policy_number'], POLICIES_DB)

    # Business Rule: Check if claim should be escalated
    escalation_result = should_escalate_claim(claim, policy)

    # Business Rule: Get claim priority
    priority = get_claim_priority(claim)

    # Generate recommendations
    recommendations = []
    if escalation_result['should_escalate']:
        recommendations.append("Assign to senior claims adjuster")
        recommendations.append("Request additional documentation")

        if 'High amount' in str(escalation_result['reasons']):
            recommendations.append("Conduct fraud investigation")
            recommendations.append("Verify policy coverage details")

        if 'Pending for' in str(escalation_result['reasons']):
            recommendations.append("Contact customer for status update")
            recommendations.append("Set reminder for follow-up")

    _ensure_demo_mode_enabled()

    return {
        "claim_id": claim_id,
        "claim_number": claim['claim_number'],
        "should_escalate": escalation_result['should_escalate'],
        "escalation_priority": escalation_result.get('escalation_priority', 'none'),
        "escalation_reasons": escalation_result['reasons'],
        "claim_priority": priority,
        "recommendations": recommendations,
        "claim_status": claim.get('status'),
        "claim_amount": claim.get('amount'),
        "claim_age_days": _calculate_claim_age(claim),
        "policy_coverage": f"${policy.get('coverage_amount', 0):,.2f}" if policy else "N/A"
    }


def _calculate_claim_age(claim: Dict[str, Any]) -> int:
    """Helper function to calculate claim age in days"""
    claim_date_str = claim.get('claim_date')
    if claim_date_str:
        try:
            claim_date = datetime.fromisoformat(claim_date_str.replace('Z', '+00:00'))
            return (datetime.now() - claim_date).days
        except ValueError:
            pass
    return 0


@router.post("/{claim_id}/escalate")
async def escalate_claim(
    claim_id: str,
    escalation_reason: str = Query(..., description="Reason for manual escalation"),
    assigned_to: Optional[str] = Query(None, description="Assign to specific adjuster"),
    current_user: dict = Depends(require_permission(Permission.UPDATE_LEADS)),
):
    """
    Manually escalate a claim.

    Authentication: Requires UPDATE_LEADS permission.
    Headers: X-Session-ID or X-API-Key

    Business Rules:
    - Claim status updated to 'under_review'
    - Escalation logged with timestamp and reason
    - Priority automatically set to 'high'
    - Notification sent to assigned adjuster
    """
    if claim_id not in CLAIMS_DB:
        raise HTTPException(status_code=404, detail="Claim not found")

    claim = CLAIMS_DB[claim_id]

    # Update claim with escalation details
    CLAIMS_DB[claim_id]['status'] = 'under_review'
    CLAIMS_DB[claim_id]['escalated'] = True
    CLAIMS_DB[claim_id]['escalation_date'] = datetime.now().isoformat()
    CLAIMS_DB[claim_id]['escalation_reason'] = escalation_reason
    CLAIMS_DB[claim_id]['escalated_by'] = current_user['username']

    if assigned_to:
        CLAIMS_DB[claim_id]['assigned_to'] = assigned_to

    # Set priority to high
    priority = get_claim_priority(claim)
    CLAIMS_DB[claim_id]['priority'] = 'high' if priority != 'critical' else 'critical'

    logger.info(
        f"User {current_user['username']} escalated claim {claim['claim_number']}. "
        f"Reason: {escalation_reason}, Assigned to: {assigned_to or 'unassigned'}"
    )

    # Log notification
    logger.info(
        f"NOTIFICATION: Claim {claim['claim_number']} escalated. "
        f"Assigned to: {assigned_to or 'claims team'}"
    )

    _ensure_demo_mode_enabled()

    return {
        "message": "Claim escalated successfully",
        "claim_id": claim_id,
        "claim_number": claim['claim_number'],
        "status": "under_review",
        "priority": CLAIMS_DB[claim_id]['priority'],
        "assigned_to": assigned_to or "Unassigned",
        "escalation_date": CLAIMS_DB[claim_id]['escalation_date']
    }


@router.post("/{claim_id}/approve")
async def approve_claim(
    claim_id: str,
    approved_amount: Optional[str] = Query(None, description="Approved amount (if different from requested)"),
    notes: Optional[str] = Query(None, description="Approval notes"),
    current_user: dict = Depends(require_permission(Permission.UPDATE_LEADS)),
):
    """
    Approve a claim.

    Authentication: Requires UPDATE_LEADS permission.
    Headers: X-Session-ID or X-API-Key

    Business Rules:
    - Claim status updated to 'approved'
    - Approved amount cannot exceed requested amount
    - Approval logged with timestamp
    - Customer notification sent
    """
    if claim_id not in CLAIMS_DB:
        raise HTTPException(status_code=404, detail="Claim not found")

    claim = CLAIMS_DB[claim_id]
    requested_amount = float(claim['amount'].replace('$', '').replace(',', '').strip())

    # Validate approved amount
    if approved_amount:
        approved_amt = float(approved_amount.replace('$', '').replace(',', '').strip())
        if approved_amt > requested_amount:
            raise HTTPException(
                status_code=400,
                detail=f"Approved amount (${approved_amt:,.2f}) cannot exceed requested amount (${requested_amount:,.2f})"
            )
    else:
        approved_amt = requested_amount

    # Update claim
    CLAIMS_DB[claim_id]['status'] = 'approved'
    CLAIMS_DB[claim_id]['approved_amount'] = f"${approved_amt:.2f}"
    CLAIMS_DB[claim_id]['approved_by'] = current_user['username']
    CLAIMS_DB[claim_id]['approval_date'] = datetime.now().isoformat()

    if notes:
        CLAIMS_DB[claim_id]['approval_notes'] = notes

    logger.info(
        f"User {current_user['username']} approved claim {claim['claim_number']}. "
        f"Amount: ${approved_amt:,.2f}"
    )

    # Log notification
    logger.info(
        f"NOTIFICATION: Claim {claim['claim_number']} approved for ${approved_amt:,.2f}. "
        f"Customer: {claim['customer_name']}"
    )

    _ensure_demo_mode_enabled()

    return {
        "message": "Claim approved successfully",
        "claim_id": claim_id,
        "claim_number": claim['claim_number'],
        "requested_amount": f"${requested_amount:,.2f}",
        "approved_amount": f"${approved_amt:,.2f}",
        "status": "approved",
        "approved_by": current_user['username'],
        "approval_date": CLAIMS_DB[claim_id]['approval_date']
    }


@router.post("/{claim_id}/reject")
async def reject_claim(
    claim_id: str,
    rejection_reason: str = Query(..., description="Reason for rejection"),
    current_user: dict = Depends(require_permission(Permission.UPDATE_LEADS)),
):
    """
    Reject a claim.

    Authentication: Requires UPDATE_LEADS permission.
    Headers: X-Session-ID or X-API-Key

    Business Rules:
    - Claim status updated to 'rejected'
    - Rejection reason required
    - Rejection logged with timestamp
    - Customer notification sent
    """
    if claim_id not in CLAIMS_DB:
        raise HTTPException(status_code=404, detail="Claim not found")

    claim = CLAIMS_DB[claim_id]

    # Update claim
    CLAIMS_DB[claim_id]['status'] = 'rejected'
    CLAIMS_DB[claim_id]['rejection_reason'] = rejection_reason
    CLAIMS_DB[claim_id]['rejected_by'] = current_user['username']
    CLAIMS_DB[claim_id]['rejection_date'] = datetime.now().isoformat()

    logger.info(
        f"User {current_user['username']} rejected claim {claim['claim_number']}. "
        f"Reason: {rejection_reason}"
    )

    # Log notification
    logger.info(
        f"NOTIFICATION: Claim {claim['claim_number']} rejected. "
        f"Customer: {claim['customer_name']}, Reason: {rejection_reason}"
    )

    _ensure_demo_mode_enabled()

    return {
        "message": "Claim rejected",
        "claim_id": claim_id,
        "claim_number": claim['claim_number'],
        "status": "rejected",
        "rejection_reason": rejection_reason,
        "rejected_by": current_user['username'],
        "rejection_date": CLAIMS_DB[claim_id]['rejection_date']
    }

