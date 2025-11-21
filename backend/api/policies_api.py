"""
Policies API - CRUD endpoints for policy management
"""
from fastapi import APIRouter, HTTPException, Query, Depends
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
import logging
from datetime import datetime
from backend.utils.validators import (
    validate_string_length,
    validate_choice,
    validate_non_negative_number,
    validate_amount_range,
    validate_date_string,
    ValidationError
)
from backend.utils.business_rules import (
    find_customer_by_name,
    verify_customer_active,
    validate_premium_range,
    calculate_coverage_from_premium,
    update_customer_policy_stats,
    validate_date_range,
    # Advanced business rules
    is_policy_eligible_for_renewal,
    calculate_renewal_premium,
    should_auto_renew_policy,
    can_cancel_policy
)
from backend.api.auth_dependencies import get_current_user_from_session, get_optional_user

router = APIRouter(prefix="/policies", tags=["Policies"])
logger = logging.getLogger(__name__)

# In-memory storage
POLICIES_DB = {}
POLICY_ID_COUNTER = 1

# Import customers database for business rules
from backend.api.customers_api import CUSTOMERS_DB

class PolicyCreate(BaseModel):
    customer_name: str = Field(..., min_length=2, max_length=100, description="Customer name")
    policy_type: str = Field(..., description="Type of policy (auto, home, life, health)")
    status: str = Field(default="active", description="Policy status")
    premium: str = Field(..., description="Premium amount")
    coverage_amount: Optional[float] = Field(None, ge=0, description="Coverage amount")
    start_date: Optional[str] = Field(None, description="Policy start date (ISO format)")
    end_date: Optional[str] = Field(None, description="Policy end date (ISO format)")
    renewal_date: Optional[str] = Field(None, description="Policy renewal date (ISO format)")

    @validator('customer_name')
    def validate_customer_name(cls, v):
        """Validate customer name"""
        try:
            return validate_string_length(v, min_length=2, max_length=100, field_name="Customer name")
        except ValidationError as e:
            raise ValueError(str(e))

    @validator('policy_type')
    def validate_policy_type_field(cls, v):
        """Validate policy type"""
        allowed_types = ['auto', 'home', 'life', 'health', 'business', 'travel']
        try:
            return validate_choice(v.lower(), allowed_types, field_name="Policy type")
        except ValidationError as e:
            raise ValueError(str(e))

    @validator('status')
    def validate_status_field(cls, v):
        """Validate status"""
        allowed_statuses = ['active', 'inactive', 'pending', 'expired', 'cancelled']
        try:
            return validate_choice(v.lower(), allowed_statuses, field_name="Status")
        except ValidationError as e:
            raise ValueError(str(e))

    @validator('coverage_amount')
    def validate_coverage_amount_field(cls, v):
        """Validate coverage amount"""
        if v is None:
            return v
        try:
            return validate_amount_range(v, min_amount=0, max_amount=10000000, field_name="Coverage amount")
        except ValidationError as e:
            raise ValueError(str(e))

    @validator('start_date', 'end_date', 'renewal_date')
    def validate_date_fields(cls, v):
        """Validate date format"""
        if v is None:
            return v
        try:
            return validate_date_string(v)
        except ValidationError as e:
            raise ValueError(str(e))

class PolicyUpdate(BaseModel):
    customer_name: Optional[str] = Field(None, min_length=2, max_length=100)
    policy_type: Optional[str] = None
    status: Optional[str] = None
    premium: Optional[str] = None
    coverage_amount: Optional[float] = Field(None, ge=0)
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    renewal_date: Optional[str] = None

    @validator('customer_name')
    def validate_customer_name(cls, v):
        """Validate customer name"""
        if v is None:
            return v
        try:
            return validate_string_length(v, min_length=2, max_length=100, field_name="Customer name")
        except ValidationError as e:
            raise ValueError(str(e))

    @validator('policy_type')
    def validate_policy_type_field(cls, v):
        """Validate policy type"""
        if v is None:
            return v
        allowed_types = ['auto', 'home', 'life', 'health', 'business', 'travel']
        try:
            return validate_choice(v.lower(), allowed_types, field_name="Policy type")
        except ValidationError as e:
            raise ValueError(str(e))

    @validator('status')
    def validate_status_field(cls, v):
        """Validate status"""
        if v is None:
            return v
        allowed_statuses = ['active', 'inactive', 'pending', 'expired', 'cancelled']
        try:
            return validate_choice(v.lower(), allowed_statuses, field_name="Status")
        except ValidationError as e:
            raise ValueError(str(e))

    @validator('start_date', 'end_date', 'renewal_date')
    def validate_date_fields(cls, v):
        """Validate date format"""
        if v is None:
            return v
        try:
            return validate_date_string(v)
        except ValidationError as e:
            raise ValueError(str(e))

class PolicyResponse(BaseModel):
    id: str
    policy_number: str
    customer_name: str
    policy_type: str
    status: str
    premium: str
    coverage_amount: Optional[float]
    start_date: Optional[str]
    end_date: Optional[str]
    renewal_date: Optional[str]

class PaginatedPoliciesResponse(BaseModel):
    policies: List[PolicyResponse]
    total: int
    page: int
    page_size: int
    total_pages: int

@router.get("", response_model=PaginatedPoliciesResponse)
async def get_policies(
    policy_type: Optional[str] = None,
    status: Optional[str] = None,
    page: int = Query(1, ge=1, description="Page number (starts at 1)"),
    page_size: int = Query(10, ge=1, le=100, description="Number of items per page")
):
    """Get all policies with pagination, optionally filtered by type or status"""
    policies = list(POLICIES_DB.values())

    # Apply filters
    if policy_type:
        policies = [p for p in policies if p["policy_type"].lower() == policy_type.lower()]
    if status:
        policies = [p for p in policies if p["status"] == status]

    # Calculate pagination
    total = len(policies)
    total_pages = (total + page_size - 1) // page_size if total > 0 else 1
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size

    # Get paginated results
    paginated_policies = policies[start_idx:end_idx]

    return {
        "policies": paginated_policies,
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": total_pages
    }

@router.get("/{policy_id}", response_model=PolicyResponse)
async def get_policy(policy_id: str):
    """Get a specific policy by ID"""
    if policy_id not in POLICIES_DB:
        raise HTTPException(status_code=404, detail="Policy not found")
    return POLICIES_DB[policy_id]

@router.post("", response_model=PolicyResponse)
async def create_policy(
    policy: PolicyCreate,
    current_user: dict = Depends(get_current_user_from_session)
):
    """
    Create a new policy

    Authentication: Required
    Headers: X-Session-ID or X-API-Key

    Business Rules:
    - Customer must exist
    - Customer must be active
    - Premium must be within acceptable range for policy type
    - Coverage is auto-calculated from premium if not provided
    - Start date must be before end date
    - Customer's policy count and total value are updated
    """
    global POLICY_ID_COUNTER

    # Business Rule: Verify customer exists
    customer = find_customer_by_name(policy.customer_name, CUSTOMERS_DB)
    if not customer:
        raise HTTPException(
            status_code=404,
            detail=f"Customer '{policy.customer_name}' not found. Please create customer first."
        )

    # Business Rule: Verify customer is active
    try:
        verify_customer_active(customer)
    except HTTPException as e:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot create policy for inactive customer. {e.detail}"
        )

    # Business Rule: Validate premium is within acceptable range
    validate_premium_range(policy.premium, policy.policy_type)

    # Business Rule: Auto-calculate coverage if not provided
    policy_dict = policy.dict()
    if not policy_dict.get('coverage_amount'):
        policy_dict['coverage_amount'] = calculate_coverage_from_premium(
            policy.premium,
            policy.policy_type
        )
        logger.info(f"Auto-calculated coverage: ${policy_dict['coverage_amount']:.2f}")

    # Business Rule: Validate date range
    if policy_dict.get('start_date') and policy_dict.get('end_date'):
        validate_date_range(policy_dict['start_date'], policy_dict['end_date'])

    # Business Rule: Set default dates if not provided
    if not policy_dict.get('start_date'):
        policy_dict['start_date'] = datetime.now().isoformat()

    if not policy_dict.get('end_date'):
        # Default to 1 year from start date
        start = datetime.fromisoformat(policy_dict['start_date'].replace('Z', '+00:00'))
        policy_dict['end_date'] = start.replace(year=start.year + 1).isoformat()

    if not policy_dict.get('renewal_date'):
        policy_dict['renewal_date'] = policy_dict['end_date']

    # Generate policy ID and number
    policy_id = f"POL-{str(POLICY_ID_COUNTER).zfill(3)}"
    policy_number = f"POL-{str(POLICY_ID_COUNTER).zfill(6)}"
    POLICY_ID_COUNTER += 1

    policy_data = {
        "id": policy_id,
        "policy_number": policy_number,
        **policy_dict
    }
    POLICIES_DB[policy_id] = policy_data

    # Business Rule: Update customer's policy statistics
    update_customer_policy_stats(customer['id'], CUSTOMERS_DB, POLICIES_DB)

    logger.info(
        f"User {current_user['username']} created policy {policy_id} for customer {policy.customer_name} "
        f"(${policy_dict['coverage_amount']:.2f} coverage)"
    )
    return policy_data

@router.put("/{policy_id}", response_model=PolicyResponse)
async def update_policy(
    policy_id: str,
    policy: PolicyUpdate,
    current_user: dict = Depends(get_current_user_from_session)
):
    """
    Update an existing policy

    Authentication: Required
    Headers: X-Session-ID or X-API-Key

    Business Rules:
    - Premium must be within acceptable range if changed
    - Date range must be valid if changed
    - Customer's policy statistics are updated if coverage changes
    """
    if policy_id not in POLICIES_DB:
        raise HTTPException(status_code=404, detail="Policy not found")

    existing_policy = POLICIES_DB[policy_id]
    update_data = policy.dict(exclude_unset=True)

    # Business Rule: Validate premium range if changing
    if 'premium' in update_data:
        policy_type = update_data.get('policy_type', existing_policy['policy_type'])
        validate_premium_range(update_data['premium'], policy_type)

        # Auto-recalculate coverage if premium changes
        if 'coverage_amount' not in update_data:
            update_data['coverage_amount'] = calculate_coverage_from_premium(
                update_data['premium'],
                policy_type
            )
            logger.info(f"Auto-recalculated coverage: ${update_data['coverage_amount']:.2f}")

    # Business Rule: Validate date range if changing
    start_date = update_data.get('start_date', existing_policy.get('start_date'))
    end_date = update_data.get('end_date', existing_policy.get('end_date'))
    if start_date and end_date:
        validate_date_range(start_date, end_date)

    # Apply updates
    POLICIES_DB[policy_id].update(update_data)

    # Business Rule: Update customer statistics if coverage or status changed
    if 'coverage_amount' in update_data or 'status' in update_data:
        customer = find_customer_by_name(POLICIES_DB[policy_id]['customer_name'], CUSTOMERS_DB)
        if customer:
            update_customer_policy_stats(customer['id'], CUSTOMERS_DB, POLICIES_DB)

    logger.info(f"User {current_user['username']} updated policy: {policy_id}")
    return POLICIES_DB[policy_id]

@router.delete("/{policy_id}")
async def delete_policy(
    policy_id: str,
    current_user: dict = Depends(get_current_user_from_session)
):
    """
    Delete a policy

    Authentication: Required
    Headers: X-Session-ID or X-API-Key

    Business Rules:
    - Customer's policy statistics are updated after deletion
    """
    if policy_id not in POLICIES_DB:
        raise HTTPException(status_code=404, detail="Policy not found")

    policy = POLICIES_DB[policy_id]
    customer_name = policy['customer_name']

    # Delete the policy
    del POLICIES_DB[policy_id]

    # Business Rule: Update customer statistics
    customer = find_customer_by_name(customer_name, CUSTOMERS_DB)
    if customer:
        update_customer_policy_stats(customer['id'], CUSTOMERS_DB, POLICIES_DB)

    logger.info(f"User {current_user['username']} deleted policy: {policy_id}")
    return {"message": "Policy deleted successfully", "id": policy_id}


# ============================================================================
# ADVANCED POLICY ENDPOINTS - Renewal, Cancellation, Analytics
# ============================================================================

@router.post("/{policy_id}/renew", response_model=PolicyResponse)
async def renew_policy(
    policy_id: str,
    current_user: dict = Depends(get_current_user_from_session)
):
    """
    Renew an existing policy

    Authentication: Required
    Headers: X-Session-ID or X-API-Key

    Business Rules:
    - Policy must be eligible for renewal (active/expiring, within 60 days of expiry)
    - Customer must be active
    - New premium calculated based on claims history and loyalty
    - New policy created with 1-year term
    - Old policy status updated to 'renewed'
    """
    if policy_id not in POLICIES_DB:
        raise HTTPException(status_code=404, detail="Policy not found")

    old_policy = POLICIES_DB[policy_id]

    # Business Rule: Check if policy is eligible for renewal
    if not is_policy_eligible_for_renewal(old_policy):
        raise HTTPException(
            status_code=400,
            detail=f"Policy {old_policy['policy_number']} is not eligible for renewal. "
                   f"Status: {old_policy.get('status')}, End date: {old_policy.get('end_date')}"
        )

    # Get customer
    customer = find_customer_by_name(old_policy['customer_name'], CUSTOMERS_DB)
    if not customer:
        raise HTTPException(status_code=404, detail="Customer not found")

    # Verify customer is active
    verify_customer_active(customer)

    # Get claims history for this policy (for premium calculation)
    from backend.api.claims_api import CLAIMS_DB
    policy_claims = [
        claim for claim in CLAIMS_DB.values()
        if claim.get('policy_number') == old_policy['policy_number']
    ]

    # Business Rule: Calculate renewal premium
    new_premium = calculate_renewal_premium(
        old_policy['premium'],
        old_policy['policy_type'],
        customer,
        policy_claims
    )

    # Business Rule: Calculate new coverage
    new_coverage = calculate_coverage_from_premium(
        f"${new_premium}",
        old_policy['policy_type']
    )

    # Create new policy
    global POLICY_ID_COUNTER
    new_policy_id = f"POL-{str(POLICY_ID_COUNTER).zfill(3)}"
    new_policy_number = f"POL-{str(POLICY_ID_COUNTER).zfill(6)}"
    POLICY_ID_COUNTER += 1

    # Set new dates (1 year from now)
    from datetime import timedelta
    start_date = datetime.now()
    end_date = start_date + timedelta(days=365)

    new_policy_data = {
        "id": new_policy_id,
        "policy_number": new_policy_number,
        "customer_name": old_policy['customer_name'],
        "policy_type": old_policy['policy_type'],
        "premium": f"${new_premium:.2f}",
        "coverage_amount": new_coverage,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "status": "active",
        "renewed_from": old_policy['policy_number']
    }

    POLICIES_DB[new_policy_id] = new_policy_data

    # Update old policy status
    POLICIES_DB[policy_id]['status'] = 'renewed'
    POLICIES_DB[policy_id]['renewed_to'] = new_policy_number

    # Update customer statistics
    update_customer_policy_stats(customer['id'], CUSTOMERS_DB, POLICIES_DB)

    logger.info(
        f"User {current_user['username']} renewed policy {old_policy['policy_number']} → {new_policy_number}. "
        f"Premium: {old_policy['premium']} → ${new_premium:.2f}, "
        f"Claims history: {len(policy_claims)} claims"
    )

    return new_policy_data


@router.get("/{policy_id}/renewal-quote")
async def get_renewal_quote(
    policy_id: str,
    current_user: dict = Depends(get_optional_user)
):
    """
    Get a renewal quote for a policy without actually renewing it

    Authentication: Optional (read-only)

    Returns:
    - Eligibility status
    - Estimated new premium
    - Premium breakdown (discounts/increases)
    - Renewal recommendations
    """
    if policy_id not in POLICIES_DB:
        raise HTTPException(status_code=404, detail="Policy not found")

    policy = POLICIES_DB[policy_id]

    # Check eligibility
    is_eligible = is_policy_eligible_for_renewal(policy)

    if not is_eligible:
        return {
            "eligible": False,
            "policy_number": policy['policy_number'],
            "current_premium": policy['premium'],
            "message": "Policy is not currently eligible for renewal",
            "status": policy.get('status'),
            "end_date": policy.get('end_date')
        }

    # Get customer
    customer = find_customer_by_name(policy['customer_name'], CUSTOMERS_DB)

    # Get claims history
    from backend.api.claims_api import CLAIMS_DB
    policy_claims = [
        claim for claim in CLAIMS_DB.values()
        if claim.get('policy_number') == policy['policy_number']
    ]

    # Calculate renewal premium
    current_premium = float(policy['premium'].replace('$', '').replace(',', '').strip())
    new_premium = calculate_renewal_premium(
        policy['premium'],
        policy['policy_type'],
        customer,
        policy_claims
    )

    # Calculate savings/increase
    difference = new_premium - current_premium
    percentage_change = (difference / current_premium) * 100

    return {
        "eligible": True,
        "policy_number": policy['policy_number'],
        "policy_type": policy['policy_type'],
        "current_premium": f"${current_premium:.2f}",
        "renewal_premium": f"${new_premium:.2f}",
        "difference": f"${abs(difference):.2f}",
        "percentage_change": f"{percentage_change:+.1f}%",
        "is_increase": difference > 0,
        "claims_count": len(policy_claims),
        "customer_tenure_years": self._get_customer_tenure(customer) if customer else 0,
        "message": f"Renewal premium {'increased' if difference > 0 else 'decreased'} by ${abs(difference):.2f} ({abs(percentage_change):.1f}%)"
    }


def _get_customer_tenure(customer: Dict[str, Any]) -> float:
    """Helper function to calculate customer tenure in years"""
    join_date_str = customer.get('join_date')
    if join_date_str:
        try:
            join_date = datetime.fromisoformat(join_date_str.replace('Z', '+00:00'))
            return (datetime.now() - join_date).days / 365.25
        except ValueError:
            pass
    return 0


@router.post("/{policy_id}/cancel")
async def cancel_policy(
    policy_id: str,
    reason: str = Query(..., description="Reason for cancellation"),
    current_user: dict = Depends(get_current_user_from_session)
):
    """
    Cancel a policy

    Authentication: Required
    Headers: X-Session-ID or X-API-Key

    Business Rules:
    - Cannot cancel if active claims exist
    - Early cancellation (within 1 year) may have penalty
    - Customer statistics updated
    """
    if policy_id not in POLICIES_DB:
        raise HTTPException(status_code=404, detail="Policy not found")

    policy = POLICIES_DB[policy_id]

    # Check for active claims
    from backend.api.claims_api import CLAIMS_DB
    active_claims = [
        claim for claim in CLAIMS_DB.values()
        if claim.get('policy_number') == policy['policy_number']
        and claim.get('status', '').lower() in ['pending', 'under_review']
    ]

    # Business Rule: Check if policy can be cancelled
    cancellation_result = can_cancel_policy(
        policy,
        reason,
        has_active_claims=len(active_claims) > 0
    )

    if not cancellation_result['can_cancel']:
        raise HTTPException(
            status_code=400,
            detail=cancellation_result['message']
        )

    # Cancel the policy
    POLICIES_DB[policy_id]['status'] = 'cancelled'
    POLICIES_DB[policy_id]['cancellation_reason'] = reason
    POLICIES_DB[policy_id]['cancellation_date'] = datetime.now().isoformat()
    POLICIES_DB[policy_id]['cancellation_penalty'] = cancellation_result['penalty']

    # Update customer statistics
    customer = find_customer_by_name(policy['customer_name'], CUSTOMERS_DB)
    if customer:
        update_customer_policy_stats(customer['id'], CUSTOMERS_DB, POLICIES_DB)

    logger.info(
        f"User {current_user['username']} cancelled policy {policy['policy_number']}. "
        f"Reason: {reason}, Penalty: ${cancellation_result['penalty']:.2f}"
    )

    return {
        "message": "Policy cancelled successfully",
        "policy_id": policy_id,
        "policy_number": policy['policy_number'],
        "cancellation_penalty": f"${cancellation_result['penalty']:.2f}",
        "details": cancellation_result['message']
    }

