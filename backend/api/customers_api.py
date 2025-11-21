"""
Customers API - CRUD endpoints for customer management
"""
from fastapi import APIRouter, HTTPException, Query, Depends
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
import logging
from datetime import datetime
from backend.api.auth_dependencies import get_current_user_from_session, get_optional_user
from backend.utils.validators import (
    validate_email,
    validate_phone,
    validate_non_negative_number,
    validate_string_length,
    validate_choice,
    validate_date_string,
    ValidationError
)
from backend.utils.business_rules import (
    get_customer_risk_score,
    should_offer_retention_incentive
)

router = APIRouter(prefix="/customers", tags=["Customers"])
logger = logging.getLogger(__name__)

# In-memory storage (replace with database later)
CUSTOMERS_DB = {}
CUSTOMER_ID_COUNTER = 1

class CustomerCreate(BaseModel):
    name: str = Field(..., min_length=2, max_length=100, description="Customer full name")
    email: str = Field(..., description="Customer email address")
    phone: str = Field(..., description="Customer phone number")
    status: str = Field(default="active", description="Customer status")
    policies_count: int = Field(default=0, ge=0, description="Number of policies")
    total_value: float = Field(default=0.0, ge=0.0, description="Total policy value")
    join_date: Optional[str] = Field(None, description="Date customer joined (ISO format)")
    last_active: Optional[str] = Field(None, description="Last activity date (ISO format)")
    reason: Optional[str] = Field(None, max_length=500, description="Reason for status change")

    @validator('name')
    def validate_name(cls, v):
        """Validate customer name"""
        try:
            return validate_string_length(v, min_length=2, max_length=100, field_name="Name")
        except ValidationError as e:
            raise ValueError(str(e))

    @validator('email')
    def validate_email_field(cls, v):
        """Validate email format"""
        try:
            return validate_email(v)
        except ValidationError as e:
            raise ValueError(str(e))

    @validator('phone')
    def validate_phone_field(cls, v):
        """Validate phone number format"""
        try:
            return validate_phone(v)
        except ValidationError as e:
            raise ValueError(str(e))

    @validator('status')
    def validate_status_field(cls, v):
        """Validate status is allowed value"""
        allowed_statuses = ['active', 'inactive', 'pending']
        try:
            return validate_choice(v, allowed_statuses, field_name="Status")
        except ValidationError as e:
            raise ValueError(str(e))

    @validator('policies_count')
    def validate_policies_count_field(cls, v):
        """Validate policies count is non-negative"""
        try:
            return int(validate_non_negative_number(v, field_name="Policies count"))
        except ValidationError as e:
            raise ValueError(str(e))

    @validator('total_value')
    def validate_total_value_field(cls, v):
        """Validate total value is non-negative"""
        try:
            return validate_non_negative_number(v, field_name="Total value")
        except ValidationError as e:
            raise ValueError(str(e))

    @validator('join_date', 'last_active')
    def validate_date_fields(cls, v):
        """Validate date format"""
        if v is None:
            return v
        try:
            return validate_date_string(v)
        except ValidationError as e:
            raise ValueError(str(e))

    @validator('reason')
    def validate_reason_field(cls, v):
        """Validate reason length"""
        if v is None:
            return v
        try:
            return validate_string_length(v, min_length=1, max_length=500, field_name="Reason")
        except ValidationError as e:
            raise ValueError(str(e))

class CustomerUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=2, max_length=100)
    email: Optional[str] = None
    phone: Optional[str] = None
    status: Optional[str] = None
    policies_count: Optional[int] = Field(None, ge=0)
    total_value: Optional[float] = Field(None, ge=0.0)
    join_date: Optional[str] = None
    last_active: Optional[str] = None
    reason: Optional[str] = Field(None, max_length=500)

    @validator('name')
    def validate_name(cls, v):
        """Validate customer name"""
        if v is None:
            return v
        try:
            return validate_string_length(v, min_length=2, max_length=100, field_name="Name")
        except ValidationError as e:
            raise ValueError(str(e))

    @validator('email')
    def validate_email_field(cls, v):
        """Validate email format"""
        if v is None:
            return v
        try:
            return validate_email(v)
        except ValidationError as e:
            raise ValueError(str(e))

    @validator('phone')
    def validate_phone_field(cls, v):
        """Validate phone number format"""
        if v is None:
            return v
        try:
            return validate_phone(v)
        except ValidationError as e:
            raise ValueError(str(e))

    @validator('status')
    def validate_status_field(cls, v):
        """Validate status is allowed value"""
        if v is None:
            return v
        allowed_statuses = ['active', 'inactive', 'pending']
        try:
            return validate_choice(v, allowed_statuses, field_name="Status")
        except ValidationError as e:
            raise ValueError(str(e))

    @validator('join_date', 'last_active')
    def validate_date_fields(cls, v):
        """Validate date format"""
        if v is None:
            return v
        try:
            return validate_date_string(v)
        except ValidationError as e:
            raise ValueError(str(e))

class CustomerResponse(BaseModel):
    id: str
    name: str
    email: str
    phone: str
    status: str
    policies_count: int
    total_value: float
    join_date: Optional[str]
    last_active: Optional[str]
    reason: Optional[str]

class PaginatedCustomersResponse(BaseModel):
    customers: List[CustomerResponse]
    total: int
    page: int
    page_size: int
    total_pages: int

@router.get("", response_model=PaginatedCustomersResponse)
async def get_customers(
    status: Optional[str] = None,
    page: int = Query(1, ge=1, description="Page number (starts at 1)"),
    page_size: int = Query(10, ge=1, le=100, description="Number of items per page"),
    current_user: Optional[dict] = Depends(get_optional_user)
):
    """
    Get all customers with pagination, optionally filtered by status

    Authentication: Optional (works with or without authentication)
    """
    if current_user:
        logger.info(f"User {current_user['username']} is viewing customers")

    customers = list(CUSTOMERS_DB.values())

    # Filter by status if provided
    if status:
        customers = [c for c in customers if c["status"] == status]

    # Calculate pagination
    total = len(customers)
    total_pages = (total + page_size - 1) // page_size if total > 0 else 1
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size

    # Get paginated results
    paginated_customers = customers[start_idx:end_idx]

    return {
        "customers": paginated_customers,
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": total_pages
    }

@router.get("/{customer_id}", response_model=CustomerResponse)
async def get_customer(
    customer_id: str,
    current_user: Optional[dict] = Depends(get_optional_user)
):
    """
    Get a specific customer by ID

    Authentication: Optional
    """
    if customer_id not in CUSTOMERS_DB:
        raise HTTPException(status_code=404, detail="Customer not found")
    return CUSTOMERS_DB[customer_id]

@router.post("", response_model=CustomerResponse)
async def create_customer(
    customer: CustomerCreate,
    current_user: dict = Depends(get_current_user_from_session)
):
    """
    Create a new customer

    Authentication: Required
    Headers: X-Session-ID or X-API-Key

    Validations:
    - Email must be unique
    - Phone must be unique
    - All field validations from CustomerCreate model
    """
    global CUSTOMER_ID_COUNTER

    # Business Validation: Check for duplicate email
    for existing_customer in CUSTOMERS_DB.values():
        if existing_customer['email'].lower() == customer.email.lower():
            raise HTTPException(
                status_code=400,
                detail=f"Customer with email '{customer.email}' already exists"
            )

    # Business Validation: Check for duplicate phone
    for existing_customer in CUSTOMERS_DB.values():
        if existing_customer['phone'] == customer.phone:
            raise HTTPException(
                status_code=400,
                detail=f"Customer with phone '{customer.phone}' already exists"
            )

    # Business Validation: If status is inactive, reason must be provided
    if customer.status == 'inactive' and not customer.reason:
        raise HTTPException(
            status_code=400,
            detail="Reason is required when creating an inactive customer"
        )

    # Generate customer ID
    customer_id = f"CUST-{str(CUSTOMER_ID_COUNTER).zfill(3)}"
    CUSTOMER_ID_COUNTER += 1

    # Set default dates if not provided
    customer_dict = customer.dict()
    if not customer_dict.get('join_date'):
        customer_dict['join_date'] = datetime.now().isoformat()
    if not customer_dict.get('last_active'):
        customer_dict['last_active'] = datetime.now().isoformat()

    customer_data = {
        "id": customer_id,
        **customer_dict
    }
    CUSTOMERS_DB[customer_id] = customer_data
    logger.info(f"User {current_user['username']} created customer: {customer_id} ({customer.email})")
    return customer_data

@router.put("/{customer_id}", response_model=CustomerResponse)
async def update_customer(
    customer_id: str,
    customer: CustomerUpdate,
    current_user: dict = Depends(get_current_user_from_session)
):
    """
    Update an existing customer

    Authentication: Required
    Headers: X-Session-ID or X-API-Key

    Validations:
    - Email must be unique (if changing)
    - Phone must be unique (if changing)
    - Cannot change status to inactive without reason
    - All field validations from CustomerUpdate model
    """
    if customer_id not in CUSTOMERS_DB:
        raise HTTPException(status_code=404, detail="Customer not found")

    existing_customer = CUSTOMERS_DB[customer_id]
    update_data = customer.dict(exclude_unset=True)

    # Business Validation: Check for duplicate email (if changing)
    if 'email' in update_data and update_data['email'] != existing_customer['email']:
        for cust_id, existing in CUSTOMERS_DB.items():
            if cust_id != customer_id and existing['email'].lower() == update_data['email'].lower():
                raise HTTPException(
                    status_code=400,
                    detail=f"Customer with email '{update_data['email']}' already exists"
                )

    # Business Validation: Check for duplicate phone (if changing)
    if 'phone' in update_data and update_data['phone'] != existing_customer['phone']:
        for cust_id, existing in CUSTOMERS_DB.items():
            if cust_id != customer_id and existing['phone'] == update_data['phone']:
                raise HTTPException(
                    status_code=400,
                    detail=f"Customer with phone '{update_data['phone']}' already exists"
                )

    # Business Validation: If changing status to inactive, reason must be provided
    if 'status' in update_data and update_data['status'] == 'inactive':
        # Check if reason is provided in update OR already exists
        if not update_data.get('reason') and not existing_customer.get('reason'):
            raise HTTPException(
                status_code=400,
                detail="Reason is required when changing status to inactive"
            )

    # Business Validation: Update last_active when status changes
    if 'status' in update_data and update_data['status'] != existing_customer['status']:
        update_data['last_active'] = datetime.now().isoformat()
        logger.info(f"Customer {customer_id} status changed from {existing_customer['status']} to {update_data['status']}")

    # Apply updates
    CUSTOMERS_DB[customer_id].update(update_data)
    logger.info(f"User {current_user['username']} updated customer: {customer_id}")
    return CUSTOMERS_DB[customer_id]

@router.delete("/{customer_id}")
async def delete_customer(
    customer_id: str,
    current_user: dict = Depends(get_current_user_from_session)
):
    """
    Delete a customer

    Authentication: Required
    Headers: X-Session-ID or X-API-Key
    """
    if customer_id not in CUSTOMERS_DB:
        raise HTTPException(status_code=404, detail="Customer not found")

    del CUSTOMERS_DB[customer_id]
    logger.info(f"User {current_user['username']} deleted customer: {customer_id}")
    return {"message": "Customer deleted successfully", "id": customer_id}


# ============================================================================
# CUSTOMER LIFECYCLE MANAGEMENT ENDPOINTS
# ============================================================================

@router.get("/{customer_id}/risk-score")
async def get_customer_risk_assessment(
    customer_id: str,
    current_user: dict = Depends(get_optional_user)
):
    """
    Get customer risk assessment

    Authentication: Optional (read-only)

    Returns:
    - Risk score (0-100)
    - Risk level (very_low, low, medium, high)
    - Risk factors
    - Recommendations
    """
    if customer_id not in CUSTOMERS_DB:
        raise HTTPException(status_code=404, detail="Customer not found")

    customer = CUSTOMERS_DB[customer_id]

    # Get customer's policies
    from backend.api.policies_api import POLICIES_DB
    customer_policies = [
        policy for policy in POLICIES_DB.values()
        if policy.get('customer_name', '').lower() == customer['name'].lower()
    ]

    # Get customer's claims
    from backend.api.claims_api import CLAIMS_DB
    customer_claims = [
        claim for claim in CLAIMS_DB.values()
        if claim.get('customer_name', '').lower() == customer['name'].lower()
    ]

    # Business Rule: Calculate risk score
    risk_assessment = get_customer_risk_score(customer, customer_policies, customer_claims)

    # Generate recommendations based on risk level
    recommendations = []
    if risk_assessment['risk_level'] == 'high':
        recommendations.append("Consider premium adjustment on renewal")
        recommendations.append("Review claims history for patterns")
        recommendations.append("Implement stricter approval process for new claims")
    elif risk_assessment['risk_level'] == 'medium':
        recommendations.append("Monitor claims activity")
        recommendations.append("Standard renewal process")
    else:
        recommendations.append("Offer loyalty discounts")
        recommendations.append("Consider cross-selling opportunities")
        recommendations.append("Fast-track claim approvals")

    return {
        "customer_id": customer_id,
        "customer_name": customer['name'],
        "risk_score": risk_assessment['risk_score'],
        "risk_level": risk_assessment['risk_level'],
        "risk_factors": risk_assessment['factors'],
        "recommendations": recommendations,
        "policies_count": len(customer_policies),
        "claims_count": len(customer_claims),
        "total_coverage": sum(float(p.get('coverage_amount', 0)) for p in customer_policies)
    }


@router.get("/{customer_id}/retention-analysis")
async def get_retention_analysis(
    customer_id: str,
    current_user: dict = Depends(get_optional_user)
):
    """
    Get customer retention analysis and incentive recommendations

    Authentication: Optional (read-only)

    Returns:
    - Retention risk level
    - Recommended incentives
    - Customer value metrics
    - Engagement recommendations
    """
    if customer_id not in CUSTOMERS_DB:
        raise HTTPException(status_code=404, detail="Customer not found")

    customer = CUSTOMERS_DB[customer_id]

    # Get customer's policies
    from backend.api.policies_api import POLICIES_DB
    customer_policies = [
        policy for policy in POLICIES_DB.values()
        if policy.get('customer_name', '').lower() == customer['name'].lower()
    ]

    # Business Rule: Check if should offer retention incentive
    incentive_result = should_offer_retention_incentive(customer, customer_policies)

    # Calculate customer lifetime value
    active_policies = [p for p in customer_policies if p.get('status', '').lower() == 'active']
    total_premium = sum(
        float(p.get('premium', '0').replace('$', '').replace(',', '').strip())
        for p in active_policies
    )

    # Calculate customer tenure
    join_date_str = customer.get('join_date')
    tenure_years = 0
    if join_date_str:
        try:
            join_date = datetime.fromisoformat(join_date_str.replace('Z', '+00:00'))
            tenure_years = (datetime.now() - join_date).days / 365.25
        except ValueError:
            pass

    # Estimate lifetime value (annual premium × tenure × retention multiplier)
    estimated_ltv = total_premium * max(tenure_years, 1) * 0.8

    # Determine retention risk
    if len(active_policies) == 0:
        retention_risk = 'critical'
    elif len(active_policies) == 1:
        retention_risk = 'high'
    elif total_premium < 500:
        retention_risk = 'medium'
    else:
        retention_risk = 'low'

    return {
        "customer_id": customer_id,
        "customer_name": customer['name'],
        "retention_risk": retention_risk,
        "offer_incentive": incentive_result['offer_incentive'],
        "incentive_type": incentive_result['incentive_type'],
        "incentive_value": incentive_result['incentive_value'],
        "incentive_reason": incentive_result['reason'],
        "customer_metrics": {
            "tenure_years": round(tenure_years, 1),
            "active_policies": len(active_policies),
            "total_annual_premium": f"${total_premium:,.2f}",
            "estimated_lifetime_value": f"${estimated_ltv:,.2f}",
            "total_coverage": f"${sum(float(p.get('coverage_amount', 0)) for p in active_policies):,.2f}"
        },
        "engagement_recommendations": self._get_engagement_recommendations(retention_risk, tenure_years)
    }


def _get_engagement_recommendations(retention_risk: str, tenure_years: float) -> List[str]:
    """Helper function to generate engagement recommendations"""
    recommendations = []

    if retention_risk == 'critical':
        recommendations.append("URGENT: Contact customer immediately")
        recommendations.append("Offer win-back incentive (15% discount)")
        recommendations.append("Schedule personal consultation call")
    elif retention_risk == 'high':
        recommendations.append("Reach out with policy review offer")
        recommendations.append("Highlight benefits and coverage options")
        recommendations.append("Offer multi-policy discount")
    elif retention_risk == 'medium':
        recommendations.append("Send quarterly newsletter")
        recommendations.append("Offer policy review")
    else:
        recommendations.append("Send thank you communication")
        recommendations.append("Offer referral incentive")
        recommendations.append("Provide VIP customer benefits")

    if tenure_years >= 5:
        recommendations.append(f"Recognize {int(tenure_years)}-year loyalty milestone")

    return recommendations


@router.post("/{customer_id}/send-retention-offer")
async def send_retention_offer(
    customer_id: str,
    offer_type: str = Query(..., description="Type of retention offer"),
    current_user: dict = Depends(get_current_user_from_session)
):
    """
    Send retention offer to customer

    Authentication: Required
    Headers: X-Session-ID or X-API-Key

    Business Rules:
    - Creates communication record
    - Logs offer in customer record
    - Sets follow-up reminder
    """
    if customer_id not in CUSTOMERS_DB:
        raise HTTPException(status_code=404, detail="Customer not found")

    customer = CUSTOMERS_DB[customer_id]

    # Log the retention offer in customer record
    if 'retention_offers' not in CUSTOMERS_DB[customer_id]:
        CUSTOMERS_DB[customer_id]['retention_offers'] = []

    offer_record = {
        "offer_type": offer_type,
        "sent_date": datetime.now().isoformat(),
        "sent_by": current_user['username'],
        "status": "sent"
    }

    CUSTOMERS_DB[customer_id]['retention_offers'].append(offer_record)

    # Create communication record
    from backend.api.communications_api import COMMUNICATIONS_DB, COMMUNICATION_ID_COUNTER

    comm_id = f"COMM-{str(COMMUNICATION_ID_COUNTER).zfill(3)}"

    communication_data = {
        "id": comm_id,
        "customer_name": customer['name'],
        "comm_type": "email",
        "channel": "email",
        "subject": f"Special Retention Offer - {offer_type}",
        "content": f"Retention offer sent to valued customer",
        "status": "sent",
        "priority": "high",
        "comm_date": datetime.now().isoformat()
    }

    COMMUNICATIONS_DB[comm_id] = communication_data

    logger.info(
        f"User {current_user['username']} sent retention offer to customer {customer['name']}. "
        f"Offer type: {offer_type}"
    )

    return {
        "message": "Retention offer sent successfully",
        "customer_id": customer_id,
        "customer_name": customer['name'],
        "offer_type": offer_type,
        "communication_id": comm_id,
        "sent_date": offer_record['sent_date'],
        "follow_up_date": (datetime.now() + timedelta(days=7)).isoformat()
    }


from datetime import timedelta

