"""
Business Rules Utilities - Reusable business logic functions
"""
from typing import Optional, Dict, Any, List
from fastapi import HTTPException
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class BusinessRuleError(Exception):
    """Custom exception for business rule violations"""
    def __init__(self, message: str, status_code: int = 400):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)


# Customer Business Rules
def verify_customer_exists(customer_id: str, customers_db: Dict[str, Any]) -> Dict[str, Any]:
    """
    Verify that a customer exists in the database

    Args:
        customer_id: Customer ID to verify
        customers_db: Customer database dictionary

    Returns:
        Customer data if found

    Raises:
        HTTPException: If customer not found
    """
    if customer_id not in customers_db:
        raise HTTPException(status_code=404, detail=f"Customer '{customer_id}' not found")
    return customers_db[customer_id]


def verify_customer_active(customer: Dict[str, Any]) -> None:
    """
    Verify that a customer is active

    Args:
        customer: Customer data dictionary

    Raises:
        HTTPException: If customer is not active
    """
    if customer.get('status', '').lower() != 'active':
        raise HTTPException(
            status_code=400,
            detail=f"Customer '{customer.get('name')}' is not active (status: {customer.get('status')})"
        )


def find_customer_by_email(email: str, customers_db: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Find a customer by email address

    Args:
        email: Email address to search for
        customers_db: Customer database dictionary

    Returns:
        Customer data if found, None otherwise
    """
    for customer in customers_db.values():
        if customer.get('email', '').lower() == email.lower():
            return customer
    return None


def find_customer_by_name(name: str, customers_db: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Find a customer by name (case-insensitive)

    Args:
        name: Customer name to search for
        customers_db: Customer database dictionary

    Returns:
        Customer data if found, None otherwise
    """
    for customer in customers_db.values():
        if customer.get('name', '').lower() == name.lower():
            return customer
    return None


# Policy Business Rules
def verify_policy_exists(policy_id: str, policies_db: Dict[str, Any]) -> Dict[str, Any]:
    """
    Verify that a policy exists in the database

    Args:
        policy_id: Policy ID to verify
        policies_db: Policy database dictionary

    Returns:
        Policy data if found

    Raises:
        HTTPException: If policy not found
    """
    if policy_id not in policies_db:
        raise HTTPException(status_code=404, detail=f"Policy '{policy_id}' not found")
    return policies_db[policy_id]


def verify_policy_active(policy: Dict[str, Any]) -> None:
    """
    Verify that a policy is active

    Args:
        policy: Policy data dictionary

    Raises:
        HTTPException: If policy is not active
    """
    if policy.get('status', '').lower() != 'active':
        raise HTTPException(
            status_code=400,
            detail=f"Policy '{policy.get('policy_number')}' is not active (status: {policy.get('status')})"
        )


def find_policy_by_number(policy_number: str, policies_db: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Find a policy by policy number

    Args:
        policy_number: Policy number to search for
        policies_db: Policy database dictionary

    Returns:
        Policy data if found, None otherwise
    """
    for policy in policies_db.values():
        if policy.get('policy_number', '').lower() == policy_number.lower():
            return policy
    return None


def calculate_coverage_from_premium(premium_str: str, policy_type: str) -> float:
    """
    Calculate coverage amount based on premium and policy type

    Args:
        premium_str: Premium amount as string (e.g., "$500")
        policy_type: Type of policy (auto, home, life, health)

    Returns:
        Calculated coverage amount
    """
    # Extract numeric value from premium string
    premium = float(premium_str.replace('$', '').replace(',', '').strip())

    # Coverage multipliers by policy type
    multipliers = {
        'auto': 100,      # $500 premium = $50,000 coverage
        'home': 200,      # $500 premium = $100,000 coverage
        'life': 500,      # $500 premium = $250,000 coverage
        'health': 50,     # $500 premium = $25,000 coverage
        'business': 300,  # $500 premium = $150,000 coverage
        'travel': 20      # $500 premium = $10,000 coverage
    }

    multiplier = multipliers.get(policy_type.lower(), 100)
    return premium * multiplier


def validate_premium_range(premium_str: str, policy_type: str) -> None:
    """
    Validate that premium is within acceptable range for policy type

    Args:
        premium_str: Premium amount as string (e.g., "$500")
        policy_type: Type of policy

    Raises:
        HTTPException: If premium is outside acceptable range
    """
    premium = float(premium_str.replace('$', '').replace(',', '').strip())

    # Premium ranges by policy type (min, max)
    ranges = {
        'auto': (100, 5000),
        'home': (200, 10000),
        'life': (500, 50000),
        'health': (150, 20000),
        'business': (500, 100000),
        'travel': (50, 2000)
    }

    min_premium, max_premium = ranges.get(policy_type.lower(), (100, 100000))

    if premium < min_premium:
        raise HTTPException(
            status_code=400,
            detail=f"Premium ${premium:.2f} is below minimum ${min_premium:.2f} for {policy_type} policy"
        )

    if premium > max_premium:
        raise HTTPException(
            status_code=400,
            detail=f"Premium ${premium:.2f} exceeds maximum ${max_premium:.2f} for {policy_type} policy"
        )


def update_customer_policy_stats(customer_id: str, customers_db: Dict[str, Any],
                                 policies_db: Dict[str, Any]) -> None:
    """
    Update customer's policy count and total value based on their policies

    Args:
        customer_id: Customer ID to update
        customers_db: Customer database dictionary
        policies_db: Policy database dictionary
    """
    if customer_id not in customers_db:
        return

    # Count active policies for this customer
    customer_name = customers_db[customer_id]['name']
    active_policies = [
        p for p in policies_db.values()
        if p.get('customer_name', '').lower() == customer_name.lower()
        and p.get('status', '').lower() == 'active'
    ]

    # Calculate total value
    total_value = 0.0
    for policy in active_policies:
        if policy.get('coverage_amount'):
            total_value += float(policy['coverage_amount'])

    # Update customer record
    customers_db[customer_id]['policies_count'] = len(active_policies)
    customers_db[customer_id]['total_value'] = total_value

    logger.info(f"Updated customer {customer_id}: {len(active_policies)} policies, ${total_value:.2f} total value")


# Claim Business Rules
def validate_claim_amount_vs_coverage(claim_amount_str: str, policy: Dict[str, Any]) -> None:
    """
    Validate that claim amount doesn't exceed policy coverage

    Args:
        claim_amount_str: Claim amount as string (e.g., "$2500")
        policy: Policy data dictionary

    Raises:
        HTTPException: If claim amount exceeds coverage
    """
    claim_amount = float(claim_amount_str.replace('$', '').replace(',', '').strip())
    coverage = policy.get('coverage_amount', 0)

    if coverage and claim_amount > coverage:
        raise HTTPException(
            status_code=400,
            detail=f"Claim amount ${claim_amount:.2f} exceeds policy coverage ${coverage:.2f}"
        )


def should_auto_approve_claim(claim_amount_str: str, claim_type: str) -> bool:
    """
    Determine if a claim should be auto-approved based on amount and type

    Args:
        claim_amount_str: Claim amount as string
        claim_type: Type of claim

    Returns:
        True if claim should be auto-approved, False otherwise
    """
    claim_amount = float(claim_amount_str.replace('$', '').replace(',', '').strip())

    # Auto-approve thresholds by claim type
    thresholds = {
        'accident': 1000,
        'theft': 500,
        'damage': 1500,
        'medical': 2000,
        'liability': 500,
        'other': 500
    }

    threshold = thresholds.get(claim_type.lower(), 500)
    return claim_amount <= threshold


# Communication Business Rules
def can_send_marketing_to_customer(customer: Dict[str, Any]) -> bool:
    """
    Check if marketing communications can be sent to customer

    Args:
        customer: Customer data dictionary

    Returns:
        True if marketing can be sent, False otherwise
    """
    # Don't send marketing to inactive customers
    if customer.get('status', '').lower() != 'active':
        return False

    # Don't send marketing to customers with no policies
    if customer.get('policies_count', 0) == 0:
        return False

    return True


def get_communication_priority(comm_type: str, subject: str) -> str:
    """
    Auto-determine communication priority based on type and subject

    Args:
        comm_type: Type of communication
        subject: Communication subject

    Returns:
        Priority level (high, medium, low)
    """
    subject_lower = subject.lower()

    # High priority keywords
    high_priority_keywords = ['urgent', 'claim', 'emergency', 'important', 'immediate']
    if any(keyword in subject_lower for keyword in high_priority_keywords):
        return 'high'

    # High priority communication types
    if comm_type.lower() in ['call', 'sms']:
        return 'high'

    # Medium priority for emails
    if comm_type.lower() == 'email':
        return 'medium'

    return 'low'


# Date validation business rules
def validate_date_range(start_date: Optional[str], end_date: Optional[str]) -> None:
    """
    Validate that date range is logical (end date after start date)

    Args:
        start_date: Start date in ISO format
        end_date: End date in ISO format

    Raises:
        HTTPException: If date range is invalid
    """
    if not start_date or not end_date:
        return

    try:
        start = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
        end = datetime.fromisoformat(end_date.replace('Z', '+00:00'))

        if end <= start:
            raise HTTPException(
                status_code=400,
                detail=f"End date ({end_date}) must be after start date ({start_date})"
            )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid date format: {str(e)}")


# ============================================================================
# ADVANCED BUSINESS RULES
# ============================================================================

# Policy Renewal Business Rules
def is_policy_eligible_for_renewal(policy: Dict[str, Any]) -> bool:
    """
    Check if a policy is eligible for renewal

    Eligibility Criteria:
    - Policy must be active or expiring soon (within 60 days)
    - Policy must not be cancelled
    - Customer must be active
    - No outstanding claims in 'rejected' status

    Args:
        policy: Policy data dictionary

    Returns:
        True if eligible for renewal, False otherwise
    """
    # Check policy status
    status = policy.get('status', '').lower()
    if status not in ['active', 'expiring']:
        logger.info(f"Policy {policy.get('policy_number')} not eligible: status is {status}")
        return False

    # Check if policy is expiring soon (within 60 days)
    end_date_str = policy.get('end_date')
    if end_date_str:
        try:
            end_date = datetime.fromisoformat(end_date_str.replace('Z', '+00:00'))
            days_until_expiry = (end_date - datetime.now()).days

            if days_until_expiry > 60:
                logger.info(f"Policy {policy.get('policy_number')} not eligible: {days_until_expiry} days until expiry")
                return False
        except ValueError:
            logger.warning(f"Invalid end_date format for policy {policy.get('policy_number')}")
            return False

    return True


def calculate_renewal_premium(
    current_premium_str: str,
    policy_type: str,
    customer: Optional[Dict[str, Any]] = None,
    claims_history: Optional[List[Dict[str, Any]]] = None
) -> float:
    """
    Calculate renewal premium based on current premium, policy type, and claims history

    Premium Adjustment Rules:
    - Base: Current premium
    - No claims in past year: -5% discount
    - 1-2 claims: No change
    - 3+ claims: +10% increase
    - Customer loyalty (5+ years): -3% discount
    - Policy type specific adjustments

    Args:
        current_premium_str: Current premium as string (e.g., "$500")
        policy_type: Type of policy
        customer: Customer data (optional, for loyalty discount)
        claims_history: List of claims for this policy (optional)

    Returns:
        New premium amount as float
    """
    # Parse current premium
    current_premium = float(current_premium_str.replace('$', '').replace(',', '').strip())

    # Start with current premium
    new_premium = current_premium

    # Claims history adjustment
    if claims_history:
        num_claims = len(claims_history)
        if num_claims == 0:
            new_premium *= 0.95  # 5% discount for no claims
            logger.info(f"Applied 5% no-claims discount")
        elif num_claims >= 3:
            new_premium *= 1.10  # 10% increase for multiple claims
            logger.info(f"Applied 10% increase for {num_claims} claims")

    # Customer loyalty discount
    if customer:
        join_date_str = customer.get('join_date')
        if join_date_str:
            try:
                join_date = datetime.fromisoformat(join_date_str.replace('Z', '+00:00'))
                years_with_company = (datetime.now() - join_date).days / 365.25

                if years_with_company >= 5:
                    new_premium *= 0.97  # 3% loyalty discount
                    logger.info(f"Applied 3% loyalty discount ({years_with_company:.1f} years)")
            except ValueError:
                pass

    # Policy type specific adjustments (market rates)
    type_adjustments = {
        'auto': 1.02,      # 2% increase (market trend)
        'home': 1.03,      # 3% increase (inflation)
        'life': 1.01,      # 1% increase (stable)
        'health': 1.05,    # 5% increase (healthcare costs)
        'business': 1.04,  # 4% increase (risk increase)
        'travel': 1.00     # No change (competitive market)
    }

    adjustment = type_adjustments.get(policy_type.lower(), 1.02)
    new_premium *= adjustment

    logger.info(f"Renewal premium calculated: ${current_premium:.2f} → ${new_premium:.2f}")

    return round(new_premium, 2)


def should_auto_renew_policy(
    policy: Dict[str, Any],
    customer: Dict[str, Any]
) -> bool:
    """
    Determine if a policy should be auto-renewed

    Auto-Renewal Criteria:
    - Customer has auto-renew enabled (if field exists)
    - Customer is active
    - Policy is eligible for renewal
    - Customer has valid payment method (assumed if active)
    - No fraud flags

    Args:
        policy: Policy data dictionary
        customer: Customer data dictionary

    Returns:
        True if should auto-renew, False otherwise
    """
    # Check customer status
    if customer.get('status', '').lower() != 'active':
        return False

    # Check if customer has auto-renew preference (default to True if not specified)
    auto_renew_enabled = customer.get('auto_renew', True)
    if not auto_renew_enabled:
        logger.info(f"Auto-renew disabled for customer {customer.get('name')}")
        return False

    # Check policy eligibility
    if not is_policy_eligible_for_renewal(policy):
        return False

    # Check for fraud flags
    if customer.get('fraud_flag', False):
        logger.warning(f"Auto-renew blocked: fraud flag for customer {customer.get('name')}")
        return False

    return True


# Claim Escalation Business Rules
def should_escalate_claim(claim: Dict[str, Any], policy: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Determine if a claim should be escalated and why

    Escalation Criteria:
    - High amount (>50% of coverage)
    - Pending for >30 days
    - Multiple claims from same customer in short period
    - Suspicious patterns

    Args:
        claim: Claim data dictionary
        policy: Policy data (optional, for coverage comparison)

    Returns:
        Dictionary with 'should_escalate' (bool) and 'reasons' (list)
    """
    reasons = []

    # Check claim amount vs coverage
    if policy:
        claim_amount = float(claim.get('amount', '0').replace('$', '').replace(',', '').strip())
        coverage = float(policy.get('coverage_amount', 0))

        if coverage > 0:
            percentage = (claim_amount / coverage) * 100
            if percentage > 50:
                reasons.append(f"High amount: {percentage:.1f}% of coverage (${claim_amount:,.2f} / ${coverage:,.2f})")

    # Check claim age
    claim_date_str = claim.get('claim_date')
    if claim_date_str:
        try:
            claim_date = datetime.fromisoformat(claim_date_str.replace('Z', '+00:00'))
            days_pending = (datetime.now() - claim_date).days

            if days_pending > 30 and claim.get('status', '').lower() == 'pending':
                reasons.append(f"Pending for {days_pending} days (>30 day threshold)")
        except ValueError:
            pass

    # Check for suspicious amounts (round numbers might indicate fraud)
    claim_amount_str = claim.get('amount', '0')
    claim_amount = float(claim_amount_str.replace('$', '').replace(',', '').strip())
    if claim_amount >= 10000 and claim_amount % 1000 == 0:
        reasons.append(f"Suspicious round amount: ${claim_amount:,.2f}")

    return {
        'should_escalate': len(reasons) > 0,
        'reasons': reasons,
        'escalation_priority': 'high' if len(reasons) >= 2 else 'medium'
    }


def get_claim_priority(claim: Dict[str, Any]) -> str:
    """
    Determine claim processing priority

    Priority Levels:
    - Critical: Medical claims, high amounts (>$50k), urgent keywords
    - High: Amounts >$10k, liability claims
    - Medium: Standard claims
    - Low: Small claims (<$1k)

    Args:
        claim: Claim data dictionary

    Returns:
        Priority level: 'critical', 'high', 'medium', or 'low'
    """
    claim_type = claim.get('claim_type', '').lower()
    claim_amount = float(claim.get('amount', '0').replace('$', '').replace(',', '').strip())
    description = claim.get('description', '').lower()

    # Critical priority
    if claim_type == 'medical':
        return 'critical'

    if claim_amount > 50000:
        return 'critical'

    urgent_keywords = ['urgent', 'emergency', 'critical', 'life-threatening', 'severe']
    if any(keyword in description for keyword in urgent_keywords):
        return 'critical'

    # High priority
    if claim_amount > 10000:
        return 'high'

    if claim_type in ['liability', 'accident']:
        return 'high'

    # Low priority
    if claim_amount < 1000:
        return 'low'

    # Default: medium priority
    return 'medium'


# Policy Cancellation Business Rules
def can_cancel_policy(
    policy: Dict[str, Any],
    reason: str,
    has_active_claims: bool = False
) -> Dict[str, Any]:
    """
    Determine if a policy can be cancelled and calculate any penalties

    Cancellation Rules:
    - Cannot cancel if active claims exist
    - Early cancellation (within first year) may have penalty
    - Customer-initiated vs company-initiated have different rules

    Args:
        policy: Policy data dictionary
        reason: Cancellation reason
        has_active_claims: Whether policy has active claims

    Returns:
        Dictionary with 'can_cancel' (bool), 'penalty' (float), and 'message' (str)
    """
    # Check for active claims
    if has_active_claims:
        return {
            'can_cancel': False,
            'penalty': 0,
            'message': 'Cannot cancel policy with active claims. Please resolve all claims first.'
        }

    # Check policy age for early cancellation penalty
    start_date_str = policy.get('start_date')
    penalty = 0

    if start_date_str:
        try:
            start_date = datetime.fromisoformat(start_date_str.replace('Z', '+00:00'))
            months_active = (datetime.now() - start_date).days / 30.44

            # Early cancellation penalty (within first 12 months)
            if months_active < 12:
                premium = float(policy.get('premium', '0').replace('$', '').replace(',', '').strip())
                penalty = premium * 0.25  # 25% of annual premium

                return {
                    'can_cancel': True,
                    'penalty': penalty,
                    'message': f'Early cancellation penalty applies: ${penalty:.2f} (25% of premium). Policy active for {months_active:.1f} months.'
                }
        except ValueError:
            pass

    # No penalty for cancellations after first year
    return {
        'can_cancel': True,
        'penalty': 0,
        'message': 'Policy can be cancelled without penalty.'
    }


# Customer Lifecycle Management
def get_customer_risk_score(customer: Dict[str, Any], policies: List[Dict[str, Any]], claims: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate customer risk score for retention and pricing

    Risk Factors:
    - Claims frequency and amount
    - Payment history (if available)
    - Policy count and types
    - Customer tenure

    Args:
        customer: Customer data dictionary
        policies: List of customer's policies
        claims: List of customer's claims

    Returns:
        Dictionary with 'risk_score' (0-100), 'risk_level', and 'factors'
    """
    risk_score = 50  # Start at neutral
    factors = []

    # Claims frequency risk
    if len(claims) > 0:
        claims_per_policy = len(claims) / max(len(policies), 1)
        if claims_per_policy > 2:
            risk_score += 20
            factors.append(f"High claims frequency: {claims_per_policy:.1f} claims per policy")
        elif claims_per_policy > 1:
            risk_score += 10
            factors.append(f"Moderate claims frequency: {claims_per_policy:.1f} claims per policy")
        else:
            risk_score -= 10
            factors.append(f"Low claims frequency: {claims_per_policy:.1f} claims per policy")
    else:
        risk_score -= 15
        factors.append("No claims history (good)")

    # Customer tenure (longer tenure = lower risk)
    join_date_str = customer.get('join_date')
    if join_date_str:
        try:
            join_date = datetime.fromisoformat(join_date_str.replace('Z', '+00:00'))
            years = (datetime.now() - join_date).days / 365.25

            if years >= 5:
                risk_score -= 15
                factors.append(f"Long tenure: {years:.1f} years (excellent)")
            elif years >= 2:
                risk_score -= 5
                factors.append(f"Good tenure: {years:.1f} years")
        except ValueError:
            pass

    # Policy count (more policies = more committed customer = lower risk)
    num_policies = len(policies)
    if num_policies >= 3:
        risk_score -= 10
        factors.append(f"Multiple policies: {num_policies} (loyal customer)")
    elif num_policies == 0:
        risk_score += 20
        factors.append("No active policies (high churn risk)")

    # Ensure score is within 0-100
    risk_score = max(0, min(100, risk_score))

    # Determine risk level
    if risk_score >= 70:
        risk_level = 'high'
    elif risk_score >= 50:
        risk_level = 'medium'
    elif risk_score >= 30:
        risk_level = 'low'
    else:
        risk_level = 'very_low'

    return {
        'risk_score': risk_score,
        'risk_level': risk_level,
        'factors': factors
    }


def should_offer_retention_incentive(customer: Dict[str, Any], policies: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Determine if customer should be offered retention incentive

    Incentive Criteria:
    - Customer has no active policies but had policies before
    - Customer has only 1 policy expiring soon
    - High-value customer (total coverage > $100k)
    - Long-term customer (5+ years) with decreasing policy count

    Args:
        customer: Customer data dictionary
        policies: List of customer's policies

    Returns:
        Dictionary with 'offer_incentive' (bool), 'incentive_type', and 'reason'
    """
    active_policies = [p for p in policies if p.get('status', '').lower() == 'active']
    total_value = sum(float(p.get('coverage_amount', 0)) for p in active_policies)

    # No active policies but was a customer
    if len(active_policies) == 0 and customer.get('policies_count', 0) > 0:
        return {
            'offer_incentive': True,
            'incentive_type': 'win_back',
            'incentive_value': '15% discount on new policy',
            'reason': 'Former customer with no active policies - win-back campaign'
        }

    # High-value customer
    if total_value > 100000:
        return {
            'offer_incentive': True,
            'incentive_type': 'vip_loyalty',
            'incentive_value': '10% discount on renewal + priority service',
            'reason': f'High-value customer: ${total_value:,.2f} total coverage'
        }

    # Long-term customer
    join_date_str = customer.get('join_date')
    if join_date_str:
        try:
            join_date = datetime.fromisoformat(join_date_str.replace('Z', '+00:00'))
            years = (datetime.now() - join_date).days / 365.25

            if years >= 5 and len(active_policies) <= 1:
                return {
                    'offer_incentive': True,
                    'incentive_type': 'loyalty_retention',
                    'incentive_value': '8% discount on additional policy',
                    'reason': f'Long-term customer ({years:.1f} years) with low policy count'
                }
        except ValueError:
            pass

    return {
        'offer_incentive': False,
        'incentive_type': None,
        'incentive_value': None,
        'reason': 'No retention incentive criteria met'
    }

