"""
Partner and Referral Lead Generation System

Comprehensive partner management and referral tracking system for
managing partner relationships, tracking referral leads, and
optimizing partner performance.
"""

from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import json
import asyncio
import logging
from decimal import Decimal

logger = logging.getLogger(__name__)

class PartnerType(Enum):
    INSURANCE_AGENT = "insurance_agent"
    FINANCIAL_ADVISOR = "financial_advisor"
    REAL_ESTATE_AGENT = "real_estate_agent"
    MORTGAGE_BROKER = "mortgage_broker"
    ACCOUNTANT = "accountant"
    LAWYER = "lawyer"
    BUSINESS_CONSULTANT = "business_consultant"
    TECHNOLOGY_PARTNER = "technology_partner"
    AFFILIATE = "affiliate"
    REFERRAL_NETWORK = "referral_network"

class PartnerStatus(Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    PENDING = "pending"
    SUSPENDED = "suspended"
    TERMINATED = "terminated"

class ReferralStatus(Enum):
    RECEIVED = "received"
    QUALIFIED = "qualified"
    CONTACTED = "contacted"
    CONVERTED = "converted"
    CLOSED_WON = "closed_won"
    CLOSED_LOST = "closed_lost"

class CommissionType(Enum):
    FLAT_FEE = "flat_fee"
    PERCENTAGE = "percentage"
    TIERED = "tiered"
    PERFORMANCE_BASED = "performance_based"

@dataclass
class Partner:
    """Partner/referral source information"""
    partner_id: str
    name: str
    partner_type: PartnerType
    status: PartnerStatus
    
    # Contact Information
    contact_person: str
    email: str
    phone: str
    company: Optional[str] = None
    website: Optional[str] = None
    
    # Address
    address: Dict[str, str] = field(default_factory=dict)
    
    # Partnership Details
    partnership_start_date: datetime = field(default_factory=datetime.now)
    contract_end_date: Optional[datetime] = None
    specialties: List[str] = field(default_factory=list)
    target_markets: List[str] = field(default_factory=list)
    
    # Commission Structure
    commission_type: CommissionType = CommissionType.PERCENTAGE
    commission_rate: Decimal = Decimal('0.05')  # 5% default
    commission_tiers: List[Dict[str, Any]] = field(default_factory=list)
    
    # Performance Metrics
    total_referrals: int = 0
    qualified_referrals: int = 0
    converted_referrals: int = 0
    total_revenue_generated: Decimal = Decimal('0')
    total_commissions_paid: Decimal = Decimal('0')
    
    # Quality Metrics
    average_lead_quality: float = 0.0
    conversion_rate: float = 0.0
    average_deal_size: Decimal = Decimal('0')
    
    # Relationship Management
    last_contact_date: Optional[datetime] = None
    relationship_manager: Optional[str] = None
    notes: List[str] = field(default_factory=list)
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

@dataclass
class ReferralLead:
    """Referral lead information"""
    referral_id: str
    partner_id: str
    status: ReferralStatus
    
    # Lead Information
    name: str
    email: str
    phone: Optional[str] = None
    company: Optional[str] = None
    job_title: Optional[str] = None
    
    # Referral Context
    referral_source: str  # How partner knows the lead
    referral_reason: str  # Why referring
    urgency_level: str = "medium"
    estimated_value: Optional[Decimal] = None
    
    # Product Interest
    products_interested: List[str] = field(default_factory=list)
    specific_needs: str = ""
    budget_range: Optional[str] = None
    decision_timeline: Optional[str] = None
    
    # Referral Details
    referral_date: datetime = field(default_factory=datetime.now)
    referral_method: str = "email"  # email, phone, form, etc.
    referral_notes: str = ""
    
    # Processing
    assigned_agent: Optional[str] = None
    first_contact_date: Optional[datetime] = None
    qualification_date: Optional[datetime] = None
    conversion_date: Optional[datetime] = None
    
    # Financial
    deal_value: Optional[Decimal] = None
    commission_amount: Optional[Decimal] = None
    commission_paid: bool = False
    commission_paid_date: Optional[datetime] = None
    
    # Quality Scoring
    lead_quality_score: float = 0.0
    qualification_notes: str = ""
    
    # Tracking
    utm_parameters: Dict[str, str] = field(default_factory=dict)
    referral_link: Optional[str] = None
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    raw_data: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CommissionPayment:
    """Commission payment record"""
    payment_id: str
    partner_id: str
    referral_id: str
    
    # Payment Details
    amount: Decimal
    commission_rate: Decimal
    deal_value: Decimal
    payment_date: datetime
    
    # Payment Method
    payment_method: str = "bank_transfer"
    payment_reference: Optional[str] = None
    
    # Status
    payment_status: str = "paid"
    notes: str = ""
    
    created_at: datetime = field(default_factory=datetime.now)

class PartnerReferralSystem:
    """Main partner and referral management system"""
    
    def __init__(self):
        self.partners = {}
        self.referral_leads = {}
        self.commission_payments = {}
        self.performance_analytics = PartnerAnalytics()
    
    async def register_partner(self, partner_data: Dict[str, Any]) -> Partner:
        """Register a new partner"""
        
        try:
            partner = Partner(
                partner_id=partner_data['partner_id'],
                name=partner_data['name'],
                partner_type=PartnerType(partner_data['partner_type']),
                status=PartnerStatus(partner_data.get('status', 'pending')),
                contact_person=partner_data['contact_person'],
                email=partner_data['email'],
                phone=partner_data['phone'],
                company=partner_data.get('company'),
                website=partner_data.get('website'),
                address=partner_data.get('address', {}),
                specialties=partner_data.get('specialties', []),
                target_markets=partner_data.get('target_markets', []),
                commission_type=CommissionType(partner_data.get('commission_type', 'percentage')),
                commission_rate=Decimal(str(partner_data.get('commission_rate', '0.05'))),
                commission_tiers=partner_data.get('commission_tiers', []),
                relationship_manager=partner_data.get('relationship_manager')
            )
            
            self.partners[partner.partner_id] = partner
            
            # Set up partner tracking
            await self._setup_partner_tracking(partner)
            
            # Send welcome package
            await self._send_partner_welcome_package(partner)
            
            logger.info(f"Registered new partner: {partner.name}")
            return partner
            
        except Exception as e:
            logger.error(f"Error registering partner: {e}")
            raise
    
    async def submit_referral(self, referral_data: Dict[str, Any]) -> ReferralLead:
        """Submit a new referral lead"""
        
        try:
            # Validate partner exists and is active
            partner_id = referral_data['partner_id']
            if partner_id not in self.partners:
                raise ValueError(f"Partner not found: {partner_id}")
            
            partner = self.partners[partner_id]
            if partner.status != PartnerStatus.ACTIVE:
                raise ValueError(f"Partner is not active: {partner.name}")
            
            # Create referral lead
            referral = ReferralLead(
                referral_id=f"ref_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{partner_id}",
                partner_id=partner_id,
                status=ReferralStatus.RECEIVED,
                name=referral_data['name'],
                email=referral_data['email'],
                phone=referral_data.get('phone'),
                company=referral_data.get('company'),
                job_title=referral_data.get('job_title'),
                referral_source=referral_data.get('referral_source', 'partner_network'),
                referral_reason=referral_data.get('referral_reason', ''),
                urgency_level=referral_data.get('urgency_level', 'medium'),
                estimated_value=Decimal(str(referral_data['estimated_value'])) if referral_data.get('estimated_value') else None,
                products_interested=referral_data.get('products_interested', []),
                specific_needs=referral_data.get('specific_needs', ''),
                budget_range=referral_data.get('budget_range'),
                decision_timeline=referral_data.get('decision_timeline'),
                referral_method=referral_data.get('referral_method', 'email'),
                referral_notes=referral_data.get('referral_notes', ''),
                utm_parameters=referral_data.get('utm_parameters', {}),
                referral_link=referral_data.get('referral_link'),
                raw_data=referral_data
            )
            
            # Calculate initial lead quality score
            referral.lead_quality_score = await self._calculate_referral_quality_score(referral, partner)
            
            self.referral_leads[referral.referral_id] = referral
            
            # Update partner metrics
            partner.total_referrals += 1
            
            # Auto-assign to agent
            await self._assign_referral_to_agent(referral)
            
            # Send notifications
            await self._send_referral_notifications(referral, partner)
            
            # Trigger follow-up sequence
            await self._trigger_referral_follow_up(referral)
            
            logger.info(f"Submitted referral: {referral.email} from partner {partner.name}")
            return referral
            
        except Exception as e:
            logger.error(f"Error submitting referral: {e}")
            raise
    
    async def update_referral_status(self, referral_id: str, new_status: ReferralStatus, 
                                   update_data: Dict[str, Any] = None) -> ReferralLead:
        """Update referral status and related information"""
        
        try:
            if referral_id not in self.referral_leads:
                raise ValueError(f"Referral not found: {referral_id}")
            
            referral = self.referral_leads[referral_id]
            partner = self.partners[referral.partner_id]
            old_status = referral.status
            
            # Update status
            referral.status = new_status
            referral.updated_at = datetime.now()
            
            # Update specific fields based on status
            if new_status == ReferralStatus.QUALIFIED:
                referral.qualification_date = datetime.now()
                partner.qualified_referrals += 1
                
                if update_data:
                    referral.qualification_notes = update_data.get('qualification_notes', '')
                    referral.assigned_agent = update_data.get('assigned_agent')
            
            elif new_status == ReferralStatus.CONTACTED:
                if not referral.first_contact_date:
                    referral.first_contact_date = datetime.now()
            
            elif new_status == ReferralStatus.CONVERTED:
                referral.conversion_date = datetime.now()
                partner.converted_referrals += 1
                
                if update_data:
                    referral.deal_value = Decimal(str(update_data['deal_value'])) if update_data.get('deal_value') else None
            
            elif new_status == ReferralStatus.CLOSED_WON:
                if update_data and update_data.get('deal_value'):
                    deal_value = Decimal(str(update_data['deal_value']))
                    referral.deal_value = deal_value
                    
                    # Calculate and process commission
                    commission_amount = await self._calculate_commission(partner, deal_value)
                    referral.commission_amount = commission_amount
                    
                    # Update partner metrics
                    partner.total_revenue_generated += deal_value
                    
                    # Schedule commission payment
                    await self._schedule_commission_payment(referral, partner, commission_amount)
            
            # Update partner performance metrics
            await self._update_partner_performance_metrics(partner)
            
            # Send status update notifications
            await self._send_status_update_notifications(referral, partner, old_status, new_status)
            
            logger.info(f"Updated referral {referral_id} status: {old_status.value} -> {new_status.value}")
            return referral
            
        except Exception as e:
            logger.error(f"Error updating referral status: {e}")
            raise
    
    async def get_partner_performance(self, partner_id: str, 
                                    date_range: Dict[str, datetime] = None) -> Dict[str, Any]:
        """Get comprehensive partner performance metrics"""
        
        try:
            if partner_id not in self.partners:
                raise ValueError(f"Partner not found: {partner_id}")
            
            partner = self.partners[partner_id]
            
            # Calculate date range if not provided
            if not date_range:
                end_date = datetime.now()
                start_date = end_date - timedelta(days=90)  # Last 90 days
                date_range = {"start": start_date, "end": end_date}
            
            # Get referrals in date range
            period_referrals = [
                ref for ref in self.referral_leads.values()
                if ref.partner_id == partner_id and 
                date_range["start"] <= ref.referral_date <= date_range["end"]
            ]
            
            # Calculate metrics
            total_referrals = len(period_referrals)
            qualified_referrals = len([ref for ref in period_referrals if ref.status in [ReferralStatus.QUALIFIED, ReferralStatus.CONTACTED, ReferralStatus.CONVERTED, ReferralStatus.CLOSED_WON]])
            converted_referrals = len([ref for ref in period_referrals if ref.status in [ReferralStatus.CONVERTED, ReferralStatus.CLOSED_WON]])
            won_referrals = len([ref for ref in period_referrals if ref.status == ReferralStatus.CLOSED_WON])
            
            total_revenue = sum(ref.deal_value or Decimal('0') for ref in period_referrals if ref.deal_value)
            total_commissions = sum(ref.commission_amount or Decimal('0') for ref in period_referrals if ref.commission_amount)
            
            performance = {
                "partner_info": {
                    "partner_id": partner.partner_id,
                    "name": partner.name,
                    "partner_type": partner.partner_type.value,
                    "status": partner.status.value,
                    "partnership_duration": (datetime.now() - partner.partnership_start_date).days
                },
                "period_metrics": {
                    "date_range": date_range,
                    "total_referrals": total_referrals,
                    "qualified_referrals": qualified_referrals,
                    "converted_referrals": converted_referrals,
                    "won_referrals": won_referrals,
                    "qualification_rate": qualified_referrals / max(total_referrals, 1),
                    "conversion_rate": converted_referrals / max(qualified_referrals, 1),
                    "win_rate": won_referrals / max(converted_referrals, 1)
                },
                "financial_metrics": {
                    "total_revenue_generated": float(total_revenue),
                    "total_commissions_earned": float(total_commissions),
                    "average_deal_size": float(total_revenue / max(won_referrals, 1)),
                    "revenue_per_referral": float(total_revenue / max(total_referrals, 1))
                },
                "quality_metrics": {
                    "average_lead_quality_score": sum(ref.lead_quality_score for ref in period_referrals) / max(len(period_referrals), 1),
                    "average_time_to_qualification": self._calculate_average_time_to_qualification(period_referrals),
                    "average_time_to_conversion": self._calculate_average_time_to_conversion(period_referrals)
                },
                "lifetime_metrics": {
                    "total_referrals": partner.total_referrals,
                    "qualified_referrals": partner.qualified_referrals,
                    "converted_referrals": partner.converted_referrals,
                    "total_revenue_generated": float(partner.total_revenue_generated),
                    "total_commissions_paid": float(partner.total_commissions_paid),
                    "lifetime_conversion_rate": partner.conversion_rate
                }
            }
            
            return performance
            
        except Exception as e:
            logger.error(f"Error getting partner performance: {e}")
            return {}
    
    async def get_referral_dashboard(self, date_range: Dict[str, datetime] = None) -> Dict[str, Any]:
        """Get comprehensive referral program dashboard"""
        
        try:
            if not date_range:
                end_date = datetime.now()
                start_date = end_date - timedelta(days=30)
                date_range = {"start": start_date, "end": end_date}
            
            period_referrals = [
                ref for ref in self.referral_leads.values()
                if date_range["start"] <= ref.referral_date <= date_range["end"]
            ]
            
            total_referrals = len(period_referrals)
            qualified_referrals = len([ref for ref in period_referrals if ref.status in [ReferralStatus.QUALIFIED, ReferralStatus.CONTACTED, ReferralStatus.CONVERTED, ReferralStatus.CLOSED_WON]])
            won_referrals = len([ref for ref in period_referrals if ref.status == ReferralStatus.CLOSED_WON])
            
            total_revenue = sum(ref.deal_value or Decimal('0') for ref in period_referrals if ref.deal_value)
            total_commissions = sum(ref.commission_amount or Decimal('0') for ref in period_referrals if ref.commission_amount)
            
            dashboard = {
                "summary": {
                    "total_partners": len([p for p in self.partners.values() if p.status == PartnerStatus.ACTIVE]),
                    "total_referrals": total_referrals,
                    "qualified_referrals": qualified_referrals,
                    "won_referrals": won_referrals,
                    "qualification_rate": qualified_referrals / max(total_referrals, 1),
                    "conversion_rate": won_referrals / max(qualified_referrals, 1),
                    "total_revenue": float(total_revenue),
                    "total_commissions": float(total_commissions)
                },
                "top_partners": await self._get_top_performing_partners(date_range),
                "partner_type_performance": await self._get_partner_type_performance(date_range),
                "referral_pipeline": await self._get_referral_pipeline_metrics(),
                "commission_summary": await self._get_commission_summary(date_range)
            }
            
            return dashboard
            
        except Exception as e:
            logger.error(f"Error generating referral dashboard: {e}")
            return {}

    async def _calculate_referral_quality_score(self, referral: ReferralLead, partner: Partner) -> float:
        """Calculate quality score for referral lead"""
        
        score = 0.0
        
        # Partner quality bonus
        if partner.average_lead_quality > 7.0:
            score += 20.0
        elif partner.average_lead_quality > 5.0:
            score += 10.0
        
        # Completeness bonus
        if referral.company:
            score += 15.0
        if referral.job_title:
            score += 10.0
        if referral.phone:
            score += 10.0
        if referral.specific_needs:
            score += 15.0
        
        # Urgency bonus
        urgency_scores = {"high": 20.0, "medium": 10.0, "low": 5.0}
        score += urgency_scores.get(referral.urgency_level, 5.0)
        
        # Estimated value bonus
        if referral.estimated_value:
            if referral.estimated_value > Decimal('10000'):
                score += 20.0
            elif referral.estimated_value > Decimal('5000'):
                score += 15.0
            elif referral.estimated_value > Decimal('1000'):
                score += 10.0
        
        return min(score, 100.0)

    async def _get_top_performing_partners(self, date_range: Dict[str, datetime]) -> List[Dict[str, Any]]:
        """Get top performing partners"""
        
        partner_performance = []
        for partner in self.partners.values():
            if partner.status == PartnerStatus.ACTIVE:
                performance = await self.get_partner_performance(partner.partner_id, date_range)
                partner_performance.append({
                    "partner_id": partner.partner_id,
                    "name": partner.name,
                    "referrals": performance["period_metrics"]["total_referrals"],
                    "revenue": performance["financial_metrics"]["total_revenue_generated"],
                    "conversion_rate": performance["period_metrics"]["conversion_rate"]
                })
        
        # Sort by revenue generated
        return sorted(partner_performance, key=lambda x: x["revenue"], reverse=True)[:10]

# Global partner referral system
partner_referral_system = PartnerReferralSystem()

class PartnerAnalytics:
    """Analytics engine for partner performance"""
    
    def __init__(self):
        self.performance_cache = {}
    
    async def generate_partner_insights(self, partner_id: str) -> Dict[str, Any]:
        """Generate AI-powered insights for partner performance"""
        
        return {
            "performance_trend": "improving",
            "recommendations": [
                "Focus on higher-value referrals",
                "Improve lead qualification process",
                "Increase referral frequency"
            ],
            "benchmarks": {
                "vs_peer_group": "above_average",
                "vs_top_performers": "below_average"
            }
        }
