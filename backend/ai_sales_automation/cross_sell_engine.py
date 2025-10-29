"""
Cross-sell/Upsell Engine

AI-powered system for identifying cross-sell and upsell opportunities across
product lines, customer segmentation, and revenue optimization.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import json
import redis

logger = logging.getLogger(__name__)

class OpportunityType(Enum):
    CROSS_SELL = "cross_sell"
    UPSELL = "upsell"
    RENEWAL = "renewal"
    EXPANSION = "expansion"
    RETENTION = "retention"

class ProductCategory(Enum):
    AUTO_INSURANCE = "auto_insurance"
    HOME_INSURANCE = "home_insurance"
    LIFE_INSURANCE = "life_insurance"
    HEALTH_INSURANCE = "health_insurance"
    BUSINESS_INSURANCE = "business_insurance"
    UMBRELLA_INSURANCE = "umbrella_insurance"
    DISABILITY_INSURANCE = "disability_insurance"
    TRAVEL_INSURANCE = "travel_insurance"

class CustomerSegment(Enum):
    HIGH_VALUE = "high_value"
    GROWING_FAMILY = "growing_family"
    YOUNG_PROFESSIONAL = "young_professional"
    ESTABLISHED_FAMILY = "established_family"
    EMPTY_NESTER = "empty_nester"
    RETIREE = "retiree"
    SMALL_BUSINESS = "small_business"
    PRICE_SENSITIVE = "price_sensitive"

@dataclass
class ProductRecommendation:
    """Product recommendation with reasoning"""
    product_id: str
    product_name: str
    product_category: ProductCategory
    opportunity_type: OpportunityType
    confidence_score: float
    revenue_potential: float
    reasoning: str
    key_benefits: List[str] = field(default_factory=list)
    pricing_strategy: str = ""
    timing_recommendation: str = ""
    objection_handling: Dict[str, str] = field(default_factory=dict)

@dataclass
class CustomerProfile:
    """Comprehensive customer profile for opportunity identification"""
    customer_id: str
    current_products: List[str] = field(default_factory=list)
    customer_segment: CustomerSegment = CustomerSegment.YOUNG_PROFESSIONAL
    lifetime_value: float = 0.0
    annual_premium: float = 0.0
    tenure_months: int = 0
    claim_history: List[Dict[str, Any]] = field(default_factory=list)
    life_events: List[str] = field(default_factory=list)
    engagement_score: float = 0.0
    satisfaction_score: float = 0.0
    risk_profile: str = "medium"
    financial_capacity: str = "medium"
    communication_preferences: List[str] = field(default_factory=list)

@dataclass
class OpportunityScore:
    """Opportunity scoring result"""
    customer_id: str
    opportunity_type: OpportunityType
    product_category: ProductCategory
    score: float
    revenue_potential: float
    probability: float
    timing_score: float
    competitive_risk: float
    effort_required: str

class OpportunityIdentifier:
    """AI-powered opportunity identification system"""
    
    def __init__(self):
        # ML models
        self.cross_sell_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.upsell_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.revenue_predictor = GradientBoostingRegressor(n_estimators=100, random_state=42)
        self.timing_model = RandomForestClassifier(n_estimators=50, random_state=42)
        
        # Product affinity matrix
        self.product_affinity = {}
        
        # Life event triggers
        self.life_event_triggers = {}
        
        # Initialize system
        self._initialize_product_affinity()
        self._initialize_life_event_triggers()
        
        logger.info("Opportunity Identifier initialized")
    
    def _initialize_product_affinity(self):
        """Initialize product affinity matrix"""
        
        self.product_affinity = {
            ProductCategory.AUTO_INSURANCE: {
                ProductCategory.HOME_INSURANCE: 0.85,
                ProductCategory.UMBRELLA_INSURANCE: 0.75,
                ProductCategory.LIFE_INSURANCE: 0.65,
                ProductCategory.DISABILITY_INSURANCE: 0.45
            },
            ProductCategory.HOME_INSURANCE: {
                ProductCategory.AUTO_INSURANCE: 0.90,
                ProductCategory.UMBRELLA_INSURANCE: 0.80,
                ProductCategory.LIFE_INSURANCE: 0.70,
                ProductCategory.BUSINESS_INSURANCE: 0.35
            },
            ProductCategory.LIFE_INSURANCE: {
                ProductCategory.DISABILITY_INSURANCE: 0.85,
                ProductCategory.AUTO_INSURANCE: 0.60,
                ProductCategory.HOME_INSURANCE: 0.65,
                ProductCategory.UMBRELLA_INSURANCE: 0.55
            },
            ProductCategory.BUSINESS_INSURANCE: {
                ProductCategory.UMBRELLA_INSURANCE: 0.70,
                ProductCategory.DISABILITY_INSURANCE: 0.60,
                ProductCategory.AUTO_INSURANCE: 0.40
            }
        }
    
    def _initialize_life_event_triggers(self):
        """Initialize life event to product opportunity mapping"""
        
        self.life_event_triggers = {
            "marriage": [
                (ProductCategory.LIFE_INSURANCE, 0.9),
                (ProductCategory.HOME_INSURANCE, 0.7),
                (ProductCategory.UMBRELLA_INSURANCE, 0.5)
            ],
            "new_baby": [
                (ProductCategory.LIFE_INSURANCE, 0.95),
                (ProductCategory.DISABILITY_INSURANCE, 0.8),
                (ProductCategory.UMBRELLA_INSURANCE, 0.6)
            ],
            "home_purchase": [
                (ProductCategory.HOME_INSURANCE, 1.0),
                (ProductCategory.UMBRELLA_INSURANCE, 0.8),
                (ProductCategory.AUTO_INSURANCE, 0.3)
            ],
            "new_job": [
                (ProductCategory.DISABILITY_INSURANCE, 0.7),
                (ProductCategory.LIFE_INSURANCE, 0.6),
                (ProductCategory.AUTO_INSURANCE, 0.4)
            ],
            "business_start": [
                (ProductCategory.BUSINESS_INSURANCE, 1.0),
                (ProductCategory.UMBRELLA_INSURANCE, 0.8),
                (ProductCategory.DISABILITY_INSURANCE, 0.7)
            ],
            "retirement": [
                (ProductCategory.HEALTH_INSURANCE, 0.8),
                (ProductCategory.TRAVEL_INSURANCE, 0.6),
                (ProductCategory.LIFE_INSURANCE, 0.4)
            ]
        }
    
    async def identify_opportunities(self, customer_profile: CustomerProfile) -> List[OpportunityScore]:
        """Identify all opportunities for a customer"""
        
        opportunities = []
        
        # Cross-sell opportunities
        cross_sell_ops = await self._identify_cross_sell_opportunities(customer_profile)
        opportunities.extend(cross_sell_ops)
        
        # Upsell opportunities
        upsell_ops = await self._identify_upsell_opportunities(customer_profile)
        opportunities.extend(upsell_ops)
        
        # Life event triggered opportunities
        life_event_ops = await self._identify_life_event_opportunities(customer_profile)
        opportunities.extend(life_event_ops)
        
        # Renewal opportunities
        renewal_ops = await self._identify_renewal_opportunities(customer_profile)
        opportunities.extend(renewal_ops)
        
        # Sort by score and return top opportunities
        opportunities.sort(key=lambda x: x.score, reverse=True)
        
        return opportunities[:10]  # Top 10 opportunities
    
    async def _identify_cross_sell_opportunities(self, customer_profile: CustomerProfile) -> List[OpportunityScore]:
        """Identify cross-sell opportunities"""
        
        opportunities = []
        current_categories = [ProductCategory(p) for p in customer_profile.current_products 
                            if p in [cat.value for cat in ProductCategory]]
        
        for current_category in current_categories:
            if current_category in self.product_affinity:
                for target_category, affinity in self.product_affinity[current_category].items():
                    if target_category.value not in customer_profile.current_products:
                        
                        # Calculate opportunity score
                        base_score = affinity * 100
                        
                        # Adjust for customer factors
                        segment_multiplier = await self._get_segment_multiplier(
                            customer_profile.customer_segment, target_category
                        )
                        
                        engagement_multiplier = min(1.5, customer_profile.engagement_score / 100 + 0.5)
                        tenure_multiplier = min(1.3, customer_profile.tenure_months / 24 + 0.7)
                        
                        final_score = base_score * segment_multiplier * engagement_multiplier * tenure_multiplier
                        
                        # Calculate revenue potential
                        revenue_potential = await self._estimate_revenue_potential(
                            customer_profile, target_category, OpportunityType.CROSS_SELL
                        )
                        
                        opportunity = OpportunityScore(
                            customer_id=customer_profile.customer_id,
                            opportunity_type=OpportunityType.CROSS_SELL,
                            product_category=target_category,
                            score=final_score,
                            revenue_potential=revenue_potential,
                            probability=min(0.95, final_score / 100),
                            timing_score=await self._calculate_timing_score(customer_profile, target_category),
                            competitive_risk=await self._assess_competitive_risk(customer_profile, target_category),
                            effort_required=await self._assess_effort_required(customer_profile, target_category)
                        )
                        
                        opportunities.append(opportunity)
        
        return opportunities
    
    async def _identify_upsell_opportunities(self, customer_profile: CustomerProfile) -> List[OpportunityScore]:
        """Identify upsell opportunities within existing products"""
        
        opportunities = []
        
        for product in customer_profile.current_products:
            product_category = ProductCategory(product)
            
            # Check for coverage increase opportunities
            if await self._has_upsell_potential(customer_profile, product_category):
                
                # Calculate upsell score
                base_score = 70  # Base upsell score
                
                # Adjust for customer factors
                satisfaction_multiplier = customer_profile.satisfaction_score / 100 + 0.5
                financial_multiplier = await self._get_financial_capacity_multiplier(customer_profile)
                claim_multiplier = await self._get_claim_history_multiplier(customer_profile)
                
                final_score = base_score * satisfaction_multiplier * financial_multiplier * claim_multiplier
                
                # Calculate revenue potential
                revenue_potential = await self._estimate_revenue_potential(
                    customer_profile, product_category, OpportunityType.UPSELL
                )
                
                opportunity = OpportunityScore(
                    customer_id=customer_profile.customer_id,
                    opportunity_type=OpportunityType.UPSELL,
                    product_category=product_category,
                    score=final_score,
                    revenue_potential=revenue_potential,
                    probability=min(0.85, final_score / 100),
                    timing_score=await self._calculate_timing_score(customer_profile, product_category),
                    competitive_risk=0.2,  # Lower risk for existing products
                    effort_required="low"
                )
                
                opportunities.append(opportunity)
        
        return opportunities
    
    async def _identify_life_event_opportunities(self, customer_profile: CustomerProfile) -> List[OpportunityScore]:
        """Identify opportunities triggered by life events"""
        
        opportunities = []
        
        for life_event in customer_profile.life_events:
            if life_event in self.life_event_triggers:
                for product_category, base_probability in self.life_event_triggers[life_event]:
                    
                    if product_category.value not in customer_profile.current_products:
                        
                        # Calculate life event opportunity score
                        base_score = base_probability * 100
                        
                        # Adjust for timing (recent life events score higher)
                        timing_multiplier = 1.0  # Assume recent for now
                        
                        final_score = base_score * timing_multiplier
                        
                        # Calculate revenue potential
                        revenue_potential = await self._estimate_revenue_potential(
                            customer_profile, product_category, OpportunityType.CROSS_SELL
                        )
                        
                        opportunity = OpportunityScore(
                            customer_id=customer_profile.customer_id,
                            opportunity_type=OpportunityType.CROSS_SELL,
                            product_category=product_category,
                            score=final_score,
                            revenue_potential=revenue_potential,
                            probability=base_probability,
                            timing_score=90,  # High timing score for life events
                            competitive_risk=await self._assess_competitive_risk(customer_profile, product_category),
                            effort_required="medium"
                        )
                        
                        opportunities.append(opportunity)
        
        return opportunities

class CustomerSegmentation:
    """Customer segmentation for targeted opportunities"""
    
    def __init__(self):
        self.segmentation_model = KMeans(n_clusters=8, random_state=42)
        self.segment_profiles = {}
        
        self._initialize_segment_profiles()
    
    def _initialize_segment_profiles(self):
        """Initialize customer segment profiles"""
        
        self.segment_profiles = {
            CustomerSegment.HIGH_VALUE: {
                "characteristics": ["high_income", "multiple_policies", "low_claims"],
                "preferred_products": [ProductCategory.UMBRELLA_INSURANCE, ProductCategory.BUSINESS_INSURANCE],
                "communication_style": "consultative",
                "price_sensitivity": "low"
            },
            CustomerSegment.GROWING_FAMILY: {
                "characteristics": ["young_children", "recent_home_purchase", "increasing_income"],
                "preferred_products": [ProductCategory.LIFE_INSURANCE, ProductCategory.DISABILITY_INSURANCE],
                "communication_style": "educational",
                "price_sensitivity": "medium"
            },
            CustomerSegment.YOUNG_PROFESSIONAL: {
                "characteristics": ["single", "renting", "career_focused"],
                "preferred_products": [ProductCategory.AUTO_INSURANCE, ProductCategory.DISABILITY_INSURANCE],
                "communication_style": "digital_first",
                "price_sensitivity": "high"
            },
            CustomerSegment.SMALL_BUSINESS: {
                "characteristics": ["business_owner", "commercial_needs", "liability_concerns"],
                "preferred_products": [ProductCategory.BUSINESS_INSURANCE, ProductCategory.UMBRELLA_INSURANCE],
                "communication_style": "relationship_based",
                "price_sensitivity": "medium"
            }
        }
    
    async def segment_customer(self, customer_data: Dict[str, Any]) -> CustomerSegment:
        """Segment customer based on profile data"""
        
        # Simple rule-based segmentation (would use ML in production)
        age = customer_data.get('age', 30)
        income = customer_data.get('annual_income', 50000)
        has_children = customer_data.get('has_children', False)
        owns_business = customer_data.get('owns_business', False)
        owns_home = customer_data.get('owns_home', False)
        
        if owns_business:
            return CustomerSegment.SMALL_BUSINESS
        elif income > 150000 and len(customer_data.get('current_products', [])) > 2:
            return CustomerSegment.HIGH_VALUE
        elif has_children and age < 45:
            return CustomerSegment.GROWING_FAMILY
        elif age < 35 and not owns_home:
            return CustomerSegment.YOUNG_PROFESSIONAL
        elif age > 55 and len(customer_data.get('current_products', [])) > 1:
            return CustomerSegment.ESTABLISHED_FAMILY
        else:
            return CustomerSegment.YOUNG_PROFESSIONAL

class RevenueOptimizer:
    """Revenue optimization for cross-sell/upsell strategies"""
    
    def __init__(self):
        self.pricing_models = {}
        self.bundle_strategies = {}
        
        self._initialize_pricing_strategies()
        self._initialize_bundle_strategies()
    
    def _initialize_pricing_strategies(self):
        """Initialize pricing strategies by segment"""
        
        self.pricing_models = {
            CustomerSegment.HIGH_VALUE: {
                "strategy": "value_based",
                "discount_range": (0, 5),
                "focus": "premium_features"
            },
            CustomerSegment.PRICE_SENSITIVE: {
                "strategy": "competitive",
                "discount_range": (10, 20),
                "focus": "cost_savings"
            },
            CustomerSegment.GROWING_FAMILY: {
                "strategy": "bundle_focused",
                "discount_range": (5, 15),
                "focus": "family_protection"
            }
        }
    
    def _initialize_bundle_strategies(self):
        """Initialize product bundle strategies"""
        
        self.bundle_strategies = {
            "family_protection": {
                "products": [ProductCategory.AUTO_INSURANCE, ProductCategory.HOME_INSURANCE, ProductCategory.LIFE_INSURANCE],
                "discount": 15,
                "target_segments": [CustomerSegment.GROWING_FAMILY, CustomerSegment.ESTABLISHED_FAMILY]
            },
            "comprehensive_coverage": {
                "products": [ProductCategory.AUTO_INSURANCE, ProductCategory.HOME_INSURANCE, ProductCategory.UMBRELLA_INSURANCE],
                "discount": 12,
                "target_segments": [CustomerSegment.HIGH_VALUE, CustomerSegment.ESTABLISHED_FAMILY]
            },
            "business_protection": {
                "products": [ProductCategory.BUSINESS_INSURANCE, ProductCategory.UMBRELLA_INSURANCE, ProductCategory.DISABILITY_INSURANCE],
                "discount": 10,
                "target_segments": [CustomerSegment.SMALL_BUSINESS]
            }
        }
    
    async def optimize_pricing(self, customer_profile: CustomerProfile, 
                             recommendations: List[ProductRecommendation]) -> List[ProductRecommendation]:
        """Optimize pricing for product recommendations"""
        
        segment = customer_profile.customer_segment
        pricing_strategy = self.pricing_models.get(segment, self.pricing_models[CustomerSegment.YOUNG_PROFESSIONAL])
        
        optimized_recommendations = []
        
        for rec in recommendations:
            optimized_rec = rec
            
            # Apply segment-specific pricing
            if pricing_strategy["strategy"] == "competitive":
                optimized_rec.pricing_strategy = f"Save up to {pricing_strategy['discount_range'][1]}% compared to competitors"
            elif pricing_strategy["strategy"] == "value_based":
                optimized_rec.pricing_strategy = f"Premium coverage with exclusive benefits worth ${rec.revenue_potential * 0.2:.0f}"
            elif pricing_strategy["strategy"] == "bundle_focused":
                optimized_rec.pricing_strategy = f"Bundle discount: Save {pricing_strategy['discount_range'][1]}% when combined with existing policies"
            
            optimized_recommendations.append(optimized_rec)
        
        return optimized_recommendations

class CrossSellUpsellEngine:
    """Main cross-sell/upsell engine"""
    
    def __init__(self, redis_client=None):
        self.redis_client = redis_client or redis.Redis(decode_responses=True)
        
        # Components
        self.opportunity_identifier = OpportunityIdentifier()
        self.customer_segmentation = CustomerSegmentation()
        self.revenue_optimizer = RevenueOptimizer()
        
        # Product catalog
        self.product_catalog = {}
        
        # Initialize system
        self._initialize_product_catalog()
        
        logger.info("Cross-sell/Upsell Engine initialized")
    
    def _initialize_product_catalog(self):
        """Initialize product catalog with details"""
        
        self.product_catalog = {
            ProductCategory.AUTO_INSURANCE: {
                "name": "Auto Insurance",
                "base_premium": 1200,
                "key_benefits": ["Collision coverage", "Comprehensive coverage", "Liability protection"],
                "upsell_options": ["Rental car coverage", "Gap insurance", "Roadside assistance"]
            },
            ProductCategory.HOME_INSURANCE: {
                "name": "Home Insurance",
                "base_premium": 1500,
                "key_benefits": ["Dwelling protection", "Personal property coverage", "Liability coverage"],
                "upsell_options": ["Flood insurance", "Earthquake coverage", "Identity theft protection"]
            },
            ProductCategory.LIFE_INSURANCE: {
                "name": "Life Insurance",
                "base_premium": 800,
                "key_benefits": ["Death benefit", "Cash value growth", "Tax advantages"],
                "upsell_options": ["Accelerated death benefit", "Waiver of premium", "Additional coverage"]
            },
            ProductCategory.UMBRELLA_INSURANCE: {
                "name": "Umbrella Insurance",
                "base_premium": 400,
                "key_benefits": ["Extended liability coverage", "Legal defense costs", "Worldwide coverage"],
                "upsell_options": ["Higher coverage limits", "Additional covered activities"]
            }
        }
    
    async def generate_recommendations(self, customer_id: str, 
                                     customer_data: Dict[str, Any] = None) -> List[ProductRecommendation]:
        """Generate comprehensive cross-sell/upsell recommendations"""
        
        try:
            # Create customer profile
            customer_profile = await self._create_customer_profile(customer_id, customer_data)
            
            # Identify opportunities
            opportunities = await self.opportunity_identifier.identify_opportunities(customer_profile)
            
            # Convert opportunities to recommendations
            recommendations = []
            
            for opportunity in opportunities:
                recommendation = await self._create_product_recommendation(opportunity, customer_profile)
                recommendations.append(recommendation)
            
            # Optimize pricing and messaging
            optimized_recommendations = await self.revenue_optimizer.optimize_pricing(
                customer_profile, recommendations
            )
            
            # Store recommendations
            await self._store_recommendations(customer_id, optimized_recommendations)
            
            logger.info(f"Generated {len(optimized_recommendations)} recommendations for customer {customer_id}")
            
            return optimized_recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            raise
    
    async def _create_customer_profile(self, customer_id: str, 
                                     customer_data: Dict[str, Any] = None) -> CustomerProfile:
        """Create comprehensive customer profile"""
        
        if not customer_data:
            customer_data = await self._get_customer_data(customer_id)
        
        # Segment customer
        segment = await self.customer_segmentation.segment_customer(customer_data)
        
        profile = CustomerProfile(
            customer_id=customer_id,
            current_products=customer_data.get('current_products', []),
            customer_segment=segment,
            lifetime_value=customer_data.get('lifetime_value', 0),
            annual_premium=customer_data.get('annual_premium', 0),
            tenure_months=customer_data.get('tenure_months', 0),
            claim_history=customer_data.get('claim_history', []),
            life_events=customer_data.get('life_events', []),
            engagement_score=customer_data.get('engagement_score', 50),
            satisfaction_score=customer_data.get('satisfaction_score', 75),
            risk_profile=customer_data.get('risk_profile', 'medium'),
            financial_capacity=customer_data.get('financial_capacity', 'medium')
        )
        
        return profile
    
    async def _create_product_recommendation(self, opportunity: OpportunityScore, 
                                           customer_profile: CustomerProfile) -> ProductRecommendation:
        """Create detailed product recommendation from opportunity"""
        
        product_info = self.product_catalog.get(opportunity.product_category, {})
        
        # Generate reasoning
        reasoning = await self._generate_recommendation_reasoning(opportunity, customer_profile)
        
        # Generate timing recommendation
        timing = await self._generate_timing_recommendation(opportunity, customer_profile)
        
        recommendation = ProductRecommendation(
            product_id=f"{opportunity.product_category.value}_{customer_profile.customer_id}",
            product_name=product_info.get('name', opportunity.product_category.value),
            product_category=opportunity.product_category,
            opportunity_type=opportunity.opportunity_type,
            confidence_score=opportunity.score,
            revenue_potential=opportunity.revenue_potential,
            reasoning=reasoning,
            key_benefits=product_info.get('key_benefits', []),
            timing_recommendation=timing
        )
        
        return recommendation

# Global cross-sell/upsell engine instance
cross_sell_engine = CrossSellUpsellEngine()