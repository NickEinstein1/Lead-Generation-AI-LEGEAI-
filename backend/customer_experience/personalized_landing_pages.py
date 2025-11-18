"""
Personalized Landing Pages Engine

Dynamic content generation and optimization based on lead profiles,
journey stage, and behavioral data.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone
from enum import Enum
import json
import logging

logger = logging.getLogger(__name__)

class ContentType(Enum):
    HEADLINE = "headline"
    SUBHEADLINE = "subheadline"
    HERO_IMAGE = "hero_image"
    VALUE_PROPOSITION = "value_proposition"
    TESTIMONIAL = "testimonial"
    CTA_BUTTON = "cta_button"
    FORM_FIELDS = "form_fields"
    SOCIAL_PROOF = "social_proof"
    PRICING = "pricing"
    FAQ = "faq"

class PersonalizationRule(Enum):
    DEMOGRAPHIC = "demographic"
    BEHAVIORAL = "behavioral"
    JOURNEY_STAGE = "journey_stage"
    TRAFFIC_SOURCE = "traffic_source"
    DEVICE_TYPE = "device_type"
    GEOGRAPHIC = "geographic"
    TIME_BASED = "time_based"

@dataclass
class ContentVariant:
    """Content variant for A/B testing and personalization"""
    variant_id: str
    content_type: ContentType
    content: str
    target_audience: Dict[str, Any]
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    conversion_rate: float = 0.0
    engagement_score: float = 0.0
    confidence_level: float = 0.0

@dataclass
class LandingPageTemplate:
    """Landing page template with personalization rules"""
    template_id: str
    template_name: str
    industry: str
    target_audience: Dict[str, Any]
    content_variants: Dict[ContentType, List[ContentVariant]] = field(default_factory=dict)
    personalization_rules: List[PersonalizationRule] = field(default_factory=list)
    conversion_goals: List[str] = field(default_factory=list)
    performance_metrics: Dict[str, float] = field(default_factory=dict)

@dataclass
class PersonalizedPage:
    """Generated personalized landing page"""
    page_id: str
    lead_id: str
    template_id: str
    generated_content: Dict[ContentType, str]
    personalization_applied: List[str]
    predicted_conversion_rate: float
    generation_timestamp: datetime
    performance_tracking: Dict[str, Any] = field(default_factory=dict)

class PersonalizedLandingPagesEngine:
    """Engine for creating and optimizing personalized landing pages"""
    
    def __init__(self):
        self.templates = {}
        self.content_library = {}
        self.personalization_rules = {}
        self.ab_test_results = {}
        
        self._initialize_templates()
        self._initialize_content_library()
        self._initialize_personalization_rules()
        
        logger.info("Personalized Landing Pages Engine initialized")
    
    def _initialize_templates(self):
        """Initialize landing page templates"""
        
        self.templates = {
            "insurance_quote": LandingPageTemplate(
                template_id="insurance_quote",
                template_name="Insurance Quote Landing Page",
                industry="insurance",
                target_audience={"age_range": "25-65", "income_level": "middle_to_high"},
                personalization_rules=[
                    PersonalizationRule.DEMOGRAPHIC,
                    PersonalizationRule.JOURNEY_STAGE,
                    PersonalizationRule.TRAFFIC_SOURCE
                ],
                conversion_goals=["quote_request", "phone_call", "email_signup"]
            ),
            "life_insurance": LandingPageTemplate(
                template_id="life_insurance",
                template_name="Life Insurance Landing Page",
                industry="insurance",
                target_audience={"age_range": "30-55", "family_status": "married_with_children"},
                personalization_rules=[
                    PersonalizationRule.DEMOGRAPHIC,
                    PersonalizationRule.BEHAVIORAL,
                    PersonalizationRule.GEOGRAPHIC
                ],
                conversion_goals=["consultation_request", "calculator_use", "brochure_download"]
            )
        }
    
    def _initialize_content_library(self):
        """Initialize content variants library"""
        
        self.content_library = {
            ContentType.HEADLINE: [
                ContentVariant(
                    variant_id="headline_urgency",
                    content_type=ContentType.HEADLINE,
                    content="Get Your {product_type} Quote in Under 2 Minutes",
                    target_audience={"journey_stage": "intent", "traffic_source": "paid_search"},
                    conversion_rate=0.18
                ),
                ContentVariant(
                    variant_id="headline_savings",
                    content_type=ContentType.HEADLINE,
                    content="Save Up to ${savings_amount} on {product_type} Insurance",
                    target_audience={"journey_stage": "consideration", "price_sensitive": True},
                    conversion_rate=0.22
                ),
                ContentVariant(
                    variant_id="headline_trust",
                    content_type=ContentType.HEADLINE,
                    content="Trusted by Over 100,000 Families for {product_type} Protection",
                    target_audience={"age_range": "45-65", "trust_focused": True},
                    conversion_rate=0.16
                )
            ],
            ContentType.VALUE_PROPOSITION: [
                ContentVariant(
                    variant_id="value_comprehensive",
                    content_type=ContentType.VALUE_PROPOSITION,
                    content="Comprehensive coverage with personalized service and competitive rates",
                    target_audience={"journey_stage": "evaluation", "detail_oriented": True},
                    conversion_rate=0.15
                ),
                ContentVariant(
                    variant_id="value_simple",
                    content_type=ContentType.VALUE_PROPOSITION,
                    content="Simple, affordable protection for what matters most",
                    target_audience={"age_range": "25-35", "simplicity_focused": True},
                    conversion_rate=0.19
                )
            ],
            ContentType.CTA_BUTTON: [
                ContentVariant(
                    variant_id="cta_urgent",
                    content_type=ContentType.CTA_BUTTON,
                    content="Get My Quote Now",
                    target_audience={"journey_stage": "intent", "urgency_responsive": True},
                    conversion_rate=0.24
                ),
                ContentVariant(
                    variant_id="cta_free",
                    content_type=ContentType.CTA_BUTTON,
                    content="Get Free Quote",
                    target_audience={"price_sensitive": True},
                    conversion_rate=0.21
                ),
                ContentVariant(
                    variant_id="cta_personalized",
                    content_type=ContentType.CTA_BUTTON,
                    content="See My Personalized Rates",
                    target_audience={"journey_stage": "consideration"},
                    conversion_rate=0.18
                )
            ]
        }
    
    def _initialize_personalization_rules(self):
        """Initialize personalization rules"""
        
        self.personalization_rules = {
            PersonalizationRule.DEMOGRAPHIC: {
                "age_young": {
                    "condition": "age < 35",
                    "content_preferences": ["simple", "digital_first", "mobile_optimized"],
                    "messaging_tone": "casual",
                    "value_focus": "affordability"
                },
                "age_middle": {
                    "condition": "35 <= age <= 55",
                    "content_preferences": ["comprehensive", "family_focused", "security"],
                    "messaging_tone": "professional",
                    "value_focus": "protection"
                },
                "age_senior": {
                    "condition": "age > 55",
                    "content_preferences": ["trust_focused", "detailed", "traditional"],
                    "messaging_tone": "respectful",
                    "value_focus": "reliability"
                }
            },
            PersonalizationRule.JOURNEY_STAGE: {
                "awareness": {
                    "content_focus": "education",
                    "cta_type": "soft",
                    "information_depth": "high"
                },
                "consideration": {
                    "content_focus": "comparison",
                    "cta_type": "medium",
                    "information_depth": "medium"
                },
                "intent": {
                    "content_focus": "conversion",
                    "cta_type": "strong",
                    "information_depth": "low"
                }
            },
            PersonalizationRule.TRAFFIC_SOURCE: {
                "organic_search": {
                    "content_style": "informational",
                    "trust_signals": "high",
                    "form_length": "medium"
                },
                "paid_search": {
                    "content_style": "conversion_focused",
                    "trust_signals": "medium",
                    "form_length": "short"
                },
                "social_media": {
                    "content_style": "engaging",
                    "trust_signals": "social_proof",
                    "form_length": "short"
                }
            }
        }
    
    async def generate_personalized_page(self, lead_data: Dict[str, Any], 
                                       template_id: str,
                                       traffic_data: Dict[str, Any] = None) -> PersonalizedPage:
        """Generate a personalized landing page for a lead"""
        
        try:
            # Get template
            template = self.templates.get(template_id)
            if not template:
                raise ValueError(f"Template {template_id} not found")
            
            # Analyze lead profile
            lead_profile = await self._analyze_lead_profile(lead_data, traffic_data)
            
            # Select optimal content variants
            selected_content = await self._select_content_variants(template, lead_profile)
            
            # Apply personalization
            personalized_content = await self._apply_personalization(selected_content, lead_profile)
            
            # Generate page
            page = PersonalizedPage(
                page_id=f"page_{lead_data.get('lead_id', 'unknown')}_{int(datetime.now(timezone.utc).timestamp())}",
                lead_id=lead_data.get('lead_id', 'unknown'),
                template_id=template_id,
                generated_content=personalized_content,
                personalization_applied=lead_profile.get('applied_rules', []),
                predicted_conversion_rate=await self._predict_conversion_rate(template, lead_profile),
                generation_timestamp=datetime.now(timezone.utc)
            )
            
            # Store page for tracking
            await self._store_personalized_page(page)
            
            logger.info(f"Generated personalized page {page.page_id} for lead {page.lead_id}")
            
            return page
            
        except Exception as e:
            logger.error(f"Error generating personalized page: {e}")
            raise
    
    async def _select_content_variants(self, template: LandingPageTemplate, 
                                     lead_profile: Dict[str, Any]) -> Dict[ContentType, ContentVariant]:
        """Select optimal content variants based on lead profile"""
        
        selected_variants = {}
        
        for content_type, variants in self.content_library.items():
            if content_type in template.content_variants or not template.content_variants:
                # Score each variant for this lead
                best_variant = None
                best_score = 0
                
                for variant in variants:
                    score = await self._score_content_variant(variant, lead_profile)
                    if score > best_score:
                        best_score = score
                        best_variant = variant
                
                if best_variant:
                    selected_variants[content_type] = best_variant
        
        return selected_variants
    
    async def _apply_personalization(self, content_variants: Dict[ContentType, ContentVariant],
                                   lead_profile: Dict[str, Any]) -> Dict[ContentType, str]:
        """Apply personalization to selected content variants"""
        
        personalized_content = {}
        
        # Personalization variables
        personalization_vars = {
            'first_name': lead_profile.get('first_name', 'there'),
            'product_type': lead_profile.get('product_interest', 'insurance'),
            'savings_amount': lead_profile.get('potential_savings', '500'),
            'location': lead_profile.get('city', 'your area'),
            'age_group': self._get_age_group(lead_profile.get('age', 35))
        }
        
        for content_type, variant in content_variants.items():
            # Apply variable substitution
            personalized_text = variant.content
            for var, value in personalization_vars.items():
                personalized_text = personalized_text.replace(f'{{{var}}}', str(value))
            
            personalized_content[content_type] = personalized_text
        
        return personalized_content
    
    async def track_page_performance(self, page_id: str, event_type: str, 
                                   event_data: Dict[str, Any] = None):
        """Track performance of personalized pages"""
        
        try:
            # Get page
            page = await self._get_personalized_page(page_id)
            if not page:
                return
            
            # Update performance metrics
            if 'events' not in page.performance_tracking:
                page.performance_tracking['events'] = []
            
            page.performance_tracking['events'].append({
                'event_type': event_type,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'data': event_data or {}
            })
            
            # Update conversion tracking
            if event_type in ['form_submission', 'quote_request', 'phone_call']:
                page.performance_tracking['converted'] = True
                page.performance_tracking['conversion_time'] = datetime.now(timezone.utc).isoformat()
            
            # Store updated page
            await self._store_personalized_page(page)
            
            logger.debug(f"Tracked {event_type} for page {page_id}")
            
        except Exception as e:
            logger.error(f"Error tracking page performance: {e}")
    
    async def get_personalization_dashboard(self) -> Dict[str, Any]:
        """Get personalization performance dashboard"""
        
        try:
            # This would typically query a database
            # For now, return sample analytics
            
            return {
                "summary": {
                    "total_pages_generated": 5420,
                    "avg_conversion_rate": 0.186,
                    "personalization_lift": 0.34,  # 34% improvement over generic pages
                    "top_performing_template": "insurance_quote"
                },
                "template_performance": {
                    "insurance_quote": {
                        "pages_generated": 3200,
                        "conversion_rate": 0.21,
                        "avg_engagement_time": 145
                    },
                    "life_insurance": {
                        "pages_generated": 2220,
                        "conversion_rate": 0.16,
                        "avg_engagement_time": 180
                    }
                },
                "content_variant_performance": [
                    {"variant_id": "headline_savings", "conversion_rate": 0.22, "confidence": 0.95},
                    {"variant_id": "cta_urgent", "conversion_rate": 0.24, "confidence": 0.92},
                    {"variant_id": "value_simple", "conversion_rate": 0.19, "confidence": 0.88}
                ],
                "personalization_rules_impact": {
                    "demographic": {"lift": 0.28, "usage": 0.85},
                    "journey_stage": {"lift": 0.41, "usage": 0.92},
                    "traffic_source": {"lift": 0.19, "usage": 0.78}
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating personalization dashboard: {e}")
            raise

# Global personalized landing pages engine
personalized_pages_engine = PersonalizedLandingPagesEngine()