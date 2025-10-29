"""
Intelligent Lead Nurturing Engine

AI-powered lead nurturing system that creates dynamic email sequences based on
lead behavior, engagement patterns, and predictive analytics.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import json
import redis

logger = logging.getLogger(__name__)

class NurturingStrategy(Enum):
    EDUCATIONAL = "educational"
    PROMOTIONAL = "promotional"
    CONSULTATIVE = "consultative"
    URGENCY_BASED = "urgency_based"
    RELATIONSHIP_BUILDING = "relationship_building"

class BehaviorTrigger(Enum):
    EMAIL_OPEN = "email_open"
    EMAIL_CLICK = "email_click"
    WEBSITE_VISIT = "website_visit"
    CONTENT_DOWNLOAD = "content_download"
    FORM_SUBMISSION = "form_submission"
    NO_ENGAGEMENT = "no_engagement"
    HIGH_ENGAGEMENT = "high_engagement"
    PRICE_PAGE_VISIT = "price_page_visit"
    COMPETITOR_RESEARCH = "competitor_research"

@dataclass
class EmailTemplate:
    """Email template with dynamic content"""
    template_id: str
    subject_line: str
    content: str
    personalization_fields: List[str] = field(default_factory=list)
    call_to_action: str = ""
    engagement_score_boost: int = 0
    expected_open_rate: float = 0.0
    expected_click_rate: float = 0.0

@dataclass
class NurturingSequence:
    """Dynamic nurturing sequence"""
    sequence_id: str
    lead_id: str
    strategy: NurturingStrategy
    current_step: int = 0
    total_steps: int = 0
    emails: List[EmailTemplate] = field(default_factory=list)
    behavior_triggers: Dict[BehaviorTrigger, str] = field(default_factory=dict)
    engagement_score: float = 0.0
    conversion_probability: float = 0.0
    next_send_time: Optional[datetime] = None
    is_active: bool = True

@dataclass
class LeadBehavior:
    """Lead behavior tracking"""
    lead_id: str
    email_opens: int = 0
    email_clicks: int = 0
    website_visits: int = 0
    content_downloads: int = 0
    form_submissions: int = 0
    last_engagement: Optional[datetime] = None
    engagement_score: float = 0.0
    behavior_pattern: str = "unknown"
    preferred_content_type: str = "unknown"
    optimal_send_time: str = "unknown"

class IntelligentNurturingEngine:
    """AI-powered intelligent lead nurturing engine"""
    
    def __init__(self, redis_client=None):
        self.redis_client = redis_client or redis.Redis(decode_responses=True)
        
        # ML models
        self.engagement_predictor = RandomForestClassifier(n_estimators=100, random_state=42)
        self.content_recommender = KMeans(n_clusters=5, random_state=42)
        self.timing_optimizer = RandomForestClassifier(n_estimators=50, random_state=42)
        
        # Model training status
        self.models_trained = False
        
        # Email templates library
        self.email_templates = {}
        self.nurturing_strategies = {}
        
        # Behavior tracking
        self.behavior_patterns = {}
        
        # Initialize system
        self._initialize_templates()
        self._initialize_strategies()
        
        logger.info("Intelligent Nurturing Engine initialized")
    
    def _initialize_templates(self):
        """Initialize email templates library"""
        
        self.email_templates = {
            "welcome_educational": EmailTemplate(
                template_id="welcome_educational",
                subject_line="Welcome! Here's what you need to know about {product_type} insurance",
                content="""
                Hi {first_name},
                
                Welcome to our insurance family! Based on your interest in {product_type} insurance, 
                I've put together some essential information to help you make the best decision.
                
                🔍 Key Benefits of {product_type} Insurance:
                {product_benefits}
                
                📊 Did you know? {industry_statistic}
                
                I'd love to schedule a quick 15-minute call to discuss your specific needs and 
                answer any questions you might have.
                
                Best regards,
                {agent_name}
                """,
                personalization_fields=["first_name", "product_type", "product_benefits", "industry_statistic", "agent_name"],
                call_to_action="Schedule a 15-minute consultation",
                engagement_score_boost=5,
                expected_open_rate=0.35,
                expected_click_rate=0.08
            ),
            
            "educational_deep_dive": EmailTemplate(
                template_id="educational_deep_dive",
                subject_line="The complete guide to {product_type} insurance (5-min read)",
                content="""
                Hi {first_name},
                
                Since you showed interest in learning more about {product_type} insurance, 
                I've created a comprehensive guide just for you.
                
                📖 In this guide, you'll discover:
                • How to calculate the right coverage amount
                • Common mistakes to avoid
                • Money-saving strategies
                • Real customer success stories
                
                {personalized_recommendation}
                
                Download your free guide here: {download_link}
                
                Questions? Just reply to this email - I read every response personally.
                
                Best,
                {agent_name}
                """,
                personalization_fields=["first_name", "product_type", "personalized_recommendation", "download_link", "agent_name"],
                call_to_action="Download your free guide",
                engagement_score_boost=8,
                expected_open_rate=0.28,
                expected_click_rate=0.12
            ),
            
            "urgency_limited_time": EmailTemplate(
                template_id="urgency_limited_time",
                subject_line="⏰ {first_name}, your {product_type} quote expires in 48 hours",
                content="""
                Hi {first_name},
                
                I wanted to give you a quick heads up - your personalized {product_type} 
                insurance quote is set to expire in 48 hours.
                
                💰 Your current quote: ${quote_amount}/month
                🎯 Potential savings: ${potential_savings}/year
                
                {urgency_reason}
                
                To secure this rate, simply click here: {quote_link}
                
                Or if you have questions, I'm available for a quick call today.
                
                Don't let this opportunity slip away!
                
                {agent_name}
                """,
                personalization_fields=["first_name", "product_type", "quote_amount", "potential_savings", "urgency_reason", "quote_link", "agent_name"],
                call_to_action="Secure your rate now",
                engagement_score_boost=3,
                expected_open_rate=0.42,
                expected_click_rate=0.15
            ),
            
            "social_proof": EmailTemplate(
                template_id="social_proof",
                subject_line="How {customer_name} saved ${savings_amount} on {product_type} insurance",
                content="""
                Hi {first_name},
                
                I thought you'd be interested in this success story from one of our recent customers.
                
                {customer_name}, a {customer_profession} from {customer_location}, was in a similar 
                situation to yours. They were looking for {product_type} insurance and were 
                concerned about {common_concern}.
                
                Here's what happened:
                ✅ Saved ${savings_amount} annually
                ✅ Got {coverage_improvement} better coverage
                ✅ Simplified their insurance management
                
                "{customer_testimonial}"
                
                Want to see if you can achieve similar results? Let's chat!
                
                {agent_name}
                """,
                personalization_fields=["first_name", "customer_name", "savings_amount", "product_type", "customer_profession", "customer_location", "common_concern", "coverage_improvement", "customer_testimonial", "agent_name"],
                call_to_action="Get your personalized savings analysis",
                engagement_score_boost=6,
                expected_open_rate=0.31,
                expected_click_rate=0.09
            ),
            
            "re_engagement": EmailTemplate(
                template_id="re_engagement",
                subject_line="Did I lose you? (Quick question, {first_name})",
                content="""
                Hi {first_name},
                
                I noticed you haven't opened my recent emails about {product_type} insurance, 
                and I'm wondering if I missed the mark somehow.
                
                Could you help me out with a quick question?
                
                What's the #1 thing you'd want to know about {product_type} insurance right now?
                
                A) How much it would cost for my situation
                B) What coverage options are available
                C) How to switch from my current provider
                D) Something else (just reply and tell me!)
                
                Just hit reply with your letter choice, and I'll send you exactly what you need.
                
                Thanks for your time,
                {agent_name}
                
                P.S. If you're no longer interested, just reply with "STOP" and I'll remove you from this sequence.
                """,
                personalization_fields=["first_name", "product_type", "agent_name"],
                call_to_action="Reply with your choice",
                engagement_score_boost=4,
                expected_open_rate=0.25,
                expected_click_rate=0.18
            )
        }
    
    def _initialize_strategies(self):
        """Initialize nurturing strategies"""
        
        self.nurturing_strategies = {
            NurturingStrategy.EDUCATIONAL: {
                "description": "Focus on educating the lead about insurance concepts and benefits",
                "email_sequence": ["welcome_educational", "educational_deep_dive", "social_proof", "re_engagement"],
                "timing_days": [0, 3, 7, 14],
                "target_audience": "information_seekers",
                "conversion_rate": 0.18
            },
            
            NurturingStrategy.PROMOTIONAL: {
                "description": "Emphasize deals, savings, and limited-time offers",
                "email_sequence": ["urgency_limited_time", "social_proof", "welcome_educational", "re_engagement"],
                "timing_days": [0, 2, 5, 10],
                "target_audience": "price_sensitive",
                "conversion_rate": 0.22
            },
            
            NurturingStrategy.CONSULTATIVE: {
                "description": "Position as trusted advisor with personalized recommendations",
                "email_sequence": ["welcome_educational", "social_proof", "educational_deep_dive", "re_engagement"],
                "timing_days": [0, 4, 8, 16],
                "target_audience": "relationship_focused",
                "conversion_rate": 0.25
            },
            
            NurturingStrategy.URGENCY_BASED: {
                "description": "Create urgency through time-sensitive offers and deadlines",
                "email_sequence": ["urgency_limited_time", "urgency_limited_time", "social_proof", "re_engagement"],
                "timing_days": [0, 1, 3, 7],
                "target_audience": "decision_ready",
                "conversion_rate": 0.28
            },
            
            NurturingStrategy.RELATIONSHIP_BUILDING: {
                "description": "Focus on building long-term relationships and trust",
                "email_sequence": ["welcome_educational", "educational_deep_dive", "social_proof", "social_proof", "re_engagement"],
                "timing_days": [0, 5, 10, 18, 25],
                "target_audience": "relationship_oriented",
                "conversion_rate": 0.20
            }
        }
    
    async def create_nurturing_sequence(self, lead_data: Dict[str, Any], 
                                      scoring_result: Dict[str, Any]) -> NurturingSequence:
        """Create personalized nurturing sequence for a lead"""
        
        try:
            # Analyze lead to determine best strategy
            strategy = await self._determine_nurturing_strategy(lead_data, scoring_result)
            
            # Get strategy configuration
            strategy_config = self.nurturing_strategies[strategy]
            
            # Create personalized email sequence
            emails = []
            for template_id in strategy_config["email_sequence"]:
                template = self.email_templates[template_id]
                personalized_template = await self._personalize_template(template, lead_data, scoring_result)
                emails.append(personalized_template)
            
            # Calculate timing
            timing_days = strategy_config["timing_days"]
            next_send_time = datetime.utcnow() + timedelta(days=timing_days[0])
            
            # Create sequence
            sequence = NurturingSequence(
                sequence_id=f"seq_{lead_data['lead_id']}_{int(datetime.utcnow().timestamp())}",
                lead_id=lead_data['lead_id'],
                strategy=strategy,
                total_steps=len(emails),
                emails=emails,
                conversion_probability=scoring_result.get('confidence_score', 0.5),
                next_send_time=next_send_time
            )
            
            # Set up behavior triggers
            sequence.behavior_triggers = await self._setup_behavior_triggers(sequence, lead_data)
            
            # Store sequence
            await self._store_sequence(sequence)
            
            logger.info(f"Created nurturing sequence {sequence.sequence_id} with strategy {strategy.value}")
            
            return sequence
            
        except Exception as e:
            logger.error(f"Error creating nurturing sequence: {e}")
            raise
    
    async def _determine_nurturing_strategy(self, lead_data: Dict[str, Any], 
                                          scoring_result: Dict[str, Any]) -> NurturingStrategy:
        """Determine the best nurturing strategy for a lead"""
        
        # Analyze lead characteristics
        priority_level = scoring_result.get('priority_level', 'MEDIUM')
        conversion_velocity = scoring_result.get('conversion_velocity', {})
        lead_score = scoring_result.get('overall_score', 50)
        
        # Get lead behavior if available
        behavior = await self._get_lead_behavior(lead_data['lead_id'])
        
        # Decision logic for strategy selection
        if priority_level == 'CRITICAL' and lead_score > 80:
            return NurturingStrategy.URGENCY_BASED
        
        elif behavior and behavior.engagement_score > 70:
            return NurturingStrategy.CONSULTATIVE
        
        elif lead_data.get('price_sensitive', False) or 'cost' in lead_data.get('interests', []):
            return NurturingStrategy.PROMOTIONAL
        
        elif lead_data.get('research_phase', False) or behavior and behavior.content_downloads > 2:
            return NurturingStrategy.EDUCATIONAL
        
        else:
            return NurturingStrategy.RELATIONSHIP_BUILDING
    
    async def _personalize_template(self, template: EmailTemplate, 
                                  lead_data: Dict[str, Any], 
                                  scoring_result: Dict[str, Any]) -> EmailTemplate:
        """Personalize email template with lead-specific data"""
        
        # Create personalized copy
        personalized_template = EmailTemplate(
            template_id=template.template_id,
            subject_line=template.subject_line,
            content=template.content,
            personalization_fields=template.personalization_fields,
            call_to_action=template.call_to_action,
            engagement_score_boost=template.engagement_score_boost,
            expected_open_rate=template.expected_open_rate,
            expected_click_rate=template.expected_click_rate
        )
        
        # Personalization data
        personalization_data = {
            'first_name': lead_data.get('first_name', 'there'),
            'last_name': lead_data.get('last_name', ''),
            'product_type': scoring_result.get('recommended_products', ['insurance'])[0],
            'agent_name': lead_data.get('assigned_agent_name', 'Your Insurance Advisor'),
            'quote_amount': f"{lead_data.get('estimated_premium', 150):.0f}",
            'potential_savings': f"{lead_data.get('potential_savings', 500):.0f}",
            'company_name': lead_data.get('company_name', 'our company')
        }
        
        # Add dynamic content based on lead profile
        if template.template_id == "educational_deep_dive":
            personalization_data['personalized_recommendation'] = await self._generate_personalized_recommendation(lead_data, scoring_result)
        
        elif template.template_id == "urgency_limited_time":
            personalization_data['urgency_reason'] = await self._generate_urgency_reason(lead_data, scoring_result)
        
        elif template.template_id == "social_proof":
            personalization_data.update(await self._generate_social_proof_content(lead_data, scoring_result))
        
        # Apply personalization
        for field in template.personalization_fields:
            if field in personalization_data:
                personalized_template.subject_line = personalized_template.subject_line.replace(f"{{{field}}}", str(personalization_data[field]))
                personalized_template.content = personalized_template.content.replace(f"{{{field}}}", str(personalization_data[field]))
        
        return personalized_template
    
    async def track_behavior(self, lead_id: str, behavior_type: BehaviorTrigger, 
                           metadata: Dict[str, Any] = None):
        """Track lead behavior and update nurturing sequence accordingly"""
        
        try:
            # Get current behavior
            behavior = await self._get_lead_behavior(lead_id)
            if not behavior:
                behavior = LeadBehavior(lead_id=lead_id)
            
            # Update behavior based on type
            if behavior_type == BehaviorTrigger.EMAIL_OPEN:
                behavior.email_opens += 1
                behavior.engagement_score += 2
            elif behavior_type == BehaviorTrigger.EMAIL_CLICK:
                behavior.email_clicks += 1
                behavior.engagement_score += 5
            elif behavior_type == BehaviorTrigger.WEBSITE_VISIT:
                behavior.website_visits += 1
                behavior.engagement_score += 3
            elif behavior_type == BehaviorTrigger.CONTENT_DOWNLOAD:
                behavior.content_downloads += 1
                behavior.engagement_score += 8
            elif behavior_type == BehaviorTrigger.FORM_SUBMISSION:
                behavior.form_submissions += 1
                behavior.engagement_score += 10
            
            behavior.last_engagement = datetime.utcnow()
            
            # Store updated behavior
            await self._store_lead_behavior(behavior)
            
            # Check if behavior triggers sequence changes
            await self._check_behavior_triggers(lead_id, behavior_type, metadata)
            
            logger.debug(f"Tracked {behavior_type.value} for lead {lead_id}")
            
        except Exception as e:
            logger.error(f"Error tracking behavior: {e}")
    
    async def _check_behavior_triggers(self, lead_id: str, behavior_type: BehaviorTrigger, 
                                     metadata: Dict[str, Any] = None):
        """Check if behavior should trigger sequence modifications"""
        
        # Get active sequence
        sequence = await self._get_active_sequence(lead_id)
        if not sequence:
            return
        
        # Check behavior triggers
        if behavior_type in sequence.behavior_triggers:
            action = sequence.behavior_triggers[behavior_type]
            
            if action == "accelerate":
                # Send next email sooner
                sequence.next_send_time = datetime.utcnow() + timedelta(hours=2)
                await self._store_sequence(sequence)
                
            elif action == "change_strategy":
                # Switch to more aggressive strategy
                if sequence.strategy != NurturingStrategy.URGENCY_BASED:
                    await self._modify_sequence_strategy(sequence, NurturingStrategy.URGENCY_BASED)
                    
            elif action == "add_followup":
                # Add immediate follow-up email
                await self._add_immediate_followup(sequence, behavior_type)
    
    async def get_sequence_performance(self, sequence_id: str) -> Dict[str, Any]:
        """Get performance metrics for a nurturing sequence"""
        
        try:
            sequence = await self._get_sequence(sequence_id)
            if not sequence:
                return {"error": "Sequence not found"}
            
            # Get behavior data
            behavior = await self._get_lead_behavior(sequence.lead_id)
            
            # Calculate performance metrics
            emails_sent = sequence.current_step
            emails_opened = behavior.email_opens if behavior else 0
            emails_clicked = behavior.email_clicks if behavior else 0
            
            open_rate = (emails_opened / emails_sent * 100) if emails_sent > 0 else 0
            click_rate = (emails_clicked / emails_sent * 100) if emails_sent > 0 else 0
            
            performance = {
                'sequence_id': sequence_id,
                'lead_id': sequence.lead_id,
                'strategy': sequence.strategy.value,
                'emails_sent': emails_sent,
                'emails_opened': emails_opened,
                'emails_clicked': emails_clicked,
                'open_rate': open_rate,
                'click_rate': click_rate,
                'engagement_score': behavior.engagement_score if behavior else 0,
                'conversion_probability': sequence.conversion_probability,
                'is_active': sequence.is_active,
                'next_send_time': sequence.next_send_time.isoformat() if sequence.next_send_time else None
            }
            
            return performance
            
        except Exception as e:
            logger.error(f"Error getting sequence performance: {e}")
            return {"error": str(e)}
    
    async def _generate_personalized_recommendation(self, lead_data: Dict[str, Any], 
                                                  scoring_result: Dict[str, Any]) -> str:
        """Generate personalized recommendation content"""
        
        recommended_products = scoring_result.get('recommended_products', ['insurance'])
        lead_score = scoring_result.get('overall_score', 50)
        
        if lead_score > 80:
            return f"Based on your profile, I recommend starting with our {recommended_products[0]} plan, which could save you up to 25% compared to typical market rates."
        elif lead_score > 60:
            return f"Given your situation, our {recommended_products[0]} coverage would be an excellent fit, offering comprehensive protection at competitive rates."
        else:
            return f"I've identified {recommended_products[0]} insurance as a great starting point for your needs, with flexible options to grow with you."
    
    async def _generate_urgency_reason(self, lead_data: Dict[str, Any], 
                                     scoring_result: Dict[str, Any]) -> str:
        """Generate urgency reason based on lead profile"""
        
        urgency_signals = scoring_result.get('urgency_signals', [])
        
        if "OPEN_ENROLLMENT_PERIOD" in urgency_signals:
            return "Open enrollment ends soon, and rates may increase significantly after the deadline."
        elif "RATE_INCREASE_PENDING" in urgency_signals:
            return "Industry-wide rate increases take effect next month - lock in today's rates now."
        elif "COMPETITOR_COMPARISON" in urgency_signals:
            return "This exclusive rate is only available for a limited time to help you switch from your current provider."
        else:
            return "Market conditions are changing rapidly, and this rate may not be available much longer."
    
    async def _generate_social_proof_content(self, lead_data: Dict[str, Any], 
                                           scoring_result: Dict[str, Any]) -> Dict[str, str]:
        """Generate social proof content with similar customer stories"""
        
        # This would typically pull from a database of customer success stories
        # For now, generate contextually appropriate content
        
        product_type = scoring_result.get('recommended_products', ['insurance'])[0]
        
        return {
            'customer_name': 'Sarah M.',
            'customer_profession': 'Marketing Manager',
            'customer_location': 'Austin, TX',
            'savings_amount': '850',
            'coverage_improvement': '40%',
            'common_concern': 'finding affordable coverage without sacrificing quality',
            'customer_testimonial': f'I was skeptical about switching {product_type} providers, but the process was seamless and I\'m saving hundreds every year while getting better coverage.'
        }

# Global nurturing engine instance
nurturing_engine = IntelligentNurturingEngine()