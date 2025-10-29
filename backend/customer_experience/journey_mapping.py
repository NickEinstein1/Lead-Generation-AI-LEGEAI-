"""
Lead Journey Mapping Engine

Comprehensive tracking and analysis of the complete customer lifecycle
from first touch to conversion and beyond.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from enum import Enum
import json
import logging

logger = logging.getLogger(__name__)

class TouchpointType(Enum):
    WEBSITE_VISIT = "website_visit"
    EMAIL_OPEN = "email_open"
    EMAIL_CLICK = "email_click"
    FORM_SUBMISSION = "form_submission"
    PHONE_CALL = "phone_call"
    CHAT_INTERACTION = "chat_interaction"
    SOCIAL_MEDIA = "social_media"
    AD_CLICK = "ad_click"
    CONTENT_DOWNLOAD = "content_download"
    WEBINAR_ATTENDANCE = "webinar_attendance"
    DEMO_REQUEST = "demo_request"
    QUOTE_REQUEST = "quote_request"
    PURCHASE = "purchase"

class JourneyStage(Enum):
    AWARENESS = "awareness"
    INTEREST = "interest"
    CONSIDERATION = "consideration"
    INTENT = "intent"
    EVALUATION = "evaluation"
    PURCHASE = "purchase"
    ONBOARDING = "onboarding"
    RETENTION = "retention"
    ADVOCACY = "advocacy"

@dataclass
class Touchpoint:
    """Individual customer touchpoint"""
    touchpoint_id: str
    lead_id: str
    touchpoint_type: TouchpointType
    timestamp: datetime
    channel: str
    source: str
    medium: str
    campaign: Optional[str] = None
    content: Optional[str] = None
    page_url: Optional[str] = None
    duration_seconds: Optional[int] = None
    conversion_value: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    engagement_score: float = 0.0
    attribution_weight: float = 1.0

@dataclass
class JourneyStage:
    """Customer journey stage"""
    stage_id: str
    stage_name: JourneyStage
    entry_date: datetime
    exit_date: Optional[datetime] = None
    duration_days: Optional[int] = None
    touchpoints: List[Touchpoint] = field(default_factory=list)
    conversion_probability: float = 0.0
    stage_value: float = 0.0
    next_best_actions: List[str] = field(default_factory=list)

@dataclass
class CustomerJourney:
    """Complete customer journey"""
    journey_id: str
    lead_id: str
    customer_id: Optional[str] = None
    start_date: datetime = field(default_factory=datetime.utcnow)
    current_stage: JourneyStage = JourneyStage.AWARENESS
    stages: List[JourneyStage] = field(default_factory=list)
    touchpoints: List[Touchpoint] = field(default_factory=list)
    total_touchpoints: int = 0
    journey_duration_days: int = 0
    conversion_probability: float = 0.0
    lifetime_value: float = 0.0
    acquisition_cost: float = 0.0
    roi: float = 0.0
    attribution_model: str = "linear"
    journey_health_score: float = 0.0

class JourneyMappingEngine:
    """Engine for tracking and analyzing customer journeys"""
    
    def __init__(self, redis_client=None):
        self.redis_client = redis_client
        self.active_journeys = {}
        self.journey_templates = {}
        self.attribution_models = {}
        
        self._initialize_journey_templates()
        self._initialize_attribution_models()
        
        logger.info("Journey Mapping Engine initialized")
    
    def _initialize_journey_templates(self):
        """Initialize journey stage templates"""
        
        self.journey_templates = {
            "insurance_lead": {
                "stages": [
                    {"name": JourneyStage.AWARENESS, "avg_duration": 3, "key_touchpoints": ["website_visit", "ad_click"]},
                    {"name": JourneyStage.INTEREST, "avg_duration": 7, "key_touchpoints": ["content_download", "email_open"]},
                    {"name": JourneyStage.CONSIDERATION, "avg_duration": 14, "key_touchpoints": ["quote_request", "phone_call"]},
                    {"name": JourneyStage.INTENT, "avg_duration": 5, "key_touchpoints": ["demo_request", "chat_interaction"]},
                    {"name": JourneyStage.EVALUATION, "avg_duration": 10, "key_touchpoints": ["multiple_quotes", "comparison_research"]},
                    {"name": JourneyStage.PURCHASE, "avg_duration": 2, "key_touchpoints": ["purchase", "contract_signing"]}
                ],
                "conversion_benchmarks": {
                    JourneyStage.AWARENESS: 0.15,
                    JourneyStage.INTEREST: 0.35,
                    JourneyStage.CONSIDERATION: 0.55,
                    JourneyStage.INTENT: 0.75,
                    JourneyStage.EVALUATION: 0.85,
                    JourneyStage.PURCHASE: 0.95
                }
            }
        }
    
    def _initialize_attribution_models(self):
        """Initialize attribution models for touchpoint weighting"""
        
        self.attribution_models = {
            "first_touch": lambda touchpoints: self._first_touch_attribution(touchpoints),
            "last_touch": lambda touchpoints: self._last_touch_attribution(touchpoints),
            "linear": lambda touchpoints: self._linear_attribution(touchpoints),
            "time_decay": lambda touchpoints: self._time_decay_attribution(touchpoints),
            "position_based": lambda touchpoints: self._position_based_attribution(touchpoints)
        }
    
    async def track_touchpoint(self, lead_id: str, touchpoint_data: Dict[str, Any]) -> str:
        """Track a new customer touchpoint"""
        
        try:
            # Create touchpoint
            touchpoint = Touchpoint(
                touchpoint_id=f"tp_{lead_id}_{int(datetime.utcnow().timestamp())}",
                lead_id=lead_id,
                touchpoint_type=TouchpointType(touchpoint_data.get('type', 'website_visit')),
                timestamp=datetime.utcnow(),
                channel=touchpoint_data.get('channel', 'direct'),
                source=touchpoint_data.get('source', 'unknown'),
                medium=touchpoint_data.get('medium', 'organic'),
                campaign=touchpoint_data.get('campaign'),
                content=touchpoint_data.get('content'),
                page_url=touchpoint_data.get('page_url'),
                duration_seconds=touchpoint_data.get('duration_seconds'),
                conversion_value=touchpoint_data.get('conversion_value', 0.0),
                metadata=touchpoint_data.get('metadata', {}),
                engagement_score=await self._calculate_engagement_score(touchpoint_data)
            )
            
            # Get or create journey
            journey = await self._get_or_create_journey(lead_id)
            
            # Add touchpoint to journey
            journey.touchpoints.append(touchpoint)
            journey.total_touchpoints += 1
            
            # Update journey stage if needed
            await self._update_journey_stage(journey, touchpoint)
            
            # Recalculate journey metrics
            await self._update_journey_metrics(journey)
            
            # Store updated journey
            await self._store_journey(journey)
            
            # Trigger real-time actions
            await self._trigger_journey_actions(journey, touchpoint)
            
            logger.debug(f"Tracked touchpoint {touchpoint.touchpoint_id} for lead {lead_id}")
            
            return touchpoint.touchpoint_id
            
        except Exception as e:
            logger.error(f"Error tracking touchpoint: {e}")
            raise
    
    async def get_journey_analysis(self, lead_id: str) -> Dict[str, Any]:
        """Get comprehensive journey analysis for a lead"""
        
        try:
            journey = await self._get_journey(lead_id)
            if not journey:
                return {"error": "Journey not found"}
            
            # Calculate journey metrics
            journey_metrics = await self._calculate_journey_metrics(journey)
            
            # Identify journey patterns
            patterns = await self._identify_journey_patterns(journey)
            
            # Generate recommendations
            recommendations = await self._generate_journey_recommendations(journey)
            
            # Calculate attribution
            attribution = await self._calculate_attribution(journey)
            
            return {
                "journey_id": journey.journey_id,
                "current_stage": journey.current_stage.value,
                "journey_duration": journey.journey_duration_days,
                "total_touchpoints": journey.total_touchpoints,
                "conversion_probability": journey.conversion_probability,
                "journey_health_score": journey.journey_health_score,
                "metrics": journey_metrics,
                "patterns": patterns,
                "recommendations": recommendations,
                "attribution": attribution,
                "stage_progression": [
                    {
                        "stage": stage.stage_name.value,
                        "duration_days": stage.duration_days,
                        "touchpoint_count": len(stage.touchpoints),
                        "stage_value": stage.stage_value
                    }
                    for stage in journey.stages
                ]
            }
            
        except Exception as e:
            logger.error(f"Error getting journey analysis: {e}")
            raise
    
    async def _calculate_engagement_score(self, touchpoint_data: Dict[str, Any]) -> float:
        """Calculate engagement score for a touchpoint"""
        
        base_scores = {
            TouchpointType.WEBSITE_VISIT: 1.0,
            TouchpointType.EMAIL_OPEN: 2.0,
            TouchpointType.EMAIL_CLICK: 4.0,
            TouchpointType.FORM_SUBMISSION: 8.0,
            TouchpointType.PHONE_CALL: 10.0,
            TouchpointType.CHAT_INTERACTION: 6.0,
            TouchpointType.CONTENT_DOWNLOAD: 7.0,
            TouchpointType.DEMO_REQUEST: 12.0,
            TouchpointType.QUOTE_REQUEST: 15.0,
            TouchpointType.PURCHASE: 20.0
        }
        
        touchpoint_type = TouchpointType(touchpoint_data.get('type', 'website_visit'))
        base_score = base_scores.get(touchpoint_type, 1.0)
        
        # Adjust based on duration
        duration = touchpoint_data.get('duration_seconds', 0)
        if duration > 300:  # 5+ minutes
            base_score *= 1.5
        elif duration > 60:  # 1+ minute
            base_score *= 1.2
        
        # Adjust based on conversion value
        conversion_value = touchpoint_data.get('conversion_value', 0)
        if conversion_value > 0:
            base_score *= (1 + conversion_value / 1000)
        
        return min(base_score, 25.0)  # Cap at 25
    
    async def _update_journey_stage(self, journey: CustomerJourney, touchpoint: Touchpoint):
        """Update journey stage based on new touchpoint"""
        
        # Stage progression rules
        stage_triggers = {
            JourneyStage.INTEREST: [TouchpointType.EMAIL_OPEN, TouchpointType.CONTENT_DOWNLOAD],
            JourneyStage.CONSIDERATION: [TouchpointType.QUOTE_REQUEST, TouchpointType.PHONE_CALL],
            JourneyStage.INTENT: [TouchpointType.DEMO_REQUEST, TouchpointType.CHAT_INTERACTION],
            JourneyStage.EVALUATION: [TouchpointType.MULTIPLE_QUOTES],
            JourneyStage.PURCHASE: [TouchpointType.PURCHASE]
        }
        
        # Check if touchpoint triggers stage progression
        for stage, triggers in stage_triggers.items():
            if touchpoint.touchpoint_type in triggers and journey.current_stage.value < stage.value:
                # Progress to new stage
                await self._progress_to_stage(journey, stage)
                break
    
    async def get_journey_dashboard(self, date_range: int = 30) -> Dict[str, Any]:
        """Get journey analytics dashboard"""
        
        try:
            # This would typically query a database
            # For now, return sample analytics
            
            return {
                "summary": {
                    "total_journeys": 1250,
                    "active_journeys": 890,
                    "completed_journeys": 360,
                    "avg_journey_duration": 18.5,
                    "conversion_rate": 0.288
                },
                "stage_distribution": {
                    "awareness": 35,
                    "interest": 25,
                    "consideration": 20,
                    "intent": 12,
                    "evaluation": 5,
                    "purchase": 3
                },
                "top_touchpoints": [
                    {"type": "website_visit", "count": 5420, "conversion_rate": 0.12},
                    {"type": "email_open", "count": 3210, "conversion_rate": 0.18},
                    {"type": "quote_request", "count": 890, "conversion_rate": 0.45},
                    {"type": "phone_call", "count": 560, "conversion_rate": 0.62}
                ],
                "channel_performance": {
                    "organic_search": {"journeys": 450, "conversion_rate": 0.32},
                    "paid_search": {"journeys": 320, "conversion_rate": 0.28},
                    "email": {"journeys": 280, "conversion_rate": 0.35},
                    "social": {"journeys": 200, "conversion_rate": 0.15}
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating journey dashboard: {e}")
            raise

# Global journey mapping engine
journey_mapping_engine = JourneyMappingEngine()