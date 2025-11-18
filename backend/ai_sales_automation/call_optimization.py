"""
Sales Call Optimization Engine

AI-powered system for optimizing sales calls including timing predictions,
talking points generation, and call outcome forecasting.
"""

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import json
import redis

logger = logging.getLogger(__name__)

class CallOutcome(Enum):
    SUCCESSFUL_CONVERSION = "successful_conversion"
    SCHEDULED_FOLLOWUP = "scheduled_followup"
    NEEDS_MORE_INFO = "needs_more_info"
    NOT_INTERESTED = "not_interested"
    NO_ANSWER = "no_answer"
    CALLBACK_REQUESTED = "callback_requested"

class CallType(Enum):
    INITIAL_CONTACT = "initial_contact"
    FOLLOW_UP = "follow_up"
    CLOSING_CALL = "closing_call"
    DISCOVERY_CALL = "discovery_call"
    DEMO_CALL = "demo_call"

@dataclass
class TalkingPoint:
    """Individual talking point for sales calls"""
    point_id: str
    category: str
    content: str
    priority: int
    effectiveness_score: float
    use_conditions: List[str] = field(default_factory=list)
    objection_response: str = ""

@dataclass
class CallRecommendation:
    """Complete call recommendation package"""
    lead_id: str
    optimal_call_time: datetime
    call_type: CallType
    success_probability: float
    talking_points: List[TalkingPoint] = field(default_factory=list)
    lead_context: Dict[str, Any] = field(default_factory=dict)
    objection_handling: Dict[str, str] = field(default_factory=dict)
    next_best_actions: List[str] = field(default_factory=list)
    call_duration_estimate: int = 15  # minutes

@dataclass
class CallHistory:
    """Historical call data for analysis"""
    call_id: str
    lead_id: str
    call_time: datetime
    duration_minutes: int
    outcome: CallOutcome
    talking_points_used: List[str]
    objections_encountered: List[str]
    next_action: str
    rep_id: str
    call_rating: int = 0  # 1-5 scale

class CallOptimizationEngine:
    """AI-powered sales call optimization engine"""
    
    def __init__(self, redis_client=None):
        self.redis_client = redis_client or redis.Redis(decode_responses=True)
        
        # ML models
        self.timing_predictor = RandomForestClassifier(n_estimators=100, random_state=42)
        self.success_predictor = GradientBoostingRegressor(n_estimators=100, random_state=42)
        self.outcome_predictor = RandomForestClassifier(n_estimators=100, random_state=42)
        
        # Model training status
        self.models_trained = False
        
        # Talking points library
        self.talking_points_library = {}
        self.objection_responses = {}
        
        # Call history for learning
        self.call_history = []
        
        # Initialize system
        self._initialize_talking_points()
        self._initialize_objection_responses()
        
        logger.info("Call Optimization Engine initialized")
    
    def _initialize_talking_points(self):
        """Initialize talking points library"""
        
        self.talking_points_library = {
            "opening": [
                TalkingPoint(
                    point_id="opening_warm",
                    category="opening",
                    content="Hi {first_name}, I hope you're having a great day! I'm calling about the {product_type} insurance information you requested. Do you have a quick moment to chat?",
                    priority=1,
                    effectiveness_score=0.85,
                    use_conditions=["warm_lead", "form_submission"]
                ),
                TalkingPoint(
                    point_id="opening_referral",
                    category="opening",
                    content="Hi {first_name}, {referrer_name} suggested I reach out to you about {product_type} insurance. They mentioned you might be looking for better coverage options.",
                    priority=1,
                    effectiveness_score=0.92,
                    use_conditions=["referral_lead"]
                ),
                TalkingPoint(
                    point_id="opening_cold",
                    category="opening",
                    content="Hi {first_name}, I'm calling because our records show you may be paying too much for {product_type} insurance. I have some information that could save you money - do you have 2 minutes?",
                    priority=2,
                    effectiveness_score=0.65,
                    use_conditions=["cold_lead"]
                )
            ],
            
            "value_proposition": [
                TalkingPoint(
                    point_id="value_savings",
                    category="value_proposition",
                    content="Based on your profile, I can show you how to save up to ${potential_savings} per year while getting better coverage than what you have now.",
                    priority=1,
                    effectiveness_score=0.88,
                    use_conditions=["price_sensitive", "has_current_coverage"]
                ),
                TalkingPoint(
                    point_id="value_coverage",
                    category="value_proposition",
                    content="What's unique about our {product_type} insurance is that we provide {unique_benefit} that most other companies don't offer, which means {specific_advantage} for you.",
                    priority=1,
                    effectiveness_score=0.82,
                    use_conditions=["coverage_focused", "research_phase"]
                ),
                TalkingPoint(
                    point_id="value_service",
                    category="value_proposition",
                    content="Our clients love that they get a dedicated agent - that's me - who knows their situation personally. No more calling 1-800 numbers and explaining your story every time.",
                    priority=2,
                    effectiveness_score=0.79,
                    use_conditions=["service_oriented", "bad_experience_with_current"]
                )
            ],
            
            "discovery": [
                TalkingPoint(
                    point_id="discovery_current_coverage",
                    category="discovery",
                    content="Tell me about your current {product_type} insurance - what do you like about it, and what would you change if you could?",
                    priority=1,
                    effectiveness_score=0.90,
                    use_conditions=["has_current_coverage"]
                ),
                TalkingPoint(
                    point_id="discovery_needs",
                    category="discovery",
                    content="What's most important to you in {product_type} insurance - is it the monthly cost, the coverage amount, or having great customer service when you need it?",
                    priority=1,
                    effectiveness_score=0.87,
                    use_conditions=["needs_assessment"]
                ),
                TalkingPoint(
                    point_id="discovery_timeline",
                    category="discovery",
                    content="When are you looking to make a decision on this? Is there anything driving the timeline, like a policy renewal or life change?",
                    priority=2,
                    effectiveness_score=0.83,
                    use_conditions=["timeline_unclear"]
                )
            ],
            
            "closing": [
                TalkingPoint(
                    point_id="closing_assumptive",
                    category="closing",
                    content="Based on everything we've discussed, it sounds like our {recommended_plan} is exactly what you're looking for. Should we get your coverage started today?",
                    priority=1,
                    effectiveness_score=0.85,
                    use_conditions=["high_interest", "needs_match"]
                ),
                TalkingPoint(
                    point_id="closing_alternative",
                    category="closing",
                    content="I can see you're still thinking it over. Would you prefer to start with our basic plan and upgrade later, or would you like me to call you back in a few days?",
                    priority=2,
                    effectiveness_score=0.72,
                    use_conditions=["hesitant", "needs_time"]
                ),
                TalkingPoint(
                    point_id="closing_urgency",
                    category="closing",
                    content="I want to make sure you get this rate - it's only guaranteed for the next {urgency_timeframe}. Can we at least get you a formal quote today?",
                    priority=1,
                    effectiveness_score=0.78,
                    use_conditions=["price_sensitive", "urgency_signals"]
                )
            ]
        }
    
    def _initialize_objection_responses(self):
        """Initialize objection handling responses"""
        
        self.objection_responses = {
            "too_expensive": {
                "response": "I understand cost is important. Let me show you how this actually saves you money in the long run. When you factor in {cost_benefit_analysis}, you're actually paying less for better protection.",
                "follow_up": "What if I could show you a plan that costs less than what you're paying now but gives you better coverage?",
                "effectiveness": 0.75
            },
            
            "happy_with_current": {
                "response": "That's great that you're happy! I'm curious - when was the last time you compared your coverage to what's available now? The market has changed a lot in the past few years.",
                "follow_up": "Even if you don't switch, wouldn't it be worth knowing if you could get the same coverage for less money?",
                "effectiveness": 0.68
            },
            
            "need_to_think": {
                "response": "Absolutely, this is an important decision. What specific aspects would you like to think about? Maybe I can provide some additional information to help.",
                "follow_up": "What if I send you a personalized comparison showing exactly how this would work for your situation? Then you can review it at your own pace.",
                "effectiveness": 0.82
            },
            
            "need_spouse_approval": {
                "response": "Of course! Most of our clients make this decision together. Would it be helpful if I called back when you're both available, or would you prefer I send some information you can review together?",
                "follow_up": "What questions do you think your spouse will have? I can make sure to address those upfront.",
                "effectiveness": 0.79
            },
            
            "not_interested": {
                "response": "I appreciate your honesty. Can I ask what's making you feel that way? Is it the timing, the product, or something else?",
                "follow_up": "Even if now isn't the right time, would you mind if I checked back in a few months? Situations change, and I'd hate for you to miss out if this becomes relevant later.",
                "effectiveness": 0.45
            }
        }
    
    async def generate_call_recommendation(self, lead_data: Dict[str, Any], 
                                         scoring_result: Dict[str, Any],
                                         call_type: CallType = CallType.INITIAL_CONTACT) -> CallRecommendation:
        """Generate comprehensive call recommendation for a lead"""
        
        try:
            # Predict optimal call time
            optimal_time = await self._predict_optimal_call_time(lead_data, scoring_result)
            
            # Calculate success probability
            success_probability = await self._predict_call_success(lead_data, scoring_result, call_type)
            
            # Generate talking points
            talking_points = await self._generate_talking_points(lead_data, scoring_result, call_type)
            
            # Prepare lead context
            lead_context = await self._prepare_lead_context(lead_data, scoring_result)
            
            # Get objection handling strategies
            objection_handling = await self._prepare_objection_handling(lead_data, scoring_result)
            
            # Determine next best actions
            next_actions = await self._determine_next_actions(lead_data, scoring_result, call_type)
            
            # Estimate call duration
            duration_estimate = await self._estimate_call_duration(lead_data, scoring_result, call_type)
            
            recommendation = CallRecommendation(
                lead_id=lead_data['lead_id'],
                optimal_call_time=optimal_time,
                call_type=call_type,
                success_probability=success_probability,
                talking_points=talking_points,
                lead_context=lead_context,
                objection_handling=objection_handling,
                next_best_actions=next_actions,
                call_duration_estimate=duration_estimate
            )
            
            # Store recommendation
            await self._store_call_recommendation(recommendation)
            
            logger.info(f"Generated call recommendation for lead {lead_data['lead_id']}")
            
            return recommendation
            
        except Exception as e:
            logger.error(f"Error generating call recommendation: {e}")
            raise
    
    async def _predict_optimal_call_time(self, lead_data: Dict[str, Any], 
                                       scoring_result: Dict[str, Any]) -> datetime:
        """Predict optimal time to call the lead"""
        
        # Get current time and lead timezone
        current_time = datetime.now(timezone.utc)
        lead_timezone = lead_data.get('timezone', 'UTC')
        
        # Analyze lead behavior patterns
        behavior_data = await self._get_lead_behavior_patterns(lead_data['lead_id'])
        
        # Default business hours: 9 AM - 6 PM in lead's timezone
        optimal_hour = 10  # Default to 10 AM
        
        # Adjust based on lead profile
        if behavior_data:
            # Use historical engagement patterns
            optimal_hour = behavior_data.get('preferred_contact_hour', 10)
        else:
            # Use demographic-based predictions
            age = lead_data.get('age', 35)
            profession = lead_data.get('profession', '').lower()
            
            if age > 55:
                optimal_hour = 9  # Earlier for older demographics
            elif 'executive' in profession or 'manager' in profession:
                optimal_hour = 8  # Early for executives
            elif 'teacher' in profession or 'healthcare' in profession:
                optimal_hour = 16  # After work hours
            else:
                optimal_hour = 11  # Mid-morning for general population
        
        # Calculate next optimal time
        next_optimal = current_time.replace(hour=optimal_hour, minute=0, second=0, microsecond=0)
        
        # If time has passed today, schedule for tomorrow
        if next_optimal <= current_time:
            next_optimal += timedelta(days=1)
        
        # Avoid weekends for business calls
        while next_optimal.weekday() >= 5:  # Saturday = 5, Sunday = 6
            next_optimal += timedelta(days=1)
        
        return next_optimal
    
    async def _predict_call_success(self, lead_data: Dict[str, Any], 
                                  scoring_result: Dict[str, Any], 
                                  call_type: CallType) -> float:
        """Predict probability of successful call outcome"""
        
        # Base success probability from lead score
        base_probability = scoring_result.get('confidence_score', 0.5)
        
        # Adjust based on call type
        call_type_multipliers = {
            CallType.INITIAL_CONTACT: 0.8,
            CallType.FOLLOW_UP: 1.1,
            CallType.DISCOVERY_CALL: 1.2,
            CallType.DEMO_CALL: 1.3,
            CallType.CLOSING_CALL: 1.4
        }
        
        probability = base_probability * call_type_multipliers.get(call_type, 1.0)
        
        # Adjust based on lead characteristics
        priority_level = scoring_result.get('priority_level', 'MEDIUM')
        if priority_level == 'CRITICAL':
            probability *= 1.3
        elif priority_level == 'HIGH':
            probability *= 1.1
        elif priority_level == 'LOW':
            probability *= 0.8
        
        # Adjust based on engagement history
        behavior = await self._get_lead_behavior_patterns(lead_data['lead_id'])
        if behavior:
            engagement_score = behavior.get('engagement_score', 0)
            if engagement_score > 70:
                probability *= 1.2
            elif engagement_score < 30:
                probability *= 0.7
        
        # Cap probability between 0.1 and 0.95
        return max(0.1, min(0.95, probability))
    
    async def _generate_talking_points(self, lead_data: Dict[str, Any], 
                                     scoring_result: Dict[str, Any], 
                                     call_type: CallType) -> List[TalkingPoint]:
        """Generate personalized talking points for the call"""
        
        selected_points = []
        
        # Always include opening
        opening_points = self.talking_points_library["opening"]
        best_opening = await self._select_best_talking_point(opening_points, lead_data, scoring_result)
        if best_opening:
            selected_points.append(await self._personalize_talking_point(best_opening, lead_data, scoring_result))
        
        # Add value proposition
        value_points = self.talking_points_library["value_proposition"]
        best_value = await self._select_best_talking_point(value_points, lead_data, scoring_result)
        if best_value:
            selected_points.append(await self._personalize_talking_point(best_value, lead_data, scoring_result))
        
        # Add discovery questions for appropriate call types
        if call_type in [CallType.INITIAL_CONTACT, CallType.DISCOVERY_CALL]:
            discovery_points = self.talking_points_library["discovery"]
            best_discovery = await self._select_best_talking_point(discovery_points, lead_data, scoring_result)
            if best_discovery:
                selected_points.append(await self._personalize_talking_point(best_discovery, lead_data, scoring_result))
        
        # Add closing for appropriate call types
        if call_type in [CallType.CLOSING_CALL, CallType.DEMO_CALL]:
            closing_points = self.talking_points_library["closing"]
            best_closing = await self._select_best_talking_point(closing_points, lead_data, scoring_result)
            if best_closing:
                selected_points.append(await self._personalize_talking_point(best_closing, lead_data, scoring_result))
        
        return selected_points
    
    async def _select_best_talking_point(self, points: List[TalkingPoint], 
                                       lead_data: Dict[str, Any], 
                                       scoring_result: Dict[str, Any]) -> Optional[TalkingPoint]:
        """Select the best talking point based on lead characteristics"""
        
        # Score each point based on conditions
        scored_points = []
        
        for point in points:
            score = point.effectiveness_score
            
            # Check use conditions
            for condition in point.use_conditions:
                if await self._check_condition(condition, lead_data, scoring_result):
                    score *= 1.2  # Boost score if condition matches
                else:
                    score *= 0.8  # Reduce score if condition doesn't match
            
            scored_points.append((point, score))
        
        # Return highest scoring point
        if scored_points:
            return max(scored_points, key=lambda x: x[1])[0]
        
        return None
    
    async def _check_condition(self, condition: str, lead_data: Dict[str, Any], 
                             scoring_result: Dict[str, Any]) -> bool:
        """Check if a condition is met for the lead"""
        
        condition_checks = {
            'warm_lead': lambda: scoring_result.get('overall_score', 0) > 60,
            'cold_lead': lambda: scoring_result.get('overall_score', 0) <= 40,
            'referral_lead': lambda: lead_data.get('source', '') == 'referral',
            'form_submission': lambda: lead_data.get('form_submitted', False),
            'price_sensitive': lambda: 'cost' in lead_data.get('interests', []) or lead_data.get('price_sensitive', False),
            'has_current_coverage': lambda: lead_data.get('has_insurance', False),
            'coverage_focused': lambda: 'coverage' in lead_data.get('interests', []),
            'research_phase': lambda: lead_data.get('research_phase', False),
            'service_oriented': lambda: 'service' in lead_data.get('interests', []),
            'high_interest': lambda: scoring_result.get('overall_score', 0) > 75,
            'needs_match': lambda: len(scoring_result.get('recommended_products', [])) > 0,
            'hesitant': lambda: scoring_result.get('overall_score', 0) < 55,
            'urgency_signals': lambda: len(scoring_result.get('urgency_signals', [])) > 0
        }
        
        check_func = condition_checks.get(condition)
        return check_func() if check_func else False
    
    async def track_call_outcome(self, call_data: Dict[str, Any]) -> str:
        """Track call outcome for learning and optimization"""
        
        try:
            call_history = CallHistory(
                call_id=call_data.get('call_id', f"call_{int(datetime.now(timezone.utc).timestamp())}"),
                lead_id=call_data['lead_id'],
                call_time=datetime.fromisoformat(call_data.get('call_time', datetime.now(timezone.utc).isoformat())),
                duration_minutes=call_data.get('duration_minutes', 0),
                outcome=CallOutcome(call_data.get('outcome', 'no_answer')),
                talking_points_used=call_data.get('talking_points_used', []),
                objections_encountered=call_data.get('objections_encountered', []),
                next_action=call_data.get('next_action', ''),
                rep_id=call_data.get('rep_id', ''),
                call_rating=call_data.get('call_rating', 0)
            )
            
            # Store call history
            await self._store_call_history(call_history)
            
            # Update talking point effectiveness
            await self._update_talking_point_effectiveness(call_history)
            
            # Trigger follow-up actions if needed
            await self._trigger_followup_actions(call_history)
            
            logger.info(f"Tracked call outcome: {call_history.outcome.value} for lead {call_history.lead_id}")
            
            return call_history.call_id
            
        except Exception as e:
            logger.error(f"Error tracking call outcome: {e}")
            raise

# Global call optimization engine instance
call_optimizer = CallOptimizationEngine()