"""
Automated Follow-up Scheduling Engine

AI-powered system for intelligent follow-up scheduling based on lead engagement,
behavior patterns, and optimal timing predictions.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import json
import redis

logger = logging.getLogger(__name__)

class FollowUpType(Enum):
    EMAIL = "email"
    PHONE_CALL = "phone_call"
    SMS = "sms"
    LINKEDIN_MESSAGE = "linkedin_message"
    DIRECT_MAIL = "direct_mail"
    TASK_REMINDER = "task_reminder"

class EngagementLevel(Enum):
    VERY_HIGH = "very_high"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    VERY_LOW = "very_low"
    DORMANT = "dormant"

class FollowUpPriority(Enum):
    URGENT = "urgent"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

@dataclass
class FollowUpAction:
    """Individual follow-up action"""
    action_id: str
    lead_id: str
    follow_up_type: FollowUpType
    scheduled_time: datetime
    priority: FollowUpPriority
    content: str
    trigger_event: str
    expected_response_rate: float
    rep_id: str
    is_automated: bool = True
    is_completed: bool = False
    completion_time: Optional[datetime] = None
    outcome: Optional[str] = None
    next_action_id: Optional[str] = None

@dataclass
class EngagementTracker:
    """Track lead engagement patterns"""
    lead_id: str
    last_email_open: Optional[datetime] = None
    last_email_click: Optional[datetime] = None
    last_website_visit: Optional[datetime] = None
    last_phone_call: Optional[datetime] = None
    last_response: Optional[datetime] = None
    engagement_score: float = 0.0
    engagement_level: EngagementLevel = EngagementLevel.MEDIUM
    response_velocity: float = 0.0  # Hours to respond
    preferred_channel: FollowUpType = FollowUpType.EMAIL
    optimal_contact_times: List[int] = field(default_factory=list)  # Hours of day
    engagement_trend: str = "stable"  # increasing, decreasing, stable

@dataclass
class FollowUpStrategy:
    """Follow-up strategy configuration"""
    strategy_id: str
    name: str
    description: str
    trigger_conditions: List[str]
    follow_up_sequence: List[Dict[str, Any]]
    success_rate: float
    target_engagement_level: EngagementLevel

class SmartScheduler:
    """Smart scheduling algorithm for follow-ups"""
    
    def __init__(self):
        self.timing_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.response_predictor = RandomForestRegressor(n_estimators=50, random_state=42)
        self.model_trained = False
        
    async def calculate_optimal_timing(self, lead_data: Dict[str, Any], 
                                     engagement: EngagementTracker,
                                     follow_up_type: FollowUpType) -> datetime:
        """Calculate optimal timing for follow-up"""
        
        current_time = datetime.now(datetime.UTC)
        
        # Base timing rules by follow-up type
        base_delays = {
            FollowUpType.EMAIL: timedelta(hours=24),
            FollowUpType.PHONE_CALL: timedelta(hours=4),
            FollowUpType.SMS: timedelta(hours=2),
            FollowUpType.LINKEDIN_MESSAGE: timedelta(hours=48),
            FollowUpType.DIRECT_MAIL: timedelta(days=7),
            FollowUpType.TASK_REMINDER: timedelta(hours=1)
        }
        
        base_delay = base_delays.get(follow_up_type, timedelta(hours=24))
        
        # Adjust based on engagement level
        engagement_multipliers = {
            EngagementLevel.VERY_HIGH: 0.25,  # Follow up quickly
            EngagementLevel.HIGH: 0.5,
            EngagementLevel.MEDIUM: 1.0,
            EngagementLevel.LOW: 2.0,
            EngagementLevel.VERY_LOW: 4.0,
            EngagementLevel.DORMANT: 7.0  # Space out follow-ups
        }
        
        multiplier = engagement_multipliers.get(engagement.engagement_level, 1.0)
        adjusted_delay = base_delay * multiplier
        
        # Adjust for optimal contact times
        optimal_time = current_time + adjusted_delay
        
        if engagement.optimal_contact_times:
            # Find nearest optimal hour
            target_hour = min(engagement.optimal_contact_times, 
                            key=lambda h: abs(h - optimal_time.hour))
            optimal_time = optimal_time.replace(hour=target_hour, minute=0, second=0)
        
        # Ensure business hours for calls
        if follow_up_type == FollowUpType.PHONE_CALL:
            if optimal_time.hour < 9:
                optimal_time = optimal_time.replace(hour=9)
            elif optimal_time.hour > 17:
                optimal_time = optimal_time.replace(hour=9) + timedelta(days=1)
        
        # Avoid weekends for business communications
        while optimal_time.weekday() >= 5:
            optimal_time += timedelta(days=1)
        
        return optimal_time

class FollowUpAutomationEngine:
    """AI-powered follow-up automation engine"""
    
    def __init__(self, redis_client=None):
        self.redis_client = redis_client or redis.Redis(decode_responses=True)
        
        # Components
        self.scheduler = SmartScheduler()
        
        # Follow-up strategies
        self.strategies = {}
        
        # Engagement tracking
        self.engagement_trackers = {}
        
        # Active follow-up sequences
        self.active_sequences = {}
        
        # Initialize system
        self._initialize_strategies()
        
        logger.info("Follow-up Automation Engine initialized")
    
    def _initialize_strategies(self):
        """Initialize follow-up strategies"""
        
        self.strategies = {
            "high_engagement_nurture": FollowUpStrategy(
                strategy_id="high_engagement_nurture",
                name="High Engagement Nurturing",
                description="Aggressive follow-up for highly engaged leads",
                trigger_conditions=["engagement_level:very_high", "engagement_level:high"],
                follow_up_sequence=[
                    {"type": "phone_call", "delay_hours": 2, "content_template": "immediate_call"},
                    {"type": "email", "delay_hours": 24, "content_template": "follow_up_email"},
                    {"type": "phone_call", "delay_hours": 72, "content_template": "second_call"},
                    {"type": "email", "delay_hours": 168, "content_template": "weekly_check_in"}
                ],
                success_rate=0.45,
                target_engagement_level=EngagementLevel.VERY_HIGH
            ),
            
            "medium_engagement_steady": FollowUpStrategy(
                strategy_id="medium_engagement_steady",
                name="Medium Engagement Steady",
                description="Consistent follow-up for moderately engaged leads",
                trigger_conditions=["engagement_level:medium"],
                follow_up_sequence=[
                    {"type": "email", "delay_hours": 24, "content_template": "educational_follow_up"},
                    {"type": "phone_call", "delay_hours": 96, "content_template": "check_in_call"},
                    {"type": "email", "delay_hours": 168, "content_template": "value_proposition"},
                    {"type": "phone_call", "delay_hours": 336, "content_template": "closing_call"}
                ],
                success_rate=0.28,
                target_engagement_level=EngagementLevel.MEDIUM
            ),
            
            "low_engagement_revival": FollowUpStrategy(
                strategy_id="low_engagement_revival",
                name="Low Engagement Revival",
                description="Re-engagement strategy for low-engagement leads",
                trigger_conditions=["engagement_level:low", "engagement_level:very_low"],
                follow_up_sequence=[
                    {"type": "email", "delay_hours": 72, "content_template": "re_engagement"},
                    {"type": "linkedin_message", "delay_hours": 168, "content_template": "linkedin_touch"},
                    {"type": "email", "delay_hours": 336, "content_template": "last_chance"},
                    {"type": "direct_mail", "delay_hours": 504, "content_template": "physical_mailer"}
                ],
                success_rate=0.12,
                target_engagement_level=EngagementLevel.LOW
            ),
            
            "dormant_reactivation": FollowUpStrategy(
                strategy_id="dormant_reactivation",
                name="Dormant Lead Reactivation",
                description="Long-term reactivation for dormant leads",
                trigger_conditions=["engagement_level:dormant"],
                follow_up_sequence=[
                    {"type": "email", "delay_hours": 168, "content_template": "market_update"},
                    {"type": "email", "delay_hours": 720, "content_template": "new_product_announcement"},
                    {"type": "email", "delay_hours": 1440, "content_template": "annual_review_invitation"}
                ],
                success_rate=0.08,
                target_engagement_level=EngagementLevel.DORMANT
            ),
            
            "post_call_sequence": FollowUpStrategy(
                strategy_id="post_call_sequence",
                name="Post-Call Follow-up",
                description="Structured follow-up after sales calls",
                trigger_conditions=["trigger_event:call_completed"],
                follow_up_sequence=[
                    {"type": "email", "delay_hours": 2, "content_template": "call_recap"},
                    {"type": "email", "delay_hours": 48, "content_template": "additional_resources"},
                    {"type": "phone_call", "delay_hours": 120, "content_template": "decision_check_in"},
                    {"type": "email", "delay_hours": 240, "content_template": "proposal_follow_up"}
                ],
                success_rate=0.35,
                target_engagement_level=EngagementLevel.HIGH
            )
        }
    
    async def track_engagement(self, lead_id: str, event_type: str, 
                             event_data: Dict[str, Any] = None):
        """Track lead engagement and update follow-up scheduling"""
        
        try:
            # Get or create engagement tracker
            tracker = await self._get_engagement_tracker(lead_id)
            if not tracker:
                tracker = EngagementTracker(lead_id=lead_id)
            
            current_time = datetime.now(datetime.UTC)
            
            # Update engagement based on event type
            if event_type == "email_open":
                tracker.last_email_open = current_time
                tracker.engagement_score += 2
            elif event_type == "email_click":
                tracker.last_email_click = current_time
                tracker.engagement_score += 5
            elif event_type == "website_visit":
                tracker.last_website_visit = current_time
                tracker.engagement_score += 3
            elif event_type == "phone_call_answered":
                tracker.last_phone_call = current_time
                tracker.engagement_score += 10
            elif event_type == "response_received":
                tracker.last_response = current_time
                tracker.engagement_score += 8
                # Calculate response velocity
                if event_data and 'sent_time' in event_data:
                    sent_time = datetime.fromisoformat(event_data['sent_time'])
                    response_hours = (current_time - sent_time).total_seconds() / 3600
                    tracker.response_velocity = response_hours
            
            # Update engagement level
            tracker.engagement_level = await self._calculate_engagement_level(tracker)
            
            # Update engagement trend
            tracker.engagement_trend = await self._calculate_engagement_trend(tracker)
            
            # Store updated tracker
            await self._store_engagement_tracker(tracker)
            
            # Trigger follow-up adjustments
            await self._adjust_follow_ups(tracker, event_type, event_data)
            
            logger.debug(f"Tracked engagement: {event_type} for lead {lead_id}")
            
        except Exception as e:
            logger.error(f"Error tracking engagement: {e}")
    
    async def schedule_follow_up_sequence(self, lead_id: str, trigger_event: str,
                                        lead_data: Dict[str, Any] = None,
                                        scoring_result: Dict[str, Any] = None) -> List[str]:
        """Schedule a complete follow-up sequence for a lead"""
        
        try:
            # Get engagement tracker
            tracker = await self._get_engagement_tracker(lead_id)
            if not tracker:
                tracker = EngagementTracker(lead_id=lead_id)
                await self._store_engagement_tracker(tracker)
            
            # Determine appropriate strategy
            strategy = await self._select_follow_up_strategy(tracker, trigger_event, lead_data, scoring_result)
            
            if not strategy:
                logger.warning(f"No suitable follow-up strategy found for lead {lead_id}")
                return []
            
            # Generate follow-up actions
            follow_up_actions = []
            current_time = datetime.now(datetime.UTC)
            
            for i, step in enumerate(strategy.follow_up_sequence):
                # Calculate timing
                delay = timedelta(hours=step['delay_hours'])
                scheduled_time = await self.scheduler.calculate_optimal_timing(
                    lead_data or {}, tracker, FollowUpType(step['type'])
                )
                
                # Adjust for sequence timing
                if i > 0:
                    scheduled_time = max(scheduled_time, current_time + delay)
                
                # Create follow-up action
                action = FollowUpAction(
                    action_id=f"followup_{lead_id}_{int(scheduled_time.timestamp())}_{i}",
                    lead_id=lead_id,
                    follow_up_type=FollowUpType(step['type']),
                    scheduled_time=scheduled_time,
                    priority=await self._determine_priority(tracker, step, scoring_result),
                    content=await self._generate_follow_up_content(step, lead_data, tracker),
                    trigger_event=trigger_event,
                    expected_response_rate=strategy.success_rate * (0.9 ** i),  # Diminishing returns
                    rep_id=lead_data.get('assigned_rep_id', 'auto') if lead_data else 'auto'
                )
                
                follow_up_actions.append(action)
                
                # Link actions in sequence
                if i > 0:
                    follow_up_actions[i-1].next_action_id = action.action_id
            
            # Store follow-up actions
            action_ids = []
            for action in follow_up_actions:
                await self._store_follow_up_action(action)
                action_ids.append(action.action_id)
            
            # Store sequence reference
            sequence_id = f"seq_{lead_id}_{int(current_time.timestamp())}"
            await self._store_follow_up_sequence(sequence_id, action_ids, strategy.strategy_id)
            
            logger.info(f"Scheduled {len(follow_up_actions)} follow-up actions for lead {lead_id} using strategy {strategy.strategy_id}")
            
            return action_ids
            
        except Exception as e:
            logger.error(f"Error scheduling follow-up sequence: {e}")
            raise
    
    async def _select_follow_up_strategy(self, tracker: EngagementTracker, 
                                       trigger_event: str,
                                       lead_data: Dict[str, Any] = None,
                                       scoring_result: Dict[str, Any] = None) -> Optional[FollowUpStrategy]:
        """Select the most appropriate follow-up strategy"""
        
        # Check for specific trigger-based strategies first
        if trigger_event == "call_completed":
            return self.strategies.get("post_call_sequence")
        
        # Select based on engagement level
        engagement_strategies = {
            EngagementLevel.VERY_HIGH: "high_engagement_nurture",
            EngagementLevel.HIGH: "high_engagement_nurture",
            EngagementLevel.MEDIUM: "medium_engagement_steady",
            EngagementLevel.LOW: "low_engagement_revival",
            EngagementLevel.VERY_LOW: "low_engagement_revival",
            EngagementLevel.DORMANT: "dormant_reactivation"
        }
        
        strategy_id = engagement_strategies.get(tracker.engagement_level)
        return self.strategies.get(strategy_id) if strategy_id else None
    
    async def _calculate_engagement_level(self, tracker: EngagementTracker) -> EngagementLevel:
        """Calculate engagement level based on recent activity"""
        
        current_time = datetime.now(datetime.UTC)
        
        # Check for recent activity (last 7 days)
        recent_activity_score = 0
        
        if tracker.last_email_open and (current_time - tracker.last_email_open).days <= 7:
            recent_activity_score += 10
        if tracker.last_email_click and (current_time - tracker.last_email_click).days <= 7:
            recent_activity_score += 20
        if tracker.last_website_visit and (current_time - tracker.last_website_visit).days <= 7:
            recent_activity_score += 15
        if tracker.last_phone_call and (current_time - tracker.last_phone_call).days <= 7:
            recent_activity_score += 30
        if tracker.last_response and (current_time - tracker.last_response).days <= 7:
            recent_activity_score += 25
        
        # Combine with overall engagement score
        total_score = (tracker.engagement_score * 0.7) + (recent_activity_score * 0.3)
        
        # Determine level
        if total_score >= 80:
            return EngagementLevel.VERY_HIGH
        elif total_score >= 60:
            return EngagementLevel.HIGH
        elif total_score >= 40:
            return EngagementLevel.MEDIUM
        elif total_score >= 20:
            return EngagementLevel.LOW
        elif total_score >= 10:
            return EngagementLevel.VERY_LOW
        else:
            return EngagementLevel.DORMANT
    
    async def execute_follow_up_action(self, action_id: str) -> Dict[str, Any]:
        """Execute a scheduled follow-up action"""
        
        try:
            # Get follow-up action
            action = await self._get_follow_up_action(action_id)
            if not action:
                return {"error": "Follow-up action not found"}
            
            if action.is_completed:
                return {"error": "Follow-up action already completed"}
            
            # Execute based on type
            result = {}
            
            if action.follow_up_type == FollowUpType.EMAIL:
                result = await self._send_follow_up_email(action)
            elif action.follow_up_type == FollowUpType.PHONE_CALL:
                result = await self._schedule_follow_up_call(action)
            elif action.follow_up_type == FollowUpType.SMS:
                result = await self._send_follow_up_sms(action)
            elif action.follow_up_type == FollowUpType.LINKEDIN_MESSAGE:
                result = await self._send_linkedin_message(action)
            elif action.follow_up_type == FollowUpType.TASK_REMINDER:
                result = await self._create_task_reminder(action)
            
            # Mark as completed
            action.is_completed = True
            action.completion_time = datetime.now(datetime.UTC)
            action.outcome = result.get('status', 'completed')
            
            await self._store_follow_up_action(action)
            
            # Schedule next action in sequence if exists
            if action.next_action_id:
                await self._activate_next_action(action.next_action_id)
            
            logger.info(f"Executed follow-up action {action_id}: {action.follow_up_type.value}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error executing follow-up action: {e}")
            return {"error": str(e)}
    
    async def get_follow_up_dashboard(self, rep_id: str = None, 
                                    date_range: Tuple[datetime, datetime] = None) -> Dict[str, Any]:
        """Get follow-up dashboard data"""
        
        try:
            current_time = datetime.now(datetime.UTC)
            
            if not date_range:
                start_date = current_time - timedelta(days=30)
                end_date = current_time
            else:
                start_date, end_date = date_range
            
            # Get follow-up actions in date range
            actions = await self._get_follow_up_actions_in_range(start_date, end_date, rep_id)
            
            # Calculate metrics
            total_actions = len(actions)
            completed_actions = len([a for a in actions if a.is_completed])
            pending_actions = total_actions - completed_actions
            
            # Response rates by type
            response_rates = {}
            for follow_up_type in FollowUpType:
                type_actions = [a for a in actions if a.follow_up_type == follow_up_type]
                if type_actions:
                    responded = len([a for a in type_actions if a.outcome == 'responded'])
                    response_rates[follow_up_type.value] = responded / len(type_actions) * 100
            
            # Upcoming actions (next 7 days)
            upcoming_cutoff = current_time + timedelta(days=7)
            upcoming_actions = await self._get_upcoming_actions(current_time, upcoming_cutoff, rep_id)
            
            # Overdue actions
            overdue_actions = await self._get_overdue_actions(current_time, rep_id)
            
            dashboard = {
                'summary': {
                    'total_actions': total_actions,
                    'completed_actions': completed_actions,
                    'pending_actions': pending_actions,
                    'completion_rate': (completed_actions / total_actions * 100) if total_actions > 0 else 0,
                    'overdue_count': len(overdue_actions)
                },
                'response_rates': response_rates,
                'upcoming_actions': [
                    {
                        'action_id': a.action_id,
                        'lead_id': a.lead_id,
                        'type': a.follow_up_type.value,
                        'scheduled_time': a.scheduled_time.isoformat(),
                        'priority': a.priority.value,
                        'content_preview': a.content[:100] + '...' if len(a.content) > 100 else a.content
                    }
                    for a in upcoming_actions[:10]  # Top 10 upcoming
                ],
                'overdue_actions': [
                    {
                        'action_id': a.action_id,
                        'lead_id': a.lead_id,
                        'type': a.follow_up_type.value,
                        'scheduled_time': a.scheduled_time.isoformat(),
                        'days_overdue': (current_time - a.scheduled_time).days
                    }
                    for a in overdue_actions[:5]  # Top 5 overdue
                ]
            }
            
            return dashboard
            
        except Exception as e:
            logger.error(f"Error generating follow-up dashboard: {e}")
            return {"error": str(e)}

# Global follow-up automation engine instance
followup_engine = FollowUpAutomationEngine()