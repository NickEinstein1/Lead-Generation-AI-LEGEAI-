"""
Sales AI Coordinator

Central coordination system that orchestrates all AI-powered sales automation
components and provides unified insights and recommendations.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import redis

from .intelligent_nurturing import nurturing_engine, NurturingStrategy
from .call_optimization import call_optimizer, CallType
from .follow_up_automation import followup_engine, FollowUpType
from .cross_sell_engine import cross_sell_engine, OpportunityType

logger = logging.getLogger(__name__)

class ActionType(Enum):
    SEND_EMAIL = "send_email"
    SCHEDULE_CALL = "schedule_call"
    CREATE_TASK = "create_task"
    UPDATE_NURTURING = "update_nurturing"
    SCHEDULE_FOLLOWUP = "schedule_followup"
    GENERATE_QUOTE = "generate_quote"
    ESCALATE_TO_MANAGER = "escalate_to_manager"

class InsightType(Enum):
    ENGAGEMENT_TREND = "engagement_trend"
    CONVERSION_OPPORTUNITY = "conversion_opportunity"
    RISK_ALERT = "risk_alert"
    REVENUE_OPPORTUNITY = "revenue_opportunity"
    PERFORMANCE_INSIGHT = "performance_insight"

@dataclass
class ActionRecommendation:
    """AI-generated action recommendation"""
    action_id: str
    action_type: ActionType
    priority: int  # 1-10 scale
    confidence: float
    description: str
    reasoning: str
    expected_outcome: str
    effort_required: str
    timeline: str
    success_probability: float
    revenue_impact: float
    parameters: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SalesInsight:
    """AI-generated sales insight"""
    insight_id: str
    insight_type: InsightType
    title: str
    description: str
    impact_level: str  # high, medium, low
    confidence: float
    data_points: List[str] = field(default_factory=list)
    recommended_actions: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)

@dataclass
class LeadSalesProfile:
    """Comprehensive sales profile for a lead"""
    lead_id: str
    overall_score: float
    conversion_probability: float
    revenue_potential: float
    engagement_level: str
    nurturing_stage: str
    last_interaction: Optional[datetime]
    next_best_action: str
    risk_factors: List[str] = field(default_factory=list)
    opportunities: List[str] = field(default_factory=list)
    sales_velocity: float = 0.0
    competitive_threats: List[str] = field(default_factory=list)

class PerformanceOptimizer:
    """Performance optimization and analytics"""
    
    def __init__(self):
        self.performance_metrics = {}
        self.optimization_rules = {}
        
        self._initialize_optimization_rules()
    
    def _initialize_optimization_rules(self):
        """Initialize performance optimization rules"""
        
        self.optimization_rules = {
            "low_email_open_rate": {
                "condition": "email_open_rate < 0.20",
                "action": "optimize_subject_lines",
                "priority": 8
            },
            "high_engagement_no_call": {
                "condition": "engagement_score > 70 AND last_call > 7_days",
                "action": "schedule_immediate_call",
                "priority": 9
            },
            "stalled_nurturing": {
                "condition": "nurturing_stage_duration > 14_days AND no_progression",
                "action": "change_nurturing_strategy",
                "priority": 7
            },
            "high_value_at_risk": {
                "condition": "revenue_potential > 5000 AND engagement_declining",
                "action": "escalate_to_senior_rep",
                "priority": 10
            }
        }
    
    async def analyze_performance(self, lead_profiles: List[LeadSalesProfile]) -> List[SalesInsight]:
        """Analyze sales performance and generate insights"""
        
        insights = []
        
        # Analyze engagement trends
        engagement_insight = await self._analyze_engagement_trends(lead_profiles)
        if engagement_insight:
            insights.append(engagement_insight)
        
        # Identify conversion opportunities
        conversion_insights = await self._identify_conversion_opportunities(lead_profiles)
        insights.extend(conversion_insights)
        
        # Detect risk factors
        risk_insights = await self._detect_risk_factors(lead_profiles)
        insights.extend(risk_insights)
        
        # Revenue optimization opportunities
        revenue_insights = await self._identify_revenue_opportunities(lead_profiles)
        insights.extend(revenue_insights)
        
        return insights
    
    async def _analyze_engagement_trends(self, lead_profiles: List[LeadSalesProfile]) -> Optional[SalesInsight]:
        """Analyze overall engagement trends"""
        
        if not lead_profiles:
            return None
        
        # Calculate engagement metrics
        high_engagement = len([p for p in lead_profiles if p.engagement_level == "high"])
        total_leads = len(lead_profiles)
        engagement_rate = high_engagement / total_leads if total_leads > 0 else 0
        
        # Determine trend
        if engagement_rate > 0.6:
            impact = "high"
            description = f"Excellent engagement: {engagement_rate:.1%} of leads are highly engaged"
        elif engagement_rate > 0.4:
            impact = "medium"
            description = f"Good engagement: {engagement_rate:.1%} of leads are highly engaged"
        else:
            impact = "high"
            description = f"Low engagement alert: Only {engagement_rate:.1%} of leads are highly engaged"
        
        return SalesInsight(
            insight_id=f"engagement_trend_{int(datetime.utcnow().timestamp())}",
            insight_type=InsightType.ENGAGEMENT_TREND,
            title="Lead Engagement Analysis",
            description=description,
            impact_level=impact,
            confidence=0.85,
            data_points=[f"Total leads: {total_leads}", f"High engagement: {high_engagement}"],
            recommended_actions=["optimize_nurturing_sequences", "review_content_strategy"]
        )

class SalesAICoordinator:
    """Central AI coordinator for sales automation"""
    
    def __init__(self, redis_client=None):
        self.redis_client = redis_client or redis.Redis(decode_responses=True)
        
        # Component references
        self.nurturing_engine = nurturing_engine
        self.call_optimizer = call_optimizer
        self.followup_engine = followup_engine
        self.cross_sell_engine = cross_sell_engine
        
        # Performance optimizer
        self.performance_optimizer = PerformanceOptimizer()
        
        # Active lead profiles
        self.lead_profiles = {}
        
        # Coordination rules
        self.coordination_rules = {}
        
        # Initialize system
        self._initialize_coordination_rules()
        
        logger.info("Sales AI Coordinator initialized")
    
    def _initialize_coordination_rules(self):
        """Initialize coordination rules between components"""
        
        self.coordination_rules = {
            "high_engagement_acceleration": {
                "trigger": "engagement_score > 80",
                "actions": [
                    {"component": "call_optimizer", "action": "schedule_immediate_call"},
                    {"component": "nurturing_engine", "action": "switch_to_urgency_strategy"},
                    {"component": "cross_sell_engine", "action": "identify_immediate_opportunities"}
                ]
            },
            "low_engagement_revival": {
                "trigger": "engagement_score < 30 AND last_interaction > 7_days",
                "actions": [
                    {"component": "nurturing_engine", "action": "switch_to_revival_strategy"},
                    {"component": "followup_engine", "action": "schedule_re_engagement_sequence"}
                ]
            },
            "call_outcome_optimization": {
                "trigger": "call_completed",
                "actions": [
                    {"component": "followup_engine", "action": "schedule_post_call_sequence"},
                    {"component": "nurturing_engine", "action": "adjust_based_on_call_outcome"},
                    {"component": "cross_sell_engine", "action": "update_opportunities_from_call"}
                ]
            }
        }
    
    async def orchestrate_lead_journey(self, lead_id: str, lead_data: Dict[str, Any], 
                                     scoring_result: Dict[str, Any]) -> Dict[str, Any]:
        """Orchestrate the complete AI-powered lead journey"""
        
        try:
            # Create comprehensive lead profile
            lead_profile = await self._create_lead_sales_profile(lead_id, lead_data, scoring_result)
            
            # Generate recommendations from all components
            recommendations = await self._generate_unified_recommendations(lead_profile, lead_data, scoring_result)
            
            # Coordinate actions between components
            coordinated_actions = await self._coordinate_actions(recommendations, lead_profile)
            
            # Execute high-priority actions
            execution_results = await self._execute_priority_actions(coordinated_actions)
            
            # Generate insights
            insights = await self._generate_lead_insights(lead_profile, recommendations)
            
            # Store updated profile
            await self._store_lead_profile(lead_profile)
            
            result = {
                "lead_id": lead_id,
                "lead_profile": {
                    "overall_score": lead_profile.overall_score,
                    "conversion_probability": lead_profile.conversion_probability,
                    "revenue_potential": lead_profile.revenue_potential,
                    "engagement_level": lead_profile.engagement_level,
                    "next_best_action": lead_profile.next_best_action
                },
                "recommendations": [
                    {
                        "action_id": rec.action_id,
                        "action_type": rec.action_type.value,
                        "priority": rec.priority,
                        "description": rec.description,
                        "timeline": rec.timeline,
                        "success_probability": rec.success_probability
                    }
                    for rec in coordinated_actions[:5]  # Top 5 recommendations
                ],
                "insights": [
                    {
                        "insight_type": insight.insight_type.value,
                        "title": insight.title,
                        "description": insight.description,
                        "impact_level": insight.impact_level
                    }
                    for insight in insights
                ],
                "execution_results": execution_results,
                "coordination_summary": {
                    "nurturing_sequence_created": "nurturing_sequence" in execution_results,
                    "call_recommendation_generated": "call_recommendation" in execution_results,
                    "follow_up_scheduled": "follow_up_sequence" in execution_results,
                    "cross_sell_opportunities": len([r for r in coordinated_actions if "cross_sell" in r.description.lower()])
                }
            }
            
            logger.info(f"Orchestrated complete lead journey for {lead_id}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error orchestrating lead journey: {e}")
            raise
    
    async def _generate_unified_recommendations(self, lead_profile: LeadSalesProfile,
                                              lead_data: Dict[str, Any],
                                              scoring_result: Dict[str, Any]) -> List[ActionRecommendation]:
        """Generate unified recommendations from all AI components"""
        
        recommendations = []
        
        # Nurturing recommendations
        if lead_profile.engagement_level in ["low", "medium"]:
            nurturing_rec = ActionRecommendation(
                action_id=f"nurture_{lead_profile.lead_id}_{int(datetime.utcnow().timestamp())}",
                action_type=ActionType.UPDATE_NURTURING,
                priority=7,
                confidence=0.85,
                description="Create personalized nurturing sequence",
                reasoning="Lead shows potential but needs nurturing to increase engagement",
                expected_outcome="Increased engagement and conversion probability",
                effort_required="low",
                timeline="immediate",
                success_probability=0.65,
                revenue_impact=lead_profile.revenue_potential * 0.3,
                parameters={"strategy": "educational", "lead_data": lead_data, "scoring_result": scoring_result}
            )
            recommendations.append(nurturing_rec)
        
        # Call recommendations
        if lead_profile.engagement_level in ["high", "very_high"] or lead_profile.conversion_probability > 0.7:
            call_rec = ActionRecommendation(
                action_id=f"call_{lead_profile.lead_id}_{int(datetime.utcnow().timestamp())}",
                action_type=ActionType.SCHEDULE_CALL,
                priority=9,
                confidence=0.90,
                description="Schedule high-priority sales call",
                reasoning="Lead shows high engagement and conversion potential",
                expected_outcome="Direct conversion opportunity",
                effort_required="medium",
                timeline="within 24 hours",
                success_probability=0.75,
                revenue_impact=lead_profile.revenue_potential * 0.8,
                parameters={"call_type": "conversion", "lead_data": lead_data, "scoring_result": scoring_result}
            )
            recommendations.append(call_rec)
        
        # Cross-sell recommendations
        if lead_profile.revenue_potential > 2000:
            cross_sell_rec = ActionRecommendation(
                action_id=f"cross_sell_{lead_profile.lead_id}_{int(datetime.utcnow().timestamp())}",
                action_type=ActionType.GENERATE_QUOTE,
                priority=6,
                confidence=0.70,
                description="Identify cross-sell opportunities",
                reasoning="High-value lead with potential for additional products",
                expected_outcome="Increased revenue per customer",
                effort_required="low",
                timeline="within 48 hours",
                success_probability=0.45,
                revenue_impact=lead_profile.revenue_potential * 0.4,
                parameters={"customer_data": lead_data}
            )
            recommendations.append(cross_sell_rec)
        
        # Follow-up recommendations
        follow_up_rec = ActionRecommendation(
            action_id=f"followup_{lead_profile.lead_id}_{int(datetime.utcnow().timestamp())}",
            action_type=ActionType.SCHEDULE_FOLLOWUP,
            priority=5,
            confidence=0.80,
            description="Schedule intelligent follow-up sequence",
            reasoning="Systematic follow-up increases conversion rates",
            expected_outcome="Maintained engagement and conversion tracking",
            effort_required="low",
            timeline="ongoing",
            success_probability=0.60,
            revenue_impact=lead_profile.revenue_potential * 0.2,
            parameters={"trigger_event": "lead_scored", "lead_data": lead_data}
        )
        recommendations.append(follow_up_rec)
        
        return recommendations
    
    async def _execute_priority_actions(self, actions: List[ActionRecommendation]) -> Dict[str, Any]:
        """Execute high-priority actions automatically"""
        
        results = {}
        
        for action in sorted(actions, key=lambda x: x.priority, reverse=True)[:3]:  # Top 3 actions
            try:
                if action.action_type == ActionType.UPDATE_NURTURING:
                    sequence = await self.nurturing_engine.create_nurturing_sequence(
                        action.parameters.get("lead_data", {}),
                        action.parameters.get("scoring_result", {}),
                        strategy=action.parameters.get("strategy", "educational")
                    )
                    results["nurturing_sequence"] = sequence.sequence_id
                
                elif action.action_type == ActionType.SCHEDULE_CALL:
                    recommendation = await self.call_optimizer.generate_call_recommendation(
                        action.parameters.get("lead_data", {}),
                        action.parameters.get("scoring_result", {}),
                        CallType.INITIAL_CONTACT
                    )
                    results["call_recommendation"] = recommendation.recommendation_id
                
                elif action.action_type == ActionType.SCHEDULE_FOLLOWUP:
                    follow_up_ids = await self.followup_engine.schedule_follow_up_sequence(
                        action.parameters.get("lead_data", {}).get("lead_id", "unknown"),
                        action.parameters.get("trigger_event", "lead_scored"),
                        action.parameters.get("lead_data", {})
                    )
                    results["follow_up_sequence"] = follow_up_ids
                
                elif action.action_type == ActionType.GENERATE_QUOTE:
                    recommendations = await self.cross_sell_engine.generate_recommendations(
                        action.parameters.get("customer_data", {}).get("lead_id", "unknown"),
                        action.parameters.get("customer_data", {})
                    )
                    results["cross_sell_recommendations"] = [r.product_id for r in recommendations]
                
            except Exception as e:
                logger.error(f"Error executing action {action.action_id}: {e}")
                results[f"error_{action.action_id}"] = str(e)
        
        return results

# Global sales AI coordinator instance
sales_ai_coordinator = SalesAICoordinator()
