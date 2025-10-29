"""
AI-Powered Sales Automation System

Comprehensive AI-driven sales automation platform providing intelligent lead nurturing,
call optimization, automated follow-ups, and cross-sell/upsell recommendations.

Features:
- Intelligent lead nurturing with dynamic email sequences
- Sales call optimization with timing and talking points
- Automated follow-up scheduling based on engagement
- Cross-sell/upsell engine with opportunity identification
- Behavioral analysis and prediction models
- Real-time sales coaching and recommendations
"""

from .intelligent_nurturing import (
    IntelligentNurturingEngine, NurturingSequence, BehaviorTrigger,
    EmailTemplate, NurturingStrategy
)
from .call_optimization import (
    CallOptimizationEngine, CallRecommendation, TalkingPoint,
    OptimalTimingPredictor, CallOutcomePredictor
)
from .follow_up_automation import (
    FollowUpAutomationEngine, FollowUpAction, EngagementTracker,
    SmartScheduler, FollowUpStrategy
)
from .cross_sell_engine import (
    CrossSellUpsellEngine, OpportunityIdentifier, ProductRecommendation,
    CustomerSegmentation, RevenueOptimizer
)
from .sales_ai_coordinator import (
    SalesAICoordinator, SalesInsight, ActionRecommendation,
    PerformanceOptimizer
)

__all__ = [
    'IntelligentNurturingEngine', 'CallOptimizationEngine', 
    'FollowUpAutomationEngine', 'CrossSellUpsellEngine',
    'SalesAICoordinator'
]