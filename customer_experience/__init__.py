"""
Customer Experience Enhancement System

Comprehensive customer experience optimization platform providing:
- Complete lead journey mapping and analytics
- AI-powered personalized landing pages
- Smart form optimization with adaptive design
- Real-time chat integration with intelligent routing
"""

from .journey_mapping import (
    journey_mapping_engine, JourneyMappingEngine, CustomerJourney,
    Touchpoint, TouchpointType, JourneyStage
)
from .personalized_landing_pages import (
    personalized_pages_engine, PersonalizedLandingPagesEngine,
    PersonalizedPage, ContentVariant, LandingPageTemplate
)
from .smart_form_optimization import (
    smart_form_engine, SmartFormOptimizationEngine,
    FormConfiguration, FormField, FormSession
)
from .realtime_chat_integration import (
    realtime_chat_engine, RealtimeChatEngine,
    ChatSession, ChatMessage, ChatAgent, ChatTrigger
)

__all__ = [
    # Global engines
    'journey_mapping_engine',
    'personalized_pages_engine', 
    'smart_form_engine',
    'realtime_chat_engine',
    
    # Classes
    'JourneyMappingEngine',
    'PersonalizedLandingPagesEngine',
    'SmartFormOptimizationEngine',
    'RealtimeChatEngine',
    
    # Data classes
    'CustomerJourney',
    'Touchpoint',
    'PersonalizedPage',
    'FormConfiguration',
    'ChatSession',
    
    # Enums
    'TouchpointType',
    'JourneyStage',
    'ChatTrigger'
]

__version__ = "1.0.0"
__author__ = "Insurance Lead Scoring Platform CX Team"