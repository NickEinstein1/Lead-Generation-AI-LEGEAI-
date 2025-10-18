"""
Strategic Growth Initiatives Module

Comprehensive multi-channel lead generation system including:
- Social Media Integration (LinkedIn, Facebook, Instagram, Twitter/X)
- Content Marketing Engine (Blog, Webinar, Whitepaper lead capture)
- Partner/Referral System (Track and score referral leads)
- Event Management (Trade show, webinar lead processing)
"""

from .social_media_integration import (
    social_media_processor, social_media_analytics,
    SocialMediaLeadProcessor, SocialMediaAnalytics,
    LinkedInConnector, FacebookConnector
)
from .content_marketing_engine import (
    content_marketing_engine, ContentMarketingEngine,
    ContentAnalytics
)
from .partner_referral_system import (
    partner_referral_system, PartnerReferralSystem,
    PartnerAnalytics
)
from .event_management import (
    event_management_system, EventManagementSystem,
    EventAnalytics
)

__all__ = [
    # Global instances
    'social_media_processor',
    'social_media_analytics',
    'content_marketing_engine',
    'partner_referral_system',
    'event_management_system',
    
    # Classes
    'SocialMediaLeadProcessor',
    'SocialMediaAnalytics',
    'LinkedInConnector',
    'FacebookConnector',
    'ContentMarketingEngine',
    'ContentAnalytics',
    'PartnerReferralSystem',
    'PartnerAnalytics',
    'EventManagementSystem',
    'EventAnalytics'
]

__version__ = "1.0.0"
__author__ = "Insurance Lead Scoring Platform Growth Team"