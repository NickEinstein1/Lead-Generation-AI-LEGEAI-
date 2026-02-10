"""
Meta (Facebook) Marketing API Integration

This module provides integration with Meta's Marketing API for:
- Lead Ads (capture leads from Facebook/Instagram)
- Ads Management (create, read, update campaigns)
- Catalog Management (sync products for dynamic ads)
- Business Management (manage business assets)
- Insights & Analytics (ad performance metrics)
"""

from .client import MetaAPIClient
from .auth import MetaOAuthHandler
from .webhooks import MetaWebhookHandler
from .lead_sync import MetaLeadSync

__all__ = [
    "MetaAPIClient",
    "MetaOAuthHandler",
    "MetaWebhookHandler",
    "MetaLeadSync",
]

