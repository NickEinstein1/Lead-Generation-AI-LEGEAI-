"""
Meta OAuth Authentication Handler

Handles OAuth 2.0 flow for Meta (Facebook) authentication
"""

import os
import logging
from typing import Dict, List, Optional
import httpx
from urllib.parse import urlencode

logger = logging.getLogger(__name__)

# Meta OAuth Configuration
META_APP_ID = os.getenv("META_APP_ID", "")
META_APP_SECRET = os.getenv("META_APP_SECRET", "")
META_REDIRECT_URI = os.getenv("META_REDIRECT_URI", "http://localhost:8000/v1/integrations/meta/callback")

# OAuth URLs
META_OAUTH_DIALOG_URL = "https://www.facebook.com/v21.0/dialog/oauth"
META_OAUTH_TOKEN_URL = "https://graph.facebook.com/v21.0/oauth/access_token"


class MetaOAuthHandler:
    """Handles Meta OAuth 2.0 authentication flow"""
    
    def __init__(self, app_id: Optional[str] = None, app_secret: Optional[str] = None):
        """
        Initialize OAuth handler
        
        Args:
            app_id: Meta App ID (defaults to env var)
            app_secret: Meta App Secret (defaults to env var)
        """
        self.app_id = app_id or META_APP_ID
        self.app_secret = app_secret or META_APP_SECRET
        self.redirect_uri = META_REDIRECT_URI
        
        if not self.app_id or not self.app_secret:
            logger.warning("Meta App ID or Secret not configured")
    
    def get_authorization_url(
        self,
        state: Optional[str] = None,
        scopes: Optional[List[str]] = None
    ) -> str:
        """
        Generate OAuth authorization URL
        
        Args:
            state: Optional state parameter for CSRF protection
            scopes: List of permission scopes to request
            
        Returns:
            Authorization URL to redirect user to
        """
        if scopes is None:
            # Default scopes based on your permissions
            scopes = [
                "ads_management",
                "ads_read",
                "business_management",
                "catalog_management",
                "email",
                "pages_manage_ads",
                "pages_read_engagement",
                "pages_show_list",
                "public_profile",
            ]
        
        params = {
            "client_id": self.app_id,
            "redirect_uri": self.redirect_uri,
            "scope": ",".join(scopes),
            "response_type": "code",
        }
        
        if state:
            params["state"] = state
        
        return f"{META_OAUTH_DIALOG_URL}?{urlencode(params)}"
    
    async def exchange_code_for_token(self, code: str) -> Dict[str, str]:
        """
        Exchange authorization code for access token
        
        Args:
            code: Authorization code from OAuth callback
            
        Returns:
            Dictionary with 'access_token' and 'token_type'
            
        Raises:
            Exception: If token exchange fails
        """
        params = {
            "client_id": self.app_id,
            "client_secret": self.app_secret,
            "redirect_uri": self.redirect_uri,
            "code": code,
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.get(META_OAUTH_TOKEN_URL, params=params)
            response.raise_for_status()
            data = response.json()
            
            if "access_token" not in data:
                raise Exception("No access token in response")
            
            return {
                "access_token": data["access_token"],
                "token_type": data.get("token_type", "bearer"),
                "expires_in": data.get("expires_in"),
            }
    
    async def get_long_lived_token(self, short_lived_token: str) -> Dict[str, str]:
        """
        Exchange short-lived token for long-lived token (60 days)
        
        Args:
            short_lived_token: Short-lived access token
            
        Returns:
            Dictionary with long-lived 'access_token'
        """
        params = {
            "grant_type": "fb_exchange_token",
            "client_id": self.app_id,
            "client_secret": self.app_secret,
            "fb_exchange_token": short_lived_token,
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.get(META_OAUTH_TOKEN_URL, params=params)
            response.raise_for_status()
            data = response.json()
            
            return {
                "access_token": data["access_token"],
                "token_type": data.get("token_type", "bearer"),
                "expires_in": data.get("expires_in", 5184000),  # 60 days default
            }
    
    async def debug_token(self, access_token: str) -> Dict[str, any]:
        """
        Debug/validate an access token
        
        Args:
            access_token: Token to validate
            
        Returns:
            Token debug information including scopes, expiration, etc.
        """
        params = {
            "input_token": access_token,
            "access_token": f"{self.app_id}|{self.app_secret}",
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://graph.facebook.com/debug_token",
                params=params
            )
            response.raise_for_status()
            return response.json().get("data", {})

