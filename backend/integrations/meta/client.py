"""
Meta Marketing API Client

Handles all interactions with Meta's Marketing API including:
- Ad Account management
- Campaign CRUD operations
- Ad Set and Ad management
- Insights and analytics
- Lead retrieval
"""

import os
import logging
from typing import Dict, List, Optional, Any
import httpx
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# Meta API Configuration
META_API_VERSION = os.getenv("META_API_VERSION", "v21.0")
META_API_BASE = f"https://graph.facebook.com/{META_API_VERSION}"


class MetaAPIError(Exception):
    """Custom exception for Meta API errors"""
    pass


class MetaAPIClient:
    """Client for interacting with Meta Marketing API"""
    
    def __init__(self, access_token: str):
        """
        Initialize Meta API client
        
        Args:
            access_token: Meta access token with required permissions
        """
        self.access_token = access_token
        self.base_url = META_API_BASE
        self.client = httpx.AsyncClient(timeout=30.0)
    
    async def close(self):
        """Close the HTTP client"""
        await self.client.aclose()
    
    async def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict] = None,
        data: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Make a request to Meta API
        
        Args:
            method: HTTP method (GET, POST, DELETE, etc.)
            endpoint: API endpoint (e.g., '/me/adaccounts')
            params: Query parameters
            data: Request body data
            
        Returns:
            API response as dictionary
            
        Raises:
            MetaAPIError: If the API request fails
        """
        url = f"{self.base_url}{endpoint}"
        
        # Add access token to params
        if params is None:
            params = {}
        params["access_token"] = self.access_token
        
        try:
            response = await self.client.request(
                method=method,
                url=url,
                params=params,
                json=data if data else None,
            )
            response.raise_for_status()
            return response.json()
            
        except httpx.HTTPStatusError as e:
            error_data = e.response.json() if e.response.content else {}
            error_message = error_data.get("error", {}).get("message", str(e))
            logger.error(f"Meta API error: {error_message}")
            raise MetaAPIError(f"Meta API request failed: {error_message}")
            
        except Exception as e:
            logger.error(f"Unexpected error calling Meta API: {e}")
            raise MetaAPIError(f"Unexpected error: {str(e)}")
    
    # ============ USER & BUSINESS ============
    
    async def get_user_info(self) -> Dict[str, Any]:
        """Get current user information"""
        return await self._request("GET", "/me", params={"fields": "id,name,email"})
    
    async def get_ad_accounts(self) -> List[Dict[str, Any]]:
        """Get all ad accounts accessible to the user"""
        response = await self._request(
            "GET",
            "/me/adaccounts",
            params={"fields": "id,name,account_status,currency,timezone_name"}
        )
        return response.get("data", [])
    
    async def get_pages(self) -> List[Dict[str, Any]]:
        """Get all pages managed by the user"""
        response = await self._request(
            "GET",
            "/me/accounts",
            params={"fields": "id,name,access_token,category,followers_count"}
        )
        return response.get("data", [])
    
    # ============ CAMPAIGNS ============
    
    async def get_campaigns(
        self,
        ad_account_id: str,
        status: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Get campaigns for an ad account
        
        Args:
            ad_account_id: Ad account ID (e.g., 'act_123456')
            status: Filter by status (e.g., ['ACTIVE', 'PAUSED'])
        """
        params = {
            "fields": "id,name,status,objective,daily_budget,lifetime_budget,created_time,updated_time"
        }
        if status:
            params["filtering"] = [{"field": "status", "operator": "IN", "value": status}]
        
        response = await self._request("GET", f"/{ad_account_id}/campaigns", params=params)
        return response.get("data", [])
    
    async def create_campaign(
        self,
        ad_account_id: str,
        name: str,
        objective: str,
        status: str = "PAUSED",
        special_ad_categories: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Create a new campaign
        
        Args:
            ad_account_id: Ad account ID
            name: Campaign name
            objective: Campaign objective (e.g., 'LEAD_GENERATION', 'CONVERSIONS')
            status: Initial status ('ACTIVE' or 'PAUSED')
            special_ad_categories: For regulated industries (e.g., ['CREDIT', 'EMPLOYMENT', 'HOUSING'])
        """
        data = {
            "name": name,
            "objective": objective,
            "status": status,
        }
        if special_ad_categories:
            data["special_ad_categories"] = special_ad_categories
        
        return await self._request("POST", f"/{ad_account_id}/campaigns", data=data)

    async def update_campaign(
        self,
        campaign_id: str,
        name: Optional[str] = None,
        status: Optional[str] = None,
        daily_budget: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Update an existing campaign"""
        data = {}
        if name:
            data["name"] = name
        if status:
            data["status"] = status
        if daily_budget:
            data["daily_budget"] = daily_budget

        return await self._request("POST", f"/{campaign_id}", data=data)

    # ============ LEAD ADS ============

    async def get_lead_gen_forms(self, page_id: str) -> List[Dict[str, Any]]:
        """Get all lead generation forms for a page"""
        response = await self._request(
            "GET",
            f"/{page_id}/leadgen_forms",
            params={"fields": "id,name,status,leads_count,created_time"}
        )
        return response.get("data", [])

    async def get_leads_from_form(
        self,
        form_id: str,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get leads from a specific lead generation form

        Args:
            form_id: Lead form ID
            limit: Maximum number of leads to retrieve
        """
        response = await self._request(
            "GET",
            f"/{form_id}/leads",
            params={"fields": "id,created_time,field_data", "limit": limit}
        )
        return response.get("data", [])

    async def get_lead_details(self, lead_id: str) -> Dict[str, Any]:
        """Get detailed information about a specific lead"""
        return await self._request(
            "GET",
            f"/{lead_id}",
            params={"fields": "id,created_time,field_data,ad_id,adset_id,campaign_id,form_id"}
        )

    # ============ INSIGHTS & ANALYTICS ============

    async def get_campaign_insights(
        self,
        campaign_id: str,
        date_preset: str = "last_7d",
        fields: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Get insights/analytics for a campaign

        Args:
            campaign_id: Campaign ID
            date_preset: Time range ('today', 'yesterday', 'last_7d', 'last_30d', etc.)
            fields: Metrics to retrieve (defaults to common metrics)
        """
        if fields is None:
            fields = [
                "impressions",
                "clicks",
                "spend",
                "cpc",
                "cpm",
                "ctr",
                "reach",
                "frequency",
                "actions",
                "cost_per_action_type"
            ]

        params = {
            "fields": ",".join(fields),
            "date_preset": date_preset,
        }

        response = await self._request("GET", f"/{campaign_id}/insights", params=params)
        return response.get("data", [{}])[0] if response.get("data") else {}

    async def get_ad_account_insights(
        self,
        ad_account_id: str,
        date_preset: str = "last_30d"
    ) -> Dict[str, Any]:
        """Get overall insights for an ad account"""
        params = {
            "fields": "impressions,clicks,spend,cpc,cpm,ctr,reach,actions",
            "date_preset": date_preset,
        }

        response = await self._request("GET", f"/{ad_account_id}/insights", params=params)
        return response.get("data", [{}])[0] if response.get("data") else {}

    # ============ CATALOG MANAGEMENT ============

    async def create_catalog(
        self,
        business_id: str,
        name: str,
        vertical: str = "commerce"
    ) -> Dict[str, Any]:
        """
        Create a product catalog

        Args:
            business_id: Business Manager ID
            name: Catalog name
            vertical: Catalog type ('commerce', 'automotive', 'real_estate', etc.)
        """
        data = {
            "name": name,
            "vertical": vertical,
        }
        return await self._request("POST", f"/{business_id}/owned_product_catalogs", data=data)

    async def add_products_to_catalog(
        self,
        catalog_id: str,
        products: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Add products to a catalog

        Args:
            catalog_id: Product catalog ID
            products: List of product dictionaries with fields like:
                - retailer_id: Unique product ID
                - name: Product name
                - description: Product description
                - price: Price in cents
                - currency: Currency code (e.g., 'USD')
                - availability: 'in stock', 'out of stock', etc.
                - url: Product URL
                - image_url: Product image URL
        """
        # Batch create products
        requests = []
        for product in products:
            requests.append({
                "method": "POST",
                "relative_url": f"{catalog_id}/products",
                "body": product
            })

        return await self._request("POST", "/", data={"batch": requests})

