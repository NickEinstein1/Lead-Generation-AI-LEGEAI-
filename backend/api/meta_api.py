"""
Meta Marketing API Integration Endpoints

Provides endpoints for:
- OAuth authentication flow
- Webhook verification and processing
- Ad account and page management
- Campaign management
- Lead form integration and sync
- Insights and analytics
"""

import os
import logging
from typing import Dict, Any, Optional, List
from fastapi import APIRouter, HTTPException, Query, Depends, Request, Response
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from backend.database.connection import session_dep
from backend.api.auth_dependencies import require_permission
from backend.security.authentication import Permission
from backend.integrations.meta import (
    MetaAPIClient,
    MetaOAuthHandler,
    MetaWebhookHandler,
    MetaLeadSync,
)

router = APIRouter(prefix="/integrations/meta", tags=["Meta Integration"])
logger = logging.getLogger(__name__)

# Initialize handlers
oauth_handler = MetaOAuthHandler()
webhook_handler = MetaWebhookHandler()

# In-memory storage for tokens (replace with database in production)
# TODO: Move to database table `meta_integrations`
META_TOKENS: Dict[str, str] = {}


# ============ PYDANTIC MODELS ============

class OAuthUrlResponse(BaseModel):
    """OAuth authorization URL response"""
    authorization_url: str
    state: str


class OAuthCallbackRequest(BaseModel):
    """OAuth callback data"""
    code: str
    state: Optional[str] = None


class TokenResponse(BaseModel):
    """Access token response"""
    access_token: str
    token_type: str
    expires_in: Optional[int] = None


class AdAccountResponse(BaseModel):
    """Ad account information"""
    id: str
    name: str
    account_status: Optional[str] = None
    currency: Optional[str] = None
    timezone_name: Optional[str] = None


class PageResponse(BaseModel):
    """Page information"""
    id: str
    name: str
    category: Optional[str] = None
    followers_count: Optional[int] = None


class CampaignCreate(BaseModel):
    """Create campaign request"""
    ad_account_id: str = Field(..., description="Ad account ID (e.g., 'act_123456')")
    name: str = Field(..., description="Campaign name")
    objective: str = Field(..., description="Campaign objective (e.g., 'LEAD_GENERATION')")
    status: str = Field(default="PAUSED", description="Initial status")
    special_ad_categories: Optional[List[str]] = Field(None, description="Special ad categories")


class CampaignUpdate(BaseModel):
    """Update campaign request"""
    name: Optional[str] = None
    status: Optional[str] = None
    daily_budget: Optional[int] = None


class LeadSyncRequest(BaseModel):
    """Manual lead sync request"""
    form_id: str = Field(..., description="Lead form ID")
    page_id: str = Field(..., description="Page ID")
    limit: int = Field(default=100, ge=1, le=500, description="Max leads to sync")


class LeadSyncResponse(BaseModel):
    """Lead sync result"""
    total_leads: int
    synced: int
    duplicates: int
    errors: int


# ============ OAUTH ENDPOINTS ============

@router.get("/auth/url", response_model=OAuthUrlResponse)
async def get_oauth_url(
    state: Optional[str] = Query(None, description="CSRF state parameter"),
    current_user: Dict[str, Any] = Depends(require_permission(Permission.VIEW_LEADS)),
):
    """
    Get Meta OAuth authorization URL
    
    Returns URL to redirect user to for Meta account connection.
    After user grants permissions, Meta will redirect to callback URL.
    """
    import secrets
    
    # Generate state if not provided
    if not state:
        state = secrets.token_urlsafe(32)
    
    auth_url = oauth_handler.get_authorization_url(state=state)
    
    logger.info(f"Generated OAuth URL for user: {current_user['username']}")
    
    return {
        "authorization_url": auth_url,
        "state": state,
    }


@router.post("/callback", response_model=TokenResponse)
async def oauth_callback(
    payload: OAuthCallbackRequest,
    session: AsyncSession = Depends(session_dep),
    current_user: Dict[str, Any] = Depends(require_permission(Permission.VIEW_LEADS)),
):
    """
    OAuth callback endpoint
    
    Exchanges authorization code for access token.
    Stores token in database for future use.
    """
    try:
        # Exchange code for short-lived token
        token_data = await oauth_handler.exchange_code_for_token(payload.code)
        short_lived_token = token_data["access_token"]
        
        # Exchange for long-lived token (60 days)
        long_lived_data = await oauth_handler.get_long_lived_token(short_lived_token)
        access_token = long_lived_data["access_token"]
        
        # Store token (TODO: save to database)
        user_id = current_user["user_id"]
        META_TOKENS[user_id] = access_token
        
        logger.info(f"OAuth successful for user: {current_user['username']}")
        
        return {
            "access_token": access_token,
            "token_type": "bearer",
            "expires_in": long_lived_data.get("expires_in"),
        }
        
    except Exception as e:
        logger.error(f"OAuth callback failed: {e}")
        raise HTTPException(status_code=400, detail=f"OAuth failed: {str(e)}")


@router.post("/disconnect")
async def disconnect_meta(
    current_user: Dict[str, Any] = Depends(require_permission(Permission.VIEW_LEADS)),
):
    """
    Disconnect Meta account
    
    Removes stored access token.
    """
    user_id = current_user["user_id"]
    
    if user_id in META_TOKENS:
        del META_TOKENS[user_id]
        logger.info(f"Meta account disconnected for user: {current_user['username']}")
        return {"message": "Meta account disconnected successfully"}
    
    raise HTTPException(status_code=404, detail="No Meta account connected")


# ============ WEBHOOK ENDPOINTS ============

@router.get("/webhook")
async def verify_webhook(
    request: Request,
):
    """
    Webhook verification endpoint (GET)

    Meta sends GET request to verify webhook subscription.
    Must return the challenge parameter if verification succeeds.
    """
    params = request.query_params
    mode = params.get("hub.mode")
    token = params.get("hub.verify_token")
    challenge = params.get("hub.challenge")

    try:
        result = webhook_handler.verify_webhook(mode, token, challenge)
        return Response(content=result, media_type="text/plain")
    except HTTPException as e:
        raise e


@router.post("/webhook")
async def receive_webhook(
    request: Request,
    session: AsyncSession = Depends(session_dep),
):
    """
    Webhook receiver endpoint (POST)

    Receives real-time events from Meta (e.g., new leads).
    Verifies signature and processes events.
    """
    # Get signature from header
    signature = request.headers.get("X-Hub-Signature-256", "")

    # Get raw body
    body = await request.body()

    # Verify signature
    if not webhook_handler.verify_signature(body, signature):
        logger.warning("Webhook signature verification failed")
        raise HTTPException(status_code=403, detail="Invalid signature")

    # Parse JSON
    import json
    data = json.loads(body)

    # Register lead handler
    async def handle_leadgen(page_id: str, lead_data: Dict[str, Any]):
        """Handle new lead event"""
        lead_id = lead_data.get("lead_id")
        form_id = lead_data.get("form_id")

        logger.info(f"Processing new lead: {lead_id} from form: {form_id}")

        # Get access token for this page (TODO: lookup from database)
        # For now, use first available token
        if not META_TOKENS:
            logger.warning("No Meta tokens available for lead sync")
            return

        access_token = list(META_TOKENS.values())[0]

        # Sync lead to database
        try:
            client = MetaAPIClient(access_token)
            lead_sync = MetaLeadSync(client)

            legeai_lead_id = await lead_sync.sync_lead(
                session=session,
                lead_id=lead_id,
                form_id=form_id,
                page_id=page_id,
            )

            if legeai_lead_id:
                logger.info(f"Lead {lead_id} synced to LEGEAI lead {legeai_lead_id}")
            else:
                logger.info(f"Lead {lead_id} already exists (duplicate)")

            await client.close()

        except Exception as e:
            logger.error(f"Error syncing lead {lead_id}: {e}")

    # Register handler
    webhook_handler.register_handler("leadgen", handle_leadgen)

    # Process webhook
    try:
        await webhook_handler.process_webhook(data)
        return {"status": "ok"}
    except Exception as e:
        logger.error(f"Error processing webhook: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============ HELPER FUNCTION ============

def _get_user_token(user_id: str) -> str:
    """Get Meta access token for user"""
    if user_id not in META_TOKENS:
        raise HTTPException(
            status_code=400,
            detail="Meta account not connected. Please connect your Meta account first."
        )
    return META_TOKENS[user_id]


# ============ AD ACCOUNT & PAGE ENDPOINTS ============

@router.get("/ad-accounts", response_model=List[AdAccountResponse])
async def get_ad_accounts(
    current_user: Dict[str, Any] = Depends(require_permission(Permission.VIEW_LEADS)),
):
    """
    Get all ad accounts accessible to the user

    Returns list of ad accounts with basic information.
    """
    access_token = _get_user_token(current_user["user_id"])

    try:
        client = MetaAPIClient(access_token)
        ad_accounts = await client.get_ad_accounts()
        await client.close()

        return ad_accounts

    except Exception as e:
        logger.error(f"Error fetching ad accounts: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/pages", response_model=List[PageResponse])
async def get_pages(
    current_user: Dict[str, Any] = Depends(require_permission(Permission.VIEW_LEADS)),
):
    """
    Get all pages managed by the user

    Returns list of Facebook/Instagram pages.
    """
    access_token = _get_user_token(current_user["user_id"])

    try:
        client = MetaAPIClient(access_token)
        pages = await client.get_pages()
        await client.close()

        return pages

    except Exception as e:
        logger.error(f"Error fetching pages: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============ CAMPAIGN ENDPOINTS ============

@router.get("/campaigns")
async def get_campaigns(
    ad_account_id: str = Query(..., description="Ad account ID (e.g., 'act_123456')"),
    status: Optional[str] = Query(None, description="Filter by status (comma-separated)"),
    current_user: Dict[str, Any] = Depends(require_permission(Permission.VIEW_LEADS)),
):
    """
    Get campaigns for an ad account

    Returns list of campaigns with basic information and metrics.
    """
    access_token = _get_user_token(current_user["user_id"])

    try:
        client = MetaAPIClient(access_token)

        # Parse status filter
        status_list = status.split(",") if status else None

        campaigns = await client.get_campaigns(ad_account_id, status=status_list)
        await client.close()

        return {"campaigns": campaigns, "total": len(campaigns)}

    except Exception as e:
        logger.error(f"Error fetching campaigns: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/campaigns")
async def create_campaign(
    payload: CampaignCreate,
    current_user: Dict[str, Any] = Depends(require_permission(Permission.CREATE_LEADS)),
):
    """
    Create a new campaign

    Creates a campaign in the specified ad account.
    Campaign starts in PAUSED status by default.
    """
    access_token = _get_user_token(current_user["user_id"])

    try:
        client = MetaAPIClient(access_token)

        campaign = await client.create_campaign(
            ad_account_id=payload.ad_account_id,
            name=payload.name,
            objective=payload.objective,
            status=payload.status,
            special_ad_categories=payload.special_ad_categories,
        )

        await client.close()

        logger.info(f"Campaign created: {campaign.get('id')} by user: {current_user['username']}")

        return campaign

    except Exception as e:
        logger.error(f"Error creating campaign: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/campaigns/{campaign_id}")
async def update_campaign(
    campaign_id: str,
    payload: CampaignUpdate,
    current_user: Dict[str, Any] = Depends(require_permission(Permission.UPDATE_LEADS)),
):
    """
    Update an existing campaign

    Updates campaign name, status, or budget.
    """
    access_token = _get_user_token(current_user["user_id"])

    try:
        client = MetaAPIClient(access_token)

        result = await client.update_campaign(
            campaign_id=campaign_id,
            name=payload.name,
            status=payload.status,
            daily_budget=payload.daily_budget,
        )

        await client.close()

        logger.info(f"Campaign updated: {campaign_id} by user: {current_user['username']}")

        return result

    except Exception as e:
        logger.error(f"Error updating campaign: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============ LEAD FORM ENDPOINTS ============

@router.get("/lead-forms")
async def get_lead_forms(
    page_id: str = Query(..., description="Page ID"),
    current_user: Dict[str, Any] = Depends(require_permission(Permission.VIEW_LEADS)),
):
    """
    Get all lead generation forms for a page

    Returns list of lead forms with basic information.
    """
    access_token = _get_user_token(current_user["user_id"])

    try:
        client = MetaAPIClient(access_token)
        forms = await client.get_lead_gen_forms(page_id)
        await client.close()

        return {"forms": forms, "total": len(forms)}

    except Exception as e:
        logger.error(f"Error fetching lead forms: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sync-leads", response_model=LeadSyncResponse)
async def sync_leads(
    payload: LeadSyncRequest,
    session: AsyncSession = Depends(session_dep),
    current_user: Dict[str, Any] = Depends(require_permission(Permission.CREATE_LEADS)),
):
    """
    Manually sync leads from a lead form

    Fetches leads from Meta and syncs them to LEGEAI database.
    Skips duplicates automatically.
    """
    access_token = _get_user_token(current_user["user_id"])

    try:
        client = MetaAPIClient(access_token)
        lead_sync = MetaLeadSync(client)

        result = await lead_sync.sync_form_leads(
            session=session,
            form_id=payload.form_id,
            page_id=payload.page_id,
            limit=payload.limit,
        )

        await client.close()

        logger.info(
            f"Lead sync completed: {result['synced']} synced, "
            f"{result['duplicates']} duplicates, {result['errors']} errors"
        )

        return result

    except Exception as e:
        logger.error(f"Error syncing leads: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============ INSIGHTS ENDPOINTS ============

@router.get("/insights/campaign/{campaign_id}")
async def get_campaign_insights(
    campaign_id: str,
    date_preset: str = Query(default="last_7d", description="Time range (e.g., 'last_7d', 'last_30d')"),
    current_user: Dict[str, Any] = Depends(require_permission(Permission.VIEW_LEADS)),
):
    """
    Get insights/analytics for a campaign

    Returns metrics like impressions, clicks, spend, conversions, etc.
    """
    access_token = _get_user_token(current_user["user_id"])

    try:
        client = MetaAPIClient(access_token)
        insights = await client.get_campaign_insights(campaign_id, date_preset=date_preset)
        await client.close()

        return insights

    except Exception as e:
        logger.error(f"Error fetching campaign insights: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/insights/account/{ad_account_id}")
async def get_account_insights(
    ad_account_id: str,
    date_preset: str = Query(default="last_30d", description="Time range"),
    current_user: Dict[str, Any] = Depends(require_permission(Permission.VIEW_LEADS)),
):
    """
    Get overall insights for an ad account

    Returns aggregate metrics across all campaigns.
    """
    access_token = _get_user_token(current_user["user_id"])

    try:
        client = MetaAPIClient(access_token)
        insights = await client.get_ad_account_insights(ad_account_id, date_preset=date_preset)
        await client.close()

        return insights

    except Exception as e:
        logger.error(f"Error fetching account insights: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============ STATUS ENDPOINT ============

@router.get("/status")
async def get_meta_status(
    current_user: Dict[str, Any] = Depends(require_permission(Permission.VIEW_LEADS)),
):
    """
    Get Meta integration status

    Returns whether user has connected their Meta account.
    """
    user_id = current_user["user_id"]
    is_connected = user_id in META_TOKENS

    return {
        "connected": is_connected,
        "user_id": user_id,
        "username": current_user["username"],
    }


# ============ CATALOG MANAGEMENT ENDPOINTS ============

class CatalogCreate(BaseModel):
    """Request model for creating a catalog"""
    business_id: str
    name: str
    vertical: str = "commerce"


class ProductCreate(BaseModel):
    """Product model for catalog"""
    id: str
    name: str
    description: str
    premium: float  # Monthly premium
    coverage: float  # Coverage amount
    image_url: Optional[str] = None
    product_type: str  # auto, home, life, health


class ProductsCreate(BaseModel):
    """Request model for adding products to catalog"""
    products: List[ProductCreate]


@router.post("/catalogs")
async def create_catalog(
    payload: CatalogCreate,
    current_user: Dict[str, Any] = Depends(require_permission(Permission.CREATE_LEADS)),
):
    """
    Create a product catalog for dynamic ads

    Allows you to sync insurance products to Meta for dynamic advertising.
    """
    access_token = _get_user_token(current_user["user_id"])

    try:
        client = MetaAPIClient(access_token)
        catalog = await client.create_catalog(
            business_id=payload.business_id,
            name=payload.name,
            vertical=payload.vertical
        )
        await client.close()

        logger.info(f"Catalog created: {catalog.get('id')} by user: {current_user['username']}")

        return catalog

    except Exception as e:
        logger.error(f"Error creating catalog: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/catalogs/{catalog_id}/products")
async def add_products_to_catalog(
    catalog_id: str,
    payload: ProductsCreate,
    current_user: Dict[str, Any] = Depends(require_permission(Permission.CREATE_LEADS)),
):
    """
    Add insurance products to catalog

    Syncs insurance products to Meta catalog for dynamic ads.
    Products are formatted according to Meta's product feed spec.
    """
    access_token = _get_user_token(current_user["user_id"])

    try:
        client = MetaAPIClient(access_token)

        # Convert insurance products to Meta product format
        products = []
        for product in payload.products:
            products.append({
                "retailer_id": product.id,
                "name": product.name,
                "description": product.description,
                "price": int(product.premium * 100),  # Convert to cents
                "currency": "USD",
                "availability": "in stock",
                "url": f"https://yoursite.com/products/{product.id}",
                "image_url": product.image_url or f"https://yoursite.com/images/{product.product_type}.jpg",
                "brand": "LEGEAI Insurance",
                "condition": "new",
                "custom_label_0": product.product_type,  # auto, home, life, health
                "custom_label_1": f"coverage_{int(product.coverage)}",
            })

        result = await client.add_products_to_catalog(catalog_id, products)
        await client.close()

        logger.info(f"Added {len(products)} products to catalog {catalog_id}")

        return {
            "catalog_id": catalog_id,
            "products_added": len(products),
            "result": result
        }

    except Exception as e:
        logger.error(f"Error adding products to catalog: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/catalogs/{catalog_id}/products")
async def get_catalog_products(
    catalog_id: str,
    limit: int = Query(default=100, le=1000),
    current_user: Dict[str, Any] = Depends(require_permission(Permission.VIEW_LEADS)),
):
    """
    Get products from a catalog

    Returns list of products in the catalog.
    """
    access_token = _get_user_token(current_user["user_id"])

    try:
        client = MetaAPIClient(access_token)

        # Get products from catalog
        response = await client._request(
            "GET",
            f"/{catalog_id}/products",
            params={"limit": limit}
        )

        await client.close()

        return response

    except Exception as e:
        logger.error(f"Error fetching catalog products: {e}")
        raise HTTPException(status_code=500, detail=str(e))

