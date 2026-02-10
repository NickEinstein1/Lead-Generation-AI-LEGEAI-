# Meta Marketing API Integration Plan

## Overview

Complete integration of Meta (Facebook) Marketing API with LEGEAI platform for:
- **Lead Ads**: Capture leads from Facebook/Instagram ads in real-time
- **Ads Management**: Create, manage, and optimize ad campaigns
- **Catalog Management**: Sync insurance products for dynamic ads
- **Analytics**: Track ad performance and ROI

## Your Meta Permissions

✅ **Ads Management Standard Access**
✅ **Business Asset User Profile Access**
✅ **Permissions Granted:**
- `ads_management` - Create and manage ads
- `ads_read` - Read ads insights
- `business_management` - Manage business assets
- `catalog_management` - Manage product catalogs
- `email` - Access user email
- `pages_manage_ads` - Manage page ads
- `pages_read_engagement` - Read page engagement
- `pages_show_list` - List managed pages
- `public_profile` - Access public profile
- `threads_business_basic` - Threads integration

## Implementation Status

### ✅ Phase 1: Core Integration Modules (COMPLETE)

**Created Files:**
1. `backend/integrations/__init__.py` - Package initialization
2. `backend/integrations/meta/__init__.py` - Meta module exports
3. `backend/integrations/meta/client.py` - Meta API client (331 lines)
   - User & Business info
   - Campaign management (CRUD)
   - Lead Ads integration
   - Insights & Analytics
   - Catalog management
4. `backend/integrations/meta/auth.py` - OAuth 2.0 handler (150 lines)
   - Authorization URL generation
   - Token exchange
   - Long-lived tokens (60 days)
   - Token validation
5. `backend/integrations/meta/webhooks.py` - Webhook handler (150 lines)
   - Webhook verification
   - Signature validation (HMAC-SHA256)
   - Real-time event processing
6. `backend/integrations/meta/lead_sync.py` - Lead sync module (180 lines)
   - Sync leads from Meta to LEGEAI
   - Field mapping
   - Duplicate detection

### 🔄 Phase 2: API Routes (NEXT)

**Files to Create:**
- `backend/api/meta_api.py` - FastAPI routes for Meta integration

**Endpoints to Implement:**
```
GET  /v1/integrations/meta/auth/url          - Get OAuth URL
GET  /v1/integrations/meta/callback          - OAuth callback
POST /v1/integrations/meta/disconnect        - Disconnect account

GET  /v1/integrations/meta/webhook           - Webhook verification
POST /v1/integrations/meta/webhook           - Webhook receiver

GET  /v1/integrations/meta/ad-accounts       - List ad accounts
GET  /v1/integrations/meta/pages             - List pages
GET  /v1/integrations/meta/campaigns         - List campaigns
POST /v1/integrations/meta/campaigns         - Create campaign
PUT  /v1/integrations/meta/campaigns/:id     - Update campaign

GET  /v1/integrations/meta/lead-forms        - List lead forms
POST /v1/integrations/meta/sync-leads        - Manually sync leads
GET  /v1/integrations/meta/insights          - Get ad insights
```

### 🔄 Phase 3: Database Models (NEXT)

**Migration to Create:**
- `backend/migrations/versions/XXX_add_meta_integration.py`

**Tables to Add:**
```sql
-- Store Meta access tokens and account info
CREATE TABLE meta_integrations (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    access_token TEXT NOT NULL,
    token_expires_at TIMESTAMP,
    meta_user_id VARCHAR(100),
    meta_user_name VARCHAR(255),
    scopes TEXT[],
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Track connected ad accounts
CREATE TABLE meta_ad_accounts (
    id SERIAL PRIMARY KEY,
    integration_id INTEGER REFERENCES meta_integrations(id),
    ad_account_id VARCHAR(100) UNIQUE NOT NULL,
    ad_account_name VARCHAR(255),
    currency VARCHAR(10),
    timezone VARCHAR(100),
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Track connected pages
CREATE TABLE meta_pages (
    id SERIAL PRIMARY KEY,
    integration_id INTEGER REFERENCES meta_integrations(id),
    page_id VARCHAR(100) UNIQUE NOT NULL,
    page_name VARCHAR(255),
    page_access_token TEXT,
    category VARCHAR(100),
    followers_count INTEGER,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Track lead forms
CREATE TABLE meta_lead_forms (
    id SERIAL PRIMARY KEY,
    page_id INTEGER REFERENCES meta_pages(id),
    form_id VARCHAR(100) UNIQUE NOT NULL,
    form_name VARCHAR(255),
    status VARCHAR(50),
    leads_count INTEGER DEFAULT 0,
    auto_sync BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Map Meta leads to LEGEAI leads
CREATE TABLE meta_lead_mappings (
    id SERIAL PRIMARY KEY,
    meta_lead_id VARCHAR(100) UNIQUE NOT NULL,
    legeai_lead_id INTEGER REFERENCES leads(id),
    form_id VARCHAR(100),
    synced_at TIMESTAMP DEFAULT NOW()
);
```

### 🔄 Phase 4: Frontend UI (NEXT)

**Pages to Create:**

1. **Settings Page**: `/dashboard/settings/integrations/meta`
   - Connect Meta account button
   - Display connected status
   - List ad accounts and pages
   - Select lead forms to auto-sync
   - Disconnect option

2. **Ads Dashboard**: `/dashboard/ads/meta`
   - Campaign performance overview
   - Lead metrics from Meta
   - Spend and ROI tracking
   - Create/edit campaigns

**Components to Create:**
- `frontend/src/components/integrations/MetaConnect.tsx`
- `frontend/src/components/integrations/MetaAdAccounts.tsx`
- `frontend/src/components/integrations/MetaLeadForms.tsx`
- `frontend/src/components/ads/MetaCampaignList.tsx`
- `frontend/src/components/ads/MetaInsights.tsx`

## Environment Variables

Add to `.env`:
```env
# Meta Marketing API
META_APP_ID=your_app_id_here
META_APP_SECRET=your_app_secret_here
META_REDIRECT_URI=http://localhost:8000/v1/integrations/meta/callback
META_VERIFY_TOKEN=legeai_meta_webhook_2024
META_API_VERSION=v21.0
```

## Next Steps

1. **Get Meta App Credentials**
   - Go to https://developers.facebook.com/apps
   - Create a new app or use existing
   - Get App ID and App Secret
   - Add to `.env` file

2. **Configure Webhook**
   - In Meta App Dashboard → Webhooks
   - Subscribe to `leadgen` events
   - Set webhook URL: `https://your-domain.com/v1/integrations/meta/webhook`
   - Use verify token from `.env`

3. **Create API Routes**
   - Implement `backend/api/meta_api.py`
   - Register routes in `backend/api/main.py`

4. **Create Database Migration**
   - Run: `alembic revision -m "Add Meta integration tables"`
   - Apply: `alembic upgrade head`

5. **Build Frontend UI**
   - Create settings page for connection
   - Create ads dashboard

6. **Test Integration**
   - Connect Meta account via OAuth
   - Create test lead ad
   - Verify webhook receives leads
   - Check leads sync to database

## Testing Checklist

- [ ] OAuth flow works (authorization → callback → token)
- [ ] Long-lived token exchange works
- [ ] Webhook verification succeeds
- [ ] Webhook signature validation works
- [ ] Lead sync creates records in database
- [ ] Duplicate leads are detected
- [ ] Campaign creation works
- [ ] Insights API returns data
- [ ] Frontend displays connected status
- [ ] Frontend shows ad performance

## Documentation

- API documentation: Auto-generated via FastAPI
- User guide: How to connect Meta account
- Developer guide: How to extend integration

