# Meta Marketing API - Quick Start Guide

## What We've Built So Far

✅ **Complete Backend Integration Modules:**
- OAuth 2.0 authentication handler
- Meta API client with full CRUD operations
- Webhook handler for real-time lead capture
- Lead sync module to import leads to database

## What You Need to Do Next

### Step 1: Get Your Meta App Credentials

1. Go to https://developers.facebook.com/apps
2. Select your app (or create a new one)
3. Go to **Settings → Basic**
4. Copy your **App ID** and **App Secret**

### Step 2: Configure Environment Variables

Add these to your `.env` file:

```env
# Meta Marketing API Configuration
META_APP_ID=YOUR_APP_ID_HERE
META_APP_SECRET=YOUR_APP_SECRET_HERE
META_REDIRECT_URI=http://localhost:8000/v1/integrations/meta/callback
META_VERIFY_TOKEN=legeai_meta_webhook_2024
META_API_VERSION=v21.0
```

### Step 3: What's Next?

We need to complete these remaining tasks:

#### A. Create API Routes (`backend/api/meta_api.py`)
This will expose endpoints for:
- OAuth connection flow
- Webhook receiver
- Campaign management
- Lead sync triggers

#### B. Create Database Tables
Add tables to store:
- Meta access tokens
- Connected ad accounts
- Lead form mappings
- Lead sync history

#### C. Build Frontend UI
Create pages for:
- Connecting Meta account (OAuth button)
- Viewing connected ad accounts
- Managing lead forms
- Viewing ad performance

## How the Integration Works

### Lead Capture Flow

```
1. User creates Lead Ad on Facebook/Instagram
2. Someone fills out the lead form
3. Meta sends webhook to LEGEAI
   → POST /v1/integrations/meta/webhook
4. Webhook handler verifies signature
5. Lead sync module fetches lead details from Meta API
6. Lead is saved to LEGEAI database
7. Lead appears in LEGEAI dashboard
```

### OAuth Connection Flow

```
1. User clicks "Connect Meta Account" in LEGEAI
2. Redirected to Meta authorization page
3. User grants permissions
4. Meta redirects back to LEGEAI callback
   → GET /v1/integrations/meta/callback?code=...
5. LEGEAI exchanges code for access token
6. Token is saved to database
7. User can now sync leads and manage ads
```

## Available Meta API Methods

### Authentication (`MetaOAuthHandler`)
- `get_authorization_url()` - Generate OAuth URL
- `exchange_code_for_token()` - Get access token
- `get_long_lived_token()` - Convert to 60-day token
- `debug_token()` - Validate token

### API Client (`MetaAPIClient`)

**User & Business:**
- `get_user_info()` - Get current user
- `get_ad_accounts()` - List ad accounts
- `get_pages()` - List managed pages

**Campaigns:**
- `get_campaigns()` - List campaigns
- `create_campaign()` - Create new campaign
- `update_campaign()` - Update campaign

**Lead Ads:**
- `get_lead_gen_forms()` - List lead forms
- `get_leads_from_form()` - Get leads from form
- `get_lead_details()` - Get single lead details

**Analytics:**
- `get_campaign_insights()` - Campaign metrics
- `get_ad_account_insights()` - Account metrics

**Catalog:**
- `create_catalog()` - Create product catalog
- `add_products_to_catalog()` - Add products

### Webhooks (`MetaWebhookHandler`)
- `verify_webhook()` - Verify subscription (GET)
- `verify_signature()` - Verify HMAC signature (POST)
- `process_webhook()` - Process incoming events

### Lead Sync (`MetaLeadSync`)
- `sync_lead()` - Sync single lead
- `sync_form_leads()` - Sync all leads from form

## Example Usage

### Connect Meta Account (OAuth)

```python
from backend.integrations.meta import MetaOAuthHandler

# Generate authorization URL
oauth = MetaOAuthHandler()
auth_url = oauth.get_authorization_url(state="random_state_123")

# User visits auth_url and grants permissions
# Meta redirects to callback with code

# Exchange code for token
token_data = await oauth.exchange_code_for_token(code)
access_token = token_data["access_token"]

# Get long-lived token (60 days)
long_lived = await oauth.get_long_lived_token(access_token)
```

### Fetch Leads

```python
from backend.integrations.meta import MetaAPIClient, MetaLeadSync

# Initialize client
client = MetaAPIClient(access_token)

# Get lead forms
forms = await client.get_lead_gen_forms(page_id="123456789")

# Sync leads from a form
lead_sync = MetaLeadSync(client)
result = await lead_sync.sync_form_leads(
    session=db_session,
    form_id="form_123",
    page_id="page_123",
    limit=100
)

print(f"Synced {result['synced']} leads")
```

### Handle Webhook

```python
from backend.integrations.meta import MetaWebhookHandler

webhook = MetaWebhookHandler()

# Verify webhook (GET request)
challenge = webhook.verify_webhook(
    mode="subscribe",
    token="legeai_meta_webhook_2024",
    challenge="challenge_string"
)

# Process webhook (POST request)
is_valid = webhook.verify_signature(
    payload=request_body,
    signature=request.headers["X-Hub-Signature-256"]
)

if is_valid:
    await webhook.process_webhook(webhook_data)
```

## Ready to Continue?

**Tell me which part you want to build next:**

1. **API Routes** - Create FastAPI endpoints
2. **Database Models** - Create migration for Meta tables
3. **Frontend UI** - Build connection and dashboard pages
4. **All of the above** - Complete the full integration

Just let me know and I'll continue building! 🚀

