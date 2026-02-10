# Meta Marketing API - Testing Guide

## ✅ What's Been Completed

### Backend Integration (COMPLETE)
- ✅ `backend/integrations/meta/client.py` - Meta API client
- ✅ `backend/integrations/meta/auth.py` - OAuth handler
- ✅ `backend/integrations/meta/webhooks.py` - Webhook handler
- ✅ `backend/integrations/meta/lead_sync.py` - Lead sync module
- ✅ `backend/api/meta_api.py` - FastAPI endpoints (592 lines)
- ✅ Router registered in `backend/api/main.py`
- ✅ Environment variables added to `.env.example`

### API Endpoints Available

#### OAuth & Authentication
```
GET  /v1/integrations/meta/auth/url       - Get OAuth authorization URL
POST /v1/integrations/meta/callback       - OAuth callback (exchange code for token)
POST /v1/integrations/meta/disconnect     - Disconnect Meta account
GET  /v1/integrations/meta/status         - Check connection status
```

#### Webhooks
```
GET  /v1/integrations/meta/webhook        - Webhook verification (Meta setup)
POST /v1/integrations/meta/webhook        - Webhook receiver (real-time events)
```

#### Ad Accounts & Pages
```
GET  /v1/integrations/meta/ad-accounts    - List ad accounts
GET  /v1/integrations/meta/pages          - List managed pages
```

#### Campaigns
```
GET  /v1/integrations/meta/campaigns      - List campaigns
POST /v1/integrations/meta/campaigns      - Create campaign
PUT  /v1/integrations/meta/campaigns/:id  - Update campaign
```

#### Lead Forms & Sync
```
GET  /v1/integrations/meta/lead-forms     - List lead forms
POST /v1/integrations/meta/sync-leads     - Manually sync leads
```

#### Analytics
```
GET  /v1/integrations/meta/insights/campaign/:id  - Campaign insights
GET  /v1/integrations/meta/insights/account/:id   - Account insights
```

## 🚀 How to Test

### Step 1: Configure Environment

Add to your `.env` file:

```env
# Meta Marketing API
META_APP_ID=your_app_id_here
META_APP_SECRET=your_app_secret_here
META_REDIRECT_URI=http://localhost:8000/v1/integrations/meta/callback
META_VERIFY_TOKEN=legeai_meta_webhook_2024
META_API_VERSION=v21.0
```

### Step 2: Get Meta App Credentials

1. Go to https://developers.facebook.com/apps
2. Select your app
3. Go to **Settings → Basic**
4. Copy **App ID** and **App Secret**
5. Update `.env` file

### Step 3: Start Backend

```bash
# From project root
python -m uvicorn backend.api.main:app --reload --host 127.0.0.1 --port 8000
```

### Step 4: Test OAuth Flow

#### 4.1 Get Authorization URL

```bash
curl -X GET "http://localhost:8000/v1/integrations/meta/auth/url" \
  -H "X-Session-ID: your_session_id"
```

Response:
```json
{
  "authorization_url": "https://www.facebook.com/v21.0/dialog/oauth?...",
  "state": "random_state_string"
}
```

#### 4.2 Visit Authorization URL

1. Copy the `authorization_url` from response
2. Open in browser
3. Log in to Facebook
4. Grant permissions
5. You'll be redirected to callback URL with `code` parameter

#### 4.3 Exchange Code for Token

```bash
curl -X POST "http://localhost:8000/v1/integrations/meta/callback" \
  -H "Content-Type: application/json" \
  -H "X-Session-ID: your_session_id" \
  -d '{
    "code": "code_from_redirect",
    "state": "state_from_step_4.1"
  }'
```

Response:
```json
{
  "access_token": "long_lived_token_here",
  "token_type": "bearer",
  "expires_in": 5184000
}
```

### Step 5: Test Ad Accounts

```bash
curl -X GET "http://localhost:8000/v1/integrations/meta/ad-accounts" \
  -H "X-Session-ID: your_session_id"
```

Response:
```json
[
  {
    "id": "act_123456789",
    "name": "My Ad Account",
    "account_status": "ACTIVE",
    "currency": "USD",
    "timezone_name": "America/Los_Angeles"
  }
]
```

### Step 6: Test Pages

```bash
curl -X GET "http://localhost:8000/v1/integrations/meta/pages" \
  -H "X-Session-ID: your_session_id"
```

### Step 7: Test Campaigns

#### List Campaigns
```bash
curl -X GET "http://localhost:8000/v1/integrations/meta/campaigns?ad_account_id=act_123456789" \
  -H "X-Session-ID: your_session_id"
```

#### Create Campaign
```bash
curl -X POST "http://localhost:8000/v1/integrations/meta/campaigns" \
  -H "Content-Type: application/json" \
  -H "X-Session-ID: your_session_id" \
  -d '{
    "ad_account_id": "act_123456789",
    "name": "Test Lead Generation Campaign",
    "objective": "LEAD_GENERATION",
    "status": "PAUSED"
  }'
```

### Step 8: Test Lead Forms

```bash
curl -X GET "http://localhost:8000/v1/integrations/meta/lead-forms?page_id=123456789" \
  -H "X-Session-ID: your_session_id"
```

### Step 9: Test Lead Sync

```bash
curl -X POST "http://localhost:8000/v1/integrations/meta/sync-leads" \
  -H "Content-Type: application/json" \
  -H "X-Session-ID: your_session_id" \
  -d '{
    "form_id": "form_123456",
    "page_id": "page_123456",
    "limit": 100
  }'
```

Response:
```json
{
  "total_leads": 50,
  "synced": 45,
  "duplicates": 5,
  "errors": 0
}
```

## 📊 API Documentation

Once backend is running, visit:
- **Swagger UI**: http://localhost:8000/v1/docs
- **ReDoc**: http://localhost:8000/v1/redoc

Look for the **"Meta Integration"** tag to see all endpoints.

## 🔍 Troubleshooting

### Error: "Meta account not connected"
- Make sure you completed OAuth flow (Step 4)
- Check that token was stored successfully

### Error: "Invalid signature" (webhook)
- Verify `META_APP_SECRET` is correct in `.env`
- Check webhook signature header is present

### Error: "Meta API request failed"
- Check your Meta app has required permissions
- Verify access token is valid
- Check Meta API status: https://developers.facebook.com/status

## ✅ Next Steps

1. **Test OAuth flow** - Connect your Meta account
2. **Test ad accounts** - Verify you can list accounts
3. **Test campaigns** - Create a test campaign
4. **Set up webhook** - Configure in Meta App Dashboard
5. **Test lead sync** - Create a lead form and sync leads

Once testing is complete, we can move to:
- Database tables for storing tokens
- Frontend UI for connection
- Automated lead sync

