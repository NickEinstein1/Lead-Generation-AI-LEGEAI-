# Meta Integration - End-to-End Testing Guide

## 🎯 Complete Testing Checklist

This guide walks you through testing the entire Meta Marketing API integration from start to finish.

---

## Prerequisites

### 1. Meta App Setup
- [ ] Created Meta App at https://developers.facebook.com/apps
- [ ] Added "Marketing API" product to your app
- [ ] Configured OAuth redirect URI: `http://localhost:8000/v1/integrations/meta/callback`
- [ ] Copied App ID and App Secret

### 2. Environment Configuration
- [ ] Added Meta credentials to `.env`:
```env
META_APP_ID=your_app_id_here
META_APP_SECRET=your_app_secret_here
META_REDIRECT_URI=http://localhost:8000/v1/integrations/meta/callback
META_VERIFY_TOKEN=legeai_meta_webhook_2024
META_API_VERSION=v21.0
```

### 3. Database Setup
- [ ] PostgreSQL running on localhost:5432
- [ ] Database `leadgen` created
- [ ] Run migration: `cd backend && alembic upgrade head`

### 4. Backend Running
- [ ] Backend started: `python -m uvicorn backend.api.main:app --reload --host 127.0.0.1 --port 8000`
- [ ] API docs accessible: http://localhost:8000/v1/docs

### 5. Frontend Running
- [ ] Frontend started: `cd frontend && npm run dev`
- [ ] Frontend accessible: http://localhost:3000

---

## Test 1: OAuth Connection Flow

### Step 1.1: Get Authorization URL
```bash
curl -X GET "http://localhost:8000/v1/integrations/meta/auth/url" \
  -H "X-Session-ID: test-session-123"
```

**Expected Response:**
```json
{
  "authorization_url": "https://www.facebook.com/v21.0/dialog/oauth?...",
  "state": "random_state_string"
}
```

- [ ] Response contains `authorization_url`
- [ ] Response contains `state`
- [ ] URL contains `facebook.com/oauth`

### Step 1.2: Visit Authorization URL
1. Copy the `authorization_url` from response
2. Open in browser
3. Log in to Facebook
4. Grant all requested permissions
5. You'll be redirected to callback URL with `code` parameter

- [ ] Successfully logged in to Facebook
- [ ] Permissions screen displayed
- [ ] Redirected to callback URL
- [ ] URL contains `code` parameter

### Step 1.3: Exchange Code for Token
```bash
curl -X POST "http://localhost:8000/v1/integrations/meta/callback" \
  -H "Content-Type: application/json" \
  -H "X-Session-ID: test-session-123" \
  -d '{
    "code": "YOUR_CODE_FROM_REDIRECT",
    "state": "STATE_FROM_STEP_1.1"
  }'
```

**Expected Response:**
```json
{
  "access_token": "long_lived_token_here",
  "token_type": "bearer",
  "expires_in": 5184000
}
```

- [ ] Response contains `access_token`
- [ ] Token type is `bearer`
- [ ] Expires in 60 days (5184000 seconds)

### Step 1.4: Verify Connection Status
```bash
curl -X GET "http://localhost:8000/v1/integrations/meta/status" \
  -H "X-Session-ID: test-session-123"
```

**Expected Response:**
```json
{
  "connected": true,
  "user_id": "test-session-123",
  "username": "..."
}
```

- [ ] `connected` is `true`
- [ ] User ID matches session

---

## Test 2: Ad Accounts & Pages

### Step 2.1: List Ad Accounts
```bash
curl -X GET "http://localhost:8000/v1/integrations/meta/ad-accounts" \
  -H "X-Session-ID: test-session-123"
```

**Expected Response:**
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

- [ ] Returns array of ad accounts
- [ ] Each account has `id`, `name`, `account_status`
- [ ] At least one account returned

### Step 2.2: List Pages
```bash
curl -X GET "http://localhost:8000/v1/integrations/meta/pages" \
  -H "X-Session-ID: test-session-123"
```

**Expected Response:**
```json
[
  {
    "id": "123456789",
    "name": "My Facebook Page",
    "category": "Business",
    "access_token": "page_token_here"
  }
]
```

- [ ] Returns array of pages
- [ ] Each page has `id`, `name`, `category`
- [ ] At least one page returned

---

## Test 3: Campaign Management

### Step 3.1: List Campaigns
```bash
curl -X GET "http://localhost:8000/v1/integrations/meta/campaigns?ad_account_id=act_YOUR_ACCOUNT_ID" \
  -H "X-Session-ID: test-session-123"
```

- [ ] Returns campaigns array
- [ ] Total count matches

### Step 3.2: Create Campaign
```bash
curl -X POST "http://localhost:8000/v1/integrations/meta/campaigns" \
  -H "Content-Type: application/json" \
  -H "X-Session-ID: test-session-123" \
  -d '{
    "ad_account_id": "act_YOUR_ACCOUNT_ID",
    "name": "Test Lead Generation Campaign",
    "objective": "LEAD_GENERATION",
    "status": "PAUSED"
  }'
```

**Expected Response:**
```json
{
  "id": "campaign_id_here",
  "name": "Test Lead Generation Campaign",
  "status": "PAUSED"
}
```

- [ ] Campaign created successfully
- [ ] Returns campaign ID
- [ ] Status is PAUSED

### Step 3.3: Get Campaign Insights
```bash
curl -X GET "http://localhost:8000/v1/integrations/meta/insights/campaign/CAMPAIGN_ID?date_preset=last_7d" \
  -H "X-Session-ID: test-session-123"
```

- [ ] Returns insights data
- [ ] Contains impressions, clicks, spend
- [ ] Contains CTR, CPC, CPM

---

## Test 4: Lead Forms & Sync

### Step 4.1: List Lead Forms
```bash
curl -X GET "http://localhost:8000/v1/integrations/meta/lead-forms?page_id=YOUR_PAGE_ID" \
  -H "X-Session-ID: test-session-123"
```

- [ ] Returns lead forms array
- [ ] Each form has `id`, `name`, `status`

### Step 4.2: Manual Lead Sync
```bash
curl -X POST "http://localhost:8000/v1/integrations/meta/sync-leads" \
  -H "Content-Type: application/json" \
  -H "X-Session-ID: test-session-123" \
  -d '{
    "form_id": "YOUR_FORM_ID",
    "page_id": "YOUR_PAGE_ID",
    "limit": 100
  }'
```

**Expected Response:**
```json
{
  "total_leads": 50,
  "synced": 45,
  "duplicates": 5,
  "errors": 0
}
```

- [ ] Leads synced successfully
- [ ] Duplicates detected
- [ ] No errors

---

## Test 5: Webhooks

### Step 5.1: Verify Webhook (GET)
```bash
curl -X GET "http://localhost:8000/v1/integrations/meta/webhook?hub.mode=subscribe&hub.verify_token=legeai_meta_webhook_2024&hub.challenge=test123"
```

**Expected Response:**
```
test123
```

- [ ] Returns challenge string
- [ ] Status code 200

### Step 5.2: Configure Webhook in Meta
1. Go to Meta App Dashboard
2. Products → Webhooks
3. Subscribe to `leadgen` events
4. Callback URL: `http://YOUR_PUBLIC_URL/v1/integrations/meta/webhook`
5. Verify Token: `legeai_meta_webhook_2024`

- [ ] Webhook subscription successful
- [ ] Verification passed

### Step 5.3: Test Real-time Lead Capture
1. Create a test lead on your Lead Ad
2. Check backend logs for webhook event
3. Verify lead appears in database

- [ ] Webhook received
- [ ] Signature verified
- [ ] Lead synced to database

---

## Test 6: Frontend UI

### Step 6.1: Settings Page
1. Visit: http://localhost:3000/dashboard/settings/integrations/meta
2. Click "Connect Meta Account"
3. Complete OAuth flow
4. Verify connection status shows "Connected"

- [ ] Settings page loads
- [ ] Connect button works
- [ ] OAuth flow completes
- [ ] Connection status updates

### Step 6.2: View Ad Accounts & Pages
- [ ] Ad accounts list displayed
- [ ] Pages list displayed
- [ ] Can select page to view lead forms

### Step 6.3: Ads Dashboard
1. Visit: http://localhost:3000/dashboard/ads/meta
2. Select ad account
3. View campaigns
4. Click campaign to see insights

- [ ] Dashboard loads
- [ ] Account selector works
- [ ] Campaigns displayed
- [ ] Insights shown

---

## ✅ Success Criteria

All tests passed if:
- [x] OAuth flow completes successfully
- [x] Ad accounts and pages load
- [x] Campaigns can be created and listed
- [x] Lead forms can be synced
- [x] Webhooks verify correctly
- [x] Frontend UI works end-to-end

---

## 🐛 Troubleshooting

### Issue: "Meta account not connected"
- Verify OAuth flow completed
- Check token is stored
- Try reconnecting

### Issue: "Invalid signature" (webhook)
- Verify `META_APP_SECRET` is correct
- Check webhook payload format

### Issue: Frontend can't connect
- Verify `NEXT_PUBLIC_API_BASE_URL` is set
- Check backend is running
- Check CORS settings

---

## 📊 Test Results

Record your test results:

| Test | Status | Notes |
|------|--------|-------|
| OAuth Flow | ⏳ | |
| Ad Accounts | ⏳ | |
| Pages | ⏳ | |
| Campaigns | ⏳ | |
| Lead Sync | ⏳ | |
| Webhooks | ⏳ | |
| Frontend | ⏳ | |

Legend: ✅ Pass | ❌ Fail | ⏳ Not Tested

