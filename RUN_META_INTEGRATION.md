# 🚀 Run Meta Integration - Step-by-Step Guide

## Step 1: Get Meta App Credentials (5 minutes)

### 1.1 Create Meta App
1. Go to https://developers.facebook.com/apps
2. Click **"Create App"**
3. Select **"Business"** as app type
4. Fill in app details:
   - App Name: "LEGEAI Lead Generation"
   - Contact Email: your email
5. Click **"Create App"**

### 1.2 Add Marketing API Product
1. In your app dashboard, find **"Add Products"**
2. Click **"Set Up"** on **"Marketing API"**
3. Accept terms and conditions

### 1.3 Get App Credentials
1. Go to **Settings → Basic** (left sidebar)
2. Copy your **App ID**
3. Click **"Show"** next to **App Secret** and copy it
4. Keep these safe - you'll need them next!

### 1.4 Configure OAuth Redirect
1. Still in **Settings → Basic**
2. Scroll to **"App Domains"**
3. Add: `localhost`
4. Scroll to **"Website"**
5. Add Site URL: `http://localhost:8000`
6. Click **"Save Changes"**

---

## Step 2: Add Meta Credentials to .env

Add these lines to your `.env` file:

```bash
# Meta Marketing API Integration
META_APP_ID=YOUR_APP_ID_HERE
META_APP_SECRET=YOUR_APP_SECRET_HERE
META_REDIRECT_URI=http://localhost:8000/v1/integrations/meta/callback
META_VERIFY_TOKEN=legeai_meta_webhook_2024
META_API_VERSION=v21.0
```

**Replace:**
- `YOUR_APP_ID_HERE` with your actual App ID
- `YOUR_APP_SECRET_HERE` with your actual App Secret

---

## Step 3: Run Database Migration

```bash
# Make sure you're in the backend directory
cd backend

# Run the migration to create Meta tables
alembic upgrade head
```

**Expected output:**
```
INFO  [alembic.runtime.migration] Running upgrade 7c3e4f5a6b78 -> 21d482deb9d0, Add Meta integration tables
```

---

## Step 4: Start Backend Server

```bash
# From project root
python -m uvicorn backend.api.main:app --reload --host 127.0.0.1 --port 8000
```

**Expected output:**
```
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Started reloader process
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

**Verify backend is running:**
- Open browser: http://localhost:8000/v1/docs
- You should see the Swagger API documentation
- Look for **"Meta Integration"** section

---

## Step 5: Start Frontend Server

Open a **NEW terminal** (keep backend running):

```bash
# From project root
cd frontend

# Install dependencies (if not already done)
npm install

# Start frontend
npm run dev
```

**Expected output:**
```
  ▲ Next.js 15.5.5
  - Local:        http://localhost:3000
  - Ready in 2.3s
```

---

## Step 6: Test Meta Integration

### 6.1 Open Settings Page
1. Open browser: http://localhost:3000/dashboard/settings/integrations/meta
2. You should see the Meta integration settings page
3. Connection status should show "Not Connected"

### 6.2 Connect Meta Account
1. Click **"Connect Meta Account"** button
2. You'll be redirected to Facebook
3. Log in with your Facebook account
4. Grant all requested permissions:
   - ✅ ads_management
   - ✅ ads_read
   - ✅ business_management
   - ✅ pages_manage_ads
   - ✅ pages_read_engagement
   - ✅ And others...
5. Click **"Continue"**
6. You'll be redirected back to LEGEAI
7. Connection status should now show **"Connected ✓"**

### 6.3 View Ad Accounts
- After connecting, you should see your ad accounts listed
- Each account shows: Name, ID, Currency, Status

### 6.4 View Pages
- Your Facebook/Instagram pages should be listed
- Click **"View Lead Forms"** on any page

### 6.5 View Ads Dashboard
1. Go to: http://localhost:3000/dashboard/ads/meta
2. Select an ad account from dropdown
3. View account-level metrics (30-day overview)
4. See your campaigns listed
5. Click a campaign to view detailed insights

---

## Step 7: Test API Endpoints (Optional)

### Test OAuth URL
```bash
curl -X GET "http://localhost:8000/v1/integrations/meta/auth/url" \
  -H "X-Session-ID: test-session"
```

### Test Connection Status
```bash
curl -X GET "http://localhost:8000/v1/integrations/meta/status" \
  -H "X-Session-ID: test-session"
```

### Test Ad Accounts (after connecting)
```bash
curl -X GET "http://localhost:8000/v1/integrations/meta/ad-accounts" \
  -H "X-Session-ID: test-session"
```

---

## Step 8: Set Up Webhooks (For Real-time Lead Capture)

### 8.1 Expose Local Server (for testing)
You need a public URL for webhooks. Use ngrok:

```bash
# Install ngrok (if not installed)
brew install ngrok

# Expose port 8000
ngrok http 8000
```

Copy the HTTPS URL (e.g., `https://abc123.ngrok.io`)

### 8.2 Configure Webhook in Meta
1. Go to Meta App Dashboard
2. **Products → Webhooks**
3. Click **"Subscribe to this object"** for **Page**
4. Callback URL: `https://YOUR_NGROK_URL/v1/integrations/meta/webhook`
5. Verify Token: `legeai_meta_webhook_2024`
6. Click **"Verify and Save"**
7. Subscribe to **"leadgen"** field

### 8.3 Test Real-time Lead Capture
1. Create a test Lead Ad on Facebook
2. Fill out the form
3. Check backend logs - you should see webhook event
4. Lead should appear in LEGEAI database

---

## ✅ Success Checklist

- [ ] Meta App created
- [ ] App ID and Secret added to `.env`
- [ ] Database migration completed
- [ ] Backend running on http://localhost:8000
- [ ] Frontend running on http://localhost:3000
- [ ] Meta account connected via OAuth
- [ ] Ad accounts visible in settings
- [ ] Pages visible in settings
- [ ] Ads dashboard showing metrics
- [ ] (Optional) Webhooks configured and working

---

## 🐛 Troubleshooting

### Backend won't start
- Check PostgreSQL is running: `psql -U postgres -h localhost -p 5432 -l`
- Or set `USE_DB=false` in `.env` to run without database

### "Meta account not connected" error
- Make sure you completed OAuth flow
- Check browser console for errors
- Verify `META_APP_ID` and `META_APP_SECRET` are correct

### Frontend can't reach backend
- Verify backend is running on port 8000
- Check `NEXT_PUBLIC_API_BASE_URL=http://localhost:8000/v1` in `.env`

### OAuth redirect fails
- Verify redirect URI in Meta App settings matches: `http://localhost:8000/v1/integrations/meta/callback`
- Check app is in Development mode (not Live)

---

## 📚 Next Steps After Testing

1. **Create Lead Ads** - Set up Lead Ads on Facebook/Instagram
2. **Configure Auto-sync** - Enable automatic lead syncing
3. **Set Up Campaigns** - Create campaigns through LEGEAI
4. **Monitor Performance** - Use the Ads dashboard
5. **Sync Products** - Add insurance products to Meta catalog

---

## 🎯 Quick Start Commands

```bash
# Terminal 1 - Backend
cd /Users/einstein/Documents/Projects/Lead-Generation-AI-LEGEAI-
python -m uvicorn backend.api.main:app --reload --host 127.0.0.1 --port 8000

# Terminal 2 - Frontend
cd /Users/einstein/Documents/Projects/Lead-Generation-AI-LEGEAI-/frontend
npm run dev

# Terminal 3 - Database Migration (run once)
cd /Users/einstein/Documents/Projects/Lead-Generation-AI-LEGEAI-/backend
alembic upgrade head
```

**Then open:** http://localhost:3000/dashboard/settings/integrations/meta

---

**Ready to start! 🚀**

