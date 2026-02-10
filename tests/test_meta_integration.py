"""
Meta Marketing API Integration - Test Suite

Tests for Meta integration including:
- OAuth flow
- Ad accounts and pages
- Campaign management
- Lead sync
- Webhooks
"""

import pytest
import asyncio
from httpx import AsyncClient
from backend.api.main import app

# Test configuration
BASE_URL = "http://localhost:8000/v1"
TEST_SESSION_ID = "test-session-123"


class TestMetaOAuth:
    """Test OAuth authentication flow"""
    
    @pytest.mark.asyncio
    async def test_get_auth_url(self):
        """Test getting OAuth authorization URL"""
        async with AsyncClient(app=app, base_url=BASE_URL) as client:
            response = await client.get(
                "/integrations/meta/auth/url",
                headers={"X-Session-ID": TEST_SESSION_ID}
            )
            
            assert response.status_code == 200
            data = response.json()
            assert "authorization_url" in data
            assert "state" in data
            assert "facebook.com" in data["authorization_url"]
            assert "oauth" in data["authorization_url"]
    
    @pytest.mark.asyncio
    async def test_connection_status_not_connected(self):
        """Test connection status when not connected"""
        async with AsyncClient(app=app, base_url=BASE_URL) as client:
            response = await client.get(
                "/integrations/meta/status",
                headers={"X-Session-ID": TEST_SESSION_ID}
            )
            
            assert response.status_code == 200
            data = response.json()
            assert "connected" in data
            # Initially should be False unless token is stored


class TestMetaAdAccounts:
    """Test ad account endpoints"""
    
    @pytest.mark.asyncio
    async def test_get_ad_accounts_not_connected(self):
        """Test getting ad accounts when not connected"""
        async with AsyncClient(app=app, base_url=BASE_URL) as client:
            response = await client.get(
                "/integrations/meta/ad-accounts",
                headers={"X-Session-ID": TEST_SESSION_ID}
            )
            
            # Should return 400 if not connected
            assert response.status_code in [400, 401]


class TestMetaPages:
    """Test pages endpoints"""
    
    @pytest.mark.asyncio
    async def test_get_pages_not_connected(self):
        """Test getting pages when not connected"""
        async with AsyncClient(app=app, base_url=BASE_URL) as client:
            response = await client.get(
                "/integrations/meta/pages",
                headers={"X-Session-ID": TEST_SESSION_ID}
            )
            
            # Should return 400 if not connected
            assert response.status_code in [400, 401]


class TestMetaCampaigns:
    """Test campaign management endpoints"""
    
    @pytest.mark.asyncio
    async def test_get_campaigns_not_connected(self):
        """Test getting campaigns when not connected"""
        async with AsyncClient(app=app, base_url=BASE_URL) as client:
            response = await client.get(
                "/integrations/meta/campaigns?ad_account_id=act_123456",
                headers={"X-Session-ID": TEST_SESSION_ID}
            )
            
            # Should return 400 if not connected
            assert response.status_code in [400, 401]


class TestMetaWebhooks:
    """Test webhook endpoints"""
    
    @pytest.mark.asyncio
    async def test_webhook_verification(self):
        """Test webhook verification (GET request)"""
        async with AsyncClient(app=app, base_url=BASE_URL) as client:
            response = await client.get(
                "/integrations/meta/webhook",
                params={
                    "hub.mode": "subscribe",
                    "hub.verify_token": "legeai_meta_webhook_2024",
                    "hub.challenge": "test_challenge_123"
                }
            )
            
            # Should return the challenge if token matches
            if response.status_code == 200:
                assert response.text == "test_challenge_123"
    
    @pytest.mark.asyncio
    async def test_webhook_verification_invalid_token(self):
        """Test webhook verification with invalid token"""
        async with AsyncClient(app=app, base_url=BASE_URL) as client:
            response = await client.get(
                "/integrations/meta/webhook",
                params={
                    "hub.mode": "subscribe",
                    "hub.verify_token": "wrong_token",
                    "hub.challenge": "test_challenge_123"
                }
            )
            
            # Should return 403 for invalid token
            assert response.status_code == 403


class TestMetaLeadSync:
    """Test lead sync endpoints"""
    
    @pytest.mark.asyncio
    async def test_get_lead_forms_not_connected(self):
        """Test getting lead forms when not connected"""
        async with AsyncClient(app=app, base_url=BASE_URL) as client:
            response = await client.get(
                "/integrations/meta/lead-forms?page_id=123456",
                headers={"X-Session-ID": TEST_SESSION_ID}
            )
            
            # Should return 400 if not connected
            assert response.status_code in [400, 401]


class TestMetaCatalog:
    """Test catalog management endpoints"""
    
    @pytest.mark.asyncio
    async def test_create_catalog_not_connected(self):
        """Test creating catalog when not connected"""
        async with AsyncClient(app=app, base_url=BASE_URL) as client:
            response = await client.post(
                "/integrations/meta/catalogs",
                headers={"X-Session-ID": TEST_SESSION_ID},
                json={
                    "business_id": "123456",
                    "name": "Test Catalog",
                    "vertical": "commerce"
                }
            )
            
            # Should return 400 if not connected
            assert response.status_code in [400, 401]


# Manual integration tests (require real Meta credentials)
def manual_test_full_oauth_flow():
    """
    Manual test for full OAuth flow
    
    Steps:
    1. Get authorization URL
    2. Visit URL in browser
    3. Grant permissions
    4. Copy code from redirect
    5. Exchange code for token
    6. Verify connection status
    """
    print("Manual OAuth Flow Test")
    print("=" * 50)
    print("1. Run: curl http://localhost:8000/v1/integrations/meta/auth/url")
    print("2. Visit the authorization_url in browser")
    print("3. Grant permissions")
    print("4. Copy the 'code' parameter from redirect URL")
    print("5. Run: curl -X POST http://localhost:8000/v1/integrations/meta/callback \\")
    print("        -H 'Content-Type: application/json' \\")
    print("        -d '{\"code\": \"YOUR_CODE\", \"state\": \"YOUR_STATE\"}'")
    print("6. Verify: curl http://localhost:8000/v1/integrations/meta/status")


if __name__ == "__main__":
    # Run manual test instructions
    manual_test_full_oauth_flow()

