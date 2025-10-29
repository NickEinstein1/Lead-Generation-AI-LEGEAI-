#!/usr/bin/env python3
"""
Create test accounts for LEAGAI via API
This script creates test accounts by calling the registration endpoint
"""

import requests
import json
import time

# API Configuration
API_BASE_URL = "http://localhost:8000"
REGISTER_ENDPOINT = f"{API_BASE_URL}/v1/auth/register"

# Test account credentials
TEST_ACCOUNTS = [
    {
        "username": "admin",
        "email": "admin@leagai.dev",
        "password": "AdminPass123!",
        "role": "admin",
        "description": "Admin user with full access"
    },
    {
        "username": "manager",
        "email": "manager@leagai.dev",
        "password": "ManagerPass123!",
        "role": "manager",
        "description": "Manager user for team oversight"
    },
    {
        "username": "agent1",
        "email": "agent1@leagai.dev",
        "password": "AgentPass123!",
        "role": "agent",
        "description": "Sales agent 1"
    },
    {
        "username": "agent2",
        "email": "agent2@leagai.dev",
        "password": "AgentPass456!",
        "role": "agent",
        "description": "Sales agent 2"
    },
    {
        "username": "viewer",
        "email": "viewer@leagai.dev",
        "password": "ViewerPass123!",
        "role": "viewer",
        "description": "Read-only viewer"
    },
]

def print_credentials_table():
    """Print a formatted table of test credentials"""
    print("=" * 110)
    print("📋 Test Account Credentials")
    print("=" * 110)
    print()
    print(f"{'Username':<15} {'Email':<25} {'Password':<20} {'Role':<12} {'Description':<25}")
    print("-" * 110)
    
    for account in TEST_ACCOUNTS:
        print(f"{account['username']:<15} {account['email']:<25} {account['password']:<20} {account['role']:<12} {account['description']:<25}")
    
    print()
    print("=" * 110)
    print()

def create_test_accounts():
    """Create all test accounts via API"""
    print("=" * 70)
    print("🔐 LEAGAI Test Account Creator (via API)")
    print("=" * 70)
    print()
    
    # Check if API is running
    try:
        response = requests.get(f"{API_BASE_URL}/v1/health", timeout=5)
        print(f"✅ API is running at {API_BASE_URL}")
        print()
    except requests.exceptions.ConnectionError:
        print(f"❌ ERROR: Cannot connect to API at {API_BASE_URL}")
        print("   Make sure the backend is running:")
        print("   .\run_backend.ps1")
        print()
        return False
    
    created_count = 0
    failed_count = 0
    
    for account in TEST_ACCOUNTS:
        print(f"Creating {account['role'].upper()} account: {account['username']}")
        print(f"  Email: {account['email']}")
        print(f"  Description: {account['description']}")
        
        payload = {
            "username": account['username'],
            "email": account['email'],
            "password": account['password'],
            "role": account['role']
        }
        
        try:
            response = requests.post(
                REGISTER_ENDPOINT,
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"  ✅ SUCCESS")
                print(f"     User ID: {data.get('user_id')}")
                print(f"     Session ID: {data.get('session_id', 'N/A')[:20]}...")
                created_count += 1
            else:
                error_msg = response.json().get('detail', response.text)
                print(f"  ❌ FAILED - {error_msg}")
                failed_count += 1
        
        except requests.exceptions.Timeout:
            print(f"  ❌ FAILED - Request timeout")
            failed_count += 1
        except requests.exceptions.ConnectionError:
            print(f"  ❌ FAILED - Cannot connect to API")
            failed_count += 1
        except Exception as e:
            print(f"  ❌ FAILED - {str(e)}")
            failed_count += 1
        
        print()
        time.sleep(0.5)  # Small delay between requests
    
    print("=" * 70)
    print(f"📊 Summary: {created_count} created, {failed_count} failed")
    print("=" * 70)
    print()
    
    return created_count > 0

def test_login():
    """Test login with created accounts"""
    print("=" * 70)
    print("🔑 Testing Login Functionality")
    print("=" * 70)
    print()
    
    test_account = TEST_ACCOUNTS[0]  # Test with admin
    
    print(f"Testing login with: {test_account['username']}")
    print(f"Password: {test_account['password']}")
    print()
    
    payload = {
        "username": test_account['username'],
        "password": test_account['password']
    }
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/v1/auth/login",
            json=payload,
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Login successful!")
            print(f"  Status: {data.get('status')}")
            print(f"  User ID: {data.get('user_id')}")
            print(f"  Role: {data.get('role')}")
            print(f"  Token: {data.get('token', 'N/A')[:50]}...")
        else:
            error_msg = response.json().get('detail', response.text)
            print(f"❌ Login failed: {error_msg}")
    
    except Exception as e:
        print(f"❌ Error: {str(e)}")
    
    print()

def main():
    """Main function"""
    print()
    print_credentials_table()
    
    # Create accounts
    success = create_test_accounts()
    
    if success:
        # Test login
        test_login()
    
    print("=" * 70)
    print("✨ Test account setup complete!")
    print("=" * 70)
    print()
    print("📝 Next steps:")
    print("  1. Start the backend: cd api && uvicorn main:app --reload")
    print("  2. Start the frontend: cd frontend && npm run dev")
    print("  3. Visit: http://localhost:3000/login")
    print("  4. Use any of the credentials above to login")
    print()
    print("💡 Example login:")
    print(f"  Username: {TEST_ACCOUNTS[0]['username']}")
    print(f"  Password: {TEST_ACCOUNTS[0]['password']}")
    print()

if __name__ == "__main__":
    main()

