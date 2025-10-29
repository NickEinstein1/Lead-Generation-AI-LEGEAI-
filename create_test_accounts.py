#!/usr/bin/env python3
"""
Create test accounts for LEAGAI development and testing
Run this script to populate the system with test users
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'api'))

from security.authentication import auth_manager, UserRole, Permission
import asyncio

# Test account credentials
TEST_ACCOUNTS = [
    {
        "username": "admin",
        "email": "admin@leagai.dev",
        "password": "AdminPass123!",
        "role": UserRole.ADMIN,
        "description": "Admin user with full access"
    },
    {
        "username": "manager",
        "email": "manager@leagai.dev",
        "password": "ManagerPass123!",
        "role": UserRole.MANAGER,
        "description": "Manager user for team oversight"
    },
    {
        "username": "agent1",
        "email": "agent1@leagai.dev",
        "password": "AgentPass123!",
        "role": UserRole.AGENT,
        "description": "Sales agent 1"
    },
    {
        "username": "agent2",
        "email": "agent2@leagai.dev",
        "password": "AgentPass456!",
        "role": UserRole.AGENT,
        "description": "Sales agent 2"
    },
    {
        "username": "viewer",
        "email": "viewer@leagai.dev",
        "password": "ViewerPass123!",
        "role": UserRole.VIEWER,
        "description": "Read-only viewer"
    },
    {
        "username": "api_client",
        "email": "api@leagai.dev",
        "password": "ApiClientPass123!",
        "role": UserRole.API_CLIENT,
        "description": "API client for integrations"
    }
]

def create_test_accounts():
    """Create all test accounts"""
    print("=" * 70)
    print("🔐 LEAGAI Test Account Creator")
    print("=" * 70)
    print()
    
    created_count = 0
    failed_count = 0
    
    for account in TEST_ACCOUNTS:
        print(f"Creating {account['role'].value.upper()} account: {account['username']}")
        print(f"  Email: {account['email']}")
        print(f"  Description: {account['description']}")
        
        success, result = auth_manager.create_user(
            username=account['username'],
            email=account['email'],
            password=account['password'],
            role=account['role']
        )
        
        if success:
            print(f"  ✅ SUCCESS - User ID: {result}")
            created_count += 1
        else:
            print(f"  ❌ FAILED - {result}")
            failed_count += 1
        
        print()
    
    print("=" * 70)
    print(f"📊 Summary: {created_count} created, {failed_count} failed")
    print("=" * 70)
    print()

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
    
    success, session_id, message = auth_manager.authenticate_user(
        identifier=test_account['username'],
        password=test_account['password'],
        ip_address="127.0.0.1",
        user_agent="TestScript/1.0"
    )
    
    if success:
        print(f"✅ Login successful!")
        print(f"  Session ID: {session_id}")
        
        # Validate session
        is_valid, user = auth_manager.validate_session(session_id)
        if is_valid and user:
            print(f"  User: {user.username}")
            print(f"  Role: {user.role.value}")
            print(f"  Permissions: {[p.value for p in user.permissions]}")
            
            # Generate API token
            api_token = auth_manager.generate_jwt_token(
                user.user_id,
                user.permissions
            )
            print(f"  API Token: {api_token[:50]}...")
    else:
        print(f"❌ Login failed: {message}")
    
    print()

def print_credentials_table():
    """Print a formatted table of test credentials"""
    print("=" * 100)
    print("📋 Test Account Credentials")
    print("=" * 100)
    print()
    print(f"{'Username':<15} {'Email':<25} {'Password':<20} {'Role':<12} {'Description':<20}")
    print("-" * 100)
    
    for account in TEST_ACCOUNTS:
        print(f"{account['username']:<15} {account['email']:<25} {account['password']:<20} {account['role'].value:<12} {account['description']:<20}")
    
    print()
    print("=" * 100)
    print()

def main():
    """Main function"""
    print()
    print_credentials_table()
    
    # Create accounts
    create_test_accounts()
    
    # Test login
    test_login()
    
    print("=" * 70)
    print("✨ Test account setup complete!")
    print("=" * 70)
    print()
    print("📝 Next steps:")
    print("  1. Start the backend: uvicorn api.main:app --reload")
    print("  2. Start the frontend: cd frontend && npm run dev")
    print("  3. Visit: http://localhost:3000/login")
    print("  4. Use any of the credentials above to login")
    print()

if __name__ == "__main__":
    main()

