"""
Authentication dependencies for protecting CRUD endpoints
"""
from fastapi import Depends, HTTPException, status, Header
from typing import Optional
from backend.security.authentication import auth_manager, UserRole, Permission
import logging
import os

logger = logging.getLogger(__name__)

# Feature flags / config
ENABLE_DEMO_MODE = os.getenv("ENABLE_DEMO_MODE", "false").lower() == "true"
ALLOW_DEV_API_KEY = os.getenv("ALLOW_DEV_API_KEY", "false").lower() == "true"
DEV_API_KEY = os.getenv("DEV_API_KEY", "dev-api-key-12345")

async def get_current_user_from_session(
    x_session_id: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None),
):
    """Resolve the current user from a session ID or (optionally) a dev API key.

    - Primary: validates ``X-Session-ID`` using ``AuthenticationManager``.
    - Secondary (dev/demo only): accepts ``X-API-Key`` but only when
      ``ALLOW_DEV_API_KEY`` or ``ENABLE_DEMO_MODE`` is true.
    """

    # Allow API key for development/testing when explicitly enabled
    if x_api_key and x_api_key == DEV_API_KEY and (ALLOW_DEV_API_KEY or ENABLE_DEMO_MODE):
        logger.info("Request authenticated with dev API key (demo/demo-like mode)")
        return {
            "user_id": "api-user",
            "username": "api-user",
            "role": UserRole.API_CLIENT.value,
            "permissions": [Permission.API_ACCESS.value],
        }

    # Check session ID
    if x_session_id:
        valid, user = auth_manager.validate_session(x_session_id)
        if valid and user:
            logger.info(f"Request authenticated for user: {user.username}")
            return {
                "user_id": user.user_id,
                "username": user.username,
                "role": user.role.value,
                "permissions": [p.value for p in user.permissions],
            }

    # No valid authentication
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Not authenticated. Please provide X-Session-ID or X-API-Key header.",
        headers={"WWW-Authenticate": "Bearer"},
    )

async def get_optional_user(
    x_session_id: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None)
):
    """
    Optional authentication - returns user if authenticated, None otherwise
    Useful for endpoints that work with or without authentication
    """
    try:
        return await get_current_user_from_session(x_session_id, x_api_key)
    except HTTPException:
        return None

def require_permission(permission: Permission):
    """
    Dependency factory to require specific permission
    
    Usage:
        @router.get("/protected", dependencies=[Depends(require_permission(Permission.VIEW_LEADS))])
    """
    async def permission_checker(current_user: dict = Depends(get_current_user_from_session)):
        user_permissions = current_user.get("permissions", [])
        if permission.value not in user_permissions and current_user.get("role") != UserRole.ADMIN.value:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission denied. Required permission: {permission.value}"
            )
        return current_user
    return permission_checker

def require_role(role: UserRole):
    """
    Dependency factory to require specific role
    
    Usage:
        @router.delete("/admin-only", dependencies=[Depends(require_role(UserRole.ADMIN))])
    """
    async def role_checker(current_user: dict = Depends(get_current_user_from_session)):
        if current_user.get("role") != role.value:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied. Required role: {role.value}"
            )
        return current_user
    return role_checker

