import jwt
import bcrypt
import secrets
import hashlib
import hmac
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import redis
import logging
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import os
import asyncio
from functools import wraps
import time

logger = logging.getLogger(__name__)

class UserRole(Enum):
    ADMIN = "admin"
    MANAGER = "manager"
    AGENT = "agent"
    VIEWER = "viewer"
    API_CLIENT = "api_client"

class Permission(Enum):
    # Lead Management
    VIEW_LEADS = "view_leads"
    CREATE_LEADS = "create_leads"
    UPDATE_LEADS = "update_leads"
    DELETE_LEADS = "delete_leads"
    EXPORT_LEADS = "export_leads"
    
    # Scoring
    VIEW_SCORES = "view_scores"
    GENERATE_SCORES = "generate_scores"
    
    # Messages
    SEND_MESSAGES = "send_messages"
    VIEW_MESSAGES = "view_messages"
    
    # Admin
    MANAGE_USERS = "manage_users"
    VIEW_AUDIT_LOGS = "view_audit_logs"
    MANAGE_COMPLIANCE = "manage_compliance"
    
    # API Access
    API_ACCESS = "api_access"
    BULK_OPERATIONS = "bulk_operations"

@dataclass
class User:
    user_id: str
    username: str
    email: str
    password_hash: str
    role: UserRole
    permissions: List[Permission] = field(default_factory=list)
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_login: Optional[datetime] = None
    failed_login_attempts: int = 0
    locked_until: Optional[datetime] = None
    mfa_enabled: bool = False
    mfa_secret: Optional[str] = None

@dataclass
class Session:
    session_id: str
    user_id: str
    created_at: datetime
    expires_at: datetime
    ip_address: str
    user_agent: str
    is_active: bool = True

class SecurityConfig:
    """Security configuration"""
    
    # JWT Settings
    JWT_SECRET_KEY = os.getenv('JWT_SECRET_KEY', secrets.token_urlsafe(32))
    JWT_ALGORITHM = 'HS256'
    JWT_EXPIRATION_HOURS = 24
    
    # Password Settings
    PASSWORD_MIN_LENGTH = 12
    PASSWORD_REQUIRE_UPPERCASE = True
    PASSWORD_REQUIRE_LOWERCASE = True
    PASSWORD_REQUIRE_NUMBERS = True
    PASSWORD_REQUIRE_SPECIAL = True
    
    # Account Lockout
    MAX_LOGIN_ATTEMPTS = 5
    LOCKOUT_DURATION_MINUTES = 30
    
    # Session Settings
    SESSION_TIMEOUT_HOURS = 8
    MAX_CONCURRENT_SESSIONS = 3
    
    # Rate Limiting
    RATE_LIMIT_REQUESTS = 100
    RATE_LIMIT_WINDOW_MINUTES = 15
    
    # Encryption
    ENCRYPTION_KEY = os.getenv('ENCRYPTION_KEY', Fernet.generate_key())

class AuthenticationManager:
    """Comprehensive authentication and authorization system"""
    
    def __init__(self, redis_config: Dict[str, Any] = None):
        self.redis_config = redis_config or {
            'host': 'localhost',
            'port': 6379,
            'db': 4
        }
        
        self.redis_client = redis.Redis(**self.redis_config)
        self.fernet = Fernet(SecurityConfig.ENCRYPTION_KEY)
        
        # Role-based permissions
        self.role_permissions = self._load_role_permissions()
        
        # In-memory user store (would be database in production)
        self.users: Dict[str, User] = {}
        self.sessions: Dict[str, Session] = {}
        
        # Rate limiting
        self.rate_limits: Dict[str, List[float]] = {}
        
    def _load_role_permissions(self) -> Dict[UserRole, List[Permission]]:
        """Load role-based permissions"""
        return {
            UserRole.ADMIN: [p for p in Permission],  # All permissions
            UserRole.MANAGER: [
                Permission.VIEW_LEADS, Permission.CREATE_LEADS, Permission.UPDATE_LEADS,
                Permission.VIEW_SCORES, Permission.GENERATE_SCORES,
                Permission.SEND_MESSAGES, Permission.VIEW_MESSAGES,
                Permission.VIEW_AUDIT_LOGS, Permission.API_ACCESS
            ],
            UserRole.AGENT: [
                Permission.VIEW_LEADS, Permission.UPDATE_LEADS,
                Permission.VIEW_SCORES, Permission.GENERATE_SCORES,
                Permission.SEND_MESSAGES, Permission.VIEW_MESSAGES
            ],
            UserRole.VIEWER: [
                Permission.VIEW_LEADS, Permission.VIEW_SCORES, Permission.VIEW_MESSAGES
            ],
            UserRole.API_CLIENT: [
                Permission.API_ACCESS, Permission.GENERATE_SCORES, Permission.BULK_OPERATIONS
            ]
        }
    
    def hash_password(self, password: str) -> str:
        """Hash password using bcrypt"""
        salt = bcrypt.gensalt()
        return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')
    
    def verify_password(self, password: str, password_hash: str) -> bool:
        """Verify password against hash"""
        return bcrypt.checkpw(password.encode('utf-8'), password_hash.encode('utf-8'))
    
    def validate_password_strength(self, password: str) -> Tuple[bool, List[str]]:
        """Validate password strength"""
        errors = []
        
        if len(password) < SecurityConfig.PASSWORD_MIN_LENGTH:
            errors.append(f"Password must be at least {SecurityConfig.PASSWORD_MIN_LENGTH} characters")
        
        if SecurityConfig.PASSWORD_REQUIRE_UPPERCASE and not any(c.isupper() for c in password):
            errors.append("Password must contain at least one uppercase letter")
        
        if SecurityConfig.PASSWORD_REQUIRE_LOWERCASE and not any(c.islower() for c in password):
            errors.append("Password must contain at least one lowercase letter")
        
        if SecurityConfig.PASSWORD_REQUIRE_NUMBERS and not any(c.isdigit() for c in password):
            errors.append("Password must contain at least one number")
        
        if SecurityConfig.PASSWORD_REQUIRE_SPECIAL and not any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password):
            errors.append("Password must contain at least one special character")
        
        return len(errors) == 0, errors
    
    def create_user(self, username: str, email: str, password: str, role: UserRole) -> Tuple[bool, str]:
        """Create a new user"""
        try:
            # Validate password
            is_valid, errors = self.validate_password_strength(password)
            if not is_valid:
                return False, "; ".join(errors)
            
            # Check if user exists
            if any(u.username == username or u.email == email for u in self.users.values()):
                return False, "User already exists"
            
            # Create user
            user_id = secrets.token_urlsafe(16)
            password_hash = self.hash_password(password)
            
            user = User(
                user_id=user_id,
                username=username,
                email=email,
                password_hash=password_hash,
                role=role,
                permissions=self.role_permissions.get(role, [])
            )
            
            self.users[user_id] = user
            
            # Log user creation
            self._log_security_event("user_created", user_id, {"username": username, "role": role.value})
            
            return True, user_id
            
        except Exception as e:
            logger.error(f"Failed to create user: {e}")
            return False, "Failed to create user"
    
    def authenticate_user(self, username: str, password: str, ip_address: str, user_agent: str) -> Tuple[bool, Optional[str], str]:
        """Authenticate user and create session"""
        try:
            # Find user
            user = None
            for u in self.users.values():
                if u.username == username or u.email == username:
                    user = u
                    break
            
            if not user:
                self._log_security_event("login_failed", None, {"username": username, "reason": "user_not_found", "ip": ip_address})
                return False, None, "Invalid credentials"
            
            # Check if account is locked
            if user.locked_until and user.locked_until > datetime.utcnow():
                self._log_security_event("login_blocked", user.user_id, {"reason": "account_locked", "ip": ip_address})
                return False, None, "Account is locked"
            
            # Check if user is active
            if not user.is_active:
                self._log_security_event("login_blocked", user.user_id, {"reason": "account_disabled", "ip": ip_address})
                return False, None, "Account is disabled"
            
            # Verify password
            if not self.verify_password(password, user.password_hash):
                user.failed_login_attempts += 1
                
                # Lock account if too many failed attempts
                if user.failed_login_attempts >= SecurityConfig.MAX_LOGIN_ATTEMPTS:
                    user.locked_until = datetime.utcnow() + timedelta(minutes=SecurityConfig.LOCKOUT_DURATION_MINUTES)
                    self._log_security_event("account_locked", user.user_id, {"ip": ip_address})
                
                self._log_security_event("login_failed", user.user_id, {"reason": "invalid_password", "ip": ip_address})
                return False, None, "Invalid credentials"
            
            # Reset failed attempts on successful login
            user.failed_login_attempts = 0
            user.locked_until = None
            user.last_login = datetime.utcnow()
            
            # Create session
            session_id = self._create_session(user.user_id, ip_address, user_agent)
            
            self._log_security_event("login_success", user.user_id, {"ip": ip_address})
            
            return True, session_id, "Login successful"
            
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            return False, None, "Authentication failed"
    
    def _create_session(self, user_id: str, ip_address: str, user_agent: str) -> str:
        """Create a new session"""
        # Clean up old sessions for user
        self._cleanup_user_sessions(user_id)
        
        session_id = secrets.token_urlsafe(32)
        expires_at = datetime.utcnow() + timedelta(hours=SecurityConfig.SESSION_TIMEOUT_HOURS)
        
        session = Session(
            session_id=session_id,
            user_id=user_id,
            created_at=datetime.utcnow(),
            expires_at=expires_at,
            ip_address=ip_address,
            user_agent=user_agent
        )
        
        self.sessions[session_id] = session
        
        # Store in Redis with expiration
        session_data = {
            'user_id': user_id,
            'created_at': session.created_at.isoformat(),
            'expires_at': session.expires_at.isoformat(),
            'ip_address': ip_address,
            'user_agent': user_agent
        }
        
        self.redis_client.setex(
            f"session:{session_id}",
            int(timedelta(hours=SecurityConfig.SESSION_TIMEOUT_HOURS).total_seconds()),
            self.fernet.encrypt(str(session_data).encode())
        )
        
        return session_id
    
    def _cleanup_user_sessions(self, user_id: str):
        """Clean up old sessions for user"""
        user_sessions = [s for s in self.sessions.values() if s.user_id == user_id and s.is_active]
        
        # Keep only the most recent sessions
        if len(user_sessions) >= SecurityConfig.MAX_CONCURRENT_SESSIONS:
            # Sort by creation time and deactivate oldest
            user_sessions.sort(key=lambda x: x.created_at)
            sessions_to_remove = user_sessions[:-SecurityConfig.MAX_CONCURRENT_SESSIONS + 1]
            
            for session in sessions_to_remove:
                self.logout_user(session.session_id)
    
    def validate_session(self, session_id: str) -> Tuple[bool, Optional[User]]:
        """Validate session and return user"""
        try:
            # Check in-memory first
            session = self.sessions.get(session_id)
            
            if not session:
                # Check Redis
                session_data = self.redis_client.get(f"session:{session_id}")
                if not session_data:
                    return False, None
                
                # Decrypt and parse session data
                decrypted_data = self.fernet.decrypt(session_data).decode()
                # Would parse the session data properly in production
            
            if not session or not session.is_active:
                return False, None
            
            # Check expiration
            if session.expires_at < datetime.utcnow():
                session.is_active = False
                return False, None
            
            # Get user
            user = self.users.get(session.user_id)
            if not user or not user.is_active:
                return False, None
            
            return True, user
            
        except Exception as e:
            logger.error(f"Session validation error: {e}")
            return False, None
    
    def logout_user(self, session_id: str) -> bool:
        """Logout user and invalidate session"""
        try:
            session = self.sessions.get(session_id)
            if session:
                session.is_active = False
                self._log_security_event("logout", session.user_id, {"session_id": session_id})
            
            # Remove from Redis
            self.redis_client.delete(f"session:{session_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Logout error: {e}")
            return False
    
    def check_permission(self, user: User, permission: Permission) -> bool:
        """Check if user has specific permission"""
        return permission in user.permissions
    
    def generate_jwt_token(self, user_id: str, permissions: List[Permission]) -> str:
        """Generate JWT token for API access"""
        payload = {
            'user_id': user_id,
            'permissions': [p.value for p in permissions],
            'iat': datetime.utcnow(),
            'exp': datetime.utcnow() + timedelta(hours=SecurityConfig.JWT_EXPIRATION_HOURS)
        }
        
        return jwt.encode(payload, SecurityConfig.JWT_SECRET_KEY, algorithm=SecurityConfig.JWT_ALGORITHM)
    
    def validate_jwt_token(self, token: str) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Validate JWT token"""
        try:
            payload = jwt.decode(token, SecurityConfig.JWT_SECRET_KEY, algorithms=[SecurityConfig.JWT_ALGORITHM])
            return True, payload
        except jwt.ExpiredSignatureError:
            return False, None
        except jwt.InvalidTokenError:
            return False, None
    
    def check_rate_limit(self, identifier: str) -> bool:
        """Check rate limiting for identifier (IP, user, etc.)"""
        current_time = time.time()
        window_start = current_time - (SecurityConfig.RATE_LIMIT_WINDOW_MINUTES * 60)
        
        # Clean old requests
        if identifier in self.rate_limits:
            self.rate_limits[identifier] = [
                req_time for req_time in self.rate_limits[identifier]
                if req_time > window_start
            ]
        else:
            self.rate_limits[identifier] = []
        
        # Check if under limit
        if len(self.rate_limits[identifier]) >= SecurityConfig.RATE_LIMIT_REQUESTS:
            return False
        
        # Add current request
        self.rate_limits[identifier].append(current_time)
        return True
    
    def _log_security_event(self, event_type: str, user_id: Optional[str], metadata: Dict[str, Any]):
        """Log security events"""
        event = {
            'timestamp': datetime.utcnow().isoformat(),
            'event_type': event_type,
            'user_id': user_id,
            'metadata': metadata
        }
        
        # Store in Redis for audit trail
        self.redis_client.lpush('security_events', str(event))
        self.redis_client.ltrim('security_events', 0, 10000)  # Keep last 10k events
        
        logger.info(f"Security event: {event_type} - {metadata}")

# Authentication decorators
def require_auth(permission: Permission = None):
    """Decorator to require authentication and optional permission"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract session from request (implementation depends on framework)
            session_id = kwargs.get('session_id') or getattr(args[0], 'session_id', None)
            
            if not session_id:
                raise PermissionError("Authentication required")
            
            # Validate session
            auth_manager = AuthenticationManager()
            is_valid, user = auth_manager.validate_session(session_id)
            
            if not is_valid:
                raise PermissionError("Invalid session")
            
            # Check permission if specified
            if permission and not auth_manager.check_permission(user, permission):
                raise PermissionError(f"Permission required: {permission.value}")
            
            # Add user to kwargs
            kwargs['current_user'] = user
            
            return await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
        
        return wrapper
    return decorator

def require_api_key():
    """Decorator to require API key authentication"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract API key from request
            api_key = kwargs.get('api_key') or getattr(args[0], 'api_key', None)
            
            if not api_key:
                raise PermissionError("API key required")
            
            # Validate JWT token
            auth_manager = AuthenticationManager()
            is_valid, payload = auth_manager.validate_jwt_token(api_key)
            
            if not is_valid:
                raise PermissionError("Invalid API key")
            
            # Add payload to kwargs
            kwargs['token_payload'] = payload
            
            return await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
        
        return wrapper
    return decorator

# Global authentication manager
auth_manager = AuthenticationManager()

# Example usage
async def example_usage():
    """Example of authentication system usage"""
    
    # Create users
    success, user_id = auth_manager.create_user("admin", "admin@company.com", "SecurePass123!", UserRole.ADMIN)
    print(f"Admin created: {success}, ID: {user_id}")
    
    success, agent_id = auth_manager.create_user("agent1", "agent@company.com", "AgentPass456!", UserRole.AGENT)
    print(f"Agent created: {success}, ID: {agent_id}")
    
    # Authenticate
    success, session_id, message = auth_manager.authenticate_user("admin", "SecurePass123!", "192.168.1.100", "Mozilla/5.0")
    print(f"Login: {success}, Session: {session_id}, Message: {message}")
    
    if success:
        # Validate session
        is_valid, user = auth_manager.validate_session(session_id)
        print(f"Session valid: {is_valid}, User: {user.username if user else None}")
        
        # Check permissions
        can_manage = auth_manager.check_permission(user, Permission.MANAGE_USERS)
        print(f"Can manage users: {can_manage}")
        
        # Generate API token
        api_token = auth_manager.generate_jwt_token(user.user_id, user.permissions)
        print(f"API Token: {api_token[:50]}...")
        
        # Validate API token
        is_valid, payload = auth_manager.validate_jwt_token(api_token)
        print(f"API token valid: {is_valid}")

if __name__ == "__main__":
    asyncio.run(example_usage())