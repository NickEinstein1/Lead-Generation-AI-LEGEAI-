"""
API Security Gateway
Provides comprehensive API security including rate limiting, authentication,
input validation, and threat protection
"""

import asyncio
import json
import logging
import hashlib
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum
import redis
from functools import wraps
import re
import bleach
from urllib.parse import unquote

from .advanced_threat_protection import threat_protection, ThreatSeverity
from .authentication import auth_manager, Permission

logger = logging.getLogger(__name__)

class SecurityLevel(Enum):
    PUBLIC = "public"
    AUTHENTICATED = "authenticated"
    AUTHORIZED = "authorized"
    ADMIN_ONLY = "admin_only"

class ValidationRule(Enum):
    REQUIRED = "required"
    EMAIL = "email"
    PHONE = "phone"
    ALPHANUMERIC = "alphanumeric"
    NUMERIC = "numeric"
    LENGTH_MIN = "length_min"
    LENGTH_MAX = "length_max"
    REGEX = "regex"
    NO_HTML = "no_html"
    NO_SQL = "no_sql"
    NO_SCRIPT = "no_script"

@dataclass
class RateLimitRule:
    requests_per_minute: int
    requests_per_hour: int
    requests_per_day: int
    burst_limit: int = 10

@dataclass
class ValidationSchema:
    field_name: str
    rules: List[ValidationRule]
    rule_params: Dict[str, Any] = None
    required: bool = True

@dataclass
class SecurityPolicy:
    endpoint: str
    security_level: SecurityLevel
    required_permissions: List[Permission] = None
    rate_limit: RateLimitRule = None
    validation_schema: List[ValidationSchema] = None
    custom_validators: List[Callable] = None

class APISecurityGateway:
    """Comprehensive API security gateway"""
    
    def __init__(self, redis_config: Dict[str, Any] = None):
        self.redis_config = redis_config or {
            'host': 'localhost',
            'port': 6379,
            'db': 7
        }
        
        self.redis_client = redis.Redis(**self.redis_config)
        
        # Security policies
        self.security_policies: Dict[str, SecurityPolicy] = {}
        
        # Rate limiting
        self.rate_limits: Dict[str, Dict[str, List[float]]] = {}
        
        # Request tracking
        self.request_logs: Dict[str, List[Dict[str, Any]]] = {}
        
        # Security headers
        self.security_headers = {
            'X-Content-Type-Options': 'nosniff',
            'X-Frame-Options': 'DENY',
            'X-XSS-Protection': '1; mode=block',
            'Strict-Transport-Security': 'max-age=31536000; includeSubDomains',
            'Content-Security-Policy': "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'",
            'Referrer-Policy': 'strict-origin-when-cross-origin'
        }
        
        # Initialize default policies
        self._initialize_default_policies()
    
    def _initialize_default_policies(self):
        """Initialize default security policies"""
        
        # Authentication endpoints
        self.add_security_policy(SecurityPolicy(
            endpoint="/auth/login",
            security_level=SecurityLevel.PUBLIC,
            rate_limit=RateLimitRule(10, 50, 200, 5),
            validation_schema=[
                ValidationSchema("username", [ValidationRule.REQUIRED, ValidationRule.LENGTH_MIN], {"length_min": 3}),
                ValidationSchema("password", [ValidationRule.REQUIRED, ValidationRule.LENGTH_MIN], {"length_min": 8})
            ]
        ))
        
        # Lead scoring endpoints
        self.add_security_policy(SecurityPolicy(
            endpoint="/lead-scoring/*",
            security_level=SecurityLevel.AUTHENTICATED,
            required_permissions=[Permission.GENERATE_SCORES],
            rate_limit=RateLimitRule(100, 1000, 5000, 20)
        ))
        
        # Admin endpoints
        self.add_security_policy(SecurityPolicy(
            endpoint="/admin/*",
            security_level=SecurityLevel.ADMIN_ONLY,
            required_permissions=[Permission.MANAGE_USERS],
            rate_limit=RateLimitRule(50, 200, 1000, 10)
        ))
    
    def add_security_policy(self, policy: SecurityPolicy):
        """Add security policy for endpoint"""
        self.security_policies[policy.endpoint] = policy
    
    async def process_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process incoming request through security gateway"""
        
        start_time = time.time()
        request_id = hashlib.md5(f"{request_data.get('path', '')}{start_time}".encode()).hexdigest()[:16]
        
        try:
            # 1. Basic request validation
            if not self._validate_basic_request(request_data):
                return self._create_error_response(400, "Invalid request format", request_id)
            
            # 2. Threat analysis
            should_block, threats = await threat_protection.analyze_request(request_data)
            if should_block:
                return self._create_error_response(403, "Request blocked by threat protection", request_id)
            
            # 3. Find matching security policy
            policy = self._find_matching_policy(request_data.get('path', ''))
            
            # 4. Rate limiting
            if policy and policy.rate_limit:
                rate_limit_result = await self._check_rate_limit(
                    request_data.get('source_ip', ''), 
                    policy.rate_limit
                )
                if not rate_limit_result['allowed']:
                    return self._create_error_response(429, "Rate limit exceeded", request_id, rate_limit_result)
            
            # 5. Authentication and authorization
            auth_result = await self._check_authentication_authorization(request_data, policy)
            if not auth_result['success']:
                return self._create_error_response(401 if 'authentication' in auth_result['error'] else 403, 
                                                 auth_result['error'], request_id)
            
            # 6. Input validation
            if policy and policy.validation_schema:
                validation_result = await self._validate_input(request_data, policy.validation_schema)
                if not validation_result['valid']:
                    return self._create_error_response(400, f"Validation failed: {validation_result['errors']}", request_id)
            
            # 7. Custom validation
            if policy and policy.custom_validators:
                for validator in policy.custom_validators:
                    if not await validator(request_data):
                        return self._create_error_response(400, "Custom validation failed", request_id)
            
            # 8. Log successful request
            await self._log_request(request_data, request_id, "allowed", time.time() - start_time)
            
            # Return success with security headers
            return {
                'status': 'allowed',
                'request_id': request_id,
                'user': auth_result.get('user'),
                'security_headers': self.security_headers,
                'processing_time_ms': round((time.time() - start_time) * 1000, 2)
            }
            
        except Exception as e:
            logger.error(f"Security gateway error: {e}")
            await self._log_request(request_data, request_id, "error", time.time() - start_time)
            return self._create_error_response(500, "Internal security error", request_id)
    
    def _validate_basic_request(self, request_data: Dict[str, Any]) -> bool:
        """Validate basic request structure"""
        required_fields = ['path', 'method', 'source_ip']
        return all(field in request_data for field in required_fields)
    
    def _find_matching_policy(self, path: str) -> Optional[SecurityPolicy]:
        """Find matching security policy for path"""
        # Exact match first
        if path in self.security_policies:
            return self.security_policies[path]
        
        # Wildcard matching
        for policy_path, policy in self.security_policies.items():
            if policy_path.endswith('*'):
                prefix = policy_path[:-1]
                if path.startswith(prefix):
                    return policy
        
        return None
    
    async def _check_rate_limit(self, identifier: str, rate_limit: RateLimitRule) -> Dict[str, Any]:
        """Check rate limiting for identifier"""
        current_time = time.time()
        
        # Initialize if not exists
        if identifier not in self.rate_limits:
            self.rate_limits[identifier] = {
                'minute': [],
                'hour': [],
                'day': []
            }
        
        limits = self.rate_limits[identifier]
        
        # Clean old entries
        limits['minute'] = [t for t in limits['minute'] if current_time - t < 60]
        limits['hour'] = [t for t in limits['hour'] if current_time - t < 3600]
        limits['day'] = [t for t in limits['day'] if current_time - t < 86400]
        
        # Check limits
        if (len(limits['minute']) >= rate_limit.requests_per_minute or
            len(limits['hour']) >= rate_limit.requests_per_hour or
            len(limits['day']) >= rate_limit.requests_per_day):
            
            return {
                'allowed': False,
                'current_counts': {
                    'minute': len(limits['minute']),
                    'hour': len(limits['hour']),
                    'day': len(limits['day'])
                },
                'limits': {
                    'minute': rate_limit.requests_per_minute,
                    'hour': rate_limit.requests_per_hour,
                    'day': rate_limit.requests_per_day
                }
            }
        
        # Add current request
        limits['minute'].append(current_time)
        limits['hour'].append(current_time)
        limits['day'].append(current_time)
        
        return {'allowed': True}
    
    async def _check_authentication_authorization(self, request_data: Dict[str, Any], 
                                                policy: Optional[SecurityPolicy]) -> Dict[str, Any]:
        """Check authentication and authorization"""
        
        if not policy or policy.security_level == SecurityLevel.PUBLIC:
            return {'success': True}
        
        # Extract authentication token
        auth_header = request_data.get('headers', {}).get('authorization', '')
        session_id = request_data.get('session_id')
        api_key = request_data.get('api_key')
        
        user = None
        
        # Try session authentication
        if session_id:
            is_valid, user = auth_manager.validate_session(session_id)
            if not is_valid:
                return {'success': False, 'error': 'Invalid session'}
        
        # Try API key authentication
        elif api_key or auth_header.startswith('Bearer '):
            token = api_key or auth_header.replace('Bearer ', '')
            is_valid, payload = auth_manager.validate_jwt_token(token)
            if not is_valid:
                return {'success': False, 'error': 'Invalid API key'}
            
            # Get user from payload
            user_id = payload.get('user_id')
            if user_id:
                user = auth_manager.users.get(user_id)
        
        else:
            return {'success': False, 'error': 'Authentication required'}
        
        if not user:
            return {'success': False, 'error': 'User not found'}
        
        # Check authorization
        if policy.security_level == SecurityLevel.ADMIN_ONLY:
            if user.role.value != 'admin':
                return {'success': False, 'error': 'Admin access required'}
        
        if policy.required_permissions:
            for permission in policy.required_permissions:
                if not auth_manager.check_permission(user, permission):
                    return {'success': False, 'error': f'Permission required: {permission.value}'}
        
        return {'success': True, 'user': user}
    
    async def _validate_input(self, request_data: Dict[str, Any], 
                            validation_schema: List[ValidationSchema]) -> Dict[str, Any]:
        """Validate input according to schema"""
        
        errors = []
        request_body = request_data.get('body', {})
        
        for schema in validation_schema:
            field_value = request_body.get(schema.field_name)
            
            # Check required
            if schema.required and (field_value is None or field_value == ''):
                errors.append(f"{schema.field_name} is required")
                continue
            
            if field_value is None:
                continue
            
            # Apply validation rules
            for rule in schema.rules:
                rule_params = schema.rule_params or {}
                
                if rule == ValidationRule.EMAIL:
                    if not re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', str(field_value)):
                        errors.append(f"{schema.field_name} must be a valid email")
                
                elif rule == ValidationRule.PHONE:
                    if not re.match(r'^\+?1?[0-9]{10,15}$', re.sub(r'[^\d+]', '', str(field_value))):
                        errors.append(f"{schema.field_name} must be a valid phone number")
                
                elif rule == ValidationRule.ALPHANUMERIC:
                    if not re.match(r'^[a-zA-Z0-9]+$', str(field_value)):
                        errors.append(f"{schema.field_name} must be alphanumeric")
                
                elif rule == ValidationRule.NUMERIC:
                    try:
                        float(field_value)
                    except (ValueError, TypeError):
                        errors.append(f"{schema.field_name} must be numeric")
                
                elif rule == ValidationRule.LENGTH_MIN:
                    min_length = rule_params.get('length_min', 0)
                    if len(str(field_value)) < min_length:
                        errors.append(f"{schema.field_name} must be at least {min_length} characters")
                
                elif rule == ValidationRule.LENGTH_MAX:
                    max_length = rule_params.get('length_max', 1000)
                    if len(str(field_value)) > max_length:
                        errors.append(f"{schema.field_name} must be at most {max_length} characters")
                
                elif rule == ValidationRule.NO_HTML:
                    if bleach.clean(str(field_value), strip=True) != str(field_value):
                        errors.append(f"{schema.field_name} cannot contain HTML")
                
                elif rule == ValidationRule.NO_SQL:
                    sql_patterns = [r'\bunion\b', r'\bselect\b', r'\binsert\b', r'\bdelete\b', r'\bdrop\b']
                    if any(re.search(pattern, str(field_value), re.IGNORECASE) for pattern in sql_patterns):
                        errors.append(f"{schema.field_name} contains potentially malicious content")
                
                elif rule == ValidationRule.NO_SCRIPT:
                    if re.search(r'<script|javascript:|on\w+\s*=', str(field_value), re.IGNORECASE):
                        errors.append(f"{schema.field_name} contains potentially malicious script")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors
        }
    
    async def _log_request(self, request_data: Dict[str, Any], request_id: str, 
                          status: str, processing_time: float):
        """Log request for audit and monitoring"""
        
        log_entry = {
            'request_id': request_id,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'source_ip': request_data.get('source_ip'),
            'method': request_data.get('method'),
            'path': request_data.get('path'),
            'user_agent': request_data.get('headers', {}).get('user-agent'),
            'status': status,
            'processing_time_ms': round(processing_time * 1000, 2)
        }
        
        # Store in Redis
        self.redis_client.lpush('api_requests', json.dumps(log_entry))
        self.redis_client.ltrim('api_requests', 0, 100000)  # Keep last 100k requests
    
    def _create_error_response(self, status_code: int, message: str, request_id: str, 
                             additional_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Create standardized error response"""
        
        response = {
            'status': 'error',
            'status_code': status_code,
            'message': message,
            'request_id': request_id,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'security_headers': self.security_headers
        }
        
        if additional_data:
            response.update(additional_data)
        
        return response
    
    def get_security_metrics(self) -> Dict[str, Any]:
        """Get security metrics and statistics"""
        
        # Get recent requests from Redis
        recent_requests = []
        for i in range(min(1000, self.redis_client.llen('api_requests'))):
            request_data = self.redis_client.lindex('api_requests', i)
            if request_data:
                recent_requests.append(json.loads(request_data))
        
        # Calculate metrics
        total_requests = len(recent_requests)
        blocked_requests = len([r for r in recent_requests if r.get('status') == 'blocked'])
        error_requests = len([r for r in recent_requests if r.get('status') == 'error'])
        
        # Top IPs
        ip_counts = {}
        for request in recent_requests:
            ip = request.get('source_ip', 'unknown')
            ip_counts[ip] = ip_counts.get(ip, 0) + 1
        
        top_ips = sorted(ip_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            'total_requests': total_requests,
            'blocked_requests': blocked_requests,
            'error_requests': error_requests,
            'success_rate': round((total_requests - blocked_requests - error_requests) / max(total_requests, 1) * 100, 2),
            'top_source_ips': top_ips,
            'active_policies': len(self.security_policies),
            'rate_limited_ips': len(self.rate_limits),
            'last_updated': datetime.now(timezone.utc).isoformat()
        }

# Global API security gateway
api_security_gateway = APISecurityGateway()