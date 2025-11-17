"""
Zero Trust Architecture Implementation
Implements "never trust, always verify" security model
"""

import json
import logging
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import redis
import asyncio
from ipaddress import ip_address, ip_network

from .authentication import auth_manager, User, UserRole
from .audit_logging import audit_logger, AuditEventType, AuditSeverity

logger = logging.getLogger(__name__)

class TrustLevel(Enum):
    UNTRUSTED = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    VERIFIED = 4

class AccessDecision(Enum):
    ALLOW = "allow"
    DENY = "deny"
    CHALLENGE = "challenge"
    MONITOR = "monitor"

class RiskFactor(Enum):
    LOCATION_ANOMALY = "location_anomaly"
    TIME_ANOMALY = "time_anomaly"
    DEVICE_UNKNOWN = "device_unknown"
    BEHAVIOR_ANOMALY = "behavior_anomaly"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    DATA_SENSITIVITY = "data_sensitivity"
    NETWORK_UNTRUSTED = "network_untrusted"

@dataclass
class DeviceFingerprint:
    device_id: str
    user_agent: str
    screen_resolution: str
    timezone: str
    language: str
    platform: str
    browser_version: str
    plugins: List[str]
    fingerprint_hash: str
    first_seen: datetime = field(default_factory=datetime.utcnow)
    last_seen: datetime = field(default_factory=datetime.utcnow)
    trust_score: float = 0.0

@dataclass
class AccessContext:
    user_id: str
    session_id: str
    source_ip: str
    device_fingerprint: DeviceFingerprint
    requested_resource: str
    requested_action: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    user_agent: Optional[str] = None
    location: Optional[Dict[str, Any]] = None

@dataclass
class RiskAssessment:
    overall_risk_score: float
    trust_level: TrustLevel
    risk_factors: List[RiskFactor]
    risk_details: Dict[str, Any]
    confidence: float
    assessment_time: datetime = field(default_factory=datetime.utcnow)

@dataclass
class AccessPolicy:
    policy_id: str
    name: str
    resource_pattern: str
    required_trust_level: TrustLevel
    allowed_roles: List[UserRole]
    risk_thresholds: Dict[str, float]
    additional_verification: List[str]  # MFA, device verification, etc.
    time_restrictions: Optional[Dict[str, Any]] = None
    location_restrictions: Optional[List[str]] = None
    active: bool = True

class ZeroTrustEngine:
    """Zero Trust security engine"""
    
    def __init__(self, redis_config: Dict[str, Any] = None):
        self.redis_config = redis_config or {
            'host': 'localhost',
            'port': 6379,
            'db': 9
        }
        
        self.redis_client = redis.Redis(**self.redis_config)
        
        # Trust and risk data
        self.device_fingerprints: Dict[str, DeviceFingerprint] = {}
        self.user_behavior_baselines: Dict[str, Dict[str, Any]] = {}
        self.access_policies: Dict[str, AccessPolicy] = {}
        
        # Configuration
        self.config = self._load_zero_trust_config()
        
        # Initialize default policies
        self._initialize_default_policies()
    
    def _load_zero_trust_config(self) -> Dict[str, Any]:
        """Load zero trust configuration"""
        return {
            'risk_scoring': {
                'location_weight': 0.3,
                'time_weight': 0.2,
                'device_weight': 0.25,
                'behavior_weight': 0.25,
                'max_risk_score': 1.0
            },
            'trust_thresholds': {
                TrustLevel.UNTRUSTED: 0.0,
                TrustLevel.LOW: 0.2,
                TrustLevel.MEDIUM: 0.5,
                TrustLevel.HIGH: 0.7,
                TrustLevel.VERIFIED: 0.9
            },
            'verification_requirements': {
                'high_risk_resources': ['admin', 'financial', 'pii'],
                'mfa_required_trust_below': TrustLevel.HIGH,
                'device_verification_required': True,
                'continuous_monitoring': True
            },
            'behavioral_analysis': {
                'learning_period_days': 30,
                'anomaly_threshold': 0.7,
                'update_frequency_hours': 1
            },
            'network_zones': {
                'trusted_networks': ['10.0.0.0/8', '192.168.0.0/16'],
                'dmz_networks': ['172.16.0.0/12'],
                'untrusted_networks': ['0.0.0.0/0']  # Everything else
            }
        }
    
    def _initialize_default_policies(self):
        """Initialize default access policies"""
        
        policies = [
            AccessPolicy(
                policy_id="admin_access",
                name="Administrative Access",
                resource_pattern="/admin/*",
                required_trust_level=TrustLevel.VERIFIED,
                allowed_roles=[UserRole.ADMIN],
                risk_thresholds={'max_risk': 0.3},
                additional_verification=['mfa', 'device_verification']
            ),
            AccessPolicy(
                policy_id="pii_access",
                name="PII Data Access",
                resource_pattern="/api/*/pii/*",
                required_trust_level=TrustLevel.HIGH,
                allowed_roles=[UserRole.ADMIN, UserRole.MANAGER],
                risk_thresholds={'max_risk': 0.4},
                additional_verification=['mfa']
            ),
            AccessPolicy(
                policy_id="lead_scoring",
                name="Lead Scoring Access",
                resource_pattern="/api/lead-scoring/*",
                required_trust_level=TrustLevel.MEDIUM,
                allowed_roles=[UserRole.ADMIN, UserRole.MANAGER, UserRole.AGENT],
                risk_thresholds={'max_risk': 0.6},
                additional_verification=[]
            ),
            AccessPolicy(
                policy_id="public_api",
                name="Public API Access",
                resource_pattern="/api/public/*",
                required_trust_level=TrustLevel.LOW,
                allowed_roles=[UserRole.API_CLIENT],
                risk_thresholds={'max_risk': 0.8},
                additional_verification=[]
            )
        ]
        
        for policy in policies:
            self.access_policies[policy.policy_id] = policy
    
    async def evaluate_access_request(self, context: AccessContext) -> Tuple[AccessDecision, RiskAssessment]:
        """Evaluate access request using zero trust principles"""
        
        # 1. Get user information
        user = auth_manager.users.get(context.user_id)
        if not user:
            return AccessDecision.DENY, RiskAssessment(
                overall_risk_score=1.0,
                trust_level=TrustLevel.UNTRUSTED,
                risk_factors=[],
                risk_details={'error': 'User not found'},
                confidence=1.0
            )
        
        # 2. Find applicable policy
        policy = self._find_applicable_policy(context.requested_resource)
        if not policy:
            return AccessDecision.DENY, RiskAssessment(
                overall_risk_score=1.0,
                trust_level=TrustLevel.UNTRUSTED,
                risk_factors=[],
                risk_details={'error': 'No applicable policy'},
                confidence=1.0
            )
        
        # 3. Check basic authorization
        if user.role not in policy.allowed_roles:
            await audit_logger.log_security_event(
                AuditEventType.SECURITY_VIOLATION,
                AuditSeverity.HIGH,
                f"Unauthorized access attempt to {context.requested_resource}",
                context.source_ip,
                {'user_id': context.user_id, 'required_roles': [r.value for r in policy.allowed_roles]}
            )
            return AccessDecision.DENY, RiskAssessment(
                overall_risk_score=1.0,
                trust_level=TrustLevel.UNTRUSTED,
                risk_factors=[],
                risk_details={'error': 'Insufficient role'},
                confidence=1.0
            )
        
        # 4. Perform risk assessment
        risk_assessment = await self._assess_risk(context, user, policy)
        
        # 5. Make access decision
        decision = self._make_access_decision(risk_assessment, policy)
        
        # 6. Log access attempt
        await audit_logger.log_user_action(
            user_id=context.user_id,
            action=f"access_request_{decision.value}",
            resource_type="zero_trust_evaluation",
            resource_id=context.requested_resource,
            outcome="success" if decision == AccessDecision.ALLOW else "blocked",
            details={
                'risk_score': risk_assessment.overall_risk_score,
                'trust_level': risk_assessment.trust_level.value,
                'policy_id': policy.policy_id,
                'risk_factors': [f.value for f in risk_assessment.risk_factors]
            },
            source_ip=context.source_ip,
            session_id=context.session_id
        )
        
        # 7. Update behavioral baselines
        await self._update_behavior_baseline(context, user)
        
        return decision, risk_assessment
    
    async def _assess_risk(self, context: AccessContext, user: User, policy: AccessPolicy) -> RiskAssessment:
        """Perform comprehensive risk assessment"""
        
        risk_factors = []
        risk_details = {}
        risk_scores = {}
        
        # 1. Device trust assessment
        device_risk, device_factors = await self._assess_device_risk(context)
        risk_scores['device'] = device_risk
        risk_factors.extend(device_factors)
        risk_details['device'] = {'risk_score': device_risk, 'factors': [f.value for f in device_factors]}
        
        # 2. Location risk assessment
        location_risk, location_factors = await self._assess_location_risk(context, user)
        risk_scores['location'] = location_risk
        risk_factors.extend(location_factors)
        risk_details['location'] = {'risk_score': location_risk, 'factors': [f.value for f in location_factors]}
        
        # 3. Time-based risk assessment
        time_risk, time_factors = await self._assess_time_risk(context, user)
        risk_scores['time'] = time_risk
        risk_factors.extend(time_factors)
        risk_details['time'] = {'risk_score': time_risk, 'factors': [f.value for f in time_factors]}
        
        # 4. Behavioral risk assessment
        behavior_risk, behavior_factors = await self._assess_behavior_risk(context, user)
        risk_scores['behavior'] = behavior_risk
        risk_factors.extend(behavior_factors)
        risk_details['behavior'] = {'risk_score': behavior_risk, 'factors': [f.value for f in behavior_factors]}
        
        # 5. Resource sensitivity assessment
        resource_risk = self._assess_resource_sensitivity(context.requested_resource, policy)
        risk_scores['resource'] = resource_risk
        risk_details['resource'] = {'sensitivity_score': resource_risk}
        
        # Calculate overall risk score
        weights = self.config['risk_scoring']
        overall_risk = (
            risk_scores['device'] * weights['device_weight'] +
            risk_scores['location'] * weights['location_weight'] +
            risk_scores['time'] * weights['time_weight'] +
            risk_scores['behavior'] * weights['behavior_weight']
        )
        
        # Adjust for resource sensitivity
        overall_risk = min(overall_risk + (resource_risk * 0.2), 1.0)
        
        # Determine trust level
        trust_level = self._calculate_trust_level(overall_risk)
        
        # Calculate confidence based on data availability
        confidence = self._calculate_confidence(context, user)
        
        return RiskAssessment(
            overall_risk_score=overall_risk,
            trust_level=trust_level,
            risk_factors=list(set(risk_factors)),  # Remove duplicates
            risk_details=risk_details,
            confidence=confidence
        )
    
    async def _assess_device_risk(self, context: AccessContext) -> Tuple[float, List[RiskFactor]]:
        """Assess device-related risk factors"""
        
        risk_score = 0.0
        risk_factors = []
        
        device = context.device_fingerprint
        
        # Check if device is known
        if device.device_id not in self.device_fingerprints:
            risk_score += 0.4
            risk_factors.append(RiskFactor.DEVICE_UNKNOWN)
        else:
            # Check device trust score
            known_device = self.device_fingerprints[device.device_id]
            if known_device.trust_score < 0.5:
                risk_score += 0.3
                risk_factors.append(RiskFactor.DEVICE_UNKNOWN)
        
        # Check for suspicious user agent
        if self._is_suspicious_user_agent(device.user_agent):
            risk_score += 0.2
        
        return min(risk_score, 1.0), risk_factors
    
    async def _assess_location_risk(self, context: AccessContext, user: User) -> Tuple[float, List[RiskFactor]]:
        """Assess location-related risk factors"""
        
        risk_score = 0.0
        risk_factors = []
        
        # Check network zone
        network_risk = self._assess_network_zone(context.source_ip)
        risk_score += network_risk
        
        if network_risk > 0.5:
            risk_factors.append(RiskFactor.NETWORK_UNTRUSTED)
        
        # Check for location anomalies (simplified)
        user_baseline = self.user_behavior_baselines.get(user.user_id, {})
        known_locations = user_baseline.get('locations', [])
        
        if context.location and known_locations:
            # This would implement actual geolocation comparison
            # For now, simplified check
            current_country = context.location.get('country', 'unknown')
            if current_country not in [loc.get('country') for loc in known_locations]:
                risk_score += 0.3
                risk_factors.append(RiskFactor.LOCATION_ANOMALY)
        
        return min(risk_score, 1.0), risk_factors
    
    async def _assess_time_risk(self, context: AccessContext, user: User) -> Tuple[float, List[RiskFactor]]:
        """Assess time-based risk factors"""
        
        risk_score = 0.0
        risk_factors = []
        
        current_hour = context.timestamp.hour
        current_day = context.timestamp.weekday()
        
        # Check against user's typical access patterns
        user_baseline = self.user_behavior_baselines.get(user.user_id, {})
        typical_hours = user_baseline.get('access_hours', list(range(9, 18)))  # Default business hours
        typical_days = user_baseline.get('access_days', list(range(0, 5)))  # Default weekdays
        
        if current_hour not in typical_hours:
            risk_score += 0.2
            risk_factors.append(RiskFactor.TIME_ANOMALY)
        
        if current_day not in typical_days:
            risk_score += 0.1
            risk_factors.append(RiskFactor.TIME_ANOMALY)
        
        return min(risk_score, 1.0), risk_factors
    
    async def _assess_behavior_risk(self, context: AccessContext, user: User) -> Tuple[float, List[RiskFactor]]:
        """Assess behavioral risk factors"""
        
        risk_score = 0.0
        risk_factors = []
        
        # Check for privilege escalation attempts
        if self._is_privilege_escalation(context, user):
            risk_score += 0.5
            risk_factors.append(RiskFactor.PRIVILEGE_ESCALATION)
        
        # Check access frequency anomalies
        recent_access_count = await self._get_recent_access_count(user.user_id)
        user_baseline = self.user_behavior_baselines.get(user.user_id, {})
        typical_access_rate = user_baseline.get('hourly_access_rate', 10)
        
        if recent_access_count > typical_access_rate * 3:
            risk_score += 0.3
            risk_factors.append(RiskFactor.BEHAVIOR_ANOMALY)
        
        return min(risk_score, 1.0), risk_factors
    
    def _assess_resource_sensitivity(self, resource: str, policy: AccessPolicy) -> float:
        """Assess sensitivity of requested resource"""
        
        sensitive_patterns = {
            '/admin/': 0.8,
            '/api/*/pii/': 0.9,
            '/api/*/financial/': 0.8,
            '/api/*/users/': 0.6,
            '/api/*/reports/': 0.5
        }
        
        for pattern, sensitivity in sensitive_patterns.items():
            if pattern.replace('*', '') in resource:
                return sensitivity
        
        return 0.2  # Default low sensitivity
    
    def _assess_network_zone(self, ip_address: str) -> float:
        """Assess risk based on network zone"""
        
        try:
            ip = ip_address(ip_address)
            
            # Check trusted networks
            for network_str in self.config['network_zones']['trusted_networks']:
                if ip in ip_network(network_str):
                    return 0.0
            
            # Check DMZ networks
            for network_str in self.config['network_zones']['dmz_networks']:
                if ip in ip_network(network_str):
                    return 0.3
            
            # Everything else is untrusted
            return 0.7
            
        except Exception:
            return 0.8  # High risk for invalid IPs
    
    def _calculate_trust_level(self, risk_score: float) -> TrustLevel:
        """Calculate trust level based on risk score"""
        
        # Invert risk score to get trust score
        trust_score = 1.0 - risk_score
        
        thresholds = self.config['trust_thresholds']
        
        if trust_score >= thresholds[TrustLevel.VERIFIED]:
            return TrustLevel.VERIFIED
        elif trust_score >= thresholds[TrustLevel.HIGH]:
            return TrustLevel.HIGH
        elif trust_score >= thresholds[TrustLevel.MEDIUM]:
            return TrustLevel.MEDIUM
        elif trust_score >= thresholds[TrustLevel.LOW]:
            return TrustLevel.LOW
        else:
            return TrustLevel.UNTRUSTED
    
    def _calculate_confidence(self, context: AccessContext, user: User) -> float:
        """Calculate confidence in risk assessment"""
        
        confidence = 0.5  # Base confidence
        
        # Increase confidence based on available data
        if context.device_fingerprint.device_id in self.device_fingerprints:
            confidence += 0.2
        
        if user.user_id in self.user_behavior_baselines:
            confidence += 0.2
        
        if context.location:
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _make_access_decision(self, risk_assessment: RiskAssessment, policy: AccessPolicy) -> AccessDecision:
        """Make final access decision"""
        
        # Check if trust level meets policy requirements
        if risk_assessment.trust_level.value < policy.required_trust_level.value:
            return AccessDecision.CHALLENGE
        
        # Check risk thresholds
        max_risk = policy.risk_thresholds.get('max_risk', 0.7)
        if risk_assessment.overall_risk_score > max_risk:
            return AccessDecision.DENY
        
        # Check for critical risk factors
        critical_factors = [RiskFactor.PRIVILEGE_ESCALATION, RiskFactor.DEVICE_UNKNOWN]
        if any(factor in risk_assessment.risk_factors for factor in critical_factors):
            if risk_assessment.overall_risk_score > 0.5:
                return AccessDecision.CHALLENGE
        
        # Allow with monitoring for medium risk
        if risk_assessment.overall_risk_score > 0.4:
            return AccessDecision.MONITOR
        
        return AccessDecision.ALLOW
    
    def _find_applicable_policy(self, resource: str) -> Optional[AccessPolicy]:
        """Find applicable access policy for resource"""
        
        # Find exact match first
        for policy in self.access_policies.values():
            if policy.resource_pattern == resource and policy.active:
                return policy
        
        # Find pattern match
        for policy in self.access_policies.values():
            if policy.active and self._matches_pattern(resource, policy.resource_pattern):
                return policy
        
        return None
    
    def _matches_pattern(self, resource: str, pattern: str) -> bool:
        """Check if resource matches pattern"""
        
        if '*' in pattern:
            # Simple wildcard matching
            prefix = pattern.split('*')[0]
            return resource.startswith(prefix)
        
        return resource == pattern
    
    def _is_suspicious_user_agent(self, user_agent: str) -> bool:
        """Check if user agent is suspicious"""
        
        suspicious_patterns = ['bot', 'crawler', 'spider', 'scraper']
        return any(pattern in user_agent.lower() for pattern in suspicious_patterns)
    
    def _is_privilege_escalation(self, context: AccessContext, user: User) -> bool:
        """Check for privilege escalation attempts"""
        
        # Check if user is trying to access resources above their role
        admin_patterns = ['/admin/', '/api/admin/']
        if user.role != UserRole.ADMIN:
            return any(pattern in context.requested_resource for pattern in admin_patterns)
        
        return False
    
    async def _get_recent_access_count(self, user_id: str) -> int:
        """Get recent access count for user"""
        
        # This would query actual access logs
        # For now, returning a mock value
        return 5
    
    async def _update_behavior_baseline(self, context: AccessContext, user: User):
        """Update user behavioral baseline"""
        
        if user.user_id not in self.user_behavior_baselines:
            self.user_behavior_baselines[user.user_id] = {
                'access_hours': [],
                'access_days': [],
                'locations': [],
                'hourly_access_rate': 0,
                'last_updated': datetime.now(datetime.UTC).isoformat()
            }
        
        baseline = self.user_behavior_baselines[user.user_id]
        
        # Update access patterns
        current_hour = context.timestamp.hour
        current_day = context.timestamp.weekday()
        
        if current_hour not in baseline['access_hours']:
            baseline['access_hours'].append(current_hour)
        
        if current_day not in baseline['access_days']:
            baseline['access_days'].append(current_day)
        
        # Update location if available
        if context.location and context.location not in baseline['locations']:
            baseline['locations'].append(context.location)
        
        baseline['last_updated'] = datetime.now(datetime.UTC).isoformat()

# Global zero trust engine
zero_trust_engine = ZeroTrustEngine()