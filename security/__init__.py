 """
Comprehensive Security Module for Insurance Lead Scoring Platform

This module provides enterprise-grade security features including:
- Multi-factor authentication and authorization
- Zero trust architecture implementation
- Comprehensive audit logging and compliance
- Automated security testing and vulnerability assessment
- Data encryption and privacy protection
- Real-time threat detection and response
"""

from .authentication import (
    auth_manager, AuthenticationManager, User, UserRole, 
    Session, MFAMethod, AuthenticationResult
)
from .authorization import (
    authorization_manager, AuthorizationManager, Permission, 
    Resource, AccessLevel, PolicyEngine
)
from .encryption import (
    encryption_manager, EncryptionManager, EncryptionType,
    KeyManager, DataClassification
)
from .audit_logging import (
    audit_logger, AuditLogger, AuditEvent, AuditEventType,
    AuditSeverity, ComplianceFramework, AuditQuery
)
from .zero_trust_architecture import (
    zero_trust_engine, ZeroTrustEngine, TrustLevel, AccessDecision,
    RiskFactor, AccessContext, RiskAssessment, AccessPolicy
)
from .security_testing import (
    security_tester, SecurityTestingFramework, TestType,
    Severity, SecurityTest, Vulnerability, run_comprehensive_security_test
)

# Global security instances
security_components = {
    'auth_manager': auth_manager,
    'authorization_manager': authorization_manager,
    'encryption_manager': encryption_manager,
    'audit_logger': audit_logger,
    'zero_trust_engine': zero_trust_engine,
    'security_tester': security_tester
}

async def initialize_security_system():
    """Initialize the complete security system"""
    
    # Initialize authentication system
    await auth_manager.initialize()
    
    # Initialize authorization policies
    await authorization_manager.load_policies()
    
    # Initialize encryption keys
    await encryption_manager.initialize_keys()
    
    # Start audit logging
    await audit_logger.log_event(
        AuditEvent(
            event_id="",
            event_type=AuditEventType.SYSTEM_ERROR,
            severity=AuditSeverity.INFO,
            user_id=None,
            session_id=None,
            source_ip="system",
            user_agent=None,
            resource_type="security_system",
            resource_id=None,
            action="security_system_initialized",
            outcome="success",
            details={"components": list(security_components.keys())}
        )
    )

async def security_health_check() -> Dict[str, Any]:
    """Perform comprehensive security health check"""
    
    health_status = {}
    
    # Check authentication system
    health_status['authentication'] = {
        'status': 'healthy' if auth_manager.is_initialized else 'unhealthy',
        'active_sessions': len(auth_manager.active_sessions),
        'users_count': len(auth_manager.users)
    }
    
    # Check authorization system
    health_status['authorization'] = {
        'status': 'healthy' if authorization_manager.policies else 'unhealthy',
        'policies_loaded': len(authorization_manager.policies),
        'permissions_count': len(authorization_manager.permissions)
    }
    
    # Check encryption system
    health_status['encryption'] = {
        'status': 'healthy' if encryption_manager.master_key else 'unhealthy',
        'keys_loaded': len(encryption_manager.encryption_keys),
        'algorithms_available': list(encryption_manager.config['algorithms'].keys())
    }
    
    # Check audit logging
    health_status['audit_logging'] = {
        'status': 'healthy',
        'recent_events_count': len(audit_logger.recent_events),
        'storage_path': str(audit_logger.storage_path)
    }
    
    # Check zero trust engine
    health_status['zero_trust'] = {
        'status': 'healthy',
        'policies_count': len(zero_trust_engine.access_policies),
        'device_fingerprints': len(zero_trust_engine.device_fingerprints)
    }
    
    # Overall status
    all_healthy = all(
        component['status'] == 'healthy' 
        for component in health_status.values()
    )
    
    return {
        'overall_status': 'healthy' if all_healthy else 'degraded',
        'components': health_status,
        'timestamp': datetime.utcnow().isoformat()
    }

__all__ = [
    # Core managers
    'auth_manager',
    'authorization_manager', 
    'encryption_manager',
    'audit_logger',
    'zero_trust_engine',
    'security_tester',
    
    # Classes
    'AuthenticationManager',
    'AuthorizationManager',
    'EncryptionManager',
    'AuditLogger',
    'ZeroTrustEngine',
    'SecurityTestingFramework',
    
    # Data classes
    'User',
    'Session',
    'Permission',
    'Resource',
    'AuditEvent',
    'AccessContext',
    'RiskAssessment',
    'SecurityTest',
    'Vulnerability',
    
    # Enums
    'UserRole',
    'MFAMethod',
    'AccessLevel',
    'EncryptionType',
    'DataClassification',
    'AuditEventType',
    'AuditSeverity',
    'ComplianceFramework',
    'TrustLevel',
    'AccessDecision',
    'RiskFactor',
    'TestType',
    'Severity',
    
    # Functions
    'initialize_security_system',
    'security_health_check',
    'run_comprehensive_security_test',
    
    # Global components
    'security_components'
]

# Version info
__version__ = "1.0.0"
__author__ = "Insurance Lead Scoring Platform Security Team"