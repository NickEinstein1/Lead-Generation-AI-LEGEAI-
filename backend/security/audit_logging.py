"""
Comprehensive Audit Logging System
Provides detailed audit trails for compliance and security monitoring
"""

import json
import logging
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import redis
import asyncio
from pathlib import Path
import gzip
import os

logger = logging.getLogger(__name__)

class AuditEventType(Enum):
    USER_LOGIN = "user_login"
    USER_LOGOUT = "user_logout"
    USER_CREATED = "user_created"
    USER_UPDATED = "user_updated"
    USER_DELETED = "user_deleted"
    PERMISSION_GRANTED = "permission_granted"
    PERMISSION_REVOKED = "permission_revoked"
    DATA_ACCESS = "data_access"
    DATA_EXPORT = "data_export"
    DATA_DELETION = "data_deletion"
    LEAD_SCORED = "lead_scored"
    MESSAGE_GENERATED = "message_generated"
    CONSENT_RECORDED = "consent_recorded"
    CONSENT_WITHDRAWN = "consent_withdrawn"
    SECURITY_VIOLATION = "security_violation"
    CONFIGURATION_CHANGED = "configuration_changed"
    API_ACCESS = "api_access"
    SYSTEM_ERROR = "system_error"

class AuditSeverity(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

class ComplianceFramework(Enum):
    GDPR = "gdpr"
    CCPA = "ccpa"
    HIPAA = "hipaa"
    SOX = "sox"
    PCI_DSS = "pci_dss"
    NAIC = "naic"

@dataclass
class AuditEvent:
    event_id: str
    event_type: AuditEventType
    severity: AuditSeverity
    user_id: Optional[str]
    session_id: Optional[str]
    source_ip: str
    user_agent: Optional[str]
    resource_type: str
    resource_id: Optional[str]
    action: str
    outcome: str  # success, failure, partial
    details: Dict[str, Any]
    compliance_frameworks: List[ComplianceFramework] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    retention_period_days: int = 2555  # 7 years default

@dataclass
class AuditQuery:
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    event_types: Optional[List[AuditEventType]] = None
    user_ids: Optional[List[str]] = None
    resource_types: Optional[List[str]] = None
    severity_levels: Optional[List[AuditSeverity]] = None
    compliance_frameworks: Optional[List[ComplianceFramework]] = None
    source_ips: Optional[List[str]] = None
    limit: int = 1000

class AuditLogger:
    """Comprehensive audit logging system"""
    
    def __init__(self, redis_config: Dict[str, Any] = None, storage_path: str = "audit_logs"):
        self.redis_config = redis_config or {
            'host': 'localhost',
            'port': 6379,
            'db': 8
        }
        
        self.redis_client = redis.Redis(**self.redis_config)
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        
        # In-memory cache for recent events
        self.recent_events: List[AuditEvent] = []
        self.max_memory_events = 10000
        
        # Configuration
        self.config = self._load_audit_config()
        
        # Initialize background tasks
        self._start_background_tasks()
    
    def _load_audit_config(self) -> Dict[str, Any]:
        """Load audit logging configuration"""
        return {
            'retention_policies': {
                'default_days': 2555,  # 7 years
                'critical_events_days': 3650,  # 10 years
                'compliance_events_days': 2555,  # 7 years
                'system_events_days': 365  # 1 year
            },
            'storage': {
                'compress_after_days': 30,
                'archive_after_days': 365,
                'backup_enabled': True,
                'backup_interval_hours': 24
            },
            'alerting': {
                'critical_events_immediate': True,
                'security_violations_immediate': True,
                'compliance_violations_immediate': True
            },
            'compliance_mapping': {
                ComplianceFramework.GDPR: [
                    AuditEventType.DATA_ACCESS,
                    AuditEventType.DATA_EXPORT,
                    AuditEventType.DATA_DELETION,
                    AuditEventType.CONSENT_RECORDED,
                    AuditEventType.CONSENT_WITHDRAWN
                ],
                ComplianceFramework.HIPAA: [
                    AuditEventType.DATA_ACCESS,
                    AuditEventType.DATA_EXPORT,
                    AuditEventType.USER_LOGIN,
                    AuditEventType.PERMISSION_GRANTED
                ],
                ComplianceFramework.SOX: [
                    AuditEventType.CONFIGURATION_CHANGED,
                    AuditEventType.PERMISSION_GRANTED,
                    AuditEventType.USER_CREATED,
                    AuditEventType.DATA_DELETION
                ]
            }
        }
    
    async def log_event(self, event: AuditEvent) -> str:
        """Log an audit event"""
        
        # Generate event ID if not provided
        if not event.event_id:
            event.event_id = self._generate_event_id(event)
        
        # Determine compliance frameworks
        if not event.compliance_frameworks:
            event.compliance_frameworks = self._determine_compliance_frameworks(event)
        
        # Set retention period based on event type and severity
        event.retention_period_days = self._calculate_retention_period(event)
        
        # Add to memory cache
        self.recent_events.append(event)
        if len(self.recent_events) > self.max_memory_events:
            self.recent_events.pop(0)
        
        # Store in Redis for immediate access
        await self._store_in_redis(event)
        
        # Queue for persistent storage
        await self._queue_for_storage(event)
        
        # Check for immediate alerts
        if self._should_alert_immediately(event):
            await self._send_immediate_alert(event)
        
        logger.info(f"Audit event logged: {event.event_id} - {event.event_type.value}")
        return event.event_id
    
    async def log_user_action(self, user_id: str, action: str, resource_type: str,
                            resource_id: Optional[str] = None, outcome: str = "success",
                            details: Dict[str, Any] = None, source_ip: str = "unknown",
                            session_id: Optional[str] = None) -> str:
        """Convenience method to log user actions"""
        
        event = AuditEvent(
            event_id="",
            event_type=AuditEventType.DATA_ACCESS,
            severity=AuditSeverity.INFO,
            user_id=user_id,
            session_id=session_id,
            source_ip=source_ip,
            user_agent=None,
            resource_type=resource_type,
            resource_id=resource_id,
            action=action,
            outcome=outcome,
            details=details or {}
        )
        
        return await self.log_event(event)
    
    async def log_security_event(self, event_type: AuditEventType, severity: AuditSeverity,
                               description: str, source_ip: str, details: Dict[str, Any] = None,
                               user_id: Optional[str] = None) -> str:
        """Log security-related events"""
        
        event = AuditEvent(
            event_id="",
            event_type=event_type,
            severity=severity,
            user_id=user_id,
            session_id=None,
            source_ip=source_ip,
            user_agent=None,
            resource_type="security",
            resource_id=None,
            action=description,
            outcome="detected",
            details=details or {}
        )
        
        return await self.log_event(event)
    
    async def query_events(self, query: AuditQuery) -> List[AuditEvent]:
        """Query audit events based on criteria"""
        
        # Start with recent events in memory
        results = []
        
        for event in self.recent_events:
            if self._matches_query(event, query):
                results.append(event)
        
        # If we need more results, query Redis
        if len(results) < query.limit:
            redis_results = await self._query_redis(query, query.limit - len(results))
            results.extend(redis_results)
        
        # If still need more, query persistent storage
        if len(results) < query.limit:
            storage_results = await self._query_storage(query, query.limit - len(results))
            results.extend(storage_results)
        
        # Sort by timestamp (newest first) and limit
        results.sort(key=lambda x: x.timestamp, reverse=True)
        return results[:query.limit]
    
    async def generate_compliance_report(self, framework: ComplianceFramework,
                                       start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Generate compliance audit report"""
        
        # Get relevant event types for framework
        relevant_events = self.config['compliance_mapping'].get(framework, [])
        
        query = AuditQuery(
            start_date=start_date,
            end_date=end_date,
            event_types=relevant_events,
            compliance_frameworks=[framework]
        )
        
        events = await self.query_events(query)
        
        # Analyze events
        total_events = len(events)
        event_type_counts = {}
        severity_counts = {}
        user_activity = {}
        
        for event in events:
            # Count by type
            event_type = event.event_type.value
            event_type_counts[event_type] = event_type_counts.get(event_type, 0) + 1
            
            # Count by severity
            severity = event.severity.value
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
            
            # Count by user
            if event.user_id:
                user_activity[event.user_id] = user_activity.get(event.user_id, 0) + 1
        
        # Identify potential compliance issues
        compliance_issues = []
        
        # Check for suspicious patterns
        if framework == ComplianceFramework.GDPR:
            # Check for data access without consent
            data_access_events = [e for e in events if e.event_type == AuditEventType.DATA_ACCESS]
            consent_events = [e for e in events if e.event_type == AuditEventType.CONSENT_RECORDED]
            
            if len(data_access_events) > len(consent_events) * 2:
                compliance_issues.append("High ratio of data access to consent events")
        
        return {
            'framework': framework.value,
            'report_period': {
                'start_date': start_date.isoformat(),
                'end_date': end_date.isoformat()
            },
            'summary': {
                'total_events': total_events,
                'event_types': event_type_counts,
                'severity_distribution': severity_counts,
                'active_users': len(user_activity),
                'top_users': sorted(user_activity.items(), key=lambda x: x[1], reverse=True)[:10]
            },
            'compliance_issues': compliance_issues,
            'generated_at': datetime.utcnow().isoformat()
        }
    
    def _generate_event_id(self, event: AuditEvent) -> str:
        """Generate unique event ID"""
        content = f"{event.event_type.value}{event.timestamp.isoformat()}{event.user_id}{event.source_ip}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def _determine_compliance_frameworks(self, event: AuditEvent) -> List[ComplianceFramework]:
        """Determine which compliance frameworks apply to this event"""
        frameworks = []
        
        for framework, event_types in self.config['compliance_mapping'].items():
            if event.event_type in event_types:
                frameworks.append(framework)
        
        return frameworks
    
    def _calculate_retention_period(self, event: AuditEvent) -> int:
        """Calculate retention period based on event characteristics"""
        
        if event.severity == AuditSeverity.CRITICAL:
            return self.config['retention_policies']['critical_events_days']
        
        if event.compliance_frameworks:
            return self.config['retention_policies']['compliance_events_days']
        
        if event.event_type in [AuditEventType.SYSTEM_ERROR, AuditEventType.API_ACCESS]:
            return self.config['retention_policies']['system_events_days']
        
        return self.config['retention_policies']['default_days']
    
    async def _store_in_redis(self, event: AuditEvent):
        """Store event in Redis for immediate access"""
        
        event_data = {
            'event_id': event.event_id,
            'event_type': event.event_type.value,
            'severity': event.severity.value,
            'user_id': event.user_id,
            'session_id': event.session_id,
            'source_ip': event.source_ip,
            'user_agent': event.user_agent,
            'resource_type': event.resource_type,
            'resource_id': event.resource_id,
            'action': event.action,
            'outcome': event.outcome,
            'details': event.details,
            'compliance_frameworks': [f.value for f in event.compliance_frameworks],
            'timestamp': event.timestamp.isoformat(),
            'retention_period_days': event.retention_period_days
        }
        
        # Store with expiration based on retention period
        self.redis_client.setex(
            f"audit_event:{event.event_id}",
            event.retention_period_days * 86400,
            json.dumps(event_data)
        )
        
        # Add to time-based indexes
        date_key = event.timestamp.strftime('%Y-%m-%d')
        self.redis_client.zadd(f"audit_events_by_date:{date_key}", {event.event_id: event.timestamp.timestamp()})
        self.redis_client.expire(f"audit_events_by_date:{date_key}", event.retention_period_days * 86400)
    
    async def _queue_for_storage(self, event: AuditEvent):
        """Queue event for persistent storage"""
        
        storage_data = {
            'event': event,
            'storage_date': datetime.utcnow().isoformat()
        }
        
        self.redis_client.lpush('audit_storage_queue', json.dumps(storage_data, default=str))
    
    def _should_alert_immediately(self, event: AuditEvent) -> bool:
        """Check if event requires immediate alerting"""
        
        if event.severity == AuditSeverity.CRITICAL and self.config['alerting']['critical_events_immediate']:
            return True
        
        if event.event_type == AuditEventType.SECURITY_VIOLATION and self.config['alerting']['security_violations_immediate']:
            return True
        
        # Check for compliance violations
        if event.compliance_frameworks and self.config['alerting']['compliance_violations_immediate']:
            if event.outcome == "failure" or event.severity in [AuditSeverity.HIGH, AuditSeverity.CRITICAL]:
                return True
        
        return False
    
    async def _send_immediate_alert(self, event: AuditEvent):
        """Send immediate alert for critical events"""
        
        alert_data = {
            'event_id': event.event_id,
            'event_type': event.event_type.value,
            'severity': event.severity.value,
            'description': f"{event.action} on {event.resource_type}",
            'user_id': event.user_id,
            'source_ip': event.source_ip,
            'timestamp': event.timestamp.isoformat(),
            'details': event.details
        }
        
        # This would integrate with notification system
        logger.critical(f"AUDIT ALERT: {json.dumps(alert_data)}")
    
    def _matches_query(self, event: AuditEvent, query: AuditQuery) -> bool:
        """Check if event matches query criteria"""
        
        if query.start_date and event.timestamp < query.start_date:
            return False
        
        if query.end_date and event.timestamp > query.end_date:
            return False
        
        if query.event_types and event.event_type not in query.event_types:
            return False
        
        if query.user_ids and event.user_id not in query.user_ids:
            return False
        
        if query.resource_types and event.resource_type not in query.resource_types:
            return False
        
        if query.severity_levels and event.severity not in query.severity_levels:
            return False
        
        if query.compliance_frameworks:
            if not any(f in event.compliance_frameworks for f in query.compliance_frameworks):
                return False
        
        if query.source_ips and event.source_ip not in query.source_ips:
            return False
        
        return True
    
    async def _query_redis(self, query: AuditQuery, limit: int) -> List[AuditEvent]:
        """Query events from Redis"""
        results = []
        
        # This would implement Redis querying logic
        # For now, returning empty list
        return results
    
    async def _query_storage(self, query: AuditQuery, limit: int) -> List[AuditEvent]:
        """Query events from persistent storage"""
        results = []
        
        # This would implement file-based querying logic
        # For now, returning empty list
        return results
    
    def _start_background_tasks(self):
        """Start background tasks for storage and maintenance"""
        
        # This would start background tasks for:
        # - Moving events from Redis to persistent storage
        # - Compressing old log files
        # - Cleaning up expired events
        # - Generating periodic reports
        pass

# Global audit logger instance
audit_logger = AuditLogger()