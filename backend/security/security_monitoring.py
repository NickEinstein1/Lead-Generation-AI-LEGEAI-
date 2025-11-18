import asyncio
import json
import logging
import hashlib
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import redis
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests
import uuid
from collections import defaultdict, deque
import threading

logger = logging.getLogger(__name__)

class ThreatLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class IncidentStatus(Enum):
    OPEN = "open"
    INVESTIGATING = "investigating"
    CONTAINED = "contained"
    RESOLVED = "resolved"
    CLOSED = "closed"

class SecurityEventType(Enum):
    AUTHENTICATION_FAILURE = "authentication_failure"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    DATA_BREACH = "data_breach"
    MALWARE_DETECTED = "malware_detected"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    POLICY_VIOLATION = "policy_violation"
    SYSTEM_COMPROMISE = "system_compromise"
    DDOS_ATTACK = "ddos_attack"
    SQL_INJECTION = "sql_injection"
    XSS_ATTEMPT = "xss_attempt"

@dataclass
class SecurityEvent:
    event_id: str
    event_type: SecurityEventType
    timestamp: datetime
    source_ip: str
    user_id: Optional[str]
    description: str
    severity: ThreatLevel
    raw_data: Dict[str, Any]
    tags: List[str] = field(default_factory=list)
    is_false_positive: bool = False

@dataclass
class SecurityIncident:
    incident_id: str
    title: str
    description: str
    severity: ThreatLevel
    status: IncidentStatus
    created_at: datetime
    updated_at: datetime
    assigned_to: Optional[str] = None
    events: List[str] = field(default_factory=list)  # Event IDs
    timeline: List[Dict[str, Any]] = field(default_factory=list)
    resolution_notes: Optional[str] = None
    lessons_learned: Optional[str] = None

@dataclass
class ThreatIntelligence:
    indicator: str
    indicator_type: str  # ip, domain, hash, etc.
    threat_type: str
    confidence: float
    source: str
    first_seen: datetime
    last_seen: datetime
    is_active: bool = True

class SecurityMonitor:
    """Comprehensive security monitoring and incident response system"""
    
    def __init__(self, redis_config: Dict[str, Any] = None):
        self.redis_config = redis_config or {
            'host': 'localhost',
            'port': 6379,
            'db': 7
        }
        
        self.redis_client = redis.Redis(**self.redis_config)
        
        # Event storage
        self.security_events: Dict[str, SecurityEvent] = {}
        self.incidents: Dict[str, SecurityIncident] = {}
        self.threat_intelligence: Dict[str, ThreatIntelligence] = {}
        
        # Monitoring state
        self.running = False
        self.event_processors: List[Callable] = []
        self.alert_handlers: List[Callable] = []
        
        # Rate limiting and anomaly detection
        self.request_counters: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.failed_login_attempts: Dict[str, List[datetime]] = defaultdict(list)
        
        # Configuration
        self.config = self._load_security_config()
        
        # Load threat intelligence
        self._load_threat_intelligence()
        
    def _load_security_config(self) -> Dict[str, Any]:
        """Load security monitoring configuration"""
        return {
            'rate_limits': {
                'requests_per_minute': 100,
                'failed_logins_per_hour': 5,
                'api_calls_per_minute': 1000
            },
            'anomaly_thresholds': {
                'unusual_login_location': 0.8,
                'unusual_access_pattern': 0.7,
                'data_exfiltration': 0.9
            },
            'incident_escalation': {
                'auto_escalate_after_minutes': 30,
                'critical_incident_immediate_alert': True
            },
            'threat_intelligence': {
                'auto_block_known_threats': True,
                'confidence_threshold': 0.7
            },
            'notifications': {
                'email_alerts': True,
                'slack_webhook': None,
                'sms_alerts': False
            }
        }
    
    def _load_threat_intelligence(self):
        """Load threat intelligence data"""
        # Sample threat intelligence (would be loaded from external sources)
        sample_threats = [
            ThreatIntelligence(
                indicator="192.168.1.100",
                indicator_type="ip",
                threat_type="known_attacker",
                confidence=0.9,
                source="internal_blacklist",
                first_seen=datetime.now(timezone.utc) - timedelta(days=30),
                last_seen=datetime.now(timezone.utc) - timedelta(days=1)
            ),
            ThreatIntelligence(
                indicator="malicious-domain.com",
                indicator_type="domain",
                threat_type="phishing",
                confidence=0.8,
                source="threat_feed",
                first_seen=datetime.now(timezone.utc) - timedelta(days=7),
                last_seen=datetime.now(timezone.utc)
            )
        ]
        
        for threat in sample_threats:
            self.threat_intelligence[threat.indicator] = threat
    
    async def start_monitoring(self):
        """Start security monitoring"""
        self.running = True
        logger.info("Security monitoring started")
        
        # Start monitoring tasks
        tasks = [
            asyncio.create_task(self._event_processing_loop()),
            asyncio.create_task(self._anomaly_detection_loop()),
            asyncio.create_task(self._threat_intelligence_loop()),
            asyncio.create_task(self._incident_management_loop())
        ]
        
        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(f"Security monitoring error: {e}")
        finally:
            self.running = False
    
    async def stop_monitoring(self):
        """Stop security monitoring"""
        self.running = False
        logger.info("Security monitoring stopped")
    
    def log_security_event(self, event_type: SecurityEventType, source_ip: str, 
                          user_id: Optional[str], description: str, 
                          raw_data: Dict[str, Any], severity: ThreatLevel = ThreatLevel.MEDIUM) -> str:
        """Log a security event"""
        event_id = str(uuid.uuid4())
        
        event = SecurityEvent(
            event_id=event_id,
            event_type=event_type,
            timestamp=datetime.now(timezone.utc),
            source_ip=source_ip,
            user_id=user_id,
            description=description,
            severity=severity,
            raw_data=raw_data
        )
        
        # Check against threat intelligence
        if self._is_known_threat(source_ip):
            event.severity = ThreatLevel.HIGH
            event.tags.append("known_threat")
        
        self.security_events[event_id] = event
        
        # Store in Redis
        event_data = {
            'event_type': event_type.value,
            'timestamp': event.timestamp.isoformat(),
            'source_ip': source_ip,
            'user_id': user_id,
            'description': description,
            'severity': severity.value,
            'raw_data': raw_data,
            'tags': event.tags
        }
        
        self.redis_client.lpush('security_events', json.dumps(event_data))
        self.redis_client.ltrim('security_events', 0, 50000)  # Keep last 50k events
        
        # Process event immediately
        asyncio.create_task(self._process_security_event(event))
        
        logger.warning(f"Security event logged: {event_type.value} from {source_ip}")
        return event_id
    
    def _is_known_threat(self, indicator: str) -> bool:
        """Check if indicator is a known threat"""
        threat = self.threat_intelligence.get(indicator)
        return threat is not None and threat.is_active and threat.confidence >= self.config['threat_intelligence']['confidence_threshold']
    
    async def _process_security_event(self, event: SecurityEvent):
        """Process a security event"""
        try:
            # Apply event processors
            for processor in self.event_processors:
                await processor(event)
            
            # Check for incident creation criteria
            if await self._should_create_incident(event):
                incident_id = await self._create_incident_from_event(event)
                logger.info(f"Incident created: {incident_id} from event {event.event_id}")
            
            # Check for immediate alerts
            if event.severity in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]:
                await self._send_immediate_alert(event)
            
        except Exception as e:
            logger.error(f"Error processing security event {event.event_id}: {e}")
    
    async def _should_create_incident(self, event: SecurityEvent) -> bool:
        """Determine if an incident should be created from an event"""
        # Critical events always create incidents
        if event.severity == ThreatLevel.CRITICAL:
            return True
        
        # Check for patterns that indicate incidents
        if event.event_type == SecurityEventType.AUTHENTICATION_FAILURE:
            # Multiple failed logins from same IP
            recent_failures = [
                e for e in self.security_events.values()
                if e.source_ip == event.source_ip 
                and e.event_type == SecurityEventType.AUTHENTICATION_FAILURE
                and e.timestamp > datetime.now(timezone.utc) - timedelta(minutes=15)
            ]
            return len(recent_failures) >= 5
        
        elif event.event_type == SecurityEventType.UNAUTHORIZED_ACCESS:
            return True
        
        elif event.event_type == SecurityEventType.DATA_BREACH:
            return True
        
        elif event.event_type == SecurityEventType.SUSPICIOUS_ACTIVITY:
            return event.severity == ThreatLevel.HIGH
        
        return False
    
    async def _create_incident_from_event(self, event: SecurityEvent) -> str:
        """Create a security incident from an event"""
        incident_id = str(uuid.uuid4())
        
        # Determine incident severity
        incident_severity = event.severity
        if event.event_type in [SecurityEventType.DATA_BREACH, SecurityEventType.SYSTEM_COMPROMISE]:
            incident_severity = ThreatLevel.CRITICAL
        
        incident = SecurityIncident(
            incident_id=incident_id,
            title=f"{event.event_type.value.replace('_', ' ').title()} - {event.source_ip}",
            description=f"Incident created from security event: {event.description}",
            severity=incident_severity,
            status=IncidentStatus.OPEN,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            events=[event.event_id]
        )
        
        # Add initial timeline entry
        incident.timeline.append({
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'action': 'incident_created',
            'description': f"Incident created from event {event.event_id}",
            'user': 'system'
        })
        
        self.incidents[incident_id] = incident
        
        # Store in Redis
        incident_data = {
            'title': incident.title,
            'description': incident.description,
            'severity': incident.severity.value,
            'status': incident.status.value,
            'created_at': incident.created_at.isoformat(),
            'events': incident.events,
            'timeline': incident.timeline
        }
        
        self.redis_client.setex(f"incident:{incident_id}", 86400 * 90, json.dumps(incident_data))
        
        return incident_id
    
    async def _send_immediate_alert(self, event: SecurityEvent):
        """Send immediate alert for high-severity events"""
        try:
            alert_message = f"""
SECURITY ALERT - {event.severity.value.upper()}

Event Type: {event.event_type.value}
Source IP: {event.source_ip}
User ID: {event.user_id or 'Unknown'}
Time: {event.timestamp.isoformat()}
Description: {event.description}

Raw Data: {json.dumps(event.raw_data, indent=2)}
            """
            
            # Send email alert
            if self.config['notifications']['email_alerts']:
                await self._send_email_alert("Security Alert", alert_message)
            
            # Send Slack alert
            if self.config['notifications']['slack_webhook']:
                await self._send_slack_alert(alert_message)
            
            logger.info(f"Immediate alert sent for event {event.event_id}")
            
        except Exception as e:
            logger.error(f"Failed to send immediate alert: {e}")
    
    async def _send_email_alert(self, subject: str, message: str):
        """Send email alert"""
        # Implementation would use actual SMTP configuration
        logger.info(f"Email alert: {subject}")
    
    async def _send_slack_alert(self, message: str):
        """Send Slack alert"""
        # Implementation would use actual Slack webhook
        logger.info(f"Slack alert: {message[:100]}...")
    
    async def _event_processing_loop(self):
        """Main event processing loop"""
        while self.running:
            try:
                # Process any queued events
                await asyncio.sleep(1)
            except Exception as e:
                logger.error(f"Event processing loop error: {e}")
                await asyncio.sleep(5)
    
    async def _anomaly_detection_loop(self):
        """Anomaly detection loop"""
        while self.running:
            try:
                await self._detect_anomalies()
                await asyncio.sleep(60)  # Check every minute
            except Exception as e:
                logger.error(f"Anomaly detection error: {e}")
                await asyncio.sleep(10)
    
    async def _detect_anomalies(self):
        """Detect security anomalies"""
        current_time = datetime.now(timezone.utc)
        
        # Detect unusual login patterns
        await self._detect_unusual_login_patterns()
        
        # Detect rate limit violations
        await self._detect_rate_limit_violations()
        
        # Detect data exfiltration patterns
        await self._detect_data_exfiltration()
        
        # Detect privilege escalation attempts
        await self._detect_privilege_escalation()
    
    async def _detect_unusual_login_patterns(self):
        """Detect unusual login patterns"""
        # Get recent login events
        recent_logins = [
            event for event in self.security_events.values()
            if event.event_type == SecurityEventType.AUTHENTICATION_FAILURE
            and event.timestamp > datetime.now(timezone.utc) - timedelta(hours=1)
        ]
        
        # Group by IP
        ip_login_counts = defaultdict(int)
        for login in recent_logins:
            ip_login_counts[login.source_ip] += 1
        
        # Check for suspicious patterns
        for ip, count in ip_login_counts.items():
            if count > self.config['rate_limits']['failed_logins_per_hour']:
                self.log_security_event(
                    SecurityEventType.SUSPICIOUS_ACTIVITY,
                    ip,
                    None,
                    f"Excessive failed login attempts: {count} in 1 hour",
                    {'failed_login_count': count, 'time_window': '1_hour'},
                    ThreatLevel.HIGH
                )
    
    async def _detect_rate_limit_violations(self):
        """Detect rate limit violations"""
        current_time = time.time()
        window_start = current_time - 60  # 1 minute window
        
        for ip, timestamps in self.request_counters.items():
            # Count requests in the last minute
            recent_requests = [ts for ts in timestamps if ts > window_start]
            
            if len(recent_requests) > self.config['rate_limits']['requests_per_minute']:
                self.log_security_event(
                    SecurityEventType.SUSPICIOUS_ACTIVITY,
                    ip,
                    None,
                    f"Rate limit exceeded: {len(recent_requests)} requests in 1 minute",
                    {'request_count': len(recent_requests), 'limit': self.config['rate_limits']['requests_per_minute']},
                    ThreatLevel.MEDIUM
                )
    
    async def _detect_data_exfiltration(self):
        """Detect potential data exfiltration"""
        # This would analyze data access patterns, large downloads, etc.
        # Implementation would depend on specific data access logging
        pass
    
    async def _detect_privilege_escalation(self):
        """Detect privilege escalation attempts"""
        # This would analyze user permission changes, admin access attempts, etc.
        # Implementation would depend on specific access control logging
        pass
    
    async def _threat_intelligence_loop(self):
        """Threat intelligence update loop"""
        while self.running:
            try:
                await self._update_threat_intelligence()
                await asyncio.sleep(3600)  # Update every hour
            except Exception as e:
                logger.error(f"Threat intelligence update error: {e}")
                await asyncio.sleep(300)
    
    async def _update_threat_intelligence(self):
        """Update threat intelligence from external sources"""
        # This would fetch updates from threat intelligence feeds
        logger.info("Updating threat intelligence...")
    
    async def _incident_management_loop(self):
        """Incident management loop"""
        while self.running:
            try:
                await self._manage_incidents()
                await asyncio.sleep(300)  # Check every 5 minutes
            except Exception as e:
                logger.error(f"Incident management error: {e}")
                await asyncio.sleep(60)
    
    async def _manage_incidents(self):
        """Manage open incidents"""
        current_time = datetime.now(timezone.utc)
        
        for incident in self.incidents.values():
            if incident.status == IncidentStatus.OPEN:
                # Auto-escalate if incident is old
                age_minutes = (current_time - incident.created_at).total_seconds() / 60
                
                if age_minutes > self.config['incident_escalation']['auto_escalate_after_minutes']:
                    await self._escalate_incident(incident.incident_id)
    
    async def _escalate_incident(self, incident_id: str):
        """Escalate an incident"""
        incident = self.incidents.get(incident_id)
        if not incident:
            return
        
        incident.status = IncidentStatus.INVESTIGATING
        incident.updated_at = datetime.now(timezone.utc)
        
        # Add timeline entry
        incident.timeline.append({
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'action': 'escalated',
            'description': 'Incident auto-escalated due to age',
            'user': 'system'
        })
        
        # Send escalation alert
        await self._send_escalation_alert(incident)
        
        logger.warning(f"Incident escalated: {incident_id}")
    
    async def _send_escalation_alert(self, incident: SecurityIncident):
        """Send incident escalation alert"""
        alert_message = f"""
INCIDENT ESCALATION

Incident ID: {incident.incident_id}
Title: {incident.title}
Severity: {incident.severity.value}
Status: {incident.status.value}
Created: {incident.created_at.isoformat()}

Description: {incident.description}

This incident requires immediate attention.
        """
        
        await self._send_email_alert("Incident Escalation", alert_message)
    
    def update_incident(self, incident_id: str, status: IncidentStatus = None, 
                       assigned_to: str = None, notes: str = None) -> bool:
        """Update an incident"""
        try:
            incident = self.incidents.get(incident_id)
            if not incident:
                return False
            
            # Update fields
            if status:
                incident.status = status
            if assigned_to:
                incident.assigned_to = assigned_to
            
            incident.updated_at = datetime.now(timezone.utc)
            
            # Add timeline entry
            timeline_entry = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'action': 'updated',
                'description': notes or 'Incident updated',
                'user': assigned_to or 'system'
            }
            
            if status:
                timeline_entry['status_change'] = status.value
            
            incident.timeline.append(timeline_entry)
            
            # Update in Redis
            incident_data = {
                'title': incident.title,
                'description': incident.description,
                'severity': incident.severity.value,
                'status': incident.status.value,
                'assigned_to': incident.assigned_to,
                'created_at': incident.created_at.isoformat(),
                'updated_at': incident.updated_at.isoformat(),
                'events': incident.events,
                'timeline': incident.timeline
            }
            
            self.redis_client.setex(f"incident:{incident_id}", 86400 * 90, json.dumps(incident_data))
            
            logger.info(f"Incident updated: {incident_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update incident {incident_id}: {e}")
            return False
    
    def get_security_dashboard(self) -> Dict[str, Any]:
        """Get security dashboard data"""
        current_time = datetime.now(timezone.utc)
        last_24h = current_time - timedelta(hours=24)
        last_7d = current_time - timedelta(days=7)
        
        # Recent events
        recent_events = [
            event for event in self.security_events.values()
            if event.timestamp > last_24h
        ]
        
        # Open incidents
        open_incidents = [
            incident for incident in self.incidents.values()
            if incident.status in [IncidentStatus.OPEN, IncidentStatus.INVESTIGATING]
        ]
        
        # Event statistics
        event_stats = defaultdict(int)
        severity_stats = defaultdict(int)
        
        for event in recent_events:
            event_stats[event.event_type.value] += 1
            severity_stats[event.severity.value] += 1
        
        return {
            'timestamp': current_time.isoformat(),
            'summary': {
                'total_events_24h': len(recent_events),
                'open_incidents': len(open_incidents),
                'critical_incidents': len([i for i in open_incidents if i.severity == ThreatLevel.CRITICAL]),
                'threat_level': self._calculate_overall_threat_level()
            },
            'event_statistics': dict(event_stats),
            'severity_distribution': dict(severity_stats),
            'recent_incidents': [
                {
                    'incident_id': incident.incident_id,
                    'title': incident.title,
                    'severity': incident.severity.value,
                    'status': incident.status.value,
                    'created_at': incident.created_at.isoformat(),
                    'assigned_to': incident.assigned_to
                }
                for incident in sorted(open_incidents, key=lambda x: x.created_at, reverse=True)[:10]
            ],
            'top_threat_sources': self._get_top_threat_sources(recent_events),
            'recommendations': self._get_security_recommendations()
        }
    
    def _calculate_overall_threat_level(self) -> str:
        """Calculate overall threat level"""
        open_incidents = [
            incident for incident in self.incidents.values()
            if incident.status in [IncidentStatus.OPEN, IncidentStatus.INVESTIGATING]
        ]
        
        if any(i.severity == ThreatLevel.CRITICAL for i in open_incidents):
            return ThreatLevel.CRITICAL.value
        elif any(i.severity == ThreatLevel.HIGH for i in open_incidents):
            return ThreatLevel.HIGH.value
        elif len(open_incidents) > 5:
            return ThreatLevel.MEDIUM.value
        else:
            return ThreatLevel.LOW.value
    
    def _get_top_threat_sources(self, events: List[SecurityEvent]) -> List[Dict[str, Any]]:
        """Get top threat sources"""
        source_counts = defaultdict(int)
        
        for event in events:
            source_counts[event.source_ip] += 1
        
        top_sources = sorted(source_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return [
            {
                'source_ip': ip,
                'event_count': count,
                'is_known_threat': self._is_known_threat(ip)
            }
            for ip, count in top_sources
        ]
    
    def _get_security_recommendations(self) -> List[str]:
        """Get security recommendations"""
        recommendations = []
        
        # Check for common issues
        recent_events = [
            event for event in self.security_events.values()
            if event.timestamp > datetime.now(timezone.utc) - timedelta(hours=24)
        ]
        
        failed_logins = [e for e in recent_events if e.event_type == SecurityEventType.AUTHENTICATION_FAILURE]
        if len(failed_logins) > 50:
            recommendations.append("Consider implementing additional authentication controls")
        
        open_incidents = [
            incident for incident in self.incidents.values()
            if incident.status in [IncidentStatus.OPEN, IncidentStatus.INVESTIGATING]
        ]
        
        if len(open_incidents) > 10:
            recommendations.append("Review incident response procedures - high number of open incidents")
        
        if any(i.severity == ThreatLevel.CRITICAL for i in open_incidents):
            recommendations.append("Address critical security incidents immediately")
        
        return recommendations

# Global security monitor
security_monitor = SecurityMonitor()

# Example usage
async def example_usage():
    """Example of security monitoring usage"""
    
    # Log some security events
    security_monitor.log_security_event(
        SecurityEventType.AUTHENTICATION_FAILURE,
        "192.168.1.100",
        "user123",
        "Failed login attempt with invalid password",
        {"username": "user123", "user_agent": "Mozilla/5.0"},
        ThreatLevel.MEDIUM
    )
    
    security_monitor.log_security_event(
        SecurityEventType.SUSPICIOUS_ACTIVITY,
        "10.0.0.50",
        None,
        "Multiple rapid API requests detected",
        {"request_count": 150, "time_window": "1_minute"},
        ThreatLevel.HIGH
    )
    
    # Get security dashboard
    dashboard = security_monitor.get_security_dashboard()
    print("Security Dashboard:")
    print(json.dumps(dashboard, indent=2))
    
    # Update an incident
    incidents = list(security_monitor.incidents.keys())
    if incidents:
        security_monitor.update_incident(
            incidents[0],
            status=IncidentStatus.INVESTIGATING,
            assigned_to="security_analyst",
            notes="Investigating suspicious activity pattern"
        )

if __name__ == "__main__":
    asyncio.run(example_usage())