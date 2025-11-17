"""
Advanced Threat Protection System
Provides real-time threat detection, prevention, and response
"""

import asyncio
import json
import logging
import hashlib
import ipaddress
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import redis
import requests
from collections import defaultdict, deque
import geoip2.database
import re

logger = logging.getLogger(__name__)

class ThreatType(Enum):
    MALWARE = "malware"
    PHISHING = "phishing"
    BRUTE_FORCE = "brute_force"
    DDoS = "ddos"
    SQL_INJECTION = "sql_injection"
    XSS = "xss"
    CSRF = "csrf"
    DATA_EXFILTRATION = "data_exfiltration"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    SUSPICIOUS_BEHAVIOR = "suspicious_behavior"

class ThreatSeverity(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

class ResponseAction(Enum):
    BLOCK = "block"
    QUARANTINE = "quarantine"
    MONITOR = "monitor"
    ALERT = "alert"
    LOG_ONLY = "log_only"

@dataclass
class ThreatSignature:
    signature_id: str
    threat_type: ThreatType
    pattern: str
    severity: ThreatSeverity
    description: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    active: bool = True

@dataclass
class ThreatDetection:
    detection_id: str
    threat_type: ThreatType
    severity: ThreatSeverity
    source_ip: str
    target: str
    description: str
    evidence: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    blocked: bool = False
    response_actions: List[ResponseAction] = field(default_factory=list)

class AdvancedThreatProtection:
    """Advanced threat protection with ML-based detection"""
    
    def __init__(self, redis_config: Dict[str, Any] = None):
        self.redis_config = redis_config or {
            'host': 'localhost',
            'port': 6379,
            'db': 6
        }
        
        self.redis_client = redis.Redis(**self.redis_config)
        
        # Threat detection state
        self.threat_signatures: Dict[str, ThreatSignature] = {}
        self.active_threats: Dict[str, ThreatDetection] = {}
        self.blocked_ips: Set[str] = set()
        self.suspicious_ips: Dict[str, float] = {}  # IP -> suspicion score
        
        # Rate limiting and behavior analysis
        self.request_patterns: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.failed_attempts: Dict[str, List[datetime]] = defaultdict(list)
        self.geo_database = None
        
        # ML models for anomaly detection
        self.anomaly_models = {}
        
        # Configuration
        self.config = self._load_threat_config()
        
        # Initialize components
        self._load_threat_signatures()
        self._initialize_geo_database()
        
    def _load_threat_config(self) -> Dict[str, Any]:
        """Load threat protection configuration"""
        return {
            'detection_thresholds': {
                'brute_force_attempts': 10,
                'request_rate_per_minute': 300,
                'failed_login_threshold': 5,
                'geo_anomaly_threshold': 0.8,
                'behavior_anomaly_threshold': 0.7
            },
            'response_actions': {
                'auto_block_critical': True,
                'auto_block_duration_hours': 24,
                'quarantine_suspicious_users': True,
                'alert_security_team': True
            },
            'threat_intelligence': {
                'update_interval_hours': 1,
                'confidence_threshold': 0.7,
                'sources': ['virustotal', 'abuseipdb', 'internal']
            },
            'ml_detection': {
                'enable_behavioral_analysis': True,
                'enable_anomaly_detection': True,
                'model_update_interval_hours': 6
            }
        }
    
    def _load_threat_signatures(self):
        """Load threat detection signatures"""
        signatures = [
            ThreatSignature(
                "sql_injection_1",
                ThreatType.SQL_INJECTION,
                r"(\bunion\b.*\bselect\b|\bselect\b.*\bunion\b)",
                ThreatSeverity.HIGH,
                "SQL injection attempt detected"
            ),
            ThreatSignature(
                "xss_1",
                ThreatType.XSS,
                r"<script[^>]*>.*?</script>",
                ThreatSeverity.MEDIUM,
                "Cross-site scripting attempt detected"
            ),
            ThreatSignature(
                "brute_force_1",
                ThreatType.BRUTE_FORCE,
                r"rapid_login_attempts",
                ThreatSeverity.HIGH,
                "Brute force attack detected"
            )
        ]
        
        for signature in signatures:
            self.threat_signatures[signature.signature_id] = signature
    
    def _initialize_geo_database(self):
        """Initialize GeoIP database for location analysis"""
        try:
            # In production, you'd download and use a real GeoIP database
            # self.geo_database = geoip2.database.Reader('GeoLite2-City.mmdb')
            pass
        except Exception as e:
            logger.warning(f"Could not initialize GeoIP database: {e}")
    
    async def analyze_request(self, request_data: Dict[str, Any]) -> Tuple[bool, List[ThreatDetection]]:
        """Analyze incoming request for threats"""
        threats_detected = []
        should_block = False
        
        source_ip = request_data.get('source_ip', '')
        user_agent = request_data.get('user_agent', '')
        request_path = request_data.get('path', '')
        request_body = request_data.get('body', '')
        
        # Check if IP is already blocked
        if source_ip in self.blocked_ips:
            return True, []
        
        # 1. Signature-based detection
        signature_threats = await self._detect_signature_threats(request_data)
        threats_detected.extend(signature_threats)
        
        # 2. Rate limiting analysis
        rate_threats = await self._analyze_rate_limiting(source_ip)
        threats_detected.extend(rate_threats)
        
        # 3. Behavioral analysis
        behavior_threats = await self._analyze_behavior(request_data)
        threats_detected.extend(behavior_threats)
        
        # 4. Geolocation analysis
        geo_threats = await self._analyze_geolocation(source_ip)
        threats_detected.extend(geo_threats)
        
        # 5. Content analysis
        content_threats = await self._analyze_content(request_body, request_path)
        threats_detected.extend(content_threats)
        
        # Determine if request should be blocked
        for threat in threats_detected:
            if threat.severity in [ThreatSeverity.CRITICAL, ThreatSeverity.HIGH]:
                should_block = True
                break
        
        # Execute response actions
        if threats_detected:
            await self._execute_response_actions(threats_detected, source_ip)
        
        return should_block, threats_detected
    
    async def _detect_signature_threats(self, request_data: Dict[str, Any]) -> List[ThreatDetection]:
        """Detect threats using signature patterns"""
        threats = []
        
        content_to_check = [
            request_data.get('path', ''),
            request_data.get('query_string', ''),
            request_data.get('body', ''),
            request_data.get('headers', {}).get('user-agent', '')
        ]
        
        full_content = ' '.join(str(c) for c in content_to_check).lower()
        
        for signature in self.threat_signatures.values():
            if not signature.active:
                continue
                
            if re.search(signature.pattern, full_content, re.IGNORECASE):
                detection = ThreatDetection(
                    detection_id=f"sig_{signature.signature_id}_{datetime.now(datetime.UTC).timestamp()}",
                    threat_type=signature.threat_type,
                    severity=signature.severity,
                    source_ip=request_data.get('source_ip', ''),
                    target=request_data.get('path', ''),
                    description=signature.description,
                    evidence={
                        'signature_id': signature.signature_id,
                        'matched_pattern': signature.pattern,
                        'request_data': request_data
                    }
                )
                threats.append(detection)
        
        return threats
    
    async def _analyze_rate_limiting(self, source_ip: str) -> List[ThreatDetection]:
        """Analyze request rates for potential attacks"""
        threats = []
        current_time = datetime.now(datetime.UTC)
        
        # Track request pattern
        self.request_patterns[source_ip].append(current_time)
        
        # Check request rate in last minute
        recent_requests = [
            req_time for req_time in self.request_patterns[source_ip]
            if current_time - req_time < timedelta(minutes=1)
        ]
        
        if len(recent_requests) > self.config['detection_thresholds']['request_rate_per_minute']:
            detection = ThreatDetection(
                detection_id=f"rate_{source_ip}_{current_time.timestamp()}",
                threat_type=ThreatType.DDoS,
                severity=ThreatSeverity.HIGH,
                source_ip=source_ip,
                target="rate_limit",
                description=f"Excessive request rate: {len(recent_requests)} requests/minute",
                evidence={
                    'request_count': len(recent_requests),
                    'time_window': '1_minute',
                    'threshold': self.config['detection_thresholds']['request_rate_per_minute']
                }
            )
            threats.append(detection)
        
        return threats
    
    async def _analyze_behavior(self, request_data: Dict[str, Any]) -> List[ThreatDetection]:
        """Analyze behavioral patterns for anomalies"""
        threats = []
        
        # This would use ML models to detect anomalous behavior
        # For now, implementing basic heuristics
        
        user_agent = request_data.get('user_agent', '')
        source_ip = request_data.get('source_ip', '')
        
        # Check for suspicious user agents
        suspicious_ua_patterns = [
            r'bot', r'crawler', r'spider', r'scraper',
            r'curl', r'wget', r'python', r'java'
        ]
        
        for pattern in suspicious_ua_patterns:
            if re.search(pattern, user_agent, re.IGNORECASE):
                detection = ThreatDetection(
                    detection_id=f"behavior_{source_ip}_{datetime.now(datetime.UTC).timestamp()}",
                    threat_type=ThreatType.SUSPICIOUS_BEHAVIOR,
                    severity=ThreatSeverity.MEDIUM,
                    source_ip=source_ip,
                    target="user_agent",
                    description=f"Suspicious user agent detected: {pattern}",
                    evidence={
                        'user_agent': user_agent,
                        'suspicious_pattern': pattern
                    }
                )
                threats.append(detection)
                break
        
        return threats
    
    async def _analyze_geolocation(self, source_ip: str) -> List[ThreatDetection]:
        """Analyze geolocation for anomalies"""
        threats = []
        
        # This would use actual GeoIP database
        # For now, implementing basic checks
        
        try:
            # Check if IP is from known malicious countries/regions
            # This is a simplified example
            if self._is_high_risk_location(source_ip):
                detection = ThreatDetection(
                    detection_id=f"geo_{source_ip}_{datetime.now(datetime.UTC).timestamp()}",
                    threat_type=ThreatType.SUSPICIOUS_BEHAVIOR,
                    severity=ThreatSeverity.MEDIUM,
                    source_ip=source_ip,
                    target="geolocation",
                    description="Request from high-risk geographic location",
                    evidence={
                        'source_ip': source_ip,
                        'risk_level': 'high'
                    }
                )
                threats.append(detection)
        
        except Exception as e:
            logger.error(f"Geolocation analysis error: {e}")
        
        return threats
    
    async def _analyze_content(self, request_body: str, request_path: str) -> List[ThreatDetection]:
        """Analyze request content for malicious patterns"""
        threats = []
        
        # Check for common attack patterns
        attack_patterns = {
            'sql_injection': [
                r"(\bunion\b.*\bselect\b|\bselect\b.*\bunion\b)",
                r"(\bdrop\b.*\btable\b|\btable\b.*\bdrop\b)",
                r"(\binsert\b.*\binto\b|\binto\b.*\binsert\b)"
            ],
            'xss': [
                r"<script[^>]*>.*?</script>",
                r"javascript:",
                r"on\w+\s*="
            ],
            'path_traversal': [
                r"\.\./",
                r"\.\.\\",
                r"%2e%2e%2f"
            ]
        }
        
        content = f"{request_path} {request_body}".lower()
        
        for attack_type, patterns in attack_patterns.items():
            for pattern in patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    detection = ThreatDetection(
                        detection_id=f"content_{attack_type}_{datetime.now(datetime.UTC).timestamp()}",
                        threat_type=ThreatType.SQL_INJECTION if attack_type == 'sql_injection' else ThreatType.XSS,
                        severity=ThreatSeverity.HIGH,
                        source_ip="unknown",
                        target=request_path,
                        description=f"{attack_type.replace('_', ' ').title()} attempt detected",
                        evidence={
                            'attack_type': attack_type,
                            'matched_pattern': pattern,
                            'content_snippet': content[:200]
                        }
                    )
                    threats.append(detection)
        
        return threats
    
    async def _execute_response_actions(self, threats: List[ThreatDetection], source_ip: str):
        """Execute response actions based on detected threats"""
        
        for threat in threats:
            # Determine response actions based on severity
            if threat.severity == ThreatSeverity.CRITICAL:
                threat.response_actions = [ResponseAction.BLOCK, ResponseAction.ALERT]
                if self.config['response_actions']['auto_block_critical']:
                    await self._block_ip(source_ip, duration_hours=24)
                    threat.blocked = True
            
            elif threat.severity == ThreatSeverity.HIGH:
                threat.response_actions = [ResponseAction.QUARANTINE, ResponseAction.ALERT]
                await self._quarantine_ip(source_ip)
            
            elif threat.severity == ThreatSeverity.MEDIUM:
                threat.response_actions = [ResponseAction.MONITOR, ResponseAction.LOG_ONLY]
            
            # Store threat detection
            self.active_threats[threat.detection_id] = threat
            
            # Log to Redis
            await self._log_threat_detection(threat)
            
            # Send alerts if configured
            if ResponseAction.ALERT in threat.response_actions:
                await self._send_security_alert(threat)
    
    async def _block_ip(self, ip_address: str, duration_hours: int = 24):
        """Block IP address for specified duration"""
        self.blocked_ips.add(ip_address)
        
        # Store in Redis with expiration
        self.redis_client.setex(
            f"blocked_ip:{ip_address}",
            duration_hours * 3600,
            json.dumps({
                'blocked_at': datetime.now(datetime.UTC).isoformat(),
                'duration_hours': duration_hours,
                'reason': 'threat_detection'
            })
        )
        
        logger.warning(f"Blocked IP {ip_address} for {duration_hours} hours")
    
    async def _quarantine_ip(self, ip_address: str):
        """Quarantine IP for enhanced monitoring"""
        self.suspicious_ips[ip_address] = 0.8  # High suspicion score
        
        # Store in Redis
        self.redis_client.setex(
            f"quarantine_ip:{ip_address}",
            86400,  # 24 hours
            json.dumps({
                'quarantined_at': datetime.now(datetime.UTC).isoformat(),
                'suspicion_score': 0.8
            })
        )
        
        logger.info(f"Quarantined IP {ip_address} for enhanced monitoring")
    
    async def _log_threat_detection(self, threat: ThreatDetection):
        """Log threat detection to Redis"""
        threat_data = {
            'detection_id': threat.detection_id,
            'threat_type': threat.threat_type.value,
            'severity': threat.severity.value,
            'source_ip': threat.source_ip,
            'target': threat.target,
            'description': threat.description,
            'evidence': threat.evidence,
            'timestamp': threat.timestamp.isoformat(),
            'blocked': threat.blocked,
            'response_actions': [action.value for action in threat.response_actions]
        }
        
        self.redis_client.lpush('threat_detections', json.dumps(threat_data))
        self.redis_client.ltrim('threat_detections', 0, 10000)  # Keep last 10k detections
    
    async def _send_security_alert(self, threat: ThreatDetection):
        """Send security alert for threat detection"""
        # This would integrate with notification system
        logger.critical(f"SECURITY ALERT: {threat.description} from {threat.source_ip}")
    
    def _is_high_risk_location(self, ip_address: str) -> bool:
        """Check if IP is from high-risk location"""
        # This would use actual GeoIP database and threat intelligence
        # For now, implementing basic check
        try:
            ip = ipaddress.ip_address(ip_address)
            # Example: Check for private/local IPs (not actually high risk)
            return ip.is_private
        except:
            return False
    
    def get_threat_summary(self) -> Dict[str, Any]:
        """Get summary of current threats"""
        current_time = datetime.now(datetime.UTC)
        recent_threats = [
            threat for threat in self.active_threats.values()
            if current_time - threat.timestamp < timedelta(hours=24)
        ]
        
        threat_counts = defaultdict(int)
        severity_counts = defaultdict(int)
        
        for threat in recent_threats:
            threat_counts[threat.threat_type.value] += 1
            severity_counts[threat.severity.value] += 1
        
        return {
            'total_threats_24h': len(recent_threats),
            'blocked_ips_count': len(self.blocked_ips),
            'quarantined_ips_count': len(self.suspicious_ips),
            'threat_types': dict(threat_counts),
            'severity_distribution': dict(severity_counts),
            'last_updated': current_time.isoformat()
        }

# Global threat protection instance
threat_protection = AdvancedThreatProtection()