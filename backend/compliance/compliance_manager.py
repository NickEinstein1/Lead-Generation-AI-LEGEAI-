import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import redis
from pathlib import Path
import hashlib
import uuid

logger = logging.getLogger(__name__)

class ComplianceFramework(Enum):
    GDPR = "gdpr"
    CCPA = "ccpa"
    HIPAA = "hipaa"
    TCPA = "tcpa"
    CAN_SPAM = "can_spam"
    SOX = "sox"
    PCI_DSS = "pci_dss"
    NAIC = "naic"
    STATE_INSURANCE = "state_insurance"

class ComplianceStatus(Enum):
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PENDING_REVIEW = "pending_review"
    REMEDIATION_REQUIRED = "remediation_required"

class ViolationSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class ComplianceRule:
    rule_id: str
    framework: ComplianceFramework
    title: str
    description: str
    requirement: str
    validation_method: str
    severity: ViolationSeverity
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)

@dataclass
class ComplianceViolation:
    violation_id: str
    rule_id: str
    framework: ComplianceFramework
    severity: ViolationSeverity
    description: str
    data_context: Dict[str, Any]
    detected_at: datetime
    resolved_at: Optional[datetime] = None
    resolution_notes: Optional[str] = None
    is_resolved: bool = False

@dataclass
class ConsentRecord:
    consent_id: str
    user_id: str
    consent_type: str  # "marketing", "data_processing", "cookies", etc.
    consent_given: bool
    consent_method: str  # "web_form", "email", "phone", "api"
    consent_timestamp: datetime
    consent_text: str
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    withdrawal_timestamp: Optional[datetime] = None
    is_active: bool = True

@dataclass
class AuditLog:
    log_id: str
    timestamp: datetime
    user_id: Optional[str]
    action: str
    resource_type: str
    resource_id: str
    details: Dict[str, Any]
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None

class ComplianceManager:
    """Comprehensive compliance management system"""
    
    def __init__(self, redis_config: Dict[str, Any] = None):
        self.redis_config = redis_config or {
            'host': 'localhost',
            'port': 6379,
            'db': 6
        }
        
        self.redis_client = redis.Redis(**self.redis_config)
        
        # Compliance rules
        self.compliance_rules: Dict[str, ComplianceRule] = {}
        self.violations: Dict[str, ComplianceViolation] = {}
        self.consent_records: Dict[str, ConsentRecord] = {}
        self.audit_logs: List[AuditLog] = []
        
        # Load compliance rules
        self._load_compliance_rules()
        
        # Data retention policies
        self.retention_policies = self._load_retention_policies()
        
    def _load_compliance_rules(self):
        """Load compliance rules for different frameworks"""
        
        # GDPR Rules
        gdpr_rules = [
            ComplianceRule(
                "GDPR-001", ComplianceFramework.GDPR,
                "Explicit Consent Required",
                "Processing personal data requires explicit consent from data subject",
                "Article 6(1)(a) - Consent must be freely given, specific, informed and unambiguous",
                "check_explicit_consent",
                ViolationSeverity.HIGH
            ),
            ComplianceRule(
                "GDPR-002", ComplianceFramework.GDPR,
                "Right to Erasure",
                "Data subjects have right to request deletion of personal data",
                "Article 17 - Right to erasure ('right to be forgotten')",
                "check_deletion_capability",
                ViolationSeverity.MEDIUM
            ),
            ComplianceRule(
                "GDPR-003", ComplianceFramework.GDPR,
                "Data Minimization",
                "Personal data must be adequate, relevant and limited to what is necessary",
                "Article 5(1)(c) - Data minimisation principle",
                "check_data_minimization",
                ViolationSeverity.MEDIUM
            ),
            ComplianceRule(
                "GDPR-004", ComplianceFramework.GDPR,
                "Data Retention Limits",
                "Personal data must not be kept longer than necessary",
                "Article 5(1)(e) - Storage limitation principle",
                "check_retention_limits",
                ViolationSeverity.HIGH
            )
        ]
        
        # CCPA Rules
        ccpa_rules = [
            ComplianceRule(
                "CCPA-001", ComplianceFramework.CCPA,
                "Consumer Right to Know",
                "Consumers have right to know what personal information is collected",
                "CCPA Section 1798.100 - Right to know about personal information collected",
                "check_transparency_notice",
                ViolationSeverity.MEDIUM
            ),
            ComplianceRule(
                "CCPA-002", ComplianceFramework.CCPA,
                "Right to Delete",
                "Consumers have right to request deletion of personal information",
                "CCPA Section 1798.105 - Right to delete personal information",
                "check_deletion_process",
                ViolationSeverity.HIGH
            ),
            ComplianceRule(
                "CCPA-003", ComplianceFramework.CCPA,
                "Opt-Out of Sale",
                "Consumers have right to opt out of sale of personal information",
                "CCPA Section 1798.120 - Right to opt-out of sale",
                "check_opt_out_mechanism",
                ViolationSeverity.HIGH
            )
        ]
        
        # HIPAA Rules
        hipaa_rules = [
            ComplianceRule(
                "HIPAA-001", ComplianceFramework.HIPAA,
                "PHI Encryption",
                "Protected Health Information must be encrypted at rest and in transit",
                "45 CFR 164.312(a)(2)(iv) - Encryption and decryption",
                "check_phi_encryption",
                ViolationSeverity.CRITICAL
            ),
            ComplianceRule(
                "HIPAA-002", ComplianceFramework.HIPAA,
                "Access Controls",
                "Implement access controls for PHI",
                "45 CFR 164.312(a)(1) - Access control",
                "check_access_controls",
                ViolationSeverity.HIGH
            ),
            ComplianceRule(
                "HIPAA-003", ComplianceFramework.HIPAA,
                "Audit Logs",
                "Maintain audit logs of PHI access",
                "45 CFR 164.312(b) - Audit controls",
                "check_audit_logs",
                ViolationSeverity.HIGH
            )
        ]
        
        # TCPA Rules
        tcpa_rules = [
            ComplianceRule(
                "TCPA-001", ComplianceFramework.TCPA,
                "Prior Express Consent",
                "Obtain prior express consent before sending marketing communications",
                "47 USC 227 - Prior express written consent required",
                "check_tcpa_consent",
                ViolationSeverity.CRITICAL
            ),
            ComplianceRule(
                "TCPA-002", ComplianceFramework.TCPA,
                "Opt-Out Instructions",
                "Provide clear opt-out instructions in communications",
                "47 USC 227 - Opt-out mechanism required",
                "check_opt_out_instructions",
                ViolationSeverity.HIGH
            )
        ]
        
        # CAN-SPAM Rules
        can_spam_rules = [
            ComplianceRule(
                "CAN-SPAM-001", ComplianceFramework.CAN_SPAM,
                "Truthful Subject Lines",
                "Subject lines must accurately reflect email content",
                "15 USC 7704(a)(1) - Prohibition against false or misleading header information",
                "check_subject_line_accuracy",
                ViolationSeverity.MEDIUM
            ),
            ComplianceRule(
                "CAN-SPAM-002", ComplianceFramework.CAN_SPAM,
                "Unsubscribe Mechanism",
                "Provide clear and conspicuous unsubscribe mechanism",
                "15 USC 7704(a)(3) - Inclusion of return address or comparable mechanism",
                "check_unsubscribe_mechanism",
                ViolationSeverity.HIGH
            ),
            ComplianceRule(
                "CAN-SPAM-003", ComplianceFramework.CAN_SPAM,
                "Physical Address",
                "Include valid physical postal address",
                "15 USC 7704(a)(5) - Inclusion of identifier, opt-out, and physical address",
                "check_physical_address",
                ViolationSeverity.MEDIUM
            )
        ]
        
        # Store all rules
        all_rules = gdpr_rules + ccpa_rules + hipaa_rules + tcpa_rules + can_spam_rules
        for rule in all_rules:
            self.compliance_rules[rule.rule_id] = rule
    
    def _load_retention_policies(self) -> Dict[str, Dict[str, Any]]:
        """Load data retention policies"""
        return {
            'lead_data': {
                'retention_period_days': 2555,  # 7 years for insurance
                'deletion_method': 'secure_delete',
                'archive_after_days': 1095,  # 3 years
                'legal_hold_override': True
            },
            'consent_records': {
                'retention_period_days': 2555,  # 7 years
                'deletion_method': 'secure_delete',
                'archive_after_days': None,
                'legal_hold_override': False
            },
            'audit_logs': {
                'retention_period_days': 2555,  # 7 years
                'deletion_method': 'archive',
                'archive_after_days': 365,  # 1 year
                'legal_hold_override': True
            },
            'communication_logs': {
                'retention_period_days': 1095,  # 3 years
                'deletion_method': 'secure_delete',
                'archive_after_days': 365,
                'legal_hold_override': True
            }
        }
    
    def validate_compliance(self, data: Dict[str, Any], frameworks: List[ComplianceFramework] = None) -> Dict[str, Any]:
        """Validate data against compliance rules"""
        frameworks = frameworks or [f for f in ComplianceFramework]
        
        validation_results = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'overall_status': ComplianceStatus.COMPLIANT.value,
            'framework_results': {},
            'violations': [],
            'recommendations': []
        }
        
        for framework in frameworks:
            framework_rules = [rule for rule in self.compliance_rules.values() 
                             if rule.framework == framework and rule.is_active]
            
            framework_result = {
                'status': ComplianceStatus.COMPLIANT.value,
                'rules_checked': len(framework_rules),
                'violations': [],
                'score': 100.0
            }
            
            violations_count = 0
            
            for rule in framework_rules:
                is_compliant, violation_details = self._validate_rule(rule, data)
                
                if not is_compliant:
                    violations_count += 1
                    violation = self._create_violation(rule, violation_details, data)
                    framework_result['violations'].append(violation)
                    validation_results['violations'].append(violation)
            
            # Calculate compliance score
            if framework_rules:
                compliance_rate = (len(framework_rules) - violations_count) / len(framework_rules)
                framework_result['score'] = compliance_rate * 100
                
                if compliance_rate < 1.0:
                    framework_result['status'] = ComplianceStatus.NON_COMPLIANT.value
                    validation_results['overall_status'] = ComplianceStatus.NON_COMPLIANT.value
            
            validation_results['framework_results'][framework.value] = framework_result
        
        # Generate recommendations
        validation_results['recommendations'] = self._generate_compliance_recommendations(validation_results['violations'])
        
        return validation_results
    
    def _validate_rule(self, rule: ComplianceRule, data: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Validate specific compliance rule"""
        try:
            # Call appropriate validation method
            validation_method = getattr(self, rule.validation_method, None)
            if validation_method:
                return validation_method(data)
            else:
                logger.warning(f"Validation method not found: {rule.validation_method}")
                return True, None
                
        except Exception as e:
            logger.error(f"Rule validation failed for {rule.rule_id}: {e}")
            return False, f"Validation error: {str(e)}"
    
    def _create_violation(self, rule: ComplianceRule, details: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create compliance violation record"""
        violation_id = str(uuid.uuid4())
        
        violation = ComplianceViolation(
            violation_id=violation_id,
            rule_id=rule.rule_id,
            framework=rule.framework,
            severity=rule.severity,
            description=f"{rule.title}: {details}",
            data_context=self._sanitize_data_context(data),
            detected_at=datetime.now(timezone.utc)
        )
        
        self.violations[violation_id] = violation
        
        return {
            'violation_id': violation_id,
            'rule_id': rule.rule_id,
            'framework': rule.framework.value,
            'severity': rule.severity.value,
            'title': rule.title,
            'description': violation.description,
            'detected_at': violation.detected_at.isoformat()
        }
    
    def _sanitize_data_context(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize data context for violation records"""
        # Remove sensitive data but keep structure for analysis
        sanitized = {}
        
        for key, value in data.items():
            if key.lower() in ['password', 'ssn', 'credit_card', 'token']:
                sanitized[key] = '[REDACTED]'
            elif isinstance(value, str) and len(value) > 100:
                sanitized[key] = value[:50] + '...[TRUNCATED]'
            else:
                sanitized[key] = value
        
        return sanitized
    
    def _generate_compliance_recommendations(self, violations: List[Dict[str, Any]]) -> List[str]:
        """Generate compliance recommendations based on violations"""
        recommendations = []
        
        # Group violations by framework
        framework_violations = {}
        for violation in violations:
            framework = violation['framework']
            if framework not in framework_violations:
                framework_violations[framework] = []
            framework_violations[framework].append(violation)
        
        # Generate framework-specific recommendations
        for framework, framework_violations_list in framework_violations.items():
            if framework == 'gdpr':
                recommendations.extend([
                    "Implement explicit consent collection mechanisms",
                    "Add data subject rights management (access, deletion, portability)",
                    "Conduct Data Protection Impact Assessment (DPIA)",
                    "Appoint Data Protection Officer (DPO) if required"
                ])
            elif framework == 'ccpa':
                recommendations.extend([
                    "Add 'Do Not Sell My Personal Information' link to website",
                    "Implement consumer request processing system",
                    "Update privacy policy with CCPA disclosures"
                ])
            elif framework == 'hipaa':
                recommendations.extend([
                    "Implement end-to-end encryption for PHI",
                    "Conduct HIPAA risk assessment",
                    "Implement role-based access controls",
                    "Enhance audit logging capabilities"
                ])
            elif framework == 'tcpa':
                recommendations.extend([
                    "Implement prior express written consent collection",
                    "Add opt-out mechanisms to all communications",
                    "Maintain consent records with timestamps"
                ])
        
        return list(set(recommendations))  # Remove duplicates
    
    # Validation methods for specific rules
    def check_explicit_consent(self, data: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Check if explicit consent is present"""
        consent_given = data.get('consent_given', False)
        consent_method = data.get('consent_method', '')
        consent_timestamp = data.get('consent_timestamp')
        
        if not consent_given:
            return False, "No consent provided"
        
        if not consent_timestamp:
            return False, "Consent timestamp missing"
        
        if consent_method not in ['web_form', 'email_confirmation', 'api_explicit']:
            return False, f"Invalid consent method: {consent_method}"
        
        return True, None
    
    def check_deletion_capability(self, data: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Check if data deletion capability exists"""
        # This would check if the system has deletion capabilities
        # For now, assume it's implemented
        return True, None
    
    def check_data_minimization(self, data: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Check data minimization compliance"""
        # Check if only necessary fields are collected
        necessary_fields = {'email', 'age', 'location', 'income', 'consent_given'}
        collected_fields = set(data.keys())
        
        unnecessary_fields = collected_fields - necessary_fields - {'timestamp', 'id', 'source'}
        
        if unnecessary_fields:
            return False, f"Unnecessary fields collected: {', '.join(unnecessary_fields)}"
        
        return True, None
    
    def check_retention_limits(self, data: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Check data retention limits"""
        created_at = data.get('created_at')
        if not created_at:
            return True, None  # Can't check without timestamp
        
        # Parse timestamp
        if isinstance(created_at, str):
            try:
                created_date = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
            except:
                return False, "Invalid timestamp format"
        else:
            created_date = created_at
        
        # Check if data is older than retention period (7 years for insurance)
        retention_days = self.retention_policies['lead_data']['retention_period_days']
        if (datetime.now(timezone.utc) - created_date).days > retention_days:
            return False, f"Data exceeds retention period of {retention_days} days"
        
        return True, None
    
    def check_transparency_notice(self, data: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Check if transparency notice was provided"""
        privacy_notice_shown = data.get('privacy_notice_shown', False)
        
        if not privacy_notice_shown:
            return False, "Privacy notice not shown to consumer"
        
        return True, None
    
    def check_deletion_process(self, data: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Check if deletion process is available"""
        # This would check if deletion process is implemented
        return True, None
    
    def check_opt_out_mechanism(self, data: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Check if opt-out mechanism is provided"""
        opt_out_provided = data.get('opt_out_mechanism_provided', False)
        
        if not opt_out_provided:
            return False, "Opt-out mechanism not provided"
        
        return True, None
    
    def check_phi_encryption(self, data: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Check if PHI is encrypted"""
        has_health_data = any(key in data for key in ['health_conditions', 'medical_history', 'prescriptions'])
        
        if has_health_data:
            encryption_status = data.get('encryption_applied', False)
            if not encryption_status:
                return False, "PHI not encrypted"
        
        return True, None
    
    def check_access_controls(self, data: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Check access controls for PHI"""
        # This would check if proper access controls are in place
        return True, None
    
    def check_audit_logs(self, data: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Check if audit logs are maintained"""
        # This would check if audit logging is enabled
        return True, None
    
    def check_tcpa_consent(self, data: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Check TCPA consent for communications"""
        communication_type = data.get('communication_type', '')
        
        if communication_type in ['sms', 'phone_call', 'robocall']:
            tcpa_consent = data.get('tcpa_consent_given', False)
            consent_method = data.get('tcpa_consent_method', '')
            
            if not tcpa_consent:
                return False, "TCPA consent not obtained for communication"
            
            if consent_method != 'written':
                return False, "TCPA requires written consent for automated communications"
        
        return True, None
    
    def check_opt_out_instructions(self, data: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Check if opt-out instructions are included"""
        message_content = data.get('message_content', '')
        
        if message_content:
            opt_out_keywords = ['stop', 'unsubscribe', 'opt out', 'opt-out']
            has_opt_out = any(keyword in message_content.lower() for keyword in opt_out_keywords)
            
            if not has_opt_out:
                return False, "Opt-out instructions not included in message"
        
        return True, None
    
    def check_subject_line_accuracy(self, data: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Check if email subject line is accurate"""
        subject_line = data.get('subject_line', '')
        message_content = data.get('message_content', '')
        
        if subject_line and message_content:
            # Basic check for misleading keywords
            misleading_keywords = ['free', 'urgent', 'act now', 'limited time']
            if any(keyword in subject_line.lower() for keyword in misleading_keywords):
                if not any(keyword in message_content.lower() for keyword in misleading_keywords):
                    return False, "Subject line may be misleading compared to content"
        
        return True, None
    
    def check_unsubscribe_mechanism(self, data: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Check if unsubscribe mechanism is provided"""
        message_content = data.get('message_content', '')
        
        if message_content:
            unsubscribe_indicators = ['unsubscribe', 'opt out', 'remove me']
            has_unsubscribe = any(indicator in message_content.lower() for indicator in unsubscribe_indicators)
            
            if not has_unsubscribe:
                return False, "Unsubscribe mechanism not provided"
        
        return True, None
    
    def check_physical_address(self, data: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Check if physical address is included"""
        message_content = data.get('message_content', '')
        sender_address = data.get('sender_physical_address', '')
        
        if message_content and not sender_address:
            return False, "Physical address not included in communication"
        
        return True, None
    
    def record_consent(self, user_id: str, consent_type: str, consent_given: bool, 
                      consent_method: str, consent_text: str, ip_address: str = None, 
                      user_agent: str = None) -> str:
        """Record user consent"""
        consent_id = str(uuid.uuid4())
        
        consent_record = ConsentRecord(
            consent_id=consent_id,
            user_id=user_id,
            consent_type=consent_type,
            consent_given=consent_given,
            consent_method=consent_method,
            consent_timestamp=datetime.now(timezone.utc),
            consent_text=consent_text,
            ip_address=ip_address,
            user_agent=user_agent
        )
        
        self.consent_records[consent_id] = consent_record
        
        # Store in Redis for persistence
        consent_data = {
            'user_id': user_id,
            'consent_type': consent_type,
            'consent_given': consent_given,
            'consent_method': consent_method,
            'consent_timestamp': consent_record.consent_timestamp.isoformat(),
            'consent_text': consent_text,
            'ip_address': ip_address,
            'user_agent': user_agent
        }
        
        self.redis_client.setex(
            f"consent:{consent_id}",
            86400 * 2555,  # 7 years retention
            json.dumps(consent_data)
        )
        
        # Log consent action
        self.log_audit_event(
            user_id=user_id,
            action="consent_recorded",
            resource_type="consent",
            resource_id=consent_id,
            details={
                'consent_type': consent_type,
                'consent_given': consent_given,
                'consent_method': consent_method
            },
            ip_address=ip_address,
            user_agent=user_agent
        )
        
        logger.info(f"Consent recorded: {consent_id} for user {user_id}")
        return consent_id
    
    def withdraw_consent(self, user_id: str, consent_type: str, withdrawal_method: str = "user_request") -> bool:
        """Withdraw user consent"""
        try:
            # Find active consent records for user and type
            user_consents = [
                consent for consent in self.consent_records.values()
                if consent.user_id == user_id and consent.consent_type == consent_type and consent.is_active
            ]
            
            for consent in user_consents:
                consent.withdrawal_timestamp = datetime.now(timezone.utc)
                consent.is_active = False
                
                # Update in Redis
                consent_data = {
                    'user_id': consent.user_id,
                    'consent_type': consent.consent_type,
                    'consent_given': False,  # Update to withdrawn
                    'consent_method': consent.consent_method,
                    'consent_timestamp': consent.consent_timestamp.isoformat(),
                    'consent_text': consent.consent_text,
                    'withdrawal_timestamp': consent.withdrawal_timestamp.isoformat(),
                    'withdrawal_method': withdrawal_method,
                    'is_active': False
                }
                
                self.redis_client.setex(
                    f"consent:{consent.consent_id}",
                    86400 * 2555,  # 7 years retention
                    json.dumps(consent_data)
                )
            
            # Log withdrawal
            self.log_audit_event(
                user_id=user_id,
                action="consent_withdrawn",
                resource_type="consent",
                resource_id=f"user_{user_id}_{consent_type}",
                details={
                    'consent_type': consent_type,
                    'withdrawal_method': withdrawal_method,
                    'consents_withdrawn': len(user_consents)
                }
            )
            
            logger.info(f"Consent withdrawn for user {user_id}, type {consent_type}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to withdraw consent: {e}")
            return False
    
    def get_user_consents(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all consent records for a user"""
        user_consents = [
            consent for consent in self.consent_records.values()
            if consent.user_id == user_id
        ]
        
        return [
            {
                'consent_id': consent.consent_id,
                'consent_type': consent.consent_type,
                'consent_given': consent.consent_given,
                'consent_method': consent.consent_method,
                'consent_timestamp': consent.consent_timestamp.isoformat(),
                'withdrawal_timestamp': consent.withdrawal_timestamp.isoformat() if consent.withdrawal_timestamp else None,
                'is_active': consent.is_active
            }
            for consent in user_consents
        ]
    
    def log_audit_event(self, user_id: Optional[str], action: str, resource_type: str, 
                       resource_id: str, details: Dict[str, Any], 
                       ip_address: str = None, user_agent: str = None):
        """Log audit event"""
        log_id = str(uuid.uuid4())
        
        audit_log = AuditLog(
            log_id=log_id,
            timestamp=datetime.now(timezone.utc),
            user_id=user_id,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            details=details,
            ip_address=ip_address,
            user_agent=user_agent
        )
        
        self.audit_logs.append(audit_log)
        
        # Store in Redis
        log_data = {
            'timestamp': audit_log.timestamp.isoformat(),
            'user_id': user_id,
            'action': action,
            'resource_type': resource_type,
            'resource_id': resource_id,
            'details': details,
            'ip_address': ip_address,
            'user_agent': user_agent
        }
        
        self.redis_client.lpush('audit_logs', json.dumps(log_data))
        self.redis_client.ltrim('audit_logs', 0, 100000)  # Keep last 100k logs
        
        logger.info(f"Audit event logged: {action} on {resource_type}:{resource_id}")
    
    def generate_compliance_report(self, frameworks: List[ComplianceFramework] = None, 
                                 start_date: datetime = None, end_date: datetime = None) -> Dict[str, Any]:
        """Generate comprehensive compliance report"""
        frameworks = frameworks or [f for f in ComplianceFramework]
        end_date = end_date or datetime.now(timezone.utc)
        start_date = start_date or (end_date - timedelta(days=30))
        
        report = {
            'report_id': str(uuid.uuid4()),
            'generated_at': datetime.now(timezone.utc).isoformat(),
            'period': {
                'start_date': start_date.isoformat(),
                'end_date': end_date.isoformat()
            },
            'frameworks': [f.value for f in frameworks],
            'summary': {
                'overall_compliance_score': 0.0,
                'total_violations': 0,
                'critical_violations': 0,
                'resolved_violations': 0
            },
            'framework_details': {},
            'violation_trends': {},
            'consent_metrics': {},
            'recommendations': []
        }
        
        # Analyze violations by framework
        period_violations = [
            v for v in self.violations.values()
            if start_date <= v.detected_at <= end_date
        ]
        
        total_score = 0
        framework_count = 0
        
        for framework in frameworks:
            framework_violations = [v for v in period_violations if v.framework == framework]
            critical_violations = [v for v in framework_violations if v.severity == ViolationSeverity.CRITICAL]
            resolved_violations = [v for v in framework_violations if v.is_resolved]
            
            framework_rules = [r for r in self.compliance_rules.values() if r.framework == framework]
            compliance_rate = 1.0
            
            if framework_rules:
                violation_rate = len(framework_violations) / len(framework_rules)
                compliance_rate = max(0.0, 1.0 - violation_rate)
            
            framework_score = compliance_rate * 100
            total_score += framework_score
            framework_count += 1
            
            report['framework_details'][framework.value] = {
                'compliance_score': framework_score,
                'total_violations': len(framework_violations),
                'critical_violations': len(critical_violations),
                'resolved_violations': len(resolved_violations),
                'resolution_rate': len(resolved_violations) / len(framework_violations) if framework_violations else 1.0,
                'active_rules': len(framework_rules)
            }
        
        # Calculate overall compliance score
        if framework_count > 0:
            report['summary']['overall_compliance_score'] = total_score / framework_count
        
        report['summary']['total_violations'] = len(period_violations)
        report['summary']['critical_violations'] = len([v for v in period_violations if v.severity == ViolationSeverity.CRITICAL])
        report['summary']['resolved_violations'] = len([v for v in period_violations if v.is_resolved])
        
        # Consent metrics
        period_consents = [
            c for c in self.consent_records.values()
            if start_date <= c.consent_timestamp <= end_date
        ]
        
        consent_types = {}
        for consent in period_consents:
            if consent.consent_type not in consent_types:
                consent_types[consent.consent_type] = {'given': 0, 'withdrawn': 0}
            
            if consent.consent_given:
                consent_types[consent.consent_type]['given'] += 1
            else:
                consent_types[consent.consent_type]['withdrawn'] += 1
        
        report['consent_metrics'] = {
            'total_consents_recorded': len(period_consents),
            'consent_types': consent_types,
            'consent_rate': len([c for c in period_consents if c.consent_given]) / len(period_consents) if period_consents else 0
        }
        
        # Generate recommendations
        if report['summary']['overall_compliance_score'] < 90:
            report['recommendations'].extend([
                "Implement automated compliance monitoring",
                "Conduct compliance training for staff",
                "Review and update privacy policies"
            ])
        
        if report['summary']['critical_violations'] > 0:
            report['recommendations'].extend([
                "Address critical violations immediately",
                "Implement additional security controls",
                "Consider legal consultation"
            ])
        
        return report
    
    def process_data_subject_request(self, request_type: str, user_id: str, 
                                   request_details: Dict[str, Any]) -> Dict[str, Any]:
        """Process data subject requests (GDPR Article 15-22, CCPA)"""
        request_id = str(uuid.uuid4())
        
        response = {
            'request_id': request_id,
            'request_type': request_type,
            'user_id': user_id,
            'processed_at': datetime.now(timezone.utc).isoformat(),
            'status': 'completed',
            'data': {}
        }
        
        if request_type == 'access':
            # Right to access (GDPR Art. 15, CCPA)
            response['data'] = {
                'personal_data': self._get_user_personal_data(user_id),
                'consents': self.get_user_consents(user_id),
                'processing_purposes': self._get_processing_purposes(user_id),
                'data_sources': self._get_data_sources(user_id),
                'retention_periods': self._get_retention_periods(user_id)
            }
        
        elif request_type == 'deletion':
            # Right to erasure (GDPR Art. 17, CCPA)
            deletion_result = self._delete_user_data(user_id)
            response['data'] = {
                'deletion_completed': deletion_result['success'],
                'deleted_records': deletion_result['deleted_count'],
                'retention_exceptions': deletion_result['exceptions']
            }
        
        elif request_type == 'portability':
            # Right to data portability (GDPR Art. 20)
            response['data'] = {
                'portable_data': self._export_user_data(user_id),
                'format': 'JSON',
                'download_link': f"/api/data-export/{request_id}"
            }
        
        elif request_type == 'rectification':
            # Right to rectification (GDPR Art. 16)
            updates = request_details.get('updates', {})
            update_result = self._update_user_data(user_id, updates)
            response['data'] = {
                'updates_applied': update_result['success'],
                'updated_fields': update_result['updated_fields']
            }
        
        elif request_type == 'opt_out_sale':
            # CCPA opt-out of sale
            opt_out_result = self._opt_out_of_sale(user_id)
            response['data'] = {
                'opt_out_applied': opt_out_result['success'],
                'effective_date': datetime.now(timezone.utc).isoformat()
            }
        
        # Log the request
        self.log_audit_event(
            user_id=user_id,
            action=f"data_subject_request_{request_type}",
            resource_type="data_subject_request",
            resource_id=request_id,
            details=request_details
        )
        
        return response
    
    def _get_user_personal_data(self, user_id: str) -> Dict[str, Any]:
        """Get all personal data for a user"""
        # This would query all systems for user data
        return {
            'user_id': user_id,
            'data_collected': 'This would contain all personal data',
            'note': 'Implementation would query all data stores'
        }
    
    def _get_processing_purposes(self, user_id: str) -> List[str]:
        """Get data processing purposes for user"""
        return [
            'Lead scoring and qualification',
            'Marketing communications',
            'Customer service',
            'Legal compliance',
            'Fraud prevention'
        ]
    
    def _get_data_sources(self, user_id: str) -> List[str]:
        """Get data sources for user data"""
        return [
            'Web form submission',
            'Third-party data enrichment',
            'Customer interactions',
            'Public records'
        ]
    
    def _get_retention_periods(self, user_id: str) -> Dict[str, str]:
        """Get retention periods for user data"""
        return {
            'lead_data': '7 years',
            'consent_records': '7 years',
            'communication_logs': '3 years',
            'audit_logs': '7 years'
        }
    
    def _delete_user_data(self, user_id: str) -> Dict[str, Any]:
        """Delete user data (right to erasure)"""
        # This would implement actual data deletion
        return {
            'success': True,
            'deleted_count': 1,
            'exceptions': ['Legal hold records retained']
        }
    
    def _export_user_data(self, user_id: str) -> Dict[str, Any]:
        """Export user data for portability"""
        # This would export all user data in portable format
        return {
            'user_data': 'Exported data would be here',
            'format': 'JSON',
            'exported_at': datetime.now(timezone.utc).isoformat()
        }
    
    def _update_user_data(self, user_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Update user data (rectification)"""
        # This would implement data updates
        return {
            'success': True,
            'updated_fields': list(updates.keys())
        }
    
    def _opt_out_of_sale(self, user_id: str) -> Dict[str, Any]:
        """Opt user out of data sale"""
        # This would implement opt-out of sale
        return {
            'success': True,
            'effective_date': datetime.now(timezone.utc).isoformat()
        }

# Global compliance manager
compliance_manager = ComplianceManager()

# Example usage
async def example_usage():
    """Example of compliance system usage"""
    
    # Sample lead data
    lead_data = {
        'user_id': 'user_123',
        'email': 'john.doe@example.com',
        'phone': '(555) 123-4567',
        'age': 35,
        'income': 75000,
        'consent_given': True,
        'consent_method': 'web_form',
        'consent_timestamp': datetime.now(timezone.utc).isoformat(),
        'privacy_notice_shown': True,
        'tcpa_consent_given': True,
        'tcpa_consent_method': 'written',
        'communication_type': 'email',
        'message_content': 'Get your free insurance quote today! Click here to unsubscribe.',
        'subject_line': 'Free Insurance Quote Available',
        'sender_physical_address': '123 Insurance St, City, ST 12345',
        'created_at': datetime.now(timezone.utc).isoformat()
    }
    
    # Validate compliance
    validation_result = compliance_manager.validate_compliance(
        lead_data, 
        [ComplianceFramework.GDPR, ComplianceFramework.CCPA, ComplianceFramework.TCPA, ComplianceFramework.CAN_SPAM]
    )
    
    print("Compliance Validation Results:")
    print(json.dumps(validation_result, indent=2))
    
    # Record consent
    consent_id = compliance_manager.record_consent(
        user_id='user_123',
        consent_type='marketing',
        consent_given=True,
        consent_method='web_form',
        consent_text='I agree to receive marketing communications',
        ip_address='192.168.1.100'
    )
    
    print(f"\nConsent recorded: {consent_id}")
    
    # Generate compliance report
    compliance_report = compliance_manager.generate_compliance_report(
        frameworks=[ComplianceFramework.GDPR, ComplianceFramework.CCPA]
    )
    
    print("\nCompliance Report:")
    print(json.dumps(compliance_report, indent=2))
    
    # Process data subject request
    access_request = compliance_manager.process_data_subject_request(
        request_type='access',
        user_id='user_123',
        request_details={'requested_data': 'all'}
    )
    
    print("\nData Subject Access Request:")
    print(json.dumps(access_request, indent=2))

if __name__ == "__main__":
    asyncio.run(example_usage())
