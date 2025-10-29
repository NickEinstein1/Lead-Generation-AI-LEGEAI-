import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
import re
import logging
from dataclasses import dataclass
from enum import Enum
import hashlib
import json

logger = logging.getLogger(__name__)

class ValidationSeverity(Enum):
    """Validation issue severity levels"""
    CRITICAL = "CRITICAL"  # Blocks processing
    HIGH = "HIGH"         # Requires attention
    MEDIUM = "MEDIUM"     # Warning
    LOW = "LOW"          # Information

class ValidationCategory(Enum):
    """Categories of validation checks"""
    DATA_QUALITY = "DATA_QUALITY"
    COMPLIANCE = "COMPLIANCE"
    BUSINESS_RULES = "BUSINESS_RULES"
    FRAUD_DETECTION = "FRAUD_DETECTION"
    COMPLETENESS = "COMPLETENESS"
    CONSISTENCY = "CONSISTENCY"

@dataclass
class ValidationResult:
    """Result of a validation check"""
    check_name: str
    category: ValidationCategory
    severity: ValidationSeverity
    passed: bool
    message: str
    details: Dict[str, Any] = None
    suggested_action: str = None
    confidence_score: float = 1.0

@dataclass
class LeadQualityScore:
    """Overall lead quality assessment"""
    overall_score: float
    quality_grade: str
    validation_results: List[ValidationResult]
    data_completeness: float
    compliance_status: str
    fraud_risk_score: float
    business_rule_violations: int
    recommended_action: str
    quality_factors: Dict[str, float]

class LeadValidator:
    """Comprehensive lead validation and quality control"""
    
    def __init__(self, config_path: str = None):
        self.validation_rules = self._load_validation_config(config_path)
        self.fraud_patterns = self._load_fraud_patterns()
        self.compliance_rules = self._load_compliance_rules()
        self.business_rules = self._load_business_rules()
        
        # Quality thresholds
        self.quality_thresholds = {
            'PREMIUM': 90,
            'HIGH': 75,
            'MEDIUM': 60,
            'LOW': 40,
            'REJECT': 0
        }
        
        # Validation statistics
        self.validation_stats = {
            'total_leads_processed': 0,
            'validation_failures': 0,
            'fraud_detections': 0,
            'compliance_violations': 0
        }
    
    def validate_lead(self, lead_data: Dict[str, Any]) -> LeadQualityScore:
        """Comprehensive lead validation"""
        logger.info(f"Validating lead: {lead_data.get('lead_id', 'UNKNOWN')}")
        
        validation_results = []
        
        # 1. Data Quality Validation
        validation_results.extend(self._validate_data_quality(lead_data))
        
        # 2. Compliance Validation
        validation_results.extend(self._validate_compliance(lead_data))
        
        # 3. Business Rules Validation
        validation_results.extend(self._validate_business_rules(lead_data))
        
        # 4. Fraud Detection
        validation_results.extend(self._detect_fraud_patterns(lead_data))
        
        # 5. Completeness Check
        validation_results.extend(self._check_data_completeness(lead_data))
        
        # 6. Consistency Validation
        validation_results.extend(self._validate_data_consistency(lead_data))
        
        # Calculate overall quality score
        quality_score = self._calculate_quality_score(validation_results, lead_data)
        
        # Update statistics
        self._update_validation_stats(quality_score)
        
        return quality_score
    
    def _validate_data_quality(self, lead_data: Dict[str, Any]) -> List[ValidationResult]:
        """Validate data quality aspects"""
        results = []
        
        # Email validation
        if 'email' in lead_data:
            email_result = self._validate_email(lead_data['email'])
            results.append(email_result)
        
        # Phone validation
        if 'phone' in lead_data:
            phone_result = self._validate_phone(lead_data['phone'])
            results.append(phone_result)
        
        # Age validation
        if 'age' in lead_data:
            age_result = self._validate_age(lead_data['age'])
            results.append(age_result)
        
        # Income validation
        if 'income' in lead_data:
            income_result = self._validate_income(lead_data['income'])
            results.append(income_result)
        
        # Credit score validation
        if 'credit_score' in lead_data:
            credit_result = self._validate_credit_score(lead_data['credit_score'])
            results.append(credit_result)
        
        # Address validation
        if any(field in lead_data for field in ['address', 'city', 'state', 'zip_code']):
            address_result = self._validate_address(lead_data)
            results.append(address_result)
        
        # Date validation
        date_fields = ['birth_date', 'lead_created_date', 'consent_timestamp']
        for field in date_fields:
            if field in lead_data:
                date_result = self._validate_date(lead_data[field], field)
                results.append(date_result)
        
        return results
    
    def _validate_compliance(self, lead_data: Dict[str, Any]) -> List[ValidationResult]:
        """Validate compliance requirements"""
        results = []
        
        # TCPA Compliance
        tcpa_result = self._validate_tcpa_compliance(lead_data)
        results.append(tcpa_result)
        
        # HIPAA Compliance (for healthcare)
        if self._is_healthcare_lead(lead_data):
            hipaa_result = self._validate_hipaa_compliance(lead_data)
            results.append(hipaa_result)
        
        # GDPR Compliance
        gdpr_result = self._validate_gdpr_compliance(lead_data)
        results.append(gdpr_result)
        
        # State-specific regulations
        if 'state' in lead_data:
            state_result = self._validate_state_regulations(lead_data)
            results.append(state_result)
        
        # Do Not Call Registry
        dnc_result = self._validate_dnc_registry(lead_data)
        results.append(dnc_result)
        
        # Consent validation
        consent_result = self._validate_consent(lead_data)
        results.append(consent_result)
        
        return results
    
    def _validate_business_rules(self, lead_data: Dict[str, Any]) -> List[ValidationResult]:
        """Validate business-specific rules"""
        results = []
        
        # Age eligibility
        age_eligibility = self._validate_age_eligibility(lead_data)
        results.append(age_eligibility)
        
        # Income requirements
        income_eligibility = self._validate_income_requirements(lead_data)
        results.append(income_eligibility)
        
        # Geographic restrictions
        geo_eligibility = self._validate_geographic_eligibility(lead_data)
        results.append(geo_eligibility)
        
        # Product-specific rules
        product_eligibility = self._validate_product_eligibility(lead_data)
        results.append(product_eligibility)
        
        # Duplicate detection
        duplicate_check = self._check_duplicate_lead(lead_data)
        results.append(duplicate_check)
        
        # Lead freshness
        freshness_check = self._validate_lead_freshness(lead_data)
        results.append(freshness_check)
        
        # Contact preferences
        contact_validation = self._validate_contact_preferences(lead_data)
        results.append(contact_validation)
        
        return results
    
    def _detect_fraud_patterns(self, lead_data: Dict[str, Any]) -> List[ValidationResult]:
        """Detect potential fraud patterns"""
        results = []
        
        # Suspicious patterns
        pattern_result = self._detect_suspicious_patterns(lead_data)
        results.append(pattern_result)
        
        # Velocity checks
        velocity_result = self._check_submission_velocity(lead_data)
        results.append(velocity_result)
        
        # Data consistency fraud indicators
        consistency_fraud = self._detect_consistency_fraud(lead_data)
        results.append(consistency_fraud)
        
        # Known fraud indicators
        known_fraud = self._check_known_fraud_indicators(lead_data)
        results.append(known_fraud)
        
        # Behavioral anomalies
        behavioral_anomalies = self._detect_behavioral_anomalies(lead_data)
        results.append(behavioral_anomalies)
        
        return results
    
    def _check_data_completeness(self, lead_data: Dict[str, Any]) -> List[ValidationResult]:
        """Check data completeness"""
        required_fields = self._get_required_fields(lead_data)
        optional_fields = self._get_optional_fields(lead_data)
        
        missing_required = [field for field in required_fields if field not in lead_data or not lead_data[field]]
        missing_optional = [field for field in optional_fields if field not in lead_data or not lead_data[field]]
        
        completeness_score = (len(required_fields) - len(missing_required)) / len(required_fields) * 100
        
        if missing_required:
            severity = ValidationSeverity.CRITICAL
            message = f"Missing required fields: {', '.join(missing_required)}"
            passed = False
        elif len(missing_optional) > len(optional_fields) * 0.5:
            severity = ValidationSeverity.MEDIUM
            message = f"Many optional fields missing: {len(missing_optional)}/{len(optional_fields)}"
            passed = True
        else:
            severity = ValidationSeverity.LOW
            message = f"Data completeness: {completeness_score:.1f}%"
            passed = True
        
        return [ValidationResult(
            check_name="data_completeness",
            category=ValidationCategory.COMPLETENESS,
            severity=severity,
            passed=passed,
            message=message,
            details={
                'completeness_score': completeness_score,
                'missing_required': missing_required,
                'missing_optional': missing_optional
            },
            suggested_action="Request missing required fields" if missing_required else "Consider enriching optional data"
        )]
    
    def _validate_data_consistency(self, lead_data: Dict[str, Any]) -> List[ValidationResult]:
        """Validate data consistency"""
        results = []
        
        # Age vs birth date consistency
        if 'age' in lead_data and 'birth_date' in lead_data:
            consistency_result = self._validate_age_birth_date_consistency(lead_data)
            results.append(consistency_result)
        
        # Income vs employment consistency
        if 'income' in lead_data and 'employment_status' in lead_data:
            employment_consistency = self._validate_income_employment_consistency(lead_data)
            results.append(employment_consistency)
        
        # Address components consistency
        address_consistency = self._validate_address_consistency(lead_data)
        results.append(address_consistency)
        
        # Contact method vs preferences consistency
        contact_consistency = self._validate_contact_consistency(lead_data)
        results.append(contact_consistency)
        
        return results
    
    def _validate_email(self, email: str) -> ValidationResult:
        """Validate email format and quality"""
        if not email:
            return ValidationResult(
                check_name="email_validation",
                category=ValidationCategory.DATA_QUALITY,
                severity=ValidationSeverity.HIGH,
                passed=False,
                message="Email is required",
                suggested_action="Request valid email address"
            )
        
        # Basic email regex
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        
        if not re.match(email_pattern, email):
            return ValidationResult(
                check_name="email_validation",
                category=ValidationCategory.DATA_QUALITY,
                severity=ValidationSeverity.HIGH,
                passed=False,
                message="Invalid email format",
                details={'email': email},
                suggested_action="Request valid email address"
            )
        
        # Check for disposable email domains
        disposable_domains = ['tempmail.com', '10minutemail.com', 'guerrillamail.com']
        domain = email.split('@')[1].lower()
        
        if domain in disposable_domains:
            return ValidationResult(
                check_name="email_validation",
                category=ValidationCategory.FRAUD_DETECTION,
                severity=ValidationSeverity.HIGH,
                passed=False,
                message="Disposable email detected",
                details={'domain': domain},
                suggested_action="Request permanent email address"
            )
        
        return ValidationResult(
            check_name="email_validation",
            category=ValidationCategory.DATA_QUALITY,
            severity=ValidationSeverity.LOW,
            passed=True,
            message="Email format valid"
        )
    
    def _validate_phone(self, phone: str) -> ValidationResult:
        """Validate phone number"""
        if not phone:
            return ValidationResult(
                check_name="phone_validation",
                category=ValidationCategory.DATA_QUALITY,
                severity=ValidationSeverity.MEDIUM,
                passed=False,
                message="Phone number missing",
                suggested_action="Request phone number"
            )
        
        # Clean phone number
        cleaned_phone = re.sub(r'[^\d]', '', phone)
        
        # US phone number validation
        if len(cleaned_phone) == 10:
            # Valid 10-digit number
            pass
        elif len(cleaned_phone) == 11 and cleaned_phone.startswith('1'):
            # Valid 11-digit with country code
            cleaned_phone = cleaned_phone[1:]
        else:
            return ValidationResult(
                check_name="phone_validation",
                category=ValidationCategory.DATA_QUALITY,
                severity=ValidationSeverity.MEDIUM,
                passed=False,
                message="Invalid phone number format",
                details={'phone': phone},
                suggested_action="Request valid 10-digit phone number"
            )
        
        # Check for invalid patterns
        invalid_patterns = ['0000000000', '1111111111', '1234567890']
        if cleaned_phone in invalid_patterns:
            return ValidationResult(
                check_name="phone_validation",
                category=ValidationCategory.FRAUD_DETECTION,
                severity=ValidationSeverity.HIGH,
                passed=False,
                message="Suspicious phone number pattern",
                details={'phone': phone},
                suggested_action="Verify phone number authenticity"
            )
        
        return ValidationResult(
            check_name="phone_validation",
            category=ValidationCategory.DATA_QUALITY,
            severity=ValidationSeverity.LOW,
            passed=True,
            message="Phone number format valid"
        )
    
    def _validate_tcpa_compliance(self, lead_data: Dict[str, Any]) -> ValidationResult:
        """Validate TCPA compliance"""
        consent_given = lead_data.get('consent_given', False)
        consent_timestamp = lead_data.get('consent_timestamp')
        consent_method = lead_data.get('consent_method', '')
        
        if not consent_given:
            return ValidationResult(
                check_name="tcpa_compliance",
                category=ValidationCategory.COMPLIANCE,
                severity=ValidationSeverity.CRITICAL,
                passed=False,
                message="TCPA consent not provided",
                suggested_action="Obtain explicit consent before contact"
            )
        
        if not consent_timestamp:
            return ValidationResult(
                check_name="tcpa_compliance",
                category=ValidationCategory.COMPLIANCE,
                severity=ValidationSeverity.HIGH,
                passed=False,
                message="TCPA consent timestamp missing",
                suggested_action="Record consent timestamp"
            )
        
        # Check consent age (must be within reasonable timeframe)
        try:
            consent_date = datetime.fromisoformat(consent_timestamp.replace('Z', '+00:00'))
            days_since_consent = (datetime.now() - consent_date.replace(tzinfo=None)).days
            
            if days_since_consent > 365:  # 1 year old consent
                return ValidationResult(
                    check_name="tcpa_compliance",
                    category=ValidationCategory.COMPLIANCE,
                    severity=ValidationSeverity.MEDIUM,
                    passed=True,
                    message="TCPA consent is old, consider re-confirmation",
                    details={'days_since_consent': days_since_consent},
                    suggested_action="Consider obtaining fresh consent"
                )
        except:
            return ValidationResult(
                check_name="tcpa_compliance",
                category=ValidationCategory.COMPLIANCE,
                severity=ValidationSeverity.HIGH,
                passed=False,
                message="Invalid consent timestamp format",
                suggested_action="Fix consent timestamp format"
            )
        
        return ValidationResult(
            check_name="tcpa_compliance",
            category=ValidationCategory.COMPLIANCE,
            severity=ValidationSeverity.LOW,
            passed=True,
            message="TCPA compliance validated"
        )
    
    def _detect_suspicious_patterns(self, lead_data: Dict[str, Any]) -> ValidationResult:
        """Detect suspicious fraud patterns"""
        fraud_score = 0
        fraud_indicators = []
        
        # Check for repeated characters in name
        name = lead_data.get('first_name', '') + lead_data.get('last_name', '')
        if name and len(set(name.lower())) < 3:
            fraud_score += 30
            fraud_indicators.append("Repeated characters in name")
        
        # Check for sequential numbers in phone/SSN
        phone = re.sub(r'[^\d]', '', lead_data.get('phone', ''))
        if phone and self._has_sequential_digits(phone):
            fraud_score += 25
            fraud_indicators.append("Sequential digits in phone")
        
        # Check for mismatched data patterns
        age = lead_data.get('age', 0)
        income = lead_data.get('income', 0)
        
        if age < 25 and income > 150000:
            fraud_score += 20
            fraud_indicators.append("Unusual age-income combination")
        
        # Check for common test data
        test_patterns = ['test', 'example', 'sample', 'demo']
        for field in ['first_name', 'last_name', 'email']:
            value = str(lead_data.get(field, '')).lower()
            if any(pattern in value for pattern in test_patterns):
                fraud_score += 40
                fraud_indicators.append(f"Test data pattern in {field}")
        
        # Determine severity based on fraud score
        if fraud_score >= 50:
            severity = ValidationSeverity.CRITICAL
            passed = False
            message = "High fraud risk detected"
            action = "Manual review required"
        elif fraud_score >= 30:
            severity = ValidationSeverity.HIGH
            passed = False
            message = "Moderate fraud risk detected"
            action = "Additional verification recommended"
        elif fraud_score > 0:
            severity = ValidationSeverity.MEDIUM
            passed = True
            message = "Low fraud risk detected"
            action = "Monitor for additional indicators"
        else:
            severity = ValidationSeverity.LOW
            passed = True
            message = "No fraud indicators detected"
            action = "Proceed normally"
        
        return ValidationResult(
            check_name="fraud_detection",
            category=ValidationCategory.FRAUD_DETECTION,
            severity=severity,
            passed=passed,
            message=message,
            details={
                'fraud_score': fraud_score,
                'indicators': fraud_indicators
            },
            suggested_action=action,
            confidence_score=min(fraud_score / 100, 1.0)
        )
    
    def _calculate_quality_score(self, validation_results: List[ValidationResult], 
                               lead_data: Dict[str, Any]) -> LeadQualityScore:
        """Calculate overall lead quality score"""
        
        # Base score
        base_score = 100.0
        
        # Deduct points based on validation failures
        for result in validation_results:
            if not result.passed:
                if result.severity == ValidationSeverity.CRITICAL:
                    base_score -= 25
                elif result.severity == ValidationSeverity.HIGH:
                    base_score -= 15
                elif result.severity == ValidationSeverity.MEDIUM:
                    base_score -= 8
                elif result.severity == ValidationSeverity.LOW:
                    base_score -= 3
        
        # Ensure score doesn't go below 0
        final_score = max(0, base_score)
        
        # Determine quality grade
        quality_grade = self._get_quality_grade(final_score)
        
        # Calculate specific metrics
        data_completeness = self._calculate_completeness_score(lead_data)
        compliance_status = self._get_compliance_status(validation_results)
        fraud_risk_score = self._get_fraud_risk_score(validation_results)
        business_rule_violations = self._count_business_rule_violations(validation_results)
        
        # Determine recommended action
        recommended_action = self._get_recommended_action(final_score, validation_results)
        
        # Quality factors breakdown
        quality_factors = {
            'data_quality': self._calculate_data_quality_factor(validation_results),
            'compliance': self._calculate_compliance_factor(validation_results),
            'completeness': data_completeness,
            'fraud_risk': 100 - fraud_risk_score,
            'business_rules': self._calculate_business_rules_factor(validation_results)
        }
        
        return LeadQualityScore(
            overall_score=final_score,
            quality_grade=quality_grade,
            validation_results=validation_results,
            data_completeness=data_completeness,
            compliance_status=compliance_status,
            fraud_risk_score=fraud_risk_score,
            business_rule_violations=business_rule_violations,
            recommended_action=recommended_action,
            quality_factors=quality_factors
        )
    
    def _get_quality_grade(self, score: float) -> str:
        """Get quality grade based on score"""
        for grade, threshold in self.quality_thresholds.items():
            if score >= threshold:
                return grade
        return 'REJECT'
    
    def _has_sequential_digits(self, number_str: str) -> bool:
        """Check if string has sequential digits"""
        if len(number_str) < 3:
            return False
        
        for i in range(len(number_str) - 2):
            if (int(number_str[i]) + 1 == int(number_str[i+1]) and 
                int(number_str[i+1]) + 1 == int(number_str[i+2])):
                return True
        return False
    
    def _load_validation_config(self, config_path: str) -> Dict:
        """Load validation configuration"""
        # Default configuration
        return {
            'required_fields': ['first_name', 'last_name', 'email', 'phone', 'consent_given'],
            'optional_fields': ['age', 'income', 'address', 'city', 'state', 'zip_code'],
            'fraud_thresholds': {'high': 50, 'medium': 30, 'low': 10},
            'compliance_requirements': ['tcpa', 'gdpr']
        }
    
    def _load_fraud_patterns(self) -> Dict:
        """Load known fraud patterns"""
        return {
            'suspicious_domains': ['tempmail.com', '10minutemail.com'],
            'test_patterns': ['test', 'example', 'sample', 'demo'],
            'invalid_phones': ['0000000000', '1111111111', '1234567890']
        }
    
    def _load_compliance_rules(self) -> Dict:
        """Load compliance rules"""
        return {
            'tcpa': {'consent_required': True, 'timestamp_required': True},
            'hipaa': {'healthcare_consent': True, 'data_encryption': True},
            'gdpr': {'explicit_consent': True, 'data_retention_limits': True}
        }
    
    def _load_business_rules(self) -> Dict:
        """Load business rules"""
        return {
            'age_limits': {'min': 18, 'max': 85},
            'income_limits': {'min': 0, 'max': 10000000},
            'geographic_restrictions': ['sanctioned_countries'],
            'product_eligibility': {'life_insurance': {'min_age': 18, 'max_age': 75}}
        }
    
    # Additional helper methods would be implemented here...
    def _validate_age(self, age: Any) -> ValidationResult:
        """Validate age field"""
        try:
            age_int = int(age)
            if age_int < 18 or age_int > 120:
                return ValidationResult(
                    check_name="age_validation",
                    category=ValidationCategory.DATA_QUALITY,
                    severity=ValidationSeverity.HIGH,
                    passed=False,
                    message=f"Age {age_int} is outside valid range (18-120)",
                    suggested_action="Verify age accuracy"
                )
            return ValidationResult(
                check_name="age_validation",
                category=ValidationCategory.DATA_QUALITY,
                severity=ValidationSeverity.LOW,
                passed=True,
                message="Age is valid"
            )
        except (ValueError, TypeError):
            return ValidationResult(
                check_name="age_validation",
                category=ValidationCategory.DATA_QUALITY,
                severity=ValidationSeverity.HIGH,
                passed=False,
                message="Age must be a valid number",
                suggested_action="Provide numeric age"
            )
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get validation statistics summary"""
        return {
            'total_processed': self.validation_stats['total_leads_processed'],
            'failure_rate': self.validation_stats['validation_failures'] / max(1, self.validation_stats['total_leads_processed']),
            'fraud_detection_rate': self.validation_stats['fraud_detections'] / max(1, self.validation_stats['total_leads_processed']),
            'compliance_violation_rate': self.validation_stats['compliance_violations'] / max(1, self.validation_stats['total_leads_processed'])
        }

# Quality Control Manager
class QualityControlManager:
    """Manages quality control processes and reporting"""
    
    def __init__(self):
        self.validator = LeadValidator()
        self.quality_reports = []
        self.quality_trends = {}
    
    def process_lead_batch(self, leads: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process a batch of leads for quality control"""
        results = []
        quality_summary = {
            'total_leads': len(leads),
            'passed': 0,
            'failed': 0,
            'quality_distribution': {},
            'common_issues': {}
        }
        
        for lead in leads:
            quality_score = self.validator.validate_lead(lead)
            results.append({
                'lead_id': lead.get('lead_id'),
                'quality_score': quality_score
            })
            
            # Update summary
            if quality_score.overall_score >= 60:
                quality_summary['passed'] += 1
            else:
                quality_summary['failed'] += 1
            
            # Track quality distribution
            grade = quality_score.quality_grade
            quality_summary['quality_distribution'][grade] = quality_summary['quality_distribution'].get(grade, 0) + 1
        
        return {
            'results': results,
            'summary': quality_summary,
            'validation_stats': self.validator.get_validation_summary()
        }

if __name__ == "__main__":
    # Example usage
    validator = LeadValidator()
    
    sample_lead = {
        'lead_id': 'LEAD_001',
        'first_name': 'John',
        'last_name': 'Doe',
        'email': 'john.doe@email.com',
        'phone': '555-123-4567',
        'age': 35,
        'income': 75000,
        'consent_given': True,
        'consent_timestamp': '2024-01-15T10:30:00Z'
    }
    
    quality_score = validator.validate_lead(sample_lead)
    print(f"Quality Score: {quality_score.overall_score}")
    print(f"Quality Grade: {quality_score.quality_grade}")
    print(f"Recommended Action: {quality_score.recommended_action}")