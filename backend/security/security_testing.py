"""
Security Testing Framework
Automated security testing and vulnerability assessment
"""

import asyncio
import json
import logging
import requests
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import re
import hashlib
import random
import string

logger = logging.getLogger(__name__)

class TestType(Enum):
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    INPUT_VALIDATION = "input_validation"
    SQL_INJECTION = "sql_injection"
    XSS = "xss"
    CSRF = "csrf"
    RATE_LIMITING = "rate_limiting"
    SESSION_MANAGEMENT = "session_management"
    ENCRYPTION = "encryption"
    API_SECURITY = "api_security"

class Severity(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

class TestStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"

@dataclass
class SecurityTest:
    test_id: str
    name: str
    description: str
    test_type: TestType
    severity: Severity
    target_endpoint: str
    test_function: Callable
    expected_result: str
    status: TestStatus = TestStatus.PENDING
    result: Optional[str] = None
    execution_time: Optional[float] = None
    error_message: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)

@dataclass
class Vulnerability:
    vuln_id: str
    name: str
    description: str
    severity: Severity
    test_type: TestType
    endpoint: str
    evidence: Dict[str, Any]
    remediation: str
    discovered_at: datetime = field(default_factory=datetime.utcnow)

@dataclass
class TestSuite:
    suite_id: str
    name: str
    description: str
    tests: List[SecurityTest]
    status: TestStatus = TestStatus.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    passed_count: int = 0
    failed_count: int = 0
    vulnerabilities: List[Vulnerability] = field(default_factory=list)

class SecurityTestingFramework:
    """Automated security testing framework"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.test_suites: Dict[str, TestSuite] = {}
        self.vulnerabilities: List[Vulnerability] = []
        
        # Test configuration
        self.config = self._load_test_config()
        
        # Initialize test suites
        self._initialize_test_suites()
    
    def _load_test_config(self) -> Dict[str, Any]:
        """Load security testing configuration"""
        return {
            'timeouts': {
                'request_timeout': 30,
                'test_timeout': 300
            },
            'rate_limiting': {
                'max_requests_per_second': 10,
                'burst_limit': 20
            },
            'payloads': {
                'sql_injection': [
                    "' OR '1'='1",
                    "'; DROP TABLE users; --",
                    "' UNION SELECT * FROM users --",
                    "1' AND 1=1 --",
                    "admin'--"
                ],
                'xss': [
                    "<script>alert('XSS')</script>",
                    "javascript:alert('XSS')",
                    "<img src=x onerror=alert('XSS')>",
                    "';alert('XSS');//",
                    "<svg onload=alert('XSS')>"
                ],
                'path_traversal': [
                    "../../../etc/passwd",
                    "..\\..\\..\\windows\\system32\\drivers\\etc\\hosts",
                    "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd"
                ]
            },
            'authentication': {
                'test_credentials': {
                    'valid': {'username': 'testuser', 'password': 'testpass123'},
                    'invalid': {'username': 'invalid', 'password': 'wrongpass'}
                }
            }
        }
    
    def _initialize_test_suites(self):
        """Initialize predefined test suites"""
        
        # Authentication test suite
        auth_tests = [
            SecurityTest(
                test_id="auth_001",
                name="Valid Login Test",
                description="Test login with valid credentials",
                test_type=TestType.AUTHENTICATION,
                severity=Severity.HIGH,
                target_endpoint="/auth/login",
                test_function=self._test_valid_login,
                expected_result="successful_login"
            ),
            SecurityTest(
                test_id="auth_002",
                name="Invalid Login Test",
                description="Test login with invalid credentials",
                test_type=TestType.AUTHENTICATION,
                severity=Severity.MEDIUM,
                target_endpoint="/auth/login",
                test_function=self._test_invalid_login,
                expected_result="login_rejected"
            ),
            SecurityTest(
                test_id="auth_003",
                name="Brute Force Protection",
                description="Test brute force protection mechanisms",
                test_type=TestType.AUTHENTICATION,
                severity=Severity.HIGH,
                target_endpoint="/auth/login",
                test_function=self._test_brute_force_protection,
                expected_result="account_locked"
            )
        ]
        
        auth_suite = TestSuite(
            suite_id="auth_suite",
            name="Authentication Security Tests",
            description="Comprehensive authentication security testing",
            tests=auth_tests
        )
        
        self.test_suites["auth_suite"] = auth_suite
        
        # Input validation test suite
        input_tests = [
            SecurityTest(
                test_id="input_001",
                name="SQL Injection Test",
                description="Test for SQL injection vulnerabilities",
                test_type=TestType.SQL_INJECTION,
                severity=Severity.CRITICAL,
                target_endpoint="/api/lead-scoring/score",
                test_function=self._test_sql_injection,
                expected_result="input_sanitized"
            ),
            SecurityTest(
                test_id="input_002",
                name="XSS Test",
                description="Test for cross-site scripting vulnerabilities",
                test_type=TestType.XSS,
                severity=Severity.HIGH,
                target_endpoint="/api/message-generation/generate",
                test_function=self._test_xss,
                expected_result="script_blocked"
            ),
            SecurityTest(
                test_id="input_003",
                name="Path Traversal Test",
                description="Test for path traversal vulnerabilities",
                test_type=TestType.INPUT_VALIDATION,
                severity=Severity.HIGH,
                target_endpoint="/api/files/",
                test_function=self._test_path_traversal,
                expected_result="access_denied"
            )
        ]
        
        input_suite = TestSuite(
            suite_id="input_suite",
            name="Input Validation Security Tests",
            description="Input validation and injection attack tests",
            tests=input_tests
        )
        
        self.test_suites["input_suite"] = input_suite
        
        # API security test suite
        api_tests = [
            SecurityTest(
                test_id="api_001",
                name="Rate Limiting Test",
                description="Test API rate limiting mechanisms",
                test_type=TestType.RATE_LIMITING,
                severity=Severity.MEDIUM,
                target_endpoint="/api/lead-scoring/score",
                test_function=self._test_rate_limiting,
                expected_result="rate_limited"
            ),
            SecurityTest(
                test_id="api_002",
                name="Authorization Test",
                description="Test API authorization controls",
                test_type=TestType.AUTHORIZATION,
                severity=Severity.HIGH,
                target_endpoint="/admin/users",
                test_function=self._test_authorization,
                expected_result="access_denied"
            ),
            SecurityTest(
                test_id="api_003",
                name="CSRF Protection Test",
                description="Test CSRF protection mechanisms",
                test_type=TestType.CSRF,
                severity=Severity.MEDIUM,
                target_endpoint="/api/users/update",
                test_function=self._test_csrf_protection,
                expected_result="csrf_token_required"
            )
        ]
        
        api_suite = TestSuite(
            suite_id="api_suite",
            name="API Security Tests",
            description="API-specific security testing",
            tests=api_tests
        )
        
        self.test_suites["api_suite"] = api_suite
    
    async def run_test_suite(self, suite_id: str) -> TestSuite:
        """Run a complete test suite"""
        
        if suite_id not in self.test_suites:
            raise ValueError(f"Test suite {suite_id} not found")
        
        suite = self.test_suites[suite_id]
        suite.status = TestStatus.RUNNING
        suite.start_time = datetime.now(datetime.UTC)
        suite.passed_count = 0
        suite.failed_count = 0
        suite.vulnerabilities = []
        
        logger.info(f"Starting test suite: {suite.name}")
        
        for test in suite.tests:
            try:
                await self._run_single_test(test)
                
                if test.status == TestStatus.PASSED:
                    suite.passed_count += 1
                elif test.status == TestStatus.FAILED:
                    suite.failed_count += 1
                    
                    # Create vulnerability if test failed
                    vulnerability = self._create_vulnerability_from_test(test)
                    suite.vulnerabilities.append(vulnerability)
                    self.vulnerabilities.append(vulnerability)
                
            except Exception as e:
                test.status = TestStatus.ERROR
                test.error_message = str(e)
                logger.error(f"Error running test {test.test_id}: {e}")
        
        suite.end_time = datetime.now(datetime.UTC)
        suite.status = TestStatus.PASSED if suite.failed_count == 0 else TestStatus.FAILED
        
        logger.info(f"Test suite completed: {suite.name} - {suite.passed_count} passed, {suite.failed_count} failed")
        
        return suite
    
    async def _run_single_test(self, test: SecurityTest):
        """Run a single security test"""
        
        test.status = TestStatus.RUNNING
        start_time = time.time()
        
        try:
            # Execute test function
            result = await test.test_function(test)
            test.result = result
            test.execution_time = time.time() - start_time
            
            # Check if result matches expected
            if result == test.expected_result:
                test.status = TestStatus.PASSED
            else:
                test.status = TestStatus.FAILED
                
        except Exception as e:
            test.status = TestStatus.ERROR
            test.error_message = str(e)
            test.execution_time = time.time() - start_time
    
    async def _test_valid_login(self, test: SecurityTest) -> str:
        """Test valid login functionality"""
        
        credentials = self.config['authentication']['test_credentials']['valid']
        
        response = requests.post(
            f"{self.base_url}{test.target_endpoint}",
            json=credentials,
            timeout=self.config['timeouts']['request_timeout']
        )
        
        if response.status_code == 200 and 'token' in response.json():
            return "successful_login"
        else:
            return "login_failed"
    
    async def _test_invalid_login(self, test: SecurityTest) -> str:
        """Test invalid login handling"""
        
        credentials = self.config['authentication']['test_credentials']['invalid']
        
        response = requests.post(
            f"{self.base_url}{test.target_endpoint}",
            json=credentials,
            timeout=self.config['timeouts']['request_timeout']
        )
        
        if response.status_code == 401:
            return "login_rejected"
        else:
            return "improper_error_handling"
    
    async def _test_brute_force_protection(self, test: SecurityTest) -> str:
        """Test brute force protection"""
        
        credentials = self.config['authentication']['test_credentials']['invalid']
        
        # Attempt multiple failed logins
        for i in range(10):
            response = requests.post(
                f"{self.base_url}{test.target_endpoint}",
                json=credentials,
                timeout=self.config['timeouts']['request_timeout']
            )
            
            # Check if account gets locked
            if response.status_code == 429 or 'locked' in response.text.lower():
                return "account_locked"
            
            await asyncio.sleep(0.1)  # Small delay between attempts
        
        return "no_brute_force_protection"
    
    async def _test_sql_injection(self, test: SecurityTest) -> str:
        """Test for SQL injection vulnerabilities"""
        
        payloads = self.config['payloads']['sql_injection']
        
        for payload in payloads:
            test_data = {
                'email': f"test{payload}@example.com",
                'name': f"Test{payload}",
                'age': 25
            }
            
            try:
                response = requests.post(
                    f"{self.base_url}{test.target_endpoint}",
                    json=test_data,
                    timeout=self.config['timeouts']['request_timeout']
                )
                
                # Check for SQL error messages or unexpected behavior
                if (response.status_code == 500 or
                    any(error in response.text.lower() for error in ['sql', 'mysql', 'postgres', 'sqlite'])):
                    return "sql_injection_vulnerable"
                    
            except Exception as e:
                logger.warning(f"SQL injection test error: {e}")
        
        return "input_sanitized"
    
    async def _test_xss(self, test: SecurityTest) -> str:
        """Test for XSS vulnerabilities"""
        
        payloads = self.config['payloads']['xss']
        
        for payload in payloads:
            test_data = {
                'message_content': payload,
                'lead_data': {'name': 'Test User'}
            }
            
            try:
                response = requests.post(
                    f"{self.base_url}{test.target_endpoint}",
                    json=test_data,
                    timeout=self.config['timeouts']['request_timeout']
                )
                
                # Check if script is reflected without encoding
                if payload in response.text and '<script>' in response.text:
                    return "xss_vulnerable"
                    
            except Exception as e:
                logger.warning(f"XSS test error: {e}")
        
        return "script_blocked"
    
    async def _test_path_traversal(self, test: SecurityTest) -> str:
        """Test for path traversal vulnerabilities"""
        
        payloads = self.config['payloads']['path_traversal']
        
        for payload in payloads:
            try:
                response = requests.get(
                    f"{self.base_url}{test.target_endpoint}{payload}",
                    timeout=self.config['timeouts']['request_timeout']
                )
                
                # Check for system file content
                if ('root:' in response.text or 
                    'localhost' in response.text or
                    response.status_code == 200):
                    return "path_traversal_vulnerable"
                    
            except Exception as e:
                logger.warning(f"Path traversal test error: {e}")
        
        return "access_denied"
    
    async def _test_rate_limiting(self, test: SecurityTest) -> str:
        """Test rate limiting mechanisms"""
        
        # Send rapid requests
        for i in range(50):
            try:
                response = requests.post(
                    f"{self.base_url}{test.target_endpoint}",
                    json={'email': 'test@example.com'},
                    timeout=self.config['timeouts']['request_timeout']
                )
                
                if response.status_code == 429:
                    return "rate_limited"
                    
            except Exception as e:
                logger.warning(f"Rate limiting test error: {e}")
        
        return "no_rate_limiting"
    
    async def _test_authorization(self, test: SecurityTest) -> str:
        """Test authorization controls"""
        
        # Try to access admin endpoint without proper authorization
        try:
            response = requests.get(
                f"{self.base_url}{test.target_endpoint}",
                timeout=self.config['timeouts']['request_timeout']
            )
            
            if response.status_code in [401, 403]:
                return "access_denied"
            elif response.status_code == 200:
                return "authorization_bypass"
                
        except Exception as e:
            logger.warning(f"Authorization test error: {e}")
        
        return "access_denied"
    
    async def _test_csrf_protection(self, test: SecurityTest) -> str:
        """Test CSRF protection mechanisms"""
        
        # Try to make state-changing request without CSRF token
        try:
            response = requests.post(
                f"{self.base_url}{test.target_endpoint}",
                json={'user_id': '123', 'role': 'admin'},
                timeout=self.config['timeouts']['request_timeout']
            )
            
            if response.status_code == 403 or 'csrf' in response.text.lower():
                return "csrf_token_required"
            elif response.status_code == 200:
                return "csrf_vulnerable"
                
        except Exception as e:
            logger.warning(f"CSRF test error: {e}")
        
        return "csrf_token_required"
    
    def _create_vulnerability_from_test(self, test: SecurityTest) -> Vulnerability:
        """Create vulnerability record from failed test"""
        
        vuln_id = hashlib.md5(f"{test.test_id}{test.target_endpoint}".encode()).hexdigest()[:12]
        
        remediation_map = {
            TestType.SQL_INJECTION: "Implement parameterized queries and input validation",
            TestType.XSS: "Implement output encoding and Content Security Policy",
            TestType.CSRF: "Implement CSRF tokens for state-changing operations",
            TestType.AUTHENTICATION: "Strengthen authentication mechanisms",
            TestType.AUTHORIZATION: "Implement proper access controls",
            TestType.RATE_LIMITING: "Implement rate limiting mechanisms"
        }
        
        return Vulnerability(
            vuln_id=vuln_id,
            name=test.name,
            description=test.description,
            severity=test.severity,
            test_type=test.test_type,
            endpoint=test.target_endpoint,
            evidence={
                'test_result': test.result,
                'expected_result': test.expected_result,
                'execution_time': test.execution_time
            },
            remediation=remediation_map.get(test.test_type, "Review and fix security issue")
        )
    
    def generate_security_report(self) -> Dict[str, Any]:
        """Generate comprehensive security report"""
        
        total_tests = sum(len(suite.tests) for suite in self.test_suites.values())
        total_passed = sum(suite.passed_count for suite in self.test_suites.values())
        total_failed = sum(suite.failed_count for suite in self.test_suites.values())
        
        # Categorize vulnerabilities by severity
        vuln_by_severity = {
            'critical': len([v for v in self.vulnerabilities if v.severity == Severity.CRITICAL]),
            'high': len([v for v in self.vulnerabilities if v.severity == Severity.HIGH]),
            'medium': len([v for v in self.vulnerabilities if v.severity == Severity.MEDIUM]),
            'low': len([v for v in self.vulnerabilities if v.severity == Severity.LOW])
        }
        
        # Calculate security score
        security_score = self._calculate_security_score(vuln_by_severity, total_tests)
        
        return {
            'report_id': hashlib.md5(str(datetime.now(datetime.UTC)).encode()).hexdigest()[:12],
            'generated_at': datetime.now(datetime.UTC).isoformat(),
            'summary': {
                'total_tests': total_tests,
                'tests_passed': total_passed,
                'tests_failed': total_failed,
                'pass_rate': (total_passed / total_tests * 100) if total_tests > 0 else 0,
                'security_score': security_score
            },
            'vulnerabilities': {
                'total': len(self.vulnerabilities),
                'by_severity': vuln_by_severity,
                'details': [
                    {
                        'id': v.vuln_id,
                        'name': v.name,
                        'severity': v.severity.value,
                        'type': v.test_type.value,
                        'endpoint': v.endpoint,
                        'remediation': v.remediation
                    } for v in self.vulnerabilities
                ]
            },
            'test_suites': {
                suite_id: {
                    'name': suite.name,
                    'status': suite.status.value,
                    'passed': suite.passed_count,
                    'failed': suite.failed_count,
                    'execution_time': (suite.end_time - suite.start_time).total_seconds() if suite.end_time and suite.start_time else 0
                } for suite_id, suite in self.test_suites.items()
            },
            'recommendations': self._generate_recommendations(vuln_by_severity)
        }
    
    def _calculate_security_score(self, vuln_by_severity: Dict[str, int], total_tests: int) -> float:
        """Calculate overall security score"""
        
        if total_tests == 0:
            return 0.0
        
        # Weight vulnerabilities by severity
        penalty = (
            vuln_by_severity['critical'] * 10 +
            vuln_by_severity['high'] * 5 +
            vuln_by_severity['medium'] * 2 +
            vuln_by_severity['low'] * 1
        )
        
        # Calculate score (0-100)
        max_penalty = total_tests * 10  # Assume all tests could be critical
        score = max(0, 100 - (penalty / max_penalty * 100))
        
        return round(score, 2)
    
    def _generate_recommendations(self, vuln_by_severity: Dict[str, int]) -> List[str]:
        """Generate security recommendations"""
        
        recommendations = []
        
        if vuln_by_severity['critical'] > 0:
            recommendations.append("URGENT: Address critical vulnerabilities immediately")
        
        if vuln_by_severity['high'] > 0:
            recommendations.append("HIGH PRIORITY: Fix high-severity vulnerabilities within 24 hours")
        
        if vuln_by_severity['medium'] > 0:
            recommendations.append("MEDIUM PRIORITY: Address medium-severity issues within 1 week")
        
        if sum(vuln_by_severity.values()) == 0:
            recommendations.append("GOOD: No vulnerabilities detected in current test scope")
        
        recommendations.extend([
            "Implement continuous security testing in CI/CD pipeline",
            "Regular security training for development team",
            "Consider third-party security audit",
            "Implement Web Application Firewall (WAF)",
            "Enable comprehensive logging and monitoring"
        ])
        
        return recommendations

# Global security testing framework
security_tester = SecurityTestingFramework()

async def run_comprehensive_security_test() -> Dict[str, Any]:
    """Run all security test suites"""
    
    results = {}
    
    for suite_id in security_tester.test_suites.keys():
        try:
            suite_result = await security_tester.run_test_suite(suite_id)
            results[suite_id] = suite_result
        except Exception as e:
            logger.error(f"Error running test suite {suite_id}: {e}")
            results[suite_id] = {'error': str(e)}
    
    # Generate comprehensive report
    report = security_tester.generate_security_report()
    
    return {
        'test_results': results,
        'security_report': report
    }

if __name__ == "__main__":
    # Run security tests
    asyncio.run(run_comprehensive_security_test())
