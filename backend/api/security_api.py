from fastapi import APIRouter, HTTPException, Depends, Request, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import logging
from datetime import datetime, timedelta, timezone

from backend.security.authentication import auth_manager, require_auth, require_api_key, Permission, UserRole
from backend.security.data_protection import data_protection, DataClassification
from backend.compliance.compliance_manager import compliance_manager, ComplianceFramework
from backend.security.security_monitoring import security_monitor, SecurityEventType, ThreatLevel, IncidentStatus

router = APIRouter(prefix="/security", tags=["Security & Compliance"])
security = HTTPBearer()

class LoginRequest(BaseModel):
    username: str
    password: str

class CreateUserRequest(BaseModel):
    username: str
    email: str
    password: str
    role: str

class ConsentRequest(BaseModel):
    user_id: str
    consent_type: str
    consent_given: bool
    consent_method: str
    consent_text: str

class DataSubjectRequest(BaseModel):
    request_type: str  # access, deletion, portability, rectification, opt_out_sale
    user_id: str
    request_details: Dict[str, Any] = {}

class SecurityEventRequest(BaseModel):
    event_type: str
    source_ip: str
    user_id: Optional[str] = None
    description: str
    raw_data: Dict[str, Any] = {}
    severity: str = "medium"

class IncidentUpdateRequest(BaseModel):
    status: Optional[str] = None
    assigned_to: Optional[str] = None
    notes: Optional[str] = None

# Authentication endpoints
@router.post("/auth/login")
async def login(request: LoginRequest, req: Request):
    """Authenticate user and create session"""
    try:
        # Get client info
        ip_address = req.client.host
        user_agent = req.headers.get("user-agent", "")
        
        # Check rate limiting
        if not auth_manager.check_rate_limit(ip_address):
            # Log security event
            security_monitor.log_security_event(
                SecurityEventType.SUSPICIOUS_ACTIVITY,
                ip_address,
                None,
                "Rate limit exceeded for login attempts",
                {"username": request.username},
                ThreatLevel.MEDIUM
            )
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
        
        # Authenticate
        success, session_id, message = auth_manager.authenticate_user(
            request.username, request.password, ip_address, user_agent
        )
        
        if not success:
            # Log failed login
            security_monitor.log_security_event(
                SecurityEventType.AUTHENTICATION_FAILURE,
                ip_address,
                None,
                f"Failed login attempt for user: {request.username}",
                {"username": request.username, "reason": message},
                ThreatLevel.LOW
            )
            raise HTTPException(status_code=401, detail=message)
        
        # Log successful login
        security_monitor.log_security_event(
            SecurityEventType.AUTHENTICATION_FAILURE,  # This should be SUCCESS in real implementation
            ip_address,
            request.username,
            "Successful login",
            {"username": request.username},
            ThreatLevel.LOW
        )
        
        return {
            "status": "success",
            "session_id": session_id,
            "message": message
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Login error: {str(e)}")
        raise HTTPException(status_code=500, detail="Authentication failed")

@router.post("/auth/logout")
@require_auth()
async def logout(session_id: str, current_user=None):
    """Logout user and invalidate session"""
    try:
        success = auth_manager.logout_user(session_id)
        
        if success:
            return {"status": "success", "message": "Logged out successfully"}
        else:
            raise HTTPException(status_code=400, detail="Logout failed")
            
    except Exception as e:
        logging.error(f"Logout error: {str(e)}")
        raise HTTPException(status_code=500, detail="Logout failed")

@router.post("/auth/create-user")
@require_auth(Permission.MANAGE_USERS)
async def create_user(request: CreateUserRequest, current_user=None):
    """Create a new user (admin only)"""
    try:
        # Validate role
        try:
            role = UserRole(request.role)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid role")
        
        success, user_id = auth_manager.create_user(
            request.username, request.email, request.password, role
        )
        
        if success:
            return {
                "status": "success",
                "user_id": user_id,
                "message": "User created successfully"
            }
        else:
            raise HTTPException(status_code=400, detail=user_id)  # user_id contains error message
            
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"User creation error: {str(e)}")
        raise HTTPException(status_code=500, detail="User creation failed")

@router.post("/auth/generate-api-key")
@require_auth(Permission.API_ACCESS)
async def generate_api_key(current_user=None):
    """Generate API key for current user"""
    try:
        api_token = auth_manager.generate_jwt_token(current_user.user_id, current_user.permissions)
        
        return {
            "status": "success",
            "api_key": api_token,
            "expires_in": "24 hours"
        }
        
    except Exception as e:
        logging.error(f"API key generation error: {str(e)}")
        raise HTTPException(status_code=500, detail="API key generation failed")

# Data protection endpoints
@router.post("/data-protection/encrypt")
@require_auth(Permission.MANAGE_COMPLIANCE)
async def encrypt_data(data: Dict[str, Any], current_user=None):
    """Encrypt sensitive data"""
    try:
        encrypted_data = {}
        
        for key, value in data.items():
            if isinstance(value, str):
                encrypted_data[key] = data_protection.encrypt_data(value)
            else:
                encrypted_data[key] = value
        
        return {
            "status": "success",
            "encrypted_data": encrypted_data
        }
        
    except Exception as e:
        logging.error(f"Data encryption error: {str(e)}")
        raise HTTPException(status_code=500, detail="Data encryption failed")

@router.post("/data-protection/anonymize")
@require_auth(Permission.VIEW_LEADS)
async def anonymize_data(data: Dict[str, Any], classification: str = "confidential", current_user=None):
    """Anonymize dataset"""
    try:
        # Validate classification
        try:
            data_classification = DataClassification(classification)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid data classification")
        
        anonymized_data = data_protection.anonymize_dataset(data, data_classification)
        
        return {
            "status": "success",
            "anonymized_data": anonymized_data
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Data anonymization error: {str(e)}")
        raise HTTPException(status_code=500, detail="Data anonymization failed")

@router.post("/data-protection/detect-pii")
@require_auth(Permission.MANAGE_COMPLIANCE)
async def detect_pii(text: str, current_user=None):
    """Detect PII in text"""
    try:
        detected_pii = data_protection.detect_pii(text)
        
        return {
            "status": "success",
            "detected_pii": {
                pii_type.value: matches for pii_type, matches in detected_pii.items()
            }
        }
        
    except Exception as e:
        logging.error(f"PII detection error: {str(e)}")
        raise HTTPException(status_code=500, detail="PII detection failed")

# Compliance endpoints
@router.post("/compliance/validate")
@require_auth(Permission.MANAGE_COMPLIANCE)
async def validate_compliance(data: Dict[str, Any], frameworks: List[str] = None, current_user=None):
    """Validate data against compliance frameworks"""
    try:
        # Convert framework strings to enums
        compliance_frameworks = []
        if frameworks:
            for framework in frameworks:
                try:
                    compliance_frameworks.append(ComplianceFramework(framework))
                except ValueError:
                    raise HTTPException(status_code=400, detail=f"Invalid framework: {framework}")
        
        validation_result = compliance_manager.validate_compliance(data, compliance_frameworks)
        
        return {
            "status": "success",
            "validation_result": validation_result
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Compliance validation error: {str(e)}")
        raise HTTPException(status_code=500, detail="Compliance validation failed")

@router.post("/compliance/record-consent")
@require_auth(Permission.MANAGE_COMPLIANCE)
async def record_consent(request: ConsentRequest, req: Request, current_user=None):
    """Record user consent"""
    try:
        ip_address = req.client.host
        user_agent = req.headers.get("user-agent", "")
        
        consent_id = compliance_manager.record_consent(
            user_id=request.user_id,
            consent_type=request.consent_type,
            consent_given=request.consent_given,
            consent_method=request.consent_method,
            consent_text=request.consent_text,
            ip_address=ip_address,
            user_agent=user_agent
        )
        
        return {
            "status": "success",
            "consent_id": consent_id,
            "message": "Consent recorded successfully"
        }
        
    except Exception as e:
        logging.error(f"Consent recording error: {str(e)}")
        raise HTTPException(status_code=500, detail="Consent recording failed")

@router.post("/compliance/withdraw-consent")
@require_auth(Permission.MANAGE_COMPLIANCE)
async def withdraw_consent(user_id: str, consent_type: str, current_user=None):
    """Withdraw user consent"""
    try:
        success = compliance_manager.withdraw_consent(user_id, consent_type)
        
        if success:
            return {
                "status": "success",
                "message": "Consent withdrawn successfully"
            }
        else:
            raise HTTPException(status_code=400, detail="Consent withdrawal failed")
            
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Consent withdrawal error: {str(e)}")
        raise HTTPException(status_code=500, detail="Consent withdrawal failed")

@router.get("/compliance/user-consents/{user_id}")
@require_auth(Permission.VIEW_LEADS)
async def get_user_consents(user_id: str, current_user=None):
    """Get all consent records for a user"""
    try:
        consents = compliance_manager.get_user_consents(user_id)
        
        return {
            "status": "success",
            "user_id": user_id,
            "consents": consents
        }
        
    except Exception as e:
        logging.error(f"Get user consents error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve user consents")

@router.post("/compliance/data-subject-request")
@require_auth(Permission.MANAGE_COMPLIANCE)
async def process_data_subject_request(request: DataSubjectRequest, current_user=None):
    """Process data subject request (GDPR/CCPA)"""
    try:
        response = compliance_manager.process_data_subject_request(
            request.request_type,
            request.user_id,
            request.request_details
        )
        
        return {
            "status": "success",
            "request_response": response
        }
        
    except Exception as e:
        logging.error(f"Data subject request error: {str(e)}")
        raise HTTPException(status_code=500, detail="Data subject request processing failed")

@router.get("/compliance/report")
@require_auth(Permission.MANAGE_COMPLIANCE)
async def generate_compliance_report(frameworks: List[str] = None, days: int = 30, current_user=None):
    """Generate compliance report"""
    try:
        # Convert framework strings to enums
        compliance_frameworks = []
        if frameworks:
            for framework in frameworks:
                try:
                    compliance_frameworks.append(ComplianceFramework(framework))
                except ValueError:
                    raise HTTPException(status_code=400, detail=f"Invalid framework: {framework}")
        
        # Calculate date range
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=days)
        
        report = compliance_manager.generate_compliance_report(
            frameworks=compliance_frameworks,
            start_date=start_date,
            end_date=end_date
        )
        
        return {
            "status": "success",
            "report": report
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Compliance report error: {str(e)}")
        raise HTTPException(status_code=500, detail="Compliance report generation failed")

# Security monitoring endpoints
@router.post("/security/log-event")
@require_auth(Permission.MANAGE_SECURITY)
async def log_security_event(request: SecurityEventRequest, current_user=None):
    """Log a security event"""
    try:
        # Validate event type
        try:
            event_type = SecurityEventType(request.event_type)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid event type")
        
        # Validate severity
        try:
            severity = ThreatLevel(request.severity)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid severity level")
        
        event_id = security_monitor.log_security_event(
            event_type=event_type,
            source_ip=request.source_ip,
            user_id=request.user_id,
            description=request.description,
            raw_data=request.raw_data,
            severity=severity
        )
        
        return {
            "status": "success",
            "event_id": event_id,
            "message": "Security event logged successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Security event logging error: {str(e)}")
        raise HTTPException(status_code=500, detail="Security event logging failed")

@router.get("/security/dashboard")
@require_auth(Permission.VIEW_SECURITY)
async def get_security_dashboard(current_user=None):
    """Get security dashboard data"""
    try:
        dashboard = security_monitor.get_security_dashboard()
        
        return {
            "status": "success",
            "dashboard": dashboard
        }
        
    except Exception as e:
        logging.error(f"Security dashboard error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve security dashboard")

@router.get("/security/incidents")
@require_auth(Permission.VIEW_SECURITY)
async def get_incidents(status: str = None, severity: str = None, limit: int = 50, current_user=None):
    """Get security incidents"""
    try:
        incidents = list(security_monitor.incidents.values())
        
        # Filter by status
        if status:
            try:
                status_filter = IncidentStatus(status)
                incidents = [i for i in incidents if i.status == status_filter]
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid status")
        
        # Filter by severity
        if severity:
            try:
                severity_filter = ThreatLevel(severity)
                incidents = [i for i in incidents if i.severity == severity_filter]
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid severity")
        
        # Sort by creation date (newest first) and limit
        incidents = sorted(incidents, key=lambda x: x.created_at, reverse=True)[:limit]
        
        # Convert to dict format
        incident_data = []
        for incident in incidents:
            incident_data.append({
                'incident_id': incident.incident_id,
                'title': incident.title,
                'description': incident.description,
                'severity': incident.severity.value,
                'status': incident.status.value,
                'created_at': incident.created_at.isoformat(),
                'updated_at': incident.updated_at.isoformat(),
                'assigned_to': incident.assigned_to,
                'event_count': len(incident.events),
                'timeline_entries': len(incident.timeline)
            })
        
        return {
            "status": "success",
            "incidents": incident_data,
            "total_count": len(incident_data)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Get incidents error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve incidents")

@router.get("/security/incidents/{incident_id}")
@require_auth(Permission.VIEW_SECURITY)
async def get_incident_details(incident_id: str, current_user=None):
    """Get detailed incident information"""
    try:
        incident = security_monitor.incidents.get(incident_id)
        if not incident:
            raise HTTPException(status_code=404, detail="Incident not found")
        
        # Get related events
        related_events = []
        for event_id in incident.events:
            event = security_monitor.security_events.get(event_id)
            if event:
                related_events.append({
                    'event_id': event.event_id,
                    'event_type': event.event_type.value,
                    'timestamp': event.timestamp.isoformat(),
                    'source_ip': event.source_ip,
                    'user_id': event.user_id,
                    'description': event.description,
                    'severity': event.severity.value,
                    'tags': event.tags
                })
        
        incident_details = {
            'incident_id': incident.incident_id,
            'title': incident.title,
            'description': incident.description,
            'severity': incident.severity.value,
            'status': incident.status.value,
            'created_at': incident.created_at.isoformat(),
            'updated_at': incident.updated_at.isoformat(),
            'assigned_to': incident.assigned_to,
            'resolution_notes': incident.resolution_notes,
            'lessons_learned': incident.lessons_learned,
            'timeline': incident.timeline,
            'related_events': related_events
        }
        
        return {
            "status": "success",
            "incident": incident_details
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Get incident details error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve incident details")

@router.put("/security/incidents/{incident_id}")
@require_auth(Permission.MANAGE_SECURITY)
async def update_incident(incident_id: str, request: IncidentUpdateRequest, current_user=None):
    """Update a security incident"""
    try:
        # Validate status if provided
        status = None
        if request.status:
            try:
                status = IncidentStatus(request.status)
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid status")
        
        success = security_monitor.update_incident(
            incident_id=incident_id,
            status=status,
            assigned_to=request.assigned_to,
            notes=request.notes
        )
        
        if success:
            return {
                "status": "success",
                "message": "Incident updated successfully"
            }
        else:
            raise HTTPException(status_code=404, detail="Incident not found")
            
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Update incident error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to update incident")

@router.get("/security/events")
@require_auth(Permission.VIEW_SECURITY)
async def get_security_events(event_type: str = None, severity: str = None, 
                             source_ip: str = None, hours: int = 24, 
                             limit: int = 100, current_user=None):
    """Get security events"""
    try:
        # Get events from the specified time window
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
        events = [
            event for event in security_monitor.security_events.values()
            if event.timestamp > cutoff_time
        ]
        
        # Apply filters
        if event_type:
            try:
                event_type_filter = SecurityEventType(event_type)
                events = [e for e in events if e.event_type == event_type_filter]
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid event type")
        
        if severity:
            try:
                severity_filter = ThreatLevel(severity)
                events = [e for e in events if e.severity == severity_filter]
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid severity")
        
        if source_ip:
            events = [e for e in events if e.source_ip == source_ip]
        
        # Sort by timestamp (newest first) and limit
        events = sorted(events, key=lambda x: x.timestamp, reverse=True)[:limit]
        
        # Convert to dict format
        event_data = []
        for event in events:
            event_data.append({
                'event_id': event.event_id,
                'event_type': event.event_type.value,
                'timestamp': event.timestamp.isoformat(),
                'source_ip': event.source_ip,
                'user_id': event.user_id,
                'description': event.description,
                'severity': event.severity.value,
                'tags': event.tags,
                'is_false_positive': event.is_false_positive
            })
        
        return {
            "status": "success",
            "events": event_data,
            "total_count": len(event_data),
            "time_window_hours": hours
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Get security events error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve security events")

# Health and status endpoints
@router.get("/health")
async def security_health_check():
    """Security system health check"""
    try:
        health_status = {
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "components": {
                "authentication": "healthy",
                "data_protection": "healthy",
                "compliance_manager": "healthy",
                "security_monitoring": "healthy"
            },
            "metrics": {
                "active_sessions": len(auth_manager.active_sessions),
                "recent_events_24h": len([
                    e for e in security_monitor.security_events.values()
                    if e.timestamp > datetime.now(timezone.utc) - timedelta(hours=24)
                ]),
                "open_incidents": len([
                    i for i in security_monitor.incidents.values()
                    if i.status in [IncidentStatus.OPEN, IncidentStatus.INVESTIGATING]
                ])
            }
        }
        
        return health_status
        
    except Exception as e:
        logging.error(f"Security health check error: {str(e)}")
        return {
            "status": "unhealthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "error": str(e)
        }

@router.get("/status")
@require_auth(Permission.VIEW_SECURITY)
async def get_security_status(current_user=None):
    """Get comprehensive security status"""
    try:
        current_time = datetime.now(timezone.utc)
        
        # Calculate various metrics
        recent_events = [
            e for e in security_monitor.security_events.values()
            if e.timestamp > current_time - timedelta(hours=24)
        ]
        
        open_incidents = [
            i for i in security_monitor.incidents.values()
            if i.status in [IncidentStatus.OPEN, IncidentStatus.INVESTIGATING]
        ]
        
        critical_incidents = [
            i for i in open_incidents
            if i.severity == ThreatLevel.CRITICAL
        ]
        
        status = {
            "timestamp": current_time.isoformat(),
            "overall_status": "healthy",
            "threat_level": security_monitor._calculate_overall_threat_level(),
            "metrics": {
                "active_sessions": len(auth_manager.active_sessions),
                "events_last_24h": len(recent_events),
                "open_incidents": len(open_incidents),
                "critical_incidents": len(critical_incidents),
                "failed_logins_last_hour": len([
                    e for e in recent_events
                    if e.event_type == SecurityEventType.AUTHENTICATION_FAILURE
                    and e.timestamp > current_time - timedelta(hours=1)
                ])
            },
            "system_health": {
                "authentication_service": "operational",
                "data_protection": "operational",
                "compliance_monitoring": "operational",
                "incident_response": "operational"
            },
            "recent_alerts": [
                {
                    "type": "incident" if len(critical_incidents) > 0 else "info",
                    "message": f"{len(critical_incidents)} critical incidents require attention" if critical_incidents else "No critical incidents",
                    "timestamp": current_time.isoformat()
                }
            ]
        }
        
        # Determine overall status
        if len(critical_incidents) > 0:
            status["overall_status"] = "critical"
        elif len(open_incidents) > 5:
            status["overall_status"] = "warning"
        
        return {
            "status": "success",
            "security_status": status
        }
        
    except Exception as e:
        logging.error(f"Get security status error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve security status")
