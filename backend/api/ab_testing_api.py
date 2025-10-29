from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import logging
from datetime import datetime
from backend.models.ab_testing.framework import ABTestManager, TestType, MetricType, TestStatus
from backend.models.ab_testing.analytics import ABTestAnalytics
from backend.models.ab_testing.reporting import ABTestReporter

router = APIRouter(prefix="/ab-testing", tags=["A/B Testing"])
logger = logging.getLogger(__name__)

# Initialize A/B testing components
ab_test_manager = ABTestManager()
ab_analytics = ABTestAnalytics()
ab_reporter = ABTestReporter()

class CreateTestRequest(BaseModel):
    name: str
    description: str
    test_type: str
    variants: List[Dict[str, Any]]
    target_sample_size: int = 1000
    confidence_level: float = 0.95
    minimum_detectable_effect: float = 0.05
    max_duration_days: int = 30
    primary_metric: str = "CONVERSION_RATE"
    secondary_metrics: List[str] = ["OPEN_RATE", "CLICK_RATE"]

class AssignVariantRequest(BaseModel):
    test_id: str
    lead_id: str
    lead_data: Optional[Dict[str, Any]] = None

class TrackEventRequest(BaseModel):
    test_id: str
    lead_id: str
    event_type: str
    value: float = 1.0
    metadata: Optional[Dict[str, Any]] = None

class TestAnalysisRequest(BaseModel):
    test_id: str
    include_segments: bool = False
    include_trends: bool = False

@router.post("/create-test")
async def create_ab_test(request: CreateTestRequest):
    """Create a new A/B test"""
    try:
        # Validate test type
        try:
            test_type = TestType(request.test_type.upper())
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid test type: {request.test_type}")
        
        # Validate metrics
        try:
            primary_metric = MetricType(request.primary_metric.upper())
            secondary_metrics = [MetricType(m.upper()) for m in request.secondary_metrics]
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Invalid metric: {str(e)}")
        
        # Create test based on type
        if test_type == TestType.MESSAGE_CONTENT:
            test_id = ab_test_manager.create_message_test(
                request.name,
                request.variants,
                request.target_sample_size
            )
        elif test_type == TestType.SEND_TIME:
            send_times = [v.get('send_time') for v in request.variants]
            test_id = ab_test_manager.create_timing_test(
                request.name,
                send_times,
                request.target_sample_size
            )
        else:
            # Generic test creation
            test_id = ab_test_manager.framework.create_test_from_config(
                request.dict()
            )
        
        return {
            "status": "success",
            "test_id": test_id,
            "message": f"A/B test '{request.name}' created successfully",
            "next_steps": [
                "Review test configuration",
                "Start test when ready",
                "Monitor test progress"
            ]
        }
        
    except Exception as e:
        logger.error(f"Error creating A/B test: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error creating test: {str(e)}")

@router.post("/start-test/{test_id}")
async def start_test(test_id: str):
    """Start an A/B test"""
    try:
        success = ab_test_manager.framework.start_test(test_id)
        
        if not success:
            raise HTTPException(status_code=400, detail="Failed to start test")
        
        return {
            "status": "success",
            "test_id": test_id,
            "message": "A/B test started successfully",
            "started_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error starting test {test_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error starting test: {str(e)}")

@router.post("/assign-variant")
async def assign_variant(request: AssignVariantRequest):
    """Assign a lead to a test variant"""
    try:
        variant_id = ab_test_manager.framework.assign_variant(
            request.test_id,
            request.lead_id,
            request.lead_data
        )
        
        if not variant_id:
            raise HTTPException(status_code=404, detail="Test not found or not active")
        
        # Get variant configuration
        test_status = ab_test_manager.framework.get_test_status(request.test_id)
        variant_config = None
        
        if test_status:
            for variant_id_key, stats in test_status['variant_stats'].items():
                if variant_id_key == variant_id:
                    variant_config = stats
                    break
        
        return {
            "status": "success",
            "test_id": request.test_id,
            "lead_id": request.lead_id,
            "assigned_variant": variant_id,
            "variant_config": variant_config,
            "assignment_timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error assigning variant: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error assigning variant: {str(e)}")

@router.post("/track-event")
async def track_event(request: TrackEventRequest):
    """Track an event for A/B test analysis"""
    try:
        success = ab_test_manager.framework.track_event(
            request.test_id,
            request.lead_id,
            request.event_type,
            request.value,
            request.metadata
        )
        
        if not success:
            raise HTTPException(status_code=400, detail="Failed to track event")
        
        return {
            "status": "success",
            "test_id": request.test_id,
            "lead_id": request.lead_id,
            "event_type": request.event_type,
            "value": request.value,
            "tracked_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error tracking event: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error tracking event: {str(e)}")

@router.get("/test-status/{test_id}")
async def get_test_status(test_id: str):
    """Get current test status and real-time metrics"""
    try:
        status = ab_test_manager.framework.get_test_status(test_id)
        
        if not status:
            raise HTTPException(status_code=404, detail="Test not found")
        
        # Add real-time analytics
        analytics = ab_analytics.get_real_time_metrics(test_id)
        
        return {
            "status": "success",
            "test_status": status,
            "real_time_analytics": analytics,
            "last_updated": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting test status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting test status: {str(e)}")

@router.post("/analyze-test")
async def analyze_test(request: TestAnalysisRequest):
    """Analyze A/B test results"""
    try:
        analysis = ab_test_manager.framework.analyze_test(request.test_id)
        
        if not analysis:
            raise HTTPException(status_code=404, detail="Test not found")
        
        # Enhanced analytics
        enhanced_analysis = ab_analytics.enhance_analysis(
            analysis,
            include_segments=request.include_segments,
            include_trends=request.include_trends
        )
        
        return {
            "status": "success",
            "analysis": enhanced_analysis,
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error analyzing test: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error analyzing test: {str(e)}")

@router.post("/stop-test/{test_id}")
async def stop_test(test_id: str, reason: str = "Manual stop"):
    """Stop an active A/B test"""
    try:
        success = ab_test_manager.framework.stop_test(test_id, reason)
        
        if not success:
            raise HTTPException(status_code=400, detail="Failed to stop test")
        
        # Generate final analysis
        final_analysis = ab_test_manager.framework.analyze_test(test_id)
        
        return {
            "status": "success",
            "test_id": test_id,
            "message": "A/B test stopped successfully",
            "stop_reason": reason,
            "stopped_at": datetime.now().isoformat(),
            "final_analysis": final_analysis
        }
        
    except Exception as e:
        logger.error(f"Error stopping test: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error stopping test: {str(e)}")

@router.get("/active-tests")
async def get_active_tests():
    """Get all active A/B tests"""
    try:
        active_tests = ab_test_manager.framework.get_active_tests()
        
        # Add summary metrics for each test
        enhanced_tests = []
        for test in active_tests:
            test_status = ab_test_manager.framework.get_test_status(test['test_id'])
            if test_status:
                test['summary_metrics'] = {
                    'total_participants': test_status['total_participants'],
                    'progress': test_status['progress'],
                    'duration_days': test_status['duration_days']
                }
            enhanced_tests.append(test)
        
        return {
            "status": "success",
            "active_tests": enhanced_tests,
            "total_active": len(enhanced_tests)
        }
        
    except Exception as e:
        logger.error(f"Error getting active tests: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting active tests: {str(e)}")

@router.get("/test-report/{test_id}")
async def generate_test_report(test_id: str, format: str = "json"):
    """Generate comprehensive test report"""
    try:
        report = ab_reporter.generate_comprehensive_report(test_id, format)
        
        if not report:
            raise HTTPException(status_code=404, detail="Test not found")
        
        return {
            "status": "success",
            "report": report,
            "format": format,
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error generating report: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating report: {str(e)}")

@router.get("/dashboard-metrics")
async def get_dashboard_metrics():
    """Get A/B testing dashboard metrics"""
    try:
        metrics = ab_analytics.get_dashboard_metrics()
        
        return {
            "status": "success",
            "dashboard_metrics": metrics,
            "last_updated": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting dashboard metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting dashboard metrics: {str(e)}")

@router.post("/bulk-assign-variants")
async def bulk_assign_variants(test_id: str, leads: List[Dict[str, Any]]):
    """Bulk assign leads to test variants"""
    try:
        assignments = []
        
        for lead in leads:
            lead_id = lead.get('lead_id')
            if not lead_id:
                continue
            
            variant_id = ab_test_manager.framework.assign_variant(
                test_id, lead_id, lead
            )
            
            if variant_id:
                assignments.append({
                    'lead_id': lead_id,
                    'variant_id': variant_id
                })
        
        return {
            "status": "success",
            "test_id": test_id,
            "total_assignments": len(assignments),
            "assignments": assignments,
            "processed_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in bulk assignment: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in bulk assignment: {str(e)}")

@router.post("/bulk-track-events")
async def bulk_track_events(events: List[TrackEventRequest]):
    """Bulk track events for multiple leads"""
    try:
        tracked_events = []
        failed_events = []
        
        for event_request in events:
            success = ab_test_manager.framework.track_event(
                event_request.test_id,
                event_request.lead_id,
                event_request.event_type,
                event_request.value,
                event_request.metadata
            )
            
            if success:
                tracked_events.append({
                    'test_id': event_request.test_id,
                    'lead_id': event_request.lead_id,
                    'event_type': event_request.event_type
                })
            else:
                failed_events.append({
                    'test_id': event_request.test_id,
                    'lead_id': event_request.lead_id,
                    'error': 'Failed to track event'
                })
        
        return {
            "status": "success",
            "total_events": len(events),
            "tracked_successfully": len(tracked_events),
            "failed_events": len(failed_events),
            "tracked_events": tracked_events,
            "failed_events": failed_events,
            "processed_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in bulk event tracking: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in bulk event tracking: {str(e)}")

# Helper endpoints for test management
@router.get("/test-templates")
async def get_test_templates():
    """Get available A/B test templates"""
    templates = {
        "message_content": {
            "name": "Message Content Test",
            "description": "Test different message content variations",
            "recommended_sample_size": 1000,
            "primary_metric": "CONVERSION_RATE",
            "secondary_metrics": ["OPEN_RATE", "CLICK_RATE"],
            "typical_duration_days": 14
        },
        "subject_line": {
            "name": "Subject Line Test",
            "description": "Test different email subject lines",
            "recommended_sample_size": 500,
            "primary_metric": "OPEN_RATE",
            "secondary_metrics": ["CONVERSION_RATE"],
            "typical_duration_days": 7
        },
        "send_timing": {
            "name": "Send Timing Test",
            "description": "Test optimal send times",
            "recommended_sample_size": 800,
            "primary_metric": "OPEN_RATE",
            "secondary_metrics": ["CONVERSION_RATE", "CLICK_RATE"],
            "typical_duration_days": 21
        },
        "personalization": {
            "name": "Personalization Test",
            "description": "Test different personalization levels",
            "recommended_sample_size": 1200,
            "primary_metric": "CONVERSION_RATE",
            "secondary_metrics": ["RESPONSE_RATE"],
            "typical_duration_days": 14
        }
    }
    
    return {
        "status": "success",
        "templates": templates
    }

@router.get("/metrics-definitions")
async def get_metrics_definitions():
    """Get definitions of available metrics"""
    metrics = {
        "CONVERSION_RATE": {
            "name": "Conversion Rate",
            "description": "Percentage of leads that convert to customers",
            "calculation": "conversions / total_leads",
            "best_for": ["message_content", "personalization", "landing_page"]
        },
        "OPEN_RATE": {
            "name": "Email Open Rate",
            "description": "Percentage of emails that are opened",
            "calculation": "opens / emails_sent",
            "best_for": ["subject_line", "send_timing"]
        },
        "CLICK_RATE": {
            "name": "Click-Through Rate",
            "description": "Percentage of emails that receive clicks",
            "calculation": "clicks / emails_sent",
            "best_for": ["call_to_action", "message_content"]
        },
        "RESPONSE_RATE": {
            "name": "Response Rate",
            "description": "Percentage of leads that respond",
            "calculation": "responses / total_contacts",
            "best_for": ["message_content", "personalization"]
        },
        "REVENUE_PER_LEAD": {
            "name": "Revenue Per Lead",
            "description": "Average revenue generated per lead",
            "calculation": "total_revenue / total_leads",
            "best_for": ["lead_scoring", "targeting"]
        }
    }
    
    return {
        "status": "success",
        "metrics": metrics
    }
