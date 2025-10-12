from fastapi import APIRouter, HTTPException, BackgroundTasks, Query
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import logging
from datetime import datetime, timedelta
from monitoring.performance_monitor import performance_monitor, MonitoredOperation
from monitoring.alerting import AlertManager
from monitoring.reporting import PerformanceReporter

router = APIRouter(prefix="/monitoring", tags=["Performance Monitoring"])
logger = logging.getLogger(__name__)

# Initialize monitoring components
alert_manager = AlertManager()
performance_reporter = PerformanceReporter()

class MetricRequest(BaseModel):
    component: str
    metric_name: str
    metric_type: str
    value: float
    metadata: Optional[Dict[str, Any]] = None
    tags: Optional[Dict[str, str]] = None

class AlertRequest(BaseModel):
    component: str
    metric_name: str
    threshold_value: float
    severity: str = "WARNING"
    comparison: str = "greater_than"  # greater_than, less_than, equals

@router.get("/dashboard")
async def get_performance_dashboard():
    """Get comprehensive performance dashboard"""
    try:
        with MonitoredOperation("api", "dashboard_generation"):
            dashboard = performance_monitor.get_performance_dashboard()
            
            return {
                "status": "success",
                "dashboard": dashboard,
                "generated_at": datetime.now().isoformat()
            }
            
    except Exception as e:
        logger.error(f"Error generating dashboard: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating dashboard: {str(e)}")

@router.get("/metrics/summary")
async def get_metrics_summary(
    time_window: str = Query("1h", description="Time window: 1m, 5m, 15m, 1h"),
    component: Optional[str] = Query(None, description="Filter by component")
):
    """Get aggregated metrics summary"""
    try:
        summary = performance_monitor.get_metrics_summary(time_window)
        
        # Filter by component if specified
        if component and "components" in summary:
            if component in summary["components"]:
                summary["components"] = {component: summary["components"][component]}
            else:
                summary["components"] = {}
        
        return {
            "status": "success",
            "summary": summary
        }
        
    except Exception as e:
        logger.error(f"Error getting metrics summary: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting metrics summary: {str(e)}")

@router.get("/health")
async def get_system_health():
    """Get overall system health status"""
    try:
        # Check health of all components
        components = ["system", "api.scoring", "model.insurance_scoring", "model.healthcare_scoring"]
        health_results = {}
        
        for component in components:
            health_check = performance_monitor.check_component_health(component)
            health_results[component] = {
                "status": health_check.status,
                "response_time": health_check.response_time,
                "checks": health_check.checks,
                "details": health_check.details
            }
        
        # Determine overall health
        statuses = [result["status"] for result in health_results.values()]
        if "unhealthy" in statuses:
            overall_status = "unhealthy"
        elif "degraded" in statuses:
            overall_status = "degraded"
        else:
            overall_status = "healthy"
        
        return {
            "status": "success",
            "overall_health": overall_status,
            "components": health_results,
            "checked_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error checking system health: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error checking system health: {str(e)}")

@router.get("/alerts")
async def get_alerts(
    active_only: bool = Query(True, description="Show only active alerts"),
    severity: Optional[str] = Query(None, description="Filter by severity"),
    component: Optional[str] = Query(None, description="Filter by component")
):
    """Get system alerts"""
    try:
        if active_only:
            alerts = performance_monitor.get_active_alerts()
        else:
            alerts = list(performance_monitor.alert_history)
        
        # Apply filters
        if severity:
            alerts = [a for a in alerts if a.severity.value == severity.upper()]
        
        if component:
            alerts = [a for a in alerts if a.component == component]
        
        # Convert to dict format
        alert_data = []
        for alert in alerts:
            alert_data.append({
                "alert_id": alert.alert_id,
                "severity": alert.severity.value,
                "component": alert.component,
                "metric_name": alert.metric_name,
                "message": alert.message,
                "value": alert.value,
                "threshold": alert.threshold,
                "timestamp": alert.timestamp.isoformat(),
                "resolved": alert.resolved,
                "resolution_time": alert.resolution_time.isoformat() if alert.resolution_time else None
            })
        
        return {
            "status": "success",
            "alerts": alert_data,
            "total_alerts": len(alert_data),
            "filters_applied": {
                "active_only": active_only,
                "severity": severity,
                "component": component
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting alerts: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting alerts: {str(e)}")

@router.post("/alerts/{alert_id}/resolve")
async def resolve_alert(alert_id: str, resolution_note: str = ""):
    """Resolve an active alert"""
    try:
        if alert_id in performance_monitor.alerts:
            alert = performance_monitor.alerts[alert_id]
            alert.resolved = True
            alert.resolution_time = datetime.now()
            
            logger.info(f"Alert {alert_id} resolved: {resolution_note}")
            
            return {
                "status": "success",
                "message": f"Alert {alert_id} resolved successfully",
                "resolution_time": alert.resolution_time.isoformat()
            }
        else:
            raise HTTPException(status_code=404, detail="Alert not found")
            
    except Exception as e:
        logger.error(f"Error resolving alert: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error resolving alert: {str(e)}")

@router.get("/metrics/anomalies")
async def detect_anomalies(
    component: str = Query(..., description="Component to analyze"),
    metric_name: str = Query(..., description="Metric to analyze"),
    window_size: int = Query(100, description="Number of recent data points to analyze")
):
    """Detect anomalies in metrics"""
    try:
        anomalies = performance_monitor.detect_anomalies(component, metric_name, window_size)
        
        return {
            "status": "success",
            "component": component,
            "metric_name": metric_name,
            "anomalies_detected": len(anomalies),
            "anomalies": anomalies,
            "analysis_window": window_size
        }
        
    except Exception as e:
        logger.error(f"Error detecting anomalies: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error detecting anomalies: {str(e)}")

@router.post("/metrics/record")
async def record_custom_metric(request: MetricRequest):
    """Record a custom metric"""
    try:
        from monitoring.performance_monitor import PerformanceMetric, MetricType
        
        metric = PerformanceMetric(
            metric_name=request.metric_name,
            metric_type=MetricType(request.metric_type.upper()),
            value=request.value,
            timestamp=datetime.now(),
            component=request.component,
            metadata=request.metadata or {},
            tags=request.tags or {}
        )
        
        performance_monitor.record_metric(metric)
        
        return {
            "status": "success",
            "message": "Metric recorded successfully",
            "metric": {
                "component": request.component,
                "metric_name": request.metric_name,
                "value": request.value,
                "timestamp": metric.timestamp.isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"Error recording metric: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error recording metric: {str(e)}")

@router.get("/performance/api")
async def get_api_performance():
    """Get detailed API performance metrics"""
    try:
        # Get API-specific metrics
        api_summary = performance_monitor._get_api_performance_summary()
        
        # Get recent API metrics for trending
        recent_metrics = [
            m for m in performance_monitor.metrics_buffer 
            if m.component.startswith("api.") and 
               m.timestamp > datetime.now() - timedelta(hours=1)
        ]
        
        # Calculate trends
        trends = {}
        for metric in recent_metrics:
            endpoint = metric.component.split(".", 1)[1]
            if endpoint not in trends:
                trends[endpoint] = {"response_times": [], "error_rates": []}
            
            if metric.metric_name == "response_time":
                trends[endpoint]["response_times"].append({
                    "timestamp": metric.timestamp.isoformat(),
                    "value": metric.value
                })
            elif metric.metric_name == "error_rate":
                trends[endpoint]["error_rates"].append({
                    "timestamp": metric.timestamp.isoformat(),
                    "value": metric.value
                })
        
        return {
            "status": "success",
            "api_performance": api_summary,
            "trends": trends,
            "analysis_period": "1 hour"
        }
        
    except Exception as e:
        logger.error(f"Error getting API performance: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting API performance: {str(e)}")

@router.get("/performance/models")
async def get_model_performance():
    """Get detailed model performance metrics"""
    try:
        model_summary = performance_monitor._get_model_performance_summary()
        
        # Get model-specific insights
        model_insights = {}
        for model_name in model_summary.keys():
            # Check for performance degradation
            recent_accuracy = [
                m for m in performance_monitor.metrics_buffer 
                if (m.component == f"model.{model_name}" and 
                    "accuracy" in m.metric_name and
                    m.timestamp > datetime.now() - timedelta(hours=6))
            ]
            
            insights = {
                "performance_trend": "stable",
                "accuracy_samples": len(recent_accuracy),
                "recommendations": []
            }
            
            if recent_accuracy:
                accuracy_values = [m.value for m in recent_accuracy]
                if len(accuracy_values) > 1:
                    trend = "improving" if accuracy_values[-1] > accuracy_values[0] else "declining"
                    insights["performance_trend"] = trend
                
                avg_accuracy = sum(accuracy_values) / len(accuracy_values)
                if avg_accuracy < 0.8:
                    insights["recommendations"].append("Model accuracy below 80% - consider retraining")
                
                if model_summary[model_name].get("avg_prediction_time", 0) > 0.1:
                    insights["recommendations"].append("Prediction latency high - consider model optimization")
            
            model_insights[model_name] = insights
        
        return {
            "status": "success",
            "model_performance": model_summary,
            "model_insights": model_insights,
            "analysis_period": "6 hours"
        }
        
    except Exception as e:
        logger.error(f"Error getting model performance: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting model performance: {str(e)}")

@router.get("/reports/performance")
async def generate_performance_report(
    start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
    format: str = Query("json", description="Report format: json, csv, pdf")
):
    """Generate comprehensive performance report"""
    try:
        # Parse dates
        if start_date:
            start_dt = datetime.fromisoformat(start_date)
        else:
            start_dt = datetime.now() - timedelta(days=7)
        
        if end_date:
            end_dt = datetime.fromisoformat(end_date)
        else:
            end_dt = datetime.now()
        
        # Generate report
        report = performance_reporter.generate_performance_report(
            start_date=start_dt,
            end_date=end_dt,
            format=format
        )
        
        return {
            "status": "success",
            "report": report,
            "period": {
                "start_date": start_dt.isoformat(),
                "end_date": end_dt.isoformat()
            },
            "format": format
        }
        
    except Exception as e:
        logger.error(f"Error generating performance report: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating performance report: {str(e)}")

@router.post("/monitoring/start")
async def start_monitoring():
    """Start performance monitoring"""
    try:
        if not performance_monitor.monitoring_active:
            performance_monitor.start_monitoring()
            message = "Performance monitoring started successfully"
        else:
            message = "Performance monitoring is already active"
        
        return {
            "status": "success",
            "message": message,
            "monitoring_active": performance_monitor.monitoring_active
        }
        
    except Exception as e:
        logger.error(f"Error starting monitoring: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error starting monitoring: {str(e)}")

@router.post("/monitoring/stop")
async def stop_monitoring():
    """Stop performance monitoring"""
    try:
        if performance_monitor.monitoring_active:
            performance_monitor.stop_monitoring()
            message = "Performance monitoring stopped successfully"
        else:
            message = "Performance monitoring is not active"
        
        return {
            "status": "success",
            "message": message,
            "monitoring_active": performance_monitor.monitoring_active
        }
        
    except Exception as e:
        logger.error(f"Error stopping monitoring: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error stopping monitoring: {str(e)}")

@router.get("/monitoring/status")
async def get_monitoring_status():
    """Get monitoring system status"""
    try:
        status = {
            "monitoring_active": performance_monitor.monitoring_active,
            "metrics_buffer_size": len(performance_monitor.metrics_buffer),
            "active_alerts": len(performance_monitor.get_active_alerts()),
            "components_monitored": len(performance_monitor.health_status),
            "uptime": "N/A",  # Would calculate actual uptime
            "last_health_check": max(
                [h.timestamp for h in performance_monitor.health_status.values()],
                default=datetime.now()
            ).isoformat()
        }
        
        return {
            "status": "success",
            "monitoring_status": status
        }
        
    except Exception as e:
        logger.error(f"Error getting monitoring status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting monitoring status: {str(e)}")

# Background task to record API metrics
@router.middleware("http")
async def monitor_api_requests(request, call_next):
    """Middleware to automatically monitor API requests"""
    start_time = time.time()
    
    try:
        response = await call_next(request)
        status_code = response.status_code
    except Exception as e:
        status_code = 500
        logger.error(f"API request failed: {e}")
        raise
    finally:
        end_time = time.time()
        response_time = end_time - start_time
        
        # Record API metrics
        endpoint = request.url.path
        method = request.method
        
        performance_monitor.record_api_call(
            endpoint=endpoint,
            method=method,
            status_code=status_code,
            response_time=response_time,
            request_size=int(request.headers.get("content-length", 0)),
            response_size=0  # Would need to calculate actual response size
        )
    
    return response