"""
Analytics API for Insurance Lead Scoring Platform

Provides REST API endpoints for advanced analytics, reporting, and business intelligence
with real-time data access and automated report generation.
"""

from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from fastapi.responses import HTMLResponse, FileResponse
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from pydantic import BaseModel
import logging

from backend.analytics.analytics_engine import analytics_engine, AnalyticsQuery, AnalyticsType, MetricType, TimeGranularity
from backend.analytics.reporting_framework import reporting_framework, ReportConfiguration, ReportType, ReportFormat
from backend.analytics.business_intelligence import bi_dashboard

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/analytics", tags=["analytics"])

class AnalyticsQueryRequest(BaseModel):
    query_type: str
    metrics: List[str]
    start_date: str
    end_date: str
    granularity: str = "daily"
    filters: Dict[str, Any] = {}
    segments: List[str] = []

class ReportGenerationRequest(BaseModel):
    report_type: str
    title: str
    description: str
    format: str = "html"
    data_sources: List[str] = []
    filters: Dict[str, Any] = {}
    delivery_methods: List[str] = []
    recipients: List[str] = []

@router.post("/query")
async def execute_analytics_query(request: AnalyticsQueryRequest):
    """Execute advanced analytics query"""
    try:
        # Parse request
        query = AnalyticsQuery(
            query_id=f"query_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            query_type=AnalyticsType(request.query_type),
            metrics=[MetricType(m) for m in request.metrics],
            time_range=(
                datetime.fromisoformat(request.start_date),
                datetime.fromisoformat(request.end_date)
            ),
            granularity=TimeGranularity(request.granularity),
            filters=request.filters,
            segments=request.segments
        )
        
        # Execute query
        result = await analytics_engine.execute_query(query)
        
        return {
            "status": "success",
            "query_id": result.query_id,
            "data": result.data,
            "insights": result.insights,
            "confidence_score": result.confidence_score,
            "generated_at": result.generated_at.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error executing analytics query: {e}")
        raise HTTPException(status_code=500, detail=f"Error executing query: {str(e)}")

@router.get("/kpis")
async def get_kpi_dashboard():
    """Get business intelligence KPI dashboard"""
    try:
        dashboard_data = await bi_dashboard.generate_executive_dashboard()
        
        return {
            "status": "success",
            "dashboard": dashboard_data
        }
        
    except Exception as e:
        logger.error(f"Error getting KPI dashboard: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting dashboard: {str(e)}")

@router.post("/kpis/update")
async def update_kpis(background_tasks: BackgroundTasks):
    """Update all KPI values"""
    try:
        # Run update in background
        background_tasks.add_task(bi_dashboard.update_kpi_values)
        
        return {
            "status": "success",
            "message": "KPI update initiated"
        }
        
    except Exception as e:
        logger.error(f"Error updating KPIs: {e}")
        raise HTTPException(status_code=500, detail=f"Error updating KPIs: {str(e)}")

@router.get("/alerts")
async def get_business_alerts(
    severity: Optional[str] = Query(None),
    acknowledged: Optional[bool] = Query(None)
):
    """Get business alerts"""
    try:
        alerts = list(bi_dashboard.alerts.values())
        
        # Filter by severity
        if severity:
            alerts = [a for a in alerts if a.severity.value == severity]
        
        # Filter by acknowledgment status
        if acknowledged is not None:
            alerts = [a for a in alerts if a.acknowledged == acknowledged]
        
        # Sort by creation time (newest first)
        alerts.sort(key=lambda x: x.created_at, reverse=True)
        
        return {
            "status": "success",
            "alerts": [
                {
                    "alert_id": alert.alert_id,
                    "kpi_id": alert.kpi_id,
                    "severity": alert.severity.value,
                    "message": alert.message,
                    "current_value": alert.current_value,
                    "threshold_value": alert.threshold_value,
                    "created_at": alert.created_at.isoformat(),
                    "acknowledged": alert.acknowledged
                }
                for alert in alerts
            ],
            "total_count": len(alerts)
        }
        
    except Exception as e:
        logger.error(f"Error getting alerts: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting alerts: {str(e)}")

@router.post("/reports/generate")
async def generate_report(request: ReportGenerationRequest, background_tasks: BackgroundTasks):
    """Generate a new report"""
    try:
        # Create report configuration
        config = ReportConfiguration(
            report_id=f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            report_type=ReportType(request.report_type),
            title=request.title,
            description=request.description,
            format=ReportFormat(request.format),
            data_sources=request.data_sources,
            filters=request.filters,
            delivery_methods=[],  # Will be set up separately
            recipients=request.recipients
        )
        
        # Create configuration
        config_id = await reporting_framework.create_report_config(config)
        
        # Generate report in background
        background_tasks.add_task(reporting_framework.generate_report, config_id)
        
        return {
            "status": "success",
            "report_id": config_id,
            "message": "Report generation initiated"
        }
        
    except Exception as e:
        logger.error(f"Error generating report: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating report: {str(e)}")

@router.get("/reports/{report_id}")
async def get_report(report_id: str):
    """Get generated report"""
    try:
        # Find the generated report
        generated_report = None
        for report in reporting_framework.generated_reports.values():
            if report.config.report_id == report_id:
                generated_report = report
                break
        
        if not generated_report:
            raise HTTPException(status_code=404, detail="Report not found")
        
        return {
            "status": "success",
            "report": {
                "report_id": generated_report.report_id,
                "title": generated_report.config.title,
                "description": generated_report.config.description,
                "format": generated_report.config.format.value,
                "content": generated_report.content,
                "metadata": generated_report.metadata,
                "generated_at": generated_report.generated_at.isoformat()
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting report: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting report: {str(e)}")

@router.get("/reports/{report_id}/download")
async def download_report(report_id: str):
    """Download report file"""
    try:
        # Find the generated report
        generated_report = None
        for report in reporting_framework.generated_reports.values():
            if report.config.report_id == report_id:
                generated_report = report
                break
        
        if not generated_report:
            raise HTTPException(status_code=404, detail="Report not found")
        
        # Return HTML content directly for now
        if generated_report.config.format == ReportFormat.HTML:
            return HTMLResponse(content=generated_report.content)
        else:
            return {
                "status": "success",
                "content": generated_report.content
            }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading report: {e}")
        raise HTTPException(status_code=500, detail=f"Error downloading report: {str(e)}")

@router.get("/insights/lead-performance")
async def get_lead_performance_insights(
    days: int = Query(30, description="Number of days to analyze"),
    insurance_type: Optional[str] = Query(None)
):
    """Get lead performance insights"""
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Create analytics query
        query = AnalyticsQuery(
            query_id=f"lead_insights_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            query_type=AnalyticsType.DIAGNOSTIC,
            metrics=[MetricType.CONVERSION_RATE, MetricType.REVENUE_PER_LEAD, MetricType.LEAD_QUALITY_SCORE],
            time_range=(start_date, end_date),
            granularity=TimeGranularity.DAILY,
            filters={"insurance_type": insurance_type} if insurance_type else {}
        )
        
        # Execute query
        result = await analytics_engine.execute_query(query)
        
        return {
            "status": "success",
            "insights": result.insights,
            "data": result.data,
            "analysis_period": f"{days} days",
            "confidence_score": result.confidence_score
        }
        
    except Exception as e:
        logger.error(f"Error getting lead performance insights: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting insights: {str(e)}")

@router.get("/insights/revenue-forecast")
async def get_revenue_forecast(
    forecast_days: int = Query(30, description="Number of days to forecast")
):
    """Get revenue forecasting insights"""
    try:
        # Create predictive analytics query
        query = AnalyticsQuery(
            query_id=f"revenue_forecast_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            query_type=AnalyticsType.PREDICTIVE,
            metrics=[MetricType.REVENUE_PER_LEAD, MetricType.CONVERSION_RATE],
            time_range=(datetime.now() - timedelta(days=90), datetime.now()),
            granularity=TimeGranularity.DAILY
        )
        
        # Execute query
        result = await analytics_engine.execute_query(query)
        
        return {
            "status": "success",
            "forecast": result.data.get('revenue_forecast', {}),
            "insights": result.insights,
            "forecast_period": f"{forecast_days} days",
            "confidence_score": result.confidence_score
        }
        
    except Exception as e:
        logger.error(f"Error getting revenue forecast: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting forecast: {str(e)}")

@router.get("/metrics/summary")
async def get_metrics_summary():
    """Get summary of all available metrics"""
    try:
        metrics_info = {}
        
        for metric_type in MetricType:
            metrics_info[metric_type.value] = {
                "name": metric_type.value.replace('_', ' ').title(),
                "description": f"Analytics for {metric_type.value.replace('_', ' ')}",
                "available": True
            }
        
        analytics_types = {
            analytics_type.value: {
                "name": analytics_type.value.title(),
                "description": f"{analytics_type.value.title()} analytics capabilities"
            }
            for analytics_type in AnalyticsType
        }
        
        return {
            "status": "success",
            "available_metrics": metrics_info,
            "analytics_types": analytics_types,
            "time_granularities": [g.value for g in TimeGranularity]
        }
        
    except Exception as e:
        logger.error(f"Error getting metrics summary: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting summary: {str(e)}")
