from fastapi import APIRouter, HTTPException, BackgroundTasks, Query, Depends
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import logging
from datetime import datetime
import asyncio

from models.integrations.api_manager import integration_manager, IntegrationStatus
from monitoring.performance_monitor import performance_monitor, MonitoredOperation

router = APIRouter(prefix="/integrations", tags=["External API Integrations"])
logger = logging.getLogger(__name__)

class EnrichmentRequest(BaseModel):
    lead_data: Dict[str, Any]
    sources: Optional[List[str]] = None
    cache_enabled: bool = True

class IntegrationTestRequest(BaseModel):
    integration_name: str
    test_data: Optional[Dict[str, Any]] = None

class IntegrationConfigUpdate(BaseModel):
    enabled: Optional[bool] = None
    rate_limit: Optional[Dict[str, Any]] = None
    timeout: Optional[int] = None

@router.post("/enrich-lead")
async def enrich_lead_data(request: EnrichmentRequest):
    """Enrich lead data using external integrations"""
    try:
        with MonitoredOperation("integrations", "lead_enrichment") as op:
            op.add_metadata(
                sources_requested=request.sources or "all",
                cache_enabled=request.cache_enabled
            )
            
            # Clear cache if disabled
            if not request.cache_enabled:
                integration_manager.clear_cache()
            
            # Enrich the lead data
            enriched_data = await integration_manager.enrich_lead_data(
                lead_data=request.lead_data,
                sources=request.sources
            )
            
            # Record enrichment metrics
            enrichment_info = enriched_data.get('_enrichment', {})
            sources_used = enrichment_info.get('sources_used', [])
            
            performance_monitor.record_metric({
                'metric_name': 'enrichment_sources_used',
                'value': len(sources_used),
                'component': 'integrations.enrichment',
                'metadata': {
                    'sources': sources_used,
                    'lead_id': request.lead_data.get('lead_id', 'unknown')
                }
            })
            
            return {
                "status": "success",
                "enriched_data": enriched_data,
                "enrichment_summary": {
                    "sources_used": sources_used,
                    "fields_added": len(enriched_data) - len(request.lead_data),
                    "timestamp": enrichment_info.get('timestamp'),
                    "cache_used": enrichment_info.get('cache_key') in integration_manager.cache
                }
            }
            
    except Exception as e:
        logger.error(f"Lead enrichment failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Enrichment failed: {str(e)}")

@router.get("/status")
async def get_integrations_status():
    """Get status of all integrations"""
    try:
        with MonitoredOperation("integrations", "status_check"):
            status = await integration_manager.get_integration_status()
            
            # Calculate overall health
            total_integrations = len(status)
            healthy_integrations = sum(1 for s in status.values() if s['health_check'])
            overall_health = "healthy" if healthy_integrations == total_integrations else "degraded"
            
            return {
                "status": "success",
                "overall_health": overall_health,
                "integrations": status,
                "summary": {
                    "total_integrations": total_integrations,
                    "healthy_integrations": healthy_integrations,
                    "unhealthy_integrations": total_integrations - healthy_integrations
                },
                "checked_at": datetime.now().isoformat()
            }
            
    except Exception as e:
        logger.error(f"Failed to get integration status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Status check failed: {str(e)}")

@router.get("/status/{integration_name}")
async def get_integration_status(integration_name: str):
    """Get detailed status of a specific integration"""
    try:
        if integration_name not in integration_manager.integrations:
            raise HTTPException(status_code=404, detail=f"Integration '{integration_name}' not found")
        
        integration = integration_manager.integrations[integration_name]
        
        # Perform health check
        health_check = await integration.health_check()
        
        return {
            "status": "success",
            "integration_name": integration_name,
            "integration_status": {
                "status": integration.status.value,
                "health_check": health_check,
                "configuration": {
                    "base_url": integration.config.base_url,
                    "timeout": integration.config.timeout,
                    "enabled": integration.config.enabled,
                    "rate_limits": {
                        "requests_per_minute": integration.config.rate_limits.requests_per_minute,
                        "requests_per_hour": integration.config.rate_limits.requests_per_hour,
                        "requests_per_day": integration.config.rate_limits.requests_per_day
                    }
                },
                "metrics": integration.metrics,
                "rate_limit_usage": {
                    "minute_requests": len(integration.rate_limiter.minute_requests),
                    "hour_requests": len(integration.rate_limiter.hour_requests),
                    "day_requests": len(integration.rate_limiter.day_requests)
                },
                "error_count": integration.error_count
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get status for {integration_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Status check failed: {str(e)}")

@router.post("/test/{integration_name}")
async def test_integration(integration_name: str, request: IntegrationTestRequest):
    """Test a specific integration"""
    try:
        with MonitoredOperation("integrations", f"test_{integration_name}"):
            result = await integration_manager.test_integration(
                integration_name=integration_name,
                test_data=request.test_data
            )
            
            return {
                "status": "success",
                "integration_name": integration_name,
                "test_result": result
            }
            
    except Exception as e:
        logger.error(f"Integration test failed for {integration_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Test failed: {str(e)}")

@router.get("/available-sources")
async def get_available_sources():
    """Get list of available data sources"""
    try:
        sources = {}
        
        for name, integration in integration_manager.integrations.items():
            sources[name] = {
                "name": integration.config.name,
                "type": integration.config.source_type.value,
                "status": integration.status.value,
                "enabled": integration.config.enabled,
                "description": _get_source_description(name),
                "data_types": _get_source_data_types(name),
                "rate_limits": {
                    "requests_per_minute": integration.config.rate_limits.requests_per_minute,
                    "requests_per_hour": integration.config.rate_limits.requests_per_hour
                }
            }
        
        return {
            "status": "success",
            "available_sources": sources,
            "total_sources": len(sources)
        }
        
    except Exception as e:
        logger.error(f"Failed to get available sources: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get sources: {str(e)}")

@router.get("/enrichment-fields")
async def get_enrichment_fields():
    """Get list of fields that can be enriched"""
    try:
        enrichment_fields = {
            "clearbit": [
                "full_name", "location", "employment", "social_profiles",
                "company_name", "company_industry", "company_size", "company_revenue"
            ],
            "fullcontact": [
                "social_profiles", "age_range", "gender", "location_general"
            ],
            "social_media": [
                "social_engagement_score", "interests", "activity_level",
                "influence_score", "platform_presence"
            ],
            "financial_data": [
                "estimated_income_range", "credit_score_range", "spending_patterns",
                "financial_stability_score", "insurance_likelihood"
            ],
            "market_intelligence": [
                "market_competition_level", "average_premium_range", "market_demand_score",
                "seasonal_trends", "local_regulations"
            ]
        }
        
        # Flatten all unique fields
        all_fields = set()
        for fields in enrichment_fields.values():
            all_fields.update(fields)
        
        return {
            "status": "success",
            "enrichment_fields": enrichment_fields,
            "all_available_fields": sorted(list(all_fields)),
            "field_count": len(all_fields)
        }
        
    except Exception as e:
        logger.error(f"Failed to get enrichment fields: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get fields: {str(e)}")

@router.post("/batch-enrich")
async def batch_enrich_leads(
    leads: List[Dict[str, Any]],
    sources: Optional[List[str]] = None,
    max_concurrent: int = Query(10, description="Maximum concurrent enrichments")
):
    """Batch enrich multiple leads"""
    try:
        if len(leads) > 100:
            raise HTTPException(status_code=400, detail="Maximum 100 leads per batch")
        
        with MonitoredOperation("integrations", "batch_enrichment") as op:
            op.add_metadata(
                batch_size=len(leads),
                sources=sources or "all",
                max_concurrent=max_concurrent
            )
            
            # Create semaphore to limit concurrent requests
            semaphore = asyncio.Semaphore(max_concurrent)
            
            async def enrich_single_lead(lead_data):
                async with semaphore:
                    try:
                        return await integration_manager.enrich_lead_data(
                            lead_data=lead_data,
                            sources=sources
                        )
                    except Exception as e:
                        logger.error(f"Failed to enrich lead {lead_data.get('lead_id', 'unknown')}: {e}")
                        return {**lead_data, '_enrichment_error': str(e)}
            
            # Process all leads concurrently
            enriched_leads = await asyncio.gather(
                *[enrich_single_lead(lead) for lead in leads],
                return_exceptions=True
            )
            
            # Count successes and failures
            successful_enrichments = 0
            failed_enrichments = 0
            
            for i, result in enumerate(enriched_leads):
                if isinstance(result, Exception):
                    enriched_leads[i] = {**leads[i], '_enrichment_error': str(result)}
                    failed_enrichments += 1
                elif '_enrichment_error' in result:
                    failed_enrichments += 1
                else:
                    successful_enrichments += 1
            
            return {
                "status": "success",
                "enriched_leads": enriched_leads,
                "batch_summary": {
                    "total_leads": len(leads),
                    "successful_enrichments": successful_enrichments,
                    "failed_enrichments": failed_enrichments,
                    "success_rate": successful_enrichments / len(leads) * 100,
                    "sources_used": sources or "all"
                }
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch enrichment failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch enrichment failed: {str(e)}")

@router.post("/cache/clear")
async def clear_enrichment_cache():
    """Clear the enrichment cache"""
    try:
        integration_manager.clear_cache()
        
        return {
            "status": "success",
            "message": "Enrichment cache cleared successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to clear cache: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to clear cache: {str(e)}")

@router.get("/metrics")
async def get_integration_metrics():
    """Get integration performance metrics"""
    try:
        metrics = {}
        
        for name, integration in integration_manager.integrations.items():
            metrics[name] = {
                "performance_metrics": integration.metrics,
                "rate_limit_usage": {
                    "minute_usage": len(integration.rate_limiter.minute_requests),
                    "hour_usage": len(integration.rate_limiter.hour_requests),
                    "day_usage": len(integration.rate_limiter.day_requests),
                    "minute_limit": integration.config.rate_limits.requests_per_minute,
                    "hour_limit": integration.config.rate_limits.requests_per_hour,
                    "day_limit": integration.config.rate_limits.requests_per_day
                },
                "health_status": integration.status.value,
                "error_count": integration.error_count
            }
        
        # Calculate overall metrics
        total_requests = sum(m["performance_metrics"]["total_requests"] for m in metrics.values())
        total_successful = sum(m["performance_metrics"]["successful_requests"] for m in metrics.values())
        total_failed = sum(m["performance_metrics"]["failed_requests"] for m in metrics.values())
        
        overall_metrics = {
            "total_requests": total_requests,
            "successful_requests": total_successful,
            "failed_requests": total_failed,
            "success_rate": (total_successful / total_requests * 100) if total_requests > 0 else 0,
            "average_response_time": sum(
                m["performance_metrics"]["avg_response_time"] for m in metrics.values()
            ) / len(metrics) if metrics else 0
        }
        
        return {
            "status": "success",
            "integration_metrics": metrics,
            "overall_metrics": overall_metrics,
            "cache_size": len(integration_manager.cache)
        }
        
    except Exception as e:
        logger.error(f"Failed to get integration metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get metrics: {str(e)}")

def _get_source_description(source_name: str) -> str:
    """Get description for a data source"""
    descriptions = {
        "clearbit": "Person and company enrichment data including employment, social profiles, and company information",
        "fullcontact": "Social media profiles, demographics, and contact information",
        "social_media": "Social media engagement scores, interests, and activity levels",
        "financial_data": "Financial indicators, credit scores, and spending patterns",
        "market_intelligence": "Market competition, pricing trends, and local regulations"
    }
    return descriptions.get(source_name, "External data enrichment source")

def _get_source_data_types(source_name: str) -> List[str]:
    """Get data types provided by a source"""
    data_types = {
        "clearbit": ["personal", "employment", "company", "social"],
        "fullcontact": ["social", "demographic", "contact"],
        "social_media": ["engagement", "interests", "activity", "influence"],
        "financial_data": ["financial", "credit", "spending", "stability"],
        "market_intelligence": ["market", "competition", "pricing", "regulations"]
    }
    return data_types.get(source_name, ["general"])