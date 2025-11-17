"""
Real-Time Pipeline API

API endpoints for managing and monitoring the real-time data pipeline,
including data ingestion, stream processing, and integration management.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime

from backend.real_time_pipeline import (
    pipeline_orchestrator, data_ingestion_manager, integration_hub,
    process_real_time_event, get_pipeline_health
)

router = APIRouter(prefix="/pipeline", tags=["Real-Time Pipeline"])
logger = logging.getLogger(__name__)

class EventData(BaseModel):
    event_type: str = Field(..., description="Type of event to process")
    data: Dict[str, Any] = Field(..., description="Event data payload")
    source_id: Optional[str] = Field(None, description="Source identifier")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)

class WebhookData(BaseModel):
    source: str = Field(..., description="Webhook source")
    event_type: str = Field(..., description="Event type")
    payload: Dict[str, Any] = Field(..., description="Webhook payload")
    signature: Optional[str] = Field(None, description="Webhook signature")

class CRMSyncRequest(BaseModel):
    lead_data: Dict[str, Any] = Field(..., description="Lead data to sync")
    sync_direction: str = Field("outbound", description="Sync direction: inbound/outbound")
    force_sync: bool = Field(False, description="Force sync even if recently synced")

@router.post("/events/ingest")
async def ingest_event(event_data: EventData, background_tasks: BackgroundTasks):
    """Ingest a real-time event into the pipeline"""
    
    try:
        # Process event through pipeline
        result = await process_real_time_event(
            event_data.event_type,
            {
                **event_data.data,
                'source_id': event_data.source_id,
                'metadata': event_data.metadata
            }
        )
        
        return {
            "status": "success",
            "event_id": result.get('event_id'),
            "processing_time": result.get('processing_time'),
            "message": "Event ingested successfully"
        }
        
    except Exception as e:
        logger.error(f"Error ingesting event: {e}")
        raise HTTPException(status_code=500, detail=f"Error ingesting event: {str(e)}")

@router.post("/webhooks/receive")
async def receive_webhook(webhook_data: WebhookData, background_tasks: BackgroundTasks):
    """Receive and process webhook data"""
    
    try:
        # Process webhook through pipeline
        result = await data_ingestion_manager.ingest_webhook_data(
            {
                'source': webhook_data.source,
                'event_type': webhook_data.event_type,
                **webhook_data.payload
            },
            source_id=webhook_data.source
        )
        
        return {
            "status": "success",
            "event_id": result,
            "message": "Webhook processed successfully"
        }
        
    except Exception as e:
        logger.error(f"Error processing webhook: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing webhook: {str(e)}")

@router.post("/crm/sync")
async def sync_with_crm(sync_request: CRMSyncRequest, background_tasks: BackgroundTasks):
    """Sync lead data with CRM system"""
    
    try:
        # Queue CRM sync operation
        await integration_hub.queue_sync_operation({
            'type': 'crm_sync',
            'operation_id': f"manual_sync_{datetime.now(datetime.UTC).timestamp()}",
            'data': sync_request.lead_data,
            'direction': sync_request.sync_direction,
            'force_sync': sync_request.force_sync
        })
        
        return {
            "status": "success",
            "message": "CRM sync queued successfully",
            "sync_direction": sync_request.sync_direction
        }
        
    except Exception as e:
        logger.error(f"Error queuing CRM sync: {e}")
        raise HTTPException(status_code=500, detail=f"Error queuing CRM sync: {str(e)}")

@router.get("/health")
async def get_pipeline_health_status():
    """Get comprehensive pipeline health status"""
    
    try:
        health_report = await get_pipeline_health()
        return health_report
        
    except Exception as e:
        logger.error(f"Error getting pipeline health: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting pipeline health: {str(e)}")

@router.get("/metrics")
async def get_pipeline_metrics():
    """Get pipeline performance metrics"""
    
    try:
        health_report = await pipeline_orchestrator.get_system_health()
        
        return {
            "status": health_report['status'],
            "metrics": health_report['metrics'],
            "uptime_seconds": health_report['uptime'],
            "timestamp": datetime.now(datetime.UTC).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting pipeline metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting pipeline metrics: {str(e)}")

@router.get("/ingestion/stats")
async def get_ingestion_stats():
    """Get data ingestion statistics"""
    
    try:
        stats = data_ingestion_manager.get_ingestion_stats()
        return stats
        
    except Exception as e:
        logger.error(f"Error getting ingestion stats: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting ingestion stats: {str(e)}")

@router.post("/pipeline/start")
async def start_pipeline():
    """Start the real-time pipeline"""
    
    try:
        if pipeline_orchestrator.status.value != "running":
            background_task = asyncio.create_task(pipeline_orchestrator.start_all_services())
            
            return {
                "status": "success",
                "message": "Pipeline startup initiated",
                "current_status": pipeline_orchestrator.status.value
            }
        else:
            return {
                "status": "info",
                "message": "Pipeline is already running",
                "current_status": pipeline_orchestrator.status.value
            }
        
    except Exception as e:
        logger.error(f"Error starting pipeline: {e}")
        raise HTTPException(status_code=500, detail=f"Error starting pipeline: {str(e)}")

@router.post("/pipeline/stop")
async def stop_pipeline():
    """Stop the real-time pipeline"""
    
    try:
        await pipeline_orchestrator.stop_all_services()
        
        return {
            "status": "success",
            "message": "Pipeline stopped successfully",
            "current_status": pipeline_orchestrator.status.value
        }
        
    except Exception as e:
        logger.error(f"Error stopping pipeline: {e}")
        raise HTTPException(status_code=500, detail=f"Error stopping pipeline: {str(e)}")

@router.get("/integrations/status")
async def get_integrations_status():
    """Get status of all integrations"""
    
    try:
        # Get CRM integration status
        crm_status = {
            "type": "crm",
            "configured": integration_hub.crm_framework.crm_config is not None,
            "enabled": True,
            "last_sync": None  # Would be populated from actual sync records
        }
        
        # Get external integrations status
        external_integrations = []
        for integration_id, integration in integration_hub.api_manager.integrations.items():
            external_integrations.append({
                "integration_id": integration_id,
                "name": integration.config.name,
                "type": integration.config.integration_type.value,
                "enabled": integration.config.enabled,
                "auth_type": integration.config.auth_type.value
            })
        
        return {
            "crm_integration": crm_status,
            "external_integrations": external_integrations,
            "total_integrations": len(external_integrations) + 1
        }
        
    except Exception as e:
        logger.error(f"Error getting integrations status: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting integrations status: {str(e)}")

# Background task for pipeline initialization
async def _initialize_pipeline_background():
    """Background task to initialize pipeline"""
    try:
        await pipeline_orchestrator.initialize_all_components()
        logger.info("Pipeline initialized successfully in background")
    except Exception as e:
        logger.error(f"Error initializing pipeline in background: {e}")

# Initialize pipeline on startup (safe in running event loop only)
import asyncio
try:
    loop = asyncio.get_event_loop()
    if loop.is_running():
        loop.create_task(_initialize_pipeline_background())
except Exception:
    # When imported outside an event loop (e.g., module import or tests), skip scheduling.
    pass