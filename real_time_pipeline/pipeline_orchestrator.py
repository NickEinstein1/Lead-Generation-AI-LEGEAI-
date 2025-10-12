"""
Pipeline Orchestrator

Central orchestrator for the real-time data pipeline, coordinating all components
and providing unified management and monitoring capabilities.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
import json

from .data_ingestion import data_ingestion_manager, event_bus, stream_processor
from .integration_framework import integration_hub
from .stream_processing import redis_stream_processor, data_transformer

logger = logging.getLogger(__name__)

class PipelineStatus(Enum):
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"
    STOPPED = "stopped"

@dataclass
class PipelineMetrics:
    events_processed: int = 0
    events_failed: int = 0
    processing_rate: float = 0.0
    average_latency: float = 0.0
    error_rate: float = 0.0
    last_updated: datetime = field(default_factory=datetime.utcnow)

class PipelineOrchestrator:
    """Central orchestrator for the real-time data pipeline"""
    
    def __init__(self):
        self.status = PipelineStatus.STOPPED
        self.components = {
            'data_ingestion': data_ingestion_manager,
            'event_bus': event_bus,
            'stream_processor': stream_processor,
            'redis_stream_processor': redis_stream_processor,
            'integration_hub': integration_hub,
            'data_transformer': data_transformer
        }
        
        self.metrics = PipelineMetrics()
        self.health_checks = {}
        self.monitoring_tasks = []
        self.error_handlers = {}
        
    async def initialize_all_components(self):
        """Initialize all pipeline components"""
        
        try:
            self.status = PipelineStatus.INITIALIZING
            logger.info("Initializing real-time data pipeline components...")
            
            # Initialize components in order
            await event_bus.initialize()
            await redis_stream_processor.initialize()
            await integration_hub.initialize()
            
            # Register event handlers
            await self._register_event_handlers()
            
            # Setup monitoring
            await self._setup_monitoring()
            
            logger.info("All pipeline components initialized successfully")
            
        except Exception as e:
            self.status = PipelineStatus.ERROR
            logger.error(f"Failed to initialize pipeline components: {e}")
            raise
    
    async def start_all_services(self):
        """Start all pipeline services"""
        
        try:
            logger.info("Starting real-time data pipeline services...")
            
            # Start core services
            await data_ingestion_manager.start_ingestion()
            await stream_processor.start_processing()
            await redis_stream_processor.start_processing([
                redis_stream_processor.StreamType.LEAD_EVENTS,
                redis_stream_processor.StreamType.EMAIL_EVENTS,
                redis_stream_processor.StreamType.WEB_EVENTS
            ])
            
            # Start monitoring tasks
            self.monitoring_tasks = [
                asyncio.create_task(self._health_monitoring_loop()),
                asyncio.create_task(self._metrics_collection_loop()),
                asyncio.create_task(self._error_monitoring_loop())
            ]
            
            self.status = PipelineStatus.RUNNING
            logger.info("All pipeline services started successfully")
            
            # Wait for all services
            await asyncio.gather(*self.monitoring_tasks, return_exceptions=True)
            
        except Exception as e:
            self.status = PipelineStatus.ERROR
            logger.error(f"Error starting pipeline services: {e}")
            raise
    
    async def stop_all_services(self):
        """Stop all pipeline services"""
        
        try:
            logger.info("Stopping real-time data pipeline services...")
            
            self.status = PipelineStatus.STOPPED
            
            # Cancel monitoring tasks
            for task in self.monitoring_tasks:
                task.cancel()
            
            # Cleanup components
            await integration_hub.cleanup()
            
            logger.info("All pipeline services stopped")
            
        except Exception as e:
            logger.error(f"Error stopping pipeline services: {e}")
    
    async def process_event(self, event_type: str, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a real-time event through the pipeline"""
        
        try:
            start_time = datetime.utcnow()
            
            # Determine event source and create event
            if event_type == "form_submission":
                event_id = await data_ingestion_manager.ingest_web_form_data(
                    event_data, 
                    source_id="web_form"
                )
            elif event_type == "api_data":
                event_id = await data_ingestion_manager.ingest_api_data(
                    event_data,
                    source_id="api_endpoint"
                )
            elif event_type == "webhook":
                event_id = await data_ingestion_manager.ingest_webhook_data(
                    event_data,
                    source_id=event_data.get('source', 'unknown')
                )
            else:
                # Generic event processing
                event_id = await data_ingestion_manager.ingest_api_data(
                    event_data,
                    source_id="generic"
                )
            
            # Calculate processing time
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            # Update metrics
            self.metrics.events_processed += 1
            self.metrics.average_latency = (
                (self.metrics.average_latency * (self.metrics.events_processed - 1) + processing_time) /
                self.metrics.events_processed
            )
            
            return {
                'success': True,
                'event_id': event_id,
                'processing_time': processing_time,
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.metrics.events_failed += 1
            logger.error(f"Error processing event {event_type}: {e}")
            
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }
    
    async def _register_event_handlers(self):
        """Register event handlers for different event types"""
        
        # Lead events
        event_bus.register_handler(
            event_bus.EventType.LEAD_CREATED,
            self._handle_lead_created
        )
        
        event_bus.register_handler(
            event_bus.EventType.LEAD_UPDATED,
            self._handle_lead_updated
        )
        
        # Email events
        event_bus.register_handler(
            event_bus.EventType.EMAIL_OPENED,
            self._handle_email_opened
        )
        
        # Form events
        event_bus.register_handler(
            event_bus.EventType.FORM_SUBMITTED,
            self._handle_form_submitted
        )
        
        logger.info("Event handlers registered")
    
    async def _handle_lead_created(self, event_data: Dict[str, Any]):
        """Handle lead created event"""
        
        try:
            # Queue CRM sync
            await integration_hub.queue_sync_operation({
                'type': 'crm_sync',
                'operation_id': f"crm_sync_{datetime.utcnow().timestamp()}",
                'data': event_data,
                'direction': 'outbound'
            })
            
            logger.info(f"Queued CRM sync for new lead: {event_data.get('lead_id')}")
            
        except Exception as e:
            logger.error(f"Error handling lead created event: {e}")
    
    async def _handle_lead_updated(self, event_data: Dict[str, Any]):
        """Handle lead updated event"""
        
        try:
            # Queue CRM sync for update
            await integration_hub.queue_sync_operation({
                'type': 'crm_sync',
                'operation_id': f"crm_update_{datetime.utcnow().timestamp()}",
                'data': event_data,
                'direction': 'outbound'
            })
            
            logger.info(f"Queued CRM update for lead: {event_data.get('lead_id')}")
            
        except Exception as e:
            logger.error(f"Error handling lead updated event: {e}")
    
    async def _handle_email_opened(self, event_data: Dict[str, Any]):
        """Handle email opened event"""
        
        try:
            # Update engagement metrics
            # This would integrate with analytics system
            logger.info(f"Email opened by lead: {event_data.get('lead_id')}")
            
        except Exception as e:
            logger.error(f"Error handling email opened event: {e}")
    
    async def _handle_form_submitted(self, event_data: Dict[str, Any]):
        """Handle form submitted event"""
        
        try:
            # Trigger lead scoring
            # This would integrate with lead scoring system
            logger.info(f"Form submitted by lead: {event_data.get('lead_id')}")
            
        except Exception as e:
            logger.error(f"Error handling form submitted event: {e}")
    
    async def _setup_monitoring(self):
        """Setup pipeline monitoring"""
        
        # Register health checks
        self.health_checks = {
            'event_bus': self._check_event_bus_health,
            'stream_processor': self._check_stream_processor_health,
            'integration_hub': self._check_integration_hub_health,
            'redis_connection': self._check_redis_connection
        }
        
        logger.info("Pipeline monitoring setup completed")
    
    async def _health_monitoring_loop(self):
        """Continuous health monitoring"""
        
        while self.status == PipelineStatus.RUNNING:
            try:
                for component, health_check in self.health_checks.items():
                    is_healthy = await health_check()
                    if not is_healthy:
                        logger.warning(f"Health check failed for {component}")
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in health monitoring: {e}")
                await asyncio.sleep(60)
    
    async def _metrics_collection_loop(self):
        """Continuous metrics collection"""
        
        while self.status == PipelineStatus.RUNNING:
            try:
                # Update processing rate
                current_time = datetime.utcnow()
                time_diff = (current_time - self.metrics.last_updated).total_seconds()
                
                if time_diff > 0:
                    self.metrics.processing_rate = self.metrics.events_processed / time_diff
                    self.metrics.error_rate = self.metrics.events_failed / max(1, self.metrics.events_processed)
                
                self.metrics.last_updated = current_time
                
                await asyncio.sleep(60)  # Update every minute
                
            except Exception as e:
                logger.error(f"Error in metrics collection: {e}")
                await asyncio.sleep(60)
    
    async def _error_monitoring_loop(self):
        """Monitor for errors and alerts"""
        
        while self.status == PipelineStatus.RUNNING:
            try:
                # Check error rate
                if self.metrics.error_rate > 0.1:  # 10% error rate threshold
                    logger.warning(f"High error rate detected: {self.metrics.error_rate:.2%}")
                
                await asyncio.sleep(120)  # Check every 2 minutes
                
            except Exception as e:
                logger.error(f"Error in error monitoring: {e}")
                await asyncio.sleep(60)
    
    async def _check_event_bus_health(self) -> bool:
        """Check event bus health"""
        try:
            return event_bus.redis_pool is not None
        except:
            return False
    
    async def _check_stream_processor_health(self) -> bool:
        """Check stream processor health"""
        try:
            return len(redis_stream_processor.processors) > 0
        except:
            return False
    
    async def _check_integration_hub_health(self) -> bool:
        """Check integration hub health"""
        try:
            return integration_hub.running
        except:
            return False
    
    async def _check_redis_connection(self) -> bool:
        """Check Redis connection"""
        try:
            import aioredis
            redis = aioredis.Redis(connection_pool=event_bus.redis_pool)
            await redis.ping()
            return True
        except:
            return False
    
    async def get_system_health(self) -> Dict[str, Any]:
        """Get comprehensive system health report"""
        
        health_report = {
            'status': self.status.value,
            'uptime': (datetime.utcnow() - self.metrics.last_updated).total_seconds(),
            'metrics': {
                'events_processed': self.metrics.events_processed,
                'events_failed': self.metrics.events_failed,
                'processing_rate': self.metrics.processing_rate,
                'average_latency': self.metrics.average_latency,
                'error_rate': self.metrics.error_rate
            },
            'components': {}
        }
        
        # Check component health
        for component, health_check in self.health_checks.items():
            try:
                is_healthy = await health_check()
                health_report['components'][component] = {
                    'status': 'healthy' if is_healthy else 'unhealthy',
                    'last_checked': datetime.utcnow().isoformat()
                }
            except Exception as e:
                health_report['components'][component] = {
                    'status': 'error',
                    'error': str(e),
                    'last_checked': datetime.utcnow().isoformat()
                }
        
        return health_report

# Global instance
pipeline_orchestrator = PipelineOrchestrator()
