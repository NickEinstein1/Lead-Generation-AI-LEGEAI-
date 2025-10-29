"""
Real-Time Data Pipeline and Integration System

Comprehensive real-time data processing pipeline for the Insurance Lead Scoring Platform
with flexible integration capabilities for custom CRM and external data sources.

Features:
- Real-time data ingestion and processing
- Event-driven architecture with message queues
- Stream processing with Apache Kafka/Redis Streams
- Data transformation and enrichment pipelines
- Integration adapters for various data sources
- Custom CRM integration framework
- Real-time analytics and monitoring
"""

from .data_ingestion import (
    data_ingestion_manager, stream_processor, event_bus,
    DataIngestionManager, StreamProcessor, EventBus
)
from .stream_processing import (
    kafka_stream_processor, redis_stream_processor, data_transformer,
    KafkaStreamProcessor, RedisStreamProcessor, DataTransformer
)
from .integration_framework import (
    integration_hub, crm_integration_framework, external_api_manager,
    IntegrationHub, CRMIntegrationFramework, ExternalAPIManager
)
from .data_transformation import (
    transformation_engine, data_validator, schema_registry,
    TransformationEngine, DataValidator, SchemaRegistry
)
from .real_time_analytics import (
    analytics_processor, metrics_aggregator, alert_manager,
    AnalyticsProcessor, MetricsAggregator, AlertManager
)
from .pipeline_orchestrator import (
    pipeline_orchestrator, PipelineOrchestrator
)

import asyncio
import logging
from typing import Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)

# Global pipeline components
pipeline_orchestrator = PipelineOrchestrator()

# Convenience functions
async def initialize_pipeline():
    """Initialize the complete real-time data pipeline"""
    await pipeline_orchestrator.initialize_all_components()

async def start_pipeline():
    """Start all pipeline services"""
    await pipeline_orchestrator.start_all_services()

async def get_pipeline_health():
    """Get pipeline system health"""
    return await pipeline_orchestrator.get_system_health()

async def process_real_time_event(event_type: str, event_data: Dict[str, Any]):
    """Process a real-time event through the pipeline"""
    return await pipeline_orchestrator.process_event(event_type, event_data)

# Export all components
__all__ = [
    # Core components
    'data_ingestion_manager',
    'stream_processor',
    'event_bus',
    'kafka_stream_processor',
    'redis_stream_processor',
    'data_transformer',
    'integration_hub',
    'crm_integration_framework',
    'external_api_manager',
    'transformation_engine',
    'data_validator',
    'schema_registry',
    'analytics_processor',
    'metrics_aggregator',
    'alert_manager',
    'pipeline_orchestrator',
    
    # Convenience functions
    'initialize_pipeline',
    'start_pipeline',
    'get_pipeline_health',
    'process_real_time_event',
    
    # Classes
    'DataIngestionManager',
    'StreamProcessor',
    'EventBus',
    'KafkaStreamProcessor',
    'RedisStreamProcessor',
    'DataTransformer',
    'IntegrationHub',
    'CRMIntegrationFramework',
    'ExternalAPIManager',
    'TransformationEngine',
    'DataValidator',
    'SchemaRegistry',
    'AnalyticsProcessor',
    'MetricsAggregator',
    'AlertManager',
    'PipelineOrchestrator'
]

__version__ = "1.0.0"
__author__ = "Insurance Lead Scoring Platform Data Team"