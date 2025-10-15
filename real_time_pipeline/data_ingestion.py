"""
Real-Time Data Ingestion System

Handles real-time data ingestion from multiple sources including:
- Web forms and landing pages
- API endpoints
- File uploads and batch imports
- Third-party integrations
- IoT devices and sensors
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
from redis import asyncio as aioredis
from concurrent.futures import ThreadPoolExecutor
import hashlib
import uuid

logger = logging.getLogger(__name__)

class DataSourceType(Enum):
    WEB_FORM = "web_form"
    API_ENDPOINT = "api_endpoint"
    FILE_UPLOAD = "file_upload"
    THIRD_PARTY_API = "third_party_api"
    DATABASE_CDC = "database_cdc"  # Change Data Capture
    MESSAGE_QUEUE = "message_queue"
    WEBHOOK = "webhook"
    STREAMING = "streaming"

class EventType(Enum):
    LEAD_CREATED = "lead_created"
    LEAD_UPDATED = "lead_updated"
    FORM_SUBMITTED = "form_submitted"
    EMAIL_OPENED = "email_opened"
    EMAIL_CLICKED = "email_clicked"
    PAGE_VISITED = "page_visited"
    DOCUMENT_DOWNLOADED = "document_downloaded"
    CALL_COMPLETED = "call_completed"
    MEETING_SCHEDULED = "meeting_scheduled"
    CUSTOM_EVENT = "custom_event"

@dataclass
class DataEvent:
    event_id: str
    event_type: EventType
    source_type: DataSourceType
    source_id: str
    data: Dict[str, Any]
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    processed: bool = False
    retry_count: int = 0

@dataclass
class DataSource:
    source_id: str
    source_type: DataSourceType
    name: str
    config: Dict[str, Any]
    enabled: bool = True
    last_sync: Optional[datetime] = None
    error_count: int = 0

class EventBus:
    """High-performance event bus for real-time data distribution"""
    
    def __init__(self, redis_config: Dict[str, Any] = None):
        self.redis_config = redis_config or {
            'host': 'localhost',
            'port': 6379,
            'db': 5
        }
        self.redis_pool = None
        self.subscribers = {}
        self.event_handlers = {}
        
    async def initialize(self):
        """Initialize event bus"""
        
        self.redis_pool = aioredis.ConnectionPool.from_url(
            f"redis://{self.redis_config['host']}:{self.redis_config['port']}/{self.redis_config['db']}",
            max_connections=50
        )
        
        logger.info("Event bus initialized")
    
    async def publish_event(self, event: DataEvent):
        """Publish event to the bus"""
        
        try:
            redis = aioredis.Redis(connection_pool=self.redis_pool)
            
            # Serialize event
            event_data = {
                'event_id': event.event_id,
                'event_type': event.event_type.value,
                'source_type': event.source_type.value,
                'source_id': event.source_id,
                'data': event.data,
                'timestamp': event.timestamp.isoformat(),
                'metadata': event.metadata
            }
            
            # Publish to Redis streams
            stream_key = f"events:{event.event_type.value}"
            await redis.xadd(stream_key, event_data)
            
            # Also publish to pub/sub for immediate processing
            await redis.publish(f"event_channel:{event.event_type.value}", json.dumps(event_data))
            
            logger.debug(f"Published event {event.event_id} to {stream_key}")
            
        except Exception as e:
            logger.error(f"Error publishing event {event.event_id}: {e}")
            raise
    
    async def subscribe_to_events(self, event_types: List[EventType], handler: Callable):
        """Subscribe to specific event types"""
        
        try:
            redis = aioredis.Redis(connection_pool=self.redis_pool)
            pubsub = redis.pubsub()
            
            # Subscribe to channels
            channels = [f"event_channel:{event_type.value}" for event_type in event_types]
            await pubsub.subscribe(*channels)
            
            # Process messages
            async for message in pubsub.listen():
                if message['type'] == 'message':
                    try:
                        event_data = json.loads(message['data'])
                        await handler(event_data)
                    except Exception as e:
                        logger.error(f"Error processing event: {e}")
            
        except Exception as e:
            logger.error(f"Error in event subscription: {e}")
    
    def register_handler(self, event_type: EventType, handler: Callable):
        """Register event handler"""
        
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        
        self.event_handlers[event_type].append(handler)
        logger.info(f"Registered handler for {event_type.value}")

class StreamProcessor:
    """High-throughput stream processor for real-time data"""
    
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self.processing_queues = {}
        self.processors = {}
        self.metrics = {
            'events_processed': 0,
            'events_failed': 0,
            'processing_time_total': 0.0
        }
        
    async def start_processing(self):
        """Start stream processing"""
        
        # Start processors for each event type
        for event_type in EventType:
            processor_task = asyncio.create_task(
                self._process_event_stream(event_type)
            )
            self.processors[event_type] = processor_task
        
        logger.info("Stream processing started for all event types")
    
    async def _process_event_stream(self, event_type: EventType):
        """Process events from a specific stream"""
        
        try:
            redis = aioredis.Redis(connection_pool=self.event_bus.redis_pool)
            stream_key = f"events:{event_type.value}"
            consumer_group = f"processors_{event_type.value}"
            consumer_name = f"processor_{uuid.uuid4().hex[:8]}"
            
            # Create consumer group if it doesn't exist
            try:
                await redis.xgroup_create(stream_key, consumer_group, id='0', mkstream=True)
            except Exception:
                pass  # Group already exists
            
            while True:
                try:
                    # Read from stream
                    messages = await redis.xreadgroup(
                        consumer_group,
                        consumer_name,
                        {stream_key: '>'},
                        count=10,
                        block=1000
                    )
                    
                    for stream, msgs in messages:
                        for msg_id, fields in msgs:
                            await self._process_single_event(event_type, msg_id, fields)
                            
                            # Acknowledge message
                            await redis.xack(stream_key, consumer_group, msg_id)
                    
                except Exception as e:
                    logger.error(f"Error processing {event_type.value} stream: {e}")
                    await asyncio.sleep(5)
                    
        except Exception as e:
            logger.error(f"Fatal error in {event_type.value} stream processor: {e}")
    
    async def _process_single_event(self, event_type: EventType, msg_id: str, fields: Dict[str, Any]):
        """Process a single event"""
        
        start_time = datetime.utcnow()
        
        try:
            # Reconstruct event
            event = DataEvent(
                event_id=fields.get('event_id'),
                event_type=EventType(fields.get('event_type')),
                source_type=DataSourceType(fields.get('source_type')),
                source_id=fields.get('source_id'),
                data=json.loads(fields.get('data', '{}')),
                timestamp=datetime.fromisoformat(fields.get('timestamp')),
                metadata=json.loads(fields.get('metadata', '{}'))
            )
            
            # Execute registered handlers
            if event_type in self.event_bus.event_handlers:
                for handler in self.event_bus.event_handlers[event_type]:
                    await handler(event)
            
            # Update metrics
            self.metrics['events_processed'] += 1
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            self.metrics['processing_time_total'] += processing_time
            
            logger.debug(f"Processed event {event.event_id} in {processing_time:.3f}s")
            
        except Exception as e:
            self.metrics['events_failed'] += 1
            logger.error(f"Error processing event {msg_id}: {e}")

class DataIngestionManager:
    """Manages data ingestion from multiple sources"""
    
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self.data_sources = {}
        self.ingestion_tasks = {}
        self.thread_executor = ThreadPoolExecutor(max_workers=20)
        
    def register_data_source(self, source: DataSource):
        """Register a new data source"""
        
        self.data_sources[source.source_id] = source
        logger.info(f"Registered data source: {source.name} ({source.source_type.value})")
    
    async def start_ingestion(self):
        """Start data ingestion from all sources"""
        
        for source_id, source in self.data_sources.items():
            if source.enabled:
                if source.source_type == DataSourceType.API_ENDPOINT:
                    # API endpoints are handled by web framework
                    continue
                elif source.source_type == DataSourceType.THIRD_PARTY_API:
                    task = asyncio.create_task(self._ingest_from_api(source))
                    self.ingestion_tasks[source_id] = task
                elif source.source_type == DataSourceType.FILE_UPLOAD:
                    task = asyncio.create_task(self._monitor_file_uploads(source))
                    self.ingestion_tasks[source_id] = task
                elif source.source_type == DataSourceType.DATABASE_CDC:
                    task = asyncio.create_task(self._monitor_database_changes(source))
                    self.ingestion_tasks[source_id] = task
        
        logger.info(f"Started ingestion for {len(self.ingestion_tasks)} data sources")
    
    async def ingest_web_form_data(self, form_data: Dict[str, Any], source_id: str = "web_form"):
        """Ingest data from web form submission"""
        
        event = DataEvent(
            event_id=str(uuid.uuid4()),
            event_type=EventType.FORM_SUBMITTED,
            source_type=DataSourceType.WEB_FORM,
            source_id=source_id,
            data=form_data,
            timestamp=datetime.utcnow(),
            metadata={
                'user_agent': form_data.get('user_agent'),
                'ip_address': form_data.get('ip_address'),
                'referrer': form_data.get('referrer')
            }
        )
        
        await self.event_bus.publish_event(event)
        return event.event_id
    
    async def ingest_api_data(self, api_data: Dict[str, Any], source_id: str, event_type: EventType = EventType.CUSTOM_EVENT):
        """Ingest data from API endpoint"""
        
        event = DataEvent(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            source_type=DataSourceType.API_ENDPOINT,
            source_id=source_id,
            data=api_data,
            timestamp=datetime.utcnow()
        )
        
        await self.event_bus.publish_event(event)
        return event.event_id
    
    async def ingest_webhook_data(self, webhook_data: Dict[str, Any], source_id: str):
        """Ingest data from webhook"""
        
        # Determine event type based on webhook data
        event_type = self._determine_event_type_from_webhook(webhook_data)
        
        event = DataEvent(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            source_type=DataSourceType.WEBHOOK,
            source_id=source_id,
            data=webhook_data,
            timestamp=datetime.utcnow()
        )
        
        await self.event_bus.publish_event(event)
        return event.event_id
    
    async def _ingest_from_api(self, source: DataSource):
        """Continuously ingest data from third-party API"""
        
        while source.enabled:
            try:
                # This would implement specific API polling logic
                # For now, simulate periodic data ingestion
                await asyncio.sleep(source.config.get('poll_interval', 300))  # 5 minutes default
                
                # Update last sync time
                source.last_sync = datetime.utcnow()
                
            except Exception as e:
                source.error_count += 1
                logger.error(f"Error ingesting from {source.name}: {e}")
                await asyncio.sleep(60)  # Wait before retry
    
    async def _monitor_file_uploads(self, source: DataSource):
        """Monitor file upload directory"""
        
        upload_dir = source.config.get('upload_directory', '/tmp/uploads')
        
        while source.enabled:
            try:
                # Monitor for new files (simplified implementation)
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                source.error_count += 1
                logger.error(f"Error monitoring file uploads for {source.name}: {e}")
                await asyncio.sleep(60)
    
    async def _monitor_database_changes(self, source: DataSource):
        """Monitor database changes using CDC"""
        
        while source.enabled:
            try:
                # This would implement database change monitoring
                # Could use database triggers, log parsing, or CDC tools
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                source.error_count += 1
                logger.error(f"Error monitoring database changes for {source.name}: {e}")
                await asyncio.sleep(60)
    
    def _determine_event_type_from_webhook(self, webhook_data: Dict[str, Any]) -> EventType:
        """Determine event type from webhook data"""
        
        # Simple logic to determine event type
        # This would be more sophisticated in practice
        
        if 'lead' in webhook_data:
            if webhook_data.get('action') == 'created':
                return EventType.LEAD_CREATED
            elif webhook_data.get('action') == 'updated':
                return EventType.LEAD_UPDATED
        
        if 'email' in webhook_data:
            if webhook_data.get('event') == 'opened':
                return EventType.EMAIL_OPENED
            elif webhook_data.get('event') == 'clicked':
                return EventType.EMAIL_CLICKED
        
        return EventType.CUSTOM_EVENT
    
    def get_ingestion_stats(self) -> Dict[str, Any]:
        """Get data ingestion statistics"""
        
        stats = {
            'total_sources': len(self.data_sources),
            'active_sources': len([s for s in self.data_sources.values() if s.enabled]),
            'sources_with_errors': len([s for s in self.data_sources.values() if s.error_count > 0]),
            'sources': {}
        }
        
        for source_id, source in self.data_sources.items():
            stats['sources'][source_id] = {
                'name': source.name,
                'type': source.source_type.value,
                'enabled': source.enabled,
                'last_sync': source.last_sync.isoformat() if source.last_sync else None,
                'error_count': source.error_count
            }
        
        return stats

# Global instances
event_bus = EventBus()
stream_processor = StreamProcessor(event_bus)
data_ingestion_manager = DataIngestionManager(event_bus)