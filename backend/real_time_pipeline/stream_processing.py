"""
Stream Processing Engine

High-performance stream processing for real-time data transformation,
enrichment, and routing using Apache Kafka and Redis Streams.
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
import uuid

logger = logging.getLogger(__name__)

class ProcessingStage(Enum):
    VALIDATION = "validation"
    ENRICHMENT = "enrichment"
    TRANSFORMATION = "transformation"
    ROUTING = "routing"
    STORAGE = "storage"

class StreamType(Enum):
    LEAD_EVENTS = "lead_events"
    EMAIL_EVENTS = "email_events"
    WEB_EVENTS = "web_events"
    API_EVENTS = "api_events"
    SYSTEM_EVENTS = "system_events"

@dataclass
class ProcessingRule:
    rule_id: str
    name: str
    condition: Dict[str, Any]
    action: Dict[str, Any]
    priority: int = 0
    enabled: bool = True

@dataclass
class StreamMessage:
    message_id: str
    stream_type: StreamType
    data: Dict[str, Any]
    timestamp: datetime
    processing_stage: ProcessingStage
    metadata: Dict[str, Any] = field(default_factory=dict)
    retry_count: int = 0

class DataTransformer:
    """Handles data transformation and enrichment"""
    
    def __init__(self):
        self.transformation_rules = {}
        self.enrichment_functions = {}
        
    def register_transformation_rule(self, rule_name: str, rule_func: Callable):
        """Register a data transformation rule"""
        
        self.transformation_rules[rule_name] = rule_func
        logger.info(f"Registered transformation rule: {rule_name}")
    
    def register_enrichment_function(self, func_name: str, func: Callable):
        """Register a data enrichment function"""
        
        self.enrichment_functions[func_name] = func
        logger.info(f"Registered enrichment function: {func_name}")
    
    async def transform_data(self, data: Dict[str, Any], rules: List[str] = None) -> Dict[str, Any]:
        """Apply transformation rules to data"""
        
        transformed_data = data.copy()
        
        # Apply specified rules or all rules
        rules_to_apply = rules or list(self.transformation_rules.keys())
        
        for rule_name in rules_to_apply:
            if rule_name in self.transformation_rules:
                try:
                    transformed_data = await self.transformation_rules[rule_name](transformed_data)
                except Exception as e:
                    logger.error(f"Error applying transformation rule {rule_name}: {e}")
        
        return transformed_data
    
    async def enrich_data(self, data: Dict[str, Any], enrichments: List[str] = None) -> Dict[str, Any]:
        """Apply enrichment functions to data"""
        
        enriched_data = data.copy()
        
        # Apply specified enrichments or all enrichments
        enrichments_to_apply = enrichments or list(self.enrichment_functions.keys())
        
        for enrichment_name in enrichments_to_apply:
            if enrichment_name in self.enrichment_functions:
                try:
                    enrichment_result = await self.enrichment_functions[enrichment_name](enriched_data)
                    if isinstance(enrichment_result, dict):
                        enriched_data.update(enrichment_result)
                except Exception as e:
                    logger.error(f"Error applying enrichment {enrichment_name}: {e}")
        
        return enriched_data

class RedisStreamProcessor:
    """Redis Streams-based stream processor"""
    
    def __init__(self, redis_config: Dict[str, Any] = None):
        self.redis_config = redis_config or {
            'host': 'localhost',
            'port': 6379,
            'db': 6
        }
        self.redis_pool = None
        self.processors = {}
        self.processing_rules = {}
        self.data_transformer = DataTransformer()
        
    async def initialize(self):
        """Initialize Redis stream processor"""
        
        self.redis_pool = aioredis.ConnectionPool.from_url(
            f"redis://{self.redis_config['host']}:{self.redis_config['port']}/{self.redis_config['db']}",
            max_connections=50
        )
        
        # Register default transformation rules
        await self._register_default_transformations()
        
        logger.info("Redis stream processor initialized")
    
    async def _register_default_transformations(self):
        """Register default data transformation rules"""
        
        # Email normalization
        self.data_transformer.register_transformation_rule(
            "normalize_email",
            self._normalize_email
        )
        
        # Phone number formatting
        self.data_transformer.register_transformation_rule(
            "format_phone",
            self._format_phone_number
        )
        
        # Lead scoring enrichment
        self.data_transformer.register_enrichment_function(
            "calculate_lead_score",
            self._calculate_lead_score
        )
        
        # Geographic enrichment
        self.data_transformer.register_enrichment_function(
            "enrich_location",
            self._enrich_location_data
        )
    
    async def start_processing(self, stream_types: List[StreamType]):
        """Start processing specified stream types"""
        
        for stream_type in stream_types:
            processor_task = asyncio.create_task(
                self._process_stream(stream_type)
            )
            self.processors[stream_type] = processor_task
        
        logger.info(f"Started processing {len(stream_types)} stream types")
    
    async def _process_stream(self, stream_type: StreamType):
        """Process messages from a specific stream"""
        
        try:
            redis = aioredis.Redis(connection_pool=self.redis_pool)
            stream_key = f"stream:{stream_type.value}"
            consumer_group = f"processors_{stream_type.value}"
            consumer_name = f"processor_{uuid.uuid4().hex[:8]}"
            
            # Create consumer group
            try:
                await redis.xgroup_create(stream_key, consumer_group, id='0', mkstream=True)
            except Exception:
                pass  # Group already exists
            
            while True:
                try:
                    # Read messages from stream
                    messages = await redis.xreadgroup(
                        consumer_group,
                        consumer_name,
                        {stream_key: '>'},
                        count=10,
                        block=1000
                    )
                    
                    for stream, msgs in messages:
                        for msg_id, fields in msgs:
                            await self._process_message(stream_type, msg_id, fields)
                            await redis.xack(stream_key, consumer_group, msg_id)
                    
                except Exception as e:
                    logger.error(f"Error processing {stream_type.value} stream: {e}")
                    await asyncio.sleep(5)
                    
        except Exception as e:
            logger.error(f"Fatal error in {stream_type.value} stream processor: {e}")
    
    async def _process_message(self, stream_type: StreamType, msg_id: str, fields: Dict[str, Any]):
        """Process a single stream message"""
        
        try:
            # Reconstruct message
            message = StreamMessage(
                message_id=msg_id,
                stream_type=stream_type,
                data=json.loads(fields.get('data', '{}')),
                timestamp=datetime.fromisoformat(fields.get('timestamp')),
                processing_stage=ProcessingStage(fields.get('stage', 'validation')),
                metadata=json.loads(fields.get('metadata', '{}'))
            )
            
            # Process through pipeline stages
            await self._process_pipeline_stages(message)
            
        except Exception as e:
            logger.error(f"Error processing message {msg_id}: {e}")
    
    async def _process_pipeline_stages(self, message: StreamMessage):
        """Process message through all pipeline stages"""
        
        stages = [
            ProcessingStage.VALIDATION,
            ProcessingStage.ENRICHMENT,
            ProcessingStage.TRANSFORMATION,
            ProcessingStage.ROUTING,
            ProcessingStage.STORAGE
        ]
        
        for stage in stages:
            try:
                if stage == ProcessingStage.VALIDATION:
                    await self._validate_message(message)
                elif stage == ProcessingStage.ENRICHMENT:
                    await self._enrich_message(message)
                elif stage == ProcessingStage.TRANSFORMATION:
                    await self._transform_message(message)
                elif stage == ProcessingStage.ROUTING:
                    await self._route_message(message)
                elif stage == ProcessingStage.STORAGE:
                    await self._store_message(message)
                
                message.processing_stage = stage
                
            except Exception as e:
                logger.error(f"Error in {stage.value} stage for message {message.message_id}: {e}")
                break
    
    async def _validate_message(self, message: StreamMessage):
        """Validate message data"""
        
        # Basic validation
        if not message.data:
            raise ValueError("Empty message data")
        
        # Stream-specific validation
        if message.stream_type == StreamType.LEAD_EVENTS:
            required_fields = ['email', 'first_name', 'last_name']
            for field in required_fields:
                if field not in message.data:
                    logger.warning(f"Missing required field {field} in lead event")
        
        logger.debug(f"Validated message {message.message_id}")
    
    async def _enrich_message(self, message: StreamMessage):
        """Enrich message data"""
        
        # Apply enrichment functions
        enriched_data = await self.data_transformer.enrich_data(message.data)
        message.data.update(enriched_data)
        
        logger.debug(f"Enriched message {message.message_id}")
    
    async def _transform_message(self, message: StreamMessage):
        """Transform message data"""
        
        # Apply transformation rules
        transformed_data = await self.data_transformer.transform_data(message.data)
        message.data = transformed_data
        
        logger.debug(f"Transformed message {message.message_id}")
    
    async def _route_message(self, message: StreamMessage):
        """Route message to appropriate destinations"""
        
        # Determine routing based on message type and content
        destinations = self._determine_routing_destinations(message)
        
        # Send to destinations
        for destination in destinations:
            await self._send_to_destination(message, destination)
        
        logger.debug(f"Routed message {message.message_id} to {len(destinations)} destinations")
    
    async def _store_message(self, message: StreamMessage):
        """Store processed message"""
        
        # Store in appropriate storage system
        # This could be database, data lake, etc.
        
        logger.debug(f"Stored message {message.message_id}")
    
    def _determine_routing_destinations(self, message: StreamMessage) -> List[str]:
        """Determine where to route the message"""
        
        destinations = []
        
        if message.stream_type == StreamType.LEAD_EVENTS:
            destinations.extend(['crm', 'lead_scoring', 'analytics'])
        elif message.stream_type == StreamType.EMAIL_EVENTS:
            destinations.extend(['email_analytics', 'engagement_tracking'])
        elif message.stream_type == StreamType.WEB_EVENTS:
            destinations.extend(['web_analytics', 'behavior_tracking'])
        
        return destinations
    
    async def _send_to_destination(self, message: StreamMessage, destination: str):
        """Send message to a specific destination"""
        
        try:
            redis = aioredis.Redis(connection_pool=self.redis_pool)
            
            # Send to destination queue
            destination_data = {
                'message_id': message.message_id,
                'stream_type': message.stream_type.value,
                'data': json.dumps(message.data),
                'timestamp': message.timestamp.isoformat(),
                'metadata': json.dumps(message.metadata)
            }
            
            await redis.lpush(f"queue:{destination}", json.dumps(destination_data))
            
        except Exception as e:
            logger.error(f"Error sending message to {destination}: {e}")
    
    # Default transformation functions
    async def _normalize_email(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize email address"""
        
        if 'email' in data:
            data['email'] = data['email'].lower().strip()
        
        return data
    
    async def _format_phone_number(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Format phone number"""
        
        if 'phone' in data:
            # Simple phone formatting (remove non-digits)
            phone = ''.join(filter(str.isdigit, data['phone']))
            if len(phone) == 10:
                data['phone'] = f"({phone[:3]}) {phone[3:6]}-{phone[6:]}"
        
        return data
    
    async def _calculate_lead_score(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate basic lead score"""
        
        score = 0
        
        # Email domain scoring
        if 'email' in data:
            domain = data['email'].split('@')[1] if '@' in data['email'] else ''
            if domain in ['gmail.com', 'yahoo.com', 'hotmail.com']:
                score += 10
            else:
                score += 20  # Business email
        
        # Company scoring
        if 'company' in data and data['company']:
            score += 15
        
        # Phone scoring
        if 'phone' in data and data['phone']:
            score += 10
        
        return {'calculated_lead_score': score}
    
    async def _enrich_location_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Enrich location data"""
        
        enrichment = {}
        
        if 'zip_code' in data:
            # Mock location enrichment
            enrichment['state'] = 'TX'  # Would lookup from zip code
            enrichment['city'] = 'Austin'
            enrichment['timezone'] = 'America/Chicago'
        
        return enrichment

class KafkaStreamProcessor:
    """Apache Kafka-based stream processor (placeholder for future implementation)"""
    
    def __init__(self, kafka_config: Dict[str, Any] = None):
        self.kafka_config = kafka_config or {
            'bootstrap_servers': ['localhost:9092'],
            'group_id': 'lead_scoring_processors'
        }
        self.consumer = None
        self.producer = None
        
    async def initialize(self):
        """Initialize Kafka stream processor"""
        
        # This would initialize Kafka consumer and producer
        # For now, just log initialization
        logger.info("Kafka stream processor initialized (placeholder)")
    
    async def start_processing(self, topics: List[str]):
        """Start processing Kafka topics"""
        
        # This would start Kafka message processing
        logger.info(f"Started Kafka processing for topics: {topics}")

# Global instances
data_transformer = DataTransformer()
redis_stream_processor = RedisStreamProcessor()
kafka_stream_processor = KafkaStreamProcessor()