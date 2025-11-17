"""
Integration Framework for Custom CRM and External Systems

Provides a flexible framework for integrating with:
- Custom CRM system (future)
- External APIs and services
- Third-party platforms
- Legacy systems
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import aiohttp
import jwt
from abc import ABC, abstractmethod
import hashlib
import hmac

logger = logging.getLogger(__name__)

class IntegrationType(Enum):
    CRM = "crm"
    EMAIL_PLATFORM = "email_platform"
    MARKETING_AUTOMATION = "marketing_automation"
    ANALYTICS = "analytics"
    COMMUNICATION = "communication"
    STORAGE = "storage"
    PAYMENT = "payment"
    CUSTOM = "custom"

class AuthenticationType(Enum):
    API_KEY = "api_key"
    OAUTH2 = "oauth2"
    JWT = "jwt"
    BASIC_AUTH = "basic_auth"
    WEBHOOK_SIGNATURE = "webhook_signature"
    CUSTOM = "custom"

class SyncDirection(Enum):
    INBOUND = "inbound"
    OUTBOUND = "outbound"
    BIDIRECTIONAL = "bidirectional"

@dataclass
class IntegrationConfig:
    integration_id: str
    name: str
    integration_type: IntegrationType
    auth_type: AuthenticationType
    auth_config: Dict[str, Any]
    endpoint_config: Dict[str, Any]
    sync_direction: SyncDirection
    enabled: bool = True
    rate_limit: Optional[Dict[str, Any]] = None
    retry_config: Optional[Dict[str, Any]] = None
    field_mappings: Dict[str, str] = field(default_factory=dict)
    webhook_config: Optional[Dict[str, Any]] = None

@dataclass
class SyncRecord:
    record_id: str
    integration_id: str
    external_id: Optional[str]
    internal_id: str
    record_type: str
    last_sync: datetime
    sync_status: str
    error_message: Optional[str] = None
    retry_count: int = 0

class BaseIntegration(ABC):
    """Base class for all integrations"""
    
    def __init__(self, config: IntegrationConfig):
        self.config = config
        self.session = None
        self.auth_token = None
        self.rate_limiter = None
        self.sync_records = {}
        
    async def initialize(self):
        """Initialize the integration"""
        
        self.session = aiohttp.ClientSession()
        await self._authenticate()
        
        if self.config.rate_limit:
            self._setup_rate_limiter()
        
        logger.info(f"Initialized integration: {self.config.name}")
    
    async def cleanup(self):
        """Cleanup resources"""
        
        if self.session:
            await self.session.close()
    
    @abstractmethod
    async def _authenticate(self):
        """Authenticate with the external system"""
        pass
    
    @abstractmethod
    async def sync_data(self, data: Dict[str, Any], direction: SyncDirection) -> Dict[str, Any]:
        """Sync data with the external system"""
        pass
    
    @abstractmethod
    async def validate_webhook(self, payload: bytes, signature: str) -> bool:
        """Validate webhook signature"""
        pass
    
    def _setup_rate_limiter(self):
        """Setup rate limiting"""
        # Implementation would depend on rate limiting strategy
        pass
    
    def _map_fields(self, data: Dict[str, Any], direction: SyncDirection) -> Dict[str, Any]:
        """Map fields between internal and external formats"""
        
        if direction == SyncDirection.OUTBOUND:
            # Map internal fields to external fields
            mapped_data = {}
            for internal_field, external_field in self.config.field_mappings.items():
                if internal_field in data:
                    mapped_data[external_field] = data[internal_field]
        else:
            # Map external fields to internal fields
            mapped_data = {}
            reverse_mappings = {v: k for k, v in self.config.field_mappings.items()}
            for external_field, internal_field in reverse_mappings.items():
                if external_field in data:
                    mapped_data[internal_field] = data[external_field]
        
        return mapped_data

class CRMIntegrationFramework:
    """Framework for integrating with custom CRM system"""
    
    def __init__(self):
        self.crm_config = None
        self.field_mappings = {}
        self.sync_handlers = {}
        self.webhook_handlers = {}
        
    def configure_crm_integration(self, config: Dict[str, Any]):
        """Configure CRM integration settings"""
        
        self.crm_config = config
        
        # Standard CRM field mappings
        self.field_mappings = {
            'lead_id': config.get('lead_id_field', 'id'),
            'first_name': config.get('first_name_field', 'first_name'),
            'last_name': config.get('last_name_field', 'last_name'),
            'email': config.get('email_field', 'email'),
            'phone': config.get('phone_field', 'phone'),
            'company': config.get('company_field', 'company'),
            'lead_score': config.get('lead_score_field', 'lead_score'),
            'lead_status': config.get('lead_status_field', 'status'),
            'assigned_rep': config.get('assigned_rep_field', 'assigned_to'),
            'created_date': config.get('created_date_field', 'created_at'),
            'updated_date': config.get('updated_date_field', 'updated_at')
        }
        
        logger.info("CRM integration framework configured")
    
    def register_sync_handler(self, record_type: str, handler: Callable):
        """Register handler for syncing specific record types"""
        
        self.sync_handlers[record_type] = handler
        logger.info(f"Registered sync handler for {record_type}")
    
    def register_webhook_handler(self, event_type: str, handler: Callable):
        """Register handler for CRM webhook events"""
        
        self.webhook_handlers[event_type] = handler
        logger.info(f"Registered webhook handler for {event_type}")
    
    async def sync_lead_to_crm(self, lead_data: Dict[str, Any]) -> Dict[str, Any]:
        """Sync lead data to CRM"""
        
        try:
            # Map fields to CRM format
            crm_data = self._map_to_crm_format(lead_data)
            
            # Call registered sync handler
            if 'lead' in self.sync_handlers:
                result = await self.sync_handlers['lead'](crm_data, SyncDirection.OUTBOUND)
            else:
                # Default sync implementation
                result = await self._default_crm_sync(crm_data)
            
            logger.info(f"Synced lead {lead_data.get('lead_id')} to CRM")
            return result
            
        except Exception as e:
            logger.error(f"Error syncing lead to CRM: {e}")
            raise
    
    async def sync_lead_from_crm(self, crm_lead_id: str) -> Dict[str, Any]:
        """Sync lead data from CRM"""
        
        try:
            # Call registered sync handler
            if 'lead' in self.sync_handlers:
                crm_data = await self.sync_handlers['lead'](
                    {'id': crm_lead_id}, 
                    SyncDirection.INBOUND
                )
            else:
                # Default sync implementation
                crm_data = await self._default_crm_fetch(crm_lead_id)
            
            # Map fields from CRM format
            lead_data = self._map_from_crm_format(crm_data)
            
            logger.info(f"Synced lead {crm_lead_id} from CRM")
            return lead_data
            
        except Exception as e:
            logger.error(f"Error syncing lead from CRM: {e}")
            raise
    
    async def handle_crm_webhook(self, event_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle webhook from CRM"""
        
        try:
            if event_type in self.webhook_handlers:
                result = await self.webhook_handlers[event_type](payload)
            else:
                result = await self._default_webhook_handler(event_type, payload)
            
            logger.info(f"Processed CRM webhook: {event_type}")
            return result
            
        except Exception as e:
            logger.error(f"Error handling CRM webhook {event_type}: {e}")
            raise
    
    def _map_to_crm_format(self, lead_data: Dict[str, Any]) -> Dict[str, Any]:
        """Map internal lead data to CRM format"""
        
        crm_data = {}
        
        for internal_field, crm_field in self.field_mappings.items():
            if internal_field in lead_data:
                crm_data[crm_field] = lead_data[internal_field]
        
        # Add any custom transformations
        if 'lead_score' in lead_data:
            crm_data[self.field_mappings['lead_score']] = int(lead_data['lead_score'])
        
        return crm_data
    
    def _map_from_crm_format(self, crm_data: Dict[str, Any]) -> Dict[str, Any]:
        """Map CRM data to internal format"""
        
        lead_data = {}
        reverse_mappings = {v: k for k, v in self.field_mappings.items()}
        
        for crm_field, internal_field in reverse_mappings.items():
            if crm_field in crm_data:
                lead_data[internal_field] = crm_data[crm_field]
        
        return lead_data
    
    async def _default_crm_sync(self, crm_data: Dict[str, Any]) -> Dict[str, Any]:
        """Default CRM sync implementation"""
        
        # This would be implemented based on your CRM's API
        # For now, return success
        return {
            'success': True,
            'crm_id': f"crm_{datetime.now(datetime.UTC).timestamp()}",
            'message': 'Synced to CRM successfully'
        }
    
    async def _default_crm_fetch(self, crm_lead_id: str) -> Dict[str, Any]:
        """Default CRM fetch implementation"""
        
        # This would fetch data from your CRM's API
        # For now, return mock data
        return {
            'id': crm_lead_id,
            'first_name': 'John',
            'last_name': 'Doe',
            'email': 'john.doe@example.com',
            'status': 'new'
        }
    
    async def _default_webhook_handler(self, event_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Default webhook handler"""
        
        logger.info(f"Received CRM webhook: {event_type}")
        return {'processed': True}

class ExternalAPIManager:
    """Manages integrations with external APIs"""
    
    def __init__(self):
        self.integrations = {}
        self.session = None
        
    async def initialize(self):
        """Initialize API manager"""
        
        self.session = aiohttp.ClientSession()
        logger.info("External API manager initialized")
    
    async def cleanup(self):
        """Cleanup resources"""
        
        if self.session:
            await self.session.close()
        
        for integration in self.integrations.values():
            await integration.cleanup()
    
    def register_integration(self, integration: BaseIntegration):
        """Register an external integration"""
        
        self.integrations[integration.config.integration_id] = integration
        logger.info(f"Registered integration: {integration.config.name}")
    
    async def sync_with_integration(self, integration_id: str, data: Dict[str, Any], 
                                  direction: SyncDirection) -> Dict[str, Any]:
        """Sync data with a specific integration"""
        
        if integration_id not in self.integrations:
            raise ValueError(f"Integration {integration_id} not found")
        
        integration = self.integrations[integration_id]
        
        if not integration.config.enabled:
            raise ValueError(f"Integration {integration_id} is disabled")
        
        return await integration.sync_data(data, direction)
    
    async def sync_with_all_integrations(self, data: Dict[str, Any], 
                                       direction: SyncDirection) -> Dict[str, Any]:
        """Sync data with all enabled integrations"""
        
        results = {}
        
        for integration_id, integration in self.integrations.items():
            if integration.config.enabled and integration.config.sync_direction in [direction, SyncDirection.BIDIRECTIONAL]:
                try:
                    result = await integration.sync_data(data, direction)
                    results[integration_id] = result
                except Exception as e:
                    logger.error(f"Error syncing with {integration_id}: {e}")
                    results[integration_id] = {'error': str(e)}
        
        return results

class IntegrationHub:
    """Central hub for managing all integrations"""
    
    def __init__(self):
        self.crm_framework = CRMIntegrationFramework()
        self.api_manager = ExternalAPIManager()
        self.sync_queue = asyncio.Queue()
        self.sync_workers = []
        self.running = False
        
    async def initialize(self):
        """Initialize integration hub"""
        
        await self.api_manager.initialize()
        
        # Start sync workers
        for i in range(5):  # 5 sync workers
            worker = asyncio.create_task(self._sync_worker(f"worker_{i}"))
            self.sync_workers.append(worker)
        
        self.running = True
        logger.info("Integration hub initialized")
    
    async def cleanup(self):
        """Cleanup resources"""
        
        self.running = False
        
        # Cancel sync workers
        for worker in self.sync_workers:
            worker.cancel()
        
        await self.api_manager.cleanup()
    
    async def queue_sync_operation(self, operation: Dict[str, Any]):
        """Queue a sync operation for processing"""
        
        await self.sync_queue.put(operation)
    
    async def _sync_worker(self, worker_name: str):
        """Background worker for processing sync operations"""
        
        while self.running:
            try:
                # Get sync operation from queue
                operation = await asyncio.wait_for(self.sync_queue.get(), timeout=1.0)
                
                # Process the operation
                await self._process_sync_operation(operation)
                
                # Mark task as done
                self.sync_queue.task_done()
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error in sync worker {worker_name}: {e}")
    
    async def _process_sync_operation(self, operation: Dict[str, Any]):
        """Process a single sync operation"""
        
        operation_type = operation.get('type')
        
        if operation_type == 'crm_sync':
            await self._process_crm_sync(operation)
        elif operation_type == 'external_sync':
            await self._process_external_sync(operation)
        else:
            logger.warning(f"Unknown sync operation type: {operation_type}")
    
    async def _process_crm_sync(self, operation: Dict[str, Any]):
        """Process CRM sync operation"""
        
        try:
            data = operation.get('data')
            direction = SyncDirection(operation.get('direction', 'outbound'))
            
            if direction == SyncDirection.OUTBOUND:
                result = await self.crm_framework.sync_lead_to_crm(data)
            else:
                result = await self.crm_framework.sync_lead_from_crm(data.get('crm_id'))
            
            logger.info(f"CRM sync completed: {operation.get('operation_id')}")
            
        except Exception as e:
            logger.error(f"CRM sync failed: {e}")
    
    async def _process_external_sync(self, operation: Dict[str, Any]):
        """Process external API sync operation"""
        
        try:
            integration_id = operation.get('integration_id')
            data = operation.get('data')
            direction = SyncDirection(operation.get('direction', 'outbound'))
            
            result = await self.api_manager.sync_with_integration(integration_id, data, direction)
            
            logger.info(f"External sync completed: {operation.get('operation_id')}")
            
        except Exception as e:
            logger.error(f"External sync failed: {e}")

# Global instances
integration_hub = IntegrationHub()
crm_integration_framework = integration_hub.crm_framework
external_api_manager = integration_hub.api_manager