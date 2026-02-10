"""
Meta Webhooks Handler

Handles real-time webhooks from Meta for:
- Lead Ads (new leads)
- Page events
- Ad account updates
"""

import os
import logging
import hmac
import hashlib
from typing import Dict, Any, Optional, Callable
from fastapi import Request, HTTPException

logger = logging.getLogger(__name__)

# Webhook Configuration
META_APP_SECRET = os.getenv("META_APP_SECRET", "")
META_VERIFY_TOKEN = os.getenv("META_VERIFY_TOKEN", "legeai_meta_webhook_2024")


class MetaWebhookHandler:
    """Handles Meta webhook verification and processing"""
    
    def __init__(
        self,
        app_secret: Optional[str] = None,
        verify_token: Optional[str] = None
    ):
        """
        Initialize webhook handler
        
        Args:
            app_secret: Meta App Secret for signature verification
            verify_token: Verification token for webhook setup
        """
        self.app_secret = app_secret or META_APP_SECRET
        self.verify_token = verify_token or META_VERIFY_TOKEN
        
        # Event handlers
        self.handlers: Dict[str, Callable] = {}
    
    def verify_webhook(self, mode: str, token: str, challenge: str) -> str:
        """
        Verify webhook subscription (GET request from Meta)
        
        Args:
            mode: Should be 'subscribe'
            token: Verification token
            challenge: Challenge string to echo back
            
        Returns:
            Challenge string if verification succeeds
            
        Raises:
            HTTPException: If verification fails
        """
        if mode == "subscribe" and token == self.verify_token:
            logger.info("Meta webhook verified successfully")
            return challenge
        else:
            logger.error(f"Webhook verification failed: mode={mode}, token={token}")
            raise HTTPException(status_code=403, detail="Verification failed")
    
    def verify_signature(self, payload: bytes, signature: str) -> bool:
        """
        Verify webhook signature (POST request from Meta)
        
        Args:
            payload: Raw request body
            signature: X-Hub-Signature-256 header value
            
        Returns:
            True if signature is valid
        """
        if not signature:
            logger.warning("No signature provided")
            return False
        
        # Signature format: sha256=<hash>
        if not signature.startswith("sha256="):
            logger.warning("Invalid signature format")
            return False
        
        expected_signature = signature.split("=")[1]
        
        # Calculate HMAC
        mac = hmac.new(
            self.app_secret.encode(),
            msg=payload,
            digestmod=hashlib.sha256
        )
        calculated_signature = mac.hexdigest()
        
        is_valid = hmac.compare_digest(calculated_signature, expected_signature)
        
        if not is_valid:
            logger.warning("Signature verification failed")
        
        return is_valid
    
    def register_handler(self, event_type: str, handler: Callable):
        """
        Register a handler for a specific event type
        
        Args:
            event_type: Event type (e.g., 'leadgen', 'page', 'ad_account')
            handler: Async function to handle the event
        """
        self.handlers[event_type] = handler
        logger.info(f"Registered handler for event type: {event_type}")
    
    async def process_webhook(self, data: Dict[str, Any]):
        """
        Process incoming webhook data
        
        Args:
            data: Webhook payload
        """
        object_type = data.get("object")
        entries = data.get("entry", [])
        
        logger.info(f"Processing webhook for object type: {object_type}")
        
        for entry in entries:
            entry_id = entry.get("id")
            changes = entry.get("changes", [])
            
            for change in changes:
                field = change.get("field")
                value = change.get("value")
                
                logger.info(f"Processing change: field={field}, entry_id={entry_id}")
                
                # Route to appropriate handler
                if field == "leadgen":
                    await self._handle_leadgen(entry_id, value)
                elif field == "page":
                    await self._handle_page_event(entry_id, value)
                elif field in self.handlers:
                    await self.handlers[field](entry_id, value)
                else:
                    logger.warning(f"No handler for field: {field}")
    
    async def _handle_leadgen(self, page_id: str, value: Dict[str, Any]):
        """Handle lead generation event"""
        lead_id = value.get("leadgen_id")
        form_id = value.get("form_id")
        created_time = value.get("created_time")
        
        logger.info(f"New lead: lead_id={lead_id}, form_id={form_id}, page_id={page_id}")
        
        # Call registered handler if exists
        if "leadgen" in self.handlers:
            await self.handlers["leadgen"](page_id, {
                "lead_id": lead_id,
                "form_id": form_id,
                "created_time": created_time,
            })
    
    async def _handle_page_event(self, page_id: str, value: Dict[str, Any]):
        """Handle page event"""
        logger.info(f"Page event: page_id={page_id}, value={value}")
        
        if "page" in self.handlers:
            await self.handlers["page"](page_id, value)

