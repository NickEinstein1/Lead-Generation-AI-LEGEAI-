"""
Meta Lead Sync Module

Syncs leads from Meta Lead Ads to LEGEAI database
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from backend.models.lead import Lead
from .client import MetaAPIClient

logger = logging.getLogger(__name__)


class MetaLeadSync:
    """Syncs leads from Meta Lead Ads to LEGEAI database"""
    
    def __init__(self, meta_client: MetaAPIClient):
        """
        Initialize lead sync
        
        Args:
            meta_client: Authenticated Meta API client
        """
        self.meta_client = meta_client
    
    async def sync_lead(
        self,
        session: AsyncSession,
        lead_id: str,
        form_id: str,
        page_id: str
    ) -> Optional[str]:
        """
        Sync a single lead from Meta to LEGEAI database
        
        Args:
            session: Database session
            lead_id: Meta lead ID
            form_id: Meta form ID
            page_id: Meta page ID
            
        Returns:
            LEGEAI lead ID if created, None if duplicate
        """
        try:
            # Get lead details from Meta
            lead_data = await self.meta_client.get_lead_details(lead_id)
            
            # Create idempotency key from Meta lead ID
            idempotency_key = f"meta_lead_{lead_id}"
            
            # Check if lead already exists
            existing = (
                await session.execute(
                    select(Lead).where(Lead.idempotency_key == idempotency_key)
                )
            ).scalar_one_or_none()
            
            if existing:
                logger.info(f"Lead {lead_id} already exists in database")
                return None
            
            # Parse Meta lead data
            field_data = lead_data.get("field_data", [])
            contact_info = self._parse_contact_info(field_data)
            attributes = self._parse_attributes(field_data)
            
            # Create new lead
            new_lead = Lead(
                idempotency_key=idempotency_key,
                channel="facebook_lead_ads",
                source=f"meta_form_{form_id}",
                product_interest=attributes.get("product_interest", "insurance"),
                contact_info=contact_info,
                consent=True,  # Meta leads have implicit consent
                lead_metadata={
                    "meta_lead_id": lead_id,
                    "meta_form_id": form_id,
                    "meta_page_id": page_id,
                    "meta_ad_id": lead_data.get("ad_id"),
                    "meta_adset_id": lead_data.get("adset_id"),
                    "meta_campaign_id": lead_data.get("campaign_id"),
                    "attributes": attributes,
                    "created_time": lead_data.get("created_time"),
                }
            )
            
            session.add(new_lead)
            await session.flush()
            await session.commit()
            
            logger.info(f"Successfully synced Meta lead {lead_id} to LEGEAI lead {new_lead.id}")
            return str(new_lead.id)
            
        except Exception as e:
            logger.error(f"Error syncing lead {lead_id}: {e}")
            await session.rollback()
            raise
    
    def _parse_contact_info(self, field_data: List[Dict[str, Any]]) -> Dict[str, str]:
        """Parse contact information from Meta field data"""
        contact_info = {}
        
        field_mapping = {
            "email": "email",
            "phone_number": "phone",
            "full_name": "full_name",
            "first_name": "first_name",
            "last_name": "last_name",
            "street_address": "address",
            "city": "city",
            "state": "state",
            "zip_code": "zip_code",
            "country": "country",
        }
        
        for field in field_data:
            field_name = field.get("name", "").lower()
            field_value = field.get("values", [None])[0]
            
            if field_name in field_mapping and field_value:
                contact_info[field_mapping[field_name]] = field_value
        
        return contact_info
    
    def _parse_attributes(self, field_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Parse custom attributes from Meta field data"""
        attributes = {}
        
        # Standard fields that aren't contact info
        custom_fields = [
            "product_interest",
            "insurance_type",
            "coverage_amount",
            "current_provider",
            "age",
            "occupation",
            "annual_income",
        ]
        
        for field in field_data:
            field_name = field.get("name", "").lower()
            field_value = field.get("values", [None])[0]
            
            if field_name in custom_fields and field_value:
                attributes[field_name] = field_value
        
        return attributes
    
    async def sync_form_leads(
        self,
        session: AsyncSession,
        form_id: str,
        page_id: str,
        limit: int = 100
    ) -> Dict[str, Any]:
        """
        Sync all leads from a specific form
        
        Args:
            session: Database session
            form_id: Meta form ID
            page_id: Meta page ID
            limit: Maximum number of leads to sync
            
        Returns:
            Summary of sync operation
        """
        try:
            leads = await self.meta_client.get_leads_from_form(form_id, limit=limit)
            
            synced_count = 0
            duplicate_count = 0
            error_count = 0
            
            for lead in leads:
                lead_id = lead.get("id")
                try:
                    result = await self.sync_lead(session, lead_id, form_id, page_id)
                    if result:
                        synced_count += 1
                    else:
                        duplicate_count += 1
                except Exception as e:
                    logger.error(f"Error syncing lead {lead_id}: {e}")
                    error_count += 1
            
            return {
                "total_leads": len(leads),
                "synced": synced_count,
                "duplicates": duplicate_count,
                "errors": error_count,
            }
            
        except Exception as e:
            logger.error(f"Error syncing leads from form {form_id}: {e}")
            raise

