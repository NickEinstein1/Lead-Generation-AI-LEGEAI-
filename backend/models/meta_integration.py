"""
Meta Marketing API Integration Models

Database models for storing Meta integration data including:
- Access tokens and user info
- Connected ad accounts
- Connected pages
- Lead forms
- Lead mappings
"""

from sqlalchemy import Column, Integer, String, Text, Boolean, DateTime, ForeignKey, ARRAY
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from backend.database.connection import Base


class MetaIntegration(Base):
    """Store Meta access tokens and account information"""
    __tablename__ = "meta_integrations"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=False, index=True)  # Reference to users table
    access_token = Column(Text, nullable=False)
    token_expires_at = Column(DateTime(timezone=True), nullable=True)
    meta_user_id = Column(String(100), nullable=True)
    meta_user_name = Column(String(255), nullable=True)
    scopes = Column(ARRAY(String), nullable=True)  # List of granted permissions
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # Relationships
    ad_accounts = relationship("MetaAdAccount", back_populates="integration", cascade="all, delete-orphan")
    pages = relationship("MetaPage", back_populates="integration", cascade="all, delete-orphan")


class MetaAdAccount(Base):
    """Track connected Meta ad accounts"""
    __tablename__ = "meta_ad_accounts"

    id = Column(Integer, primary_key=True, index=True)
    integration_id = Column(Integer, ForeignKey("meta_integrations.id", ondelete="CASCADE"), nullable=False)
    ad_account_id = Column(String(100), unique=True, nullable=False, index=True)
    ad_account_name = Column(String(255), nullable=True)
    currency = Column(String(10), nullable=True)
    timezone = Column(String(100), nullable=True)
    account_status = Column(String(50), nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # Relationships
    integration = relationship("MetaIntegration", back_populates="ad_accounts")


class MetaPage(Base):
    """Track connected Facebook/Instagram pages"""
    __tablename__ = "meta_pages"

    id = Column(Integer, primary_key=True, index=True)
    integration_id = Column(Integer, ForeignKey("meta_integrations.id", ondelete="CASCADE"), nullable=False)
    page_id = Column(String(100), unique=True, nullable=False, index=True)
    page_name = Column(String(255), nullable=True)
    page_access_token = Column(Text, nullable=True)  # Page-specific token
    category = Column(String(100), nullable=True)
    followers_count = Column(Integer, nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # Relationships
    integration = relationship("MetaIntegration", back_populates="pages")
    lead_forms = relationship("MetaLeadForm", back_populates="page", cascade="all, delete-orphan")


class MetaLeadForm(Base):
    """Track lead generation forms"""
    __tablename__ = "meta_lead_forms"

    id = Column(Integer, primary_key=True, index=True)
    page_id = Column(Integer, ForeignKey("meta_pages.id", ondelete="CASCADE"), nullable=False)
    form_id = Column(String(100), unique=True, nullable=False, index=True)
    form_name = Column(String(255), nullable=True)
    status = Column(String(50), nullable=True)  # ACTIVE, PAUSED, ARCHIVED
    leads_count = Column(Integer, default=0)
    auto_sync = Column(Boolean, default=True)  # Auto-sync new leads
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # Relationships
    page = relationship("MetaPage", back_populates="lead_forms")


class MetaLeadMapping(Base):
    """Map Meta leads to LEGEAI leads"""
    __tablename__ = "meta_lead_mappings"

    id = Column(Integer, primary_key=True, index=True)
    meta_lead_id = Column(String(100), unique=True, nullable=False, index=True)
    legeai_lead_id = Column(Integer, nullable=False, index=True)  # Reference to leads table
    form_id = Column(String(100), nullable=True)
    page_id = Column(String(100), nullable=True)
    synced_at = Column(DateTime(timezone=True), server_default=func.now())


class MetaCampaign(Base):
    """Track Meta campaigns created through LEGEAI"""
    __tablename__ = "meta_campaigns"

    id = Column(Integer, primary_key=True, index=True)
    campaign_id = Column(String(100), unique=True, nullable=False, index=True)
    ad_account_id = Column(String(100), nullable=False)
    campaign_name = Column(String(255), nullable=False)
    objective = Column(String(100), nullable=True)
    status = Column(String(50), nullable=True)  # ACTIVE, PAUSED, DELETED
    daily_budget = Column(Integer, nullable=True)  # In cents
    lifetime_budget = Column(Integer, nullable=True)  # In cents
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class MetaCatalog(Base):
    """Track Meta product catalogs for dynamic ads"""
    __tablename__ = "meta_catalogs"

    id = Column(Integer, primary_key=True, index=True)
    catalog_id = Column(String(100), unique=True, nullable=False, index=True)
    business_id = Column(String(100), nullable=False)
    catalog_name = Column(String(255), nullable=False)
    vertical = Column(String(50), nullable=True)  # commerce, travel, etc.
    product_count = Column(Integer, default=0)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

