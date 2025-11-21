"""
Marketing Automation Database Models
"""
from sqlalchemy import Column, Integer, String, Text, DateTime, Boolean, Float, JSON, ForeignKey, Enum as SQLEnum
from sqlalchemy.orm import relationship
from datetime import datetime
import enum
from backend.database import Base


class CampaignType(str, enum.Enum):
    EMAIL = "email"
    SMS = "sms"
    MULTI_CHANNEL = "multi_channel"
    DRIP = "drip"


class CampaignStatus(str, enum.Enum):
    DRAFT = "draft"
    SCHEDULED = "scheduled"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    ARCHIVED = "archived"


class TriggerType(str, enum.Enum):
    TIME_BASED = "time_based"
    EVENT_BASED = "event_based"
    BEHAVIOR_BASED = "behavior_based"
    SCORE_BASED = "score_based"


class SegmentOperator(str, enum.Enum):
    AND = "and"
    OR = "or"


class Campaign(Base):
    __tablename__ = "marketing_campaigns"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    campaign_type = Column(SQLEnum(CampaignType), nullable=False)
    status = Column(SQLEnum(CampaignStatus), default=CampaignStatus.DRAFT)
    
    # Targeting
    segment_id = Column(Integer, ForeignKey("audience_segments.id"), nullable=True)
    target_count = Column(Integer, default=0)
    
    # Content
    template_id = Column(Integer, ForeignKey("marketing_templates.id"), nullable=True)
    subject_line = Column(String(500))
    preview_text = Column(String(500))
    
    # Scheduling
    scheduled_at = Column(DateTime, nullable=True)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    
    # A/B Testing
    is_ab_test = Column(Boolean, default=False)
    ab_test_config = Column(JSON, nullable=True)  # {"variant_a": {...}, "variant_b": {...}, "split": 50}
    
    # Automation
    is_automated = Column(Boolean, default=False)
    automation_trigger_id = Column(Integer, ForeignKey("automation_triggers.id"), nullable=True)
    
    # Metadata
    created_by = Column(String(100))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    segment = relationship("AudienceSegment", back_populates="campaigns")
    template = relationship("MarketingTemplate", back_populates="campaigns")
    automation_trigger = relationship("AutomationTrigger", back_populates="campaigns")
    analytics = relationship("CampaignAnalytics", back_populates="campaign", uselist=False)
    sends = relationship("CampaignSend", back_populates="campaign")


class AudienceSegment(Base):
    __tablename__ = "audience_segments"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    
    # Segment Criteria (stored as JSON)
    criteria = Column(JSON, nullable=False)  # {"age": {"min": 25, "max": 45}, "insurance_type": ["auto", "home"], ...}
    operator = Column(SQLEnum(SegmentOperator), default=SegmentOperator.AND)
    
    # Segment Size
    estimated_size = Column(Integer, default=0)
    last_calculated_at = Column(DateTime, nullable=True)
    
    # Metadata
    created_by = Column(String(100))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    
    # Relationships
    campaigns = relationship("Campaign", back_populates="segment")


class MarketingTemplate(Base):
    __tablename__ = "marketing_templates"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    template_type = Column(SQLEnum(CampaignType), nullable=False)
    
    # Content
    subject_line = Column(String(500))  # For email
    html_content = Column(Text)  # For email
    text_content = Column(Text)  # For SMS or plain text email
    
    # Personalization tokens available
    available_tokens = Column(JSON)  # ["{{first_name}}", "{{policy_type}}", "{{renewal_date}}", ...]
    
    # Preview
    thumbnail_url = Column(String(500), nullable=True)
    
    # Metadata
    created_by = Column(String(100))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    
    # Relationships
    campaigns = relationship("Campaign", back_populates="template")


class AutomationTrigger(Base):
    __tablename__ = "automation_triggers"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    trigger_type = Column(SQLEnum(TriggerType), nullable=False)

    # Trigger Configuration
    trigger_config = Column(JSON, nullable=False)
    # Examples:
    # TIME_BASED: {"schedule": "daily", "time": "09:00", "timezone": "UTC"}
    # EVENT_BASED: {"event": "new_lead", "conditions": {"insurance_type": "auto"}}
    # BEHAVIOR_BASED: {"action": "email_opened", "campaign_id": 123, "wait_time": "2 days"}
    # SCORE_BASED: {"score_threshold": 80, "direction": "above"}

    # Metadata
    created_by = Column(String(100))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = Column(Boolean, default=True)

    # Relationships
    campaigns = relationship("Campaign", back_populates="automation_trigger")


class CampaignAnalytics(Base):
    __tablename__ = "campaign_analytics"

    id = Column(Integer, primary_key=True, index=True)
    campaign_id = Column(Integer, ForeignKey("marketing_campaigns.id"), unique=True, nullable=False)

    # Delivery Metrics
    total_sent = Column(Integer, default=0)
    total_delivered = Column(Integer, default=0)
    total_bounced = Column(Integer, default=0)
    total_failed = Column(Integer, default=0)

    # Engagement Metrics
    total_opened = Column(Integer, default=0)
    unique_opened = Column(Integer, default=0)
    total_clicked = Column(Integer, default=0)
    unique_clicked = Column(Integer, default=0)
    total_unsubscribed = Column(Integer, default=0)
    total_spam_reports = Column(Integer, default=0)

    # Conversion Metrics
    total_conversions = Column(Integer, default=0)
    total_revenue = Column(Float, default=0.0)

    # Calculated Rates
    delivery_rate = Column(Float, default=0.0)  # delivered / sent
    open_rate = Column(Float, default=0.0)  # unique_opened / delivered
    click_rate = Column(Float, default=0.0)  # unique_clicked / delivered
    click_to_open_rate = Column(Float, default=0.0)  # unique_clicked / unique_opened
    conversion_rate = Column(Float, default=0.0)  # conversions / delivered
    unsubscribe_rate = Column(Float, default=0.0)  # unsubscribed / delivered
    roi = Column(Float, default=0.0)  # (revenue - cost) / cost

    # Cost
    campaign_cost = Column(Float, default=0.0)

    # Timestamps
    last_updated = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    campaign = relationship("Campaign", back_populates="analytics")


class CampaignSend(Base):
    __tablename__ = "campaign_sends"

    id = Column(Integer, primary_key=True, index=True)
    campaign_id = Column(Integer, ForeignKey("marketing_campaigns.id"), nullable=False)

    # Recipient Info
    customer_id = Column(Integer, nullable=True)  # Link to customer if exists
    recipient_email = Column(String(255))
    recipient_phone = Column(String(50), nullable=True)
    recipient_name = Column(String(255))

    # Send Status
    status = Column(String(50))  # sent, delivered, bounced, failed, opened, clicked, converted
    sent_at = Column(DateTime, default=datetime.utcnow)
    delivered_at = Column(DateTime, nullable=True)
    opened_at = Column(DateTime, nullable=True)
    clicked_at = Column(DateTime, nullable=True)
    converted_at = Column(DateTime, nullable=True)

    # Engagement Details
    open_count = Column(Integer, default=0)
    click_count = Column(Integer, default=0)
    links_clicked = Column(JSON, nullable=True)  # [{"url": "...", "clicked_at": "..."}]

    # A/B Test Variant
    variant = Column(String(50), nullable=True)  # "A", "B", "control"

    # Error Info
    error_message = Column(Text, nullable=True)
    bounce_type = Column(String(50), nullable=True)  # hard, soft, complaint

    # Relationships
    campaign = relationship("Campaign", back_populates="sends")

