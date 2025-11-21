from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import String, Integer, ForeignKey, JSON, Text
from typing import Optional
from backend.models.base import Base, TimestampMixin

class Communication(Base, TimestampMixin):
    """Customer communication model"""
    __tablename__ = "communications"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    customer_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("customers.id", ondelete="SET NULL"), nullable=True, index=True)
    customer_name: Mapped[str] = mapped_column(String(255), nullable=False)
    
    # Communication details
    comm_type: Mapped[str] = mapped_column(String(50), nullable=False, index=True)  # email, sms, call
    channel: Mapped[str] = mapped_column(String(50), nullable=False)  # Email, SMS, Phone
    subject: Mapped[str] = mapped_column(String(500), nullable=False)
    status: Mapped[str] = mapped_column(String(32), default="sent", index=True)  # sent, delivered, completed, pending
    
    # Dates
    comm_date: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    
    # Content
    content: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    metadata: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

