from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import String, Integer, Float, Boolean, JSON
from typing import Optional
from backend.models.base import Base, TimestampMixin

class Customer(Base, TimestampMixin):
    """Customer model for insurance customers"""
    __tablename__ = "customers"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    email: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    phone: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="active", index=True)  # active, inactive
    
    # Customer details
    policies_count: Mapped[int] = mapped_column(Integer, default=0)
    total_value: Mapped[float] = mapped_column(Float, default=0.0)
    join_date: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    last_active: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    reason: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)  # For inactive customers
    
    # Additional metadata
    metadata: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

