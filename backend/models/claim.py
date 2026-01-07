from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import String, Integer, Float, ForeignKey, JSON
from typing import Optional
from backend.models.base import Base, TimestampMixin

class Claim(Base, TimestampMixin):
    """Insurance claim model"""
    __tablename__ = "claims"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    claim_number: Mapped[str] = mapped_column(String(50), unique=True, nullable=False, index=True)
    policy_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("policies.id", ondelete="SET NULL"), nullable=True, index=True)
    policy_number: Mapped[str] = mapped_column(String(50), nullable=False)
    customer_name: Mapped[str] = mapped_column(String(255), nullable=False)
    
    # Claim details
    claim_type: Mapped[str] = mapped_column(String(50), nullable=False, index=True)  # Auto, Home, Life, Health
    amount: Mapped[str] = mapped_column(String(50), nullable=False)  # e.g., "$5,000"
    status: Mapped[str] = mapped_column(String(32), default="pending", index=True)  # pending, approved, rejected
    
    # Dates
    claim_date: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    due_date: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    processed_date: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    
    # Additional details
    description: Mapped[Optional[str]] = mapped_column(String(1000), nullable=True)

    # Additional metadata (stored in DB column "metadata" but avoid reserved
    # attribute name on the SQLAlchemy declarative base)
    claim_metadata: Mapped[Optional[dict]] = mapped_column(
        "metadata", JSON, nullable=True
    )

