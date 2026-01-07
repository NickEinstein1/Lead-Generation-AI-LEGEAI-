from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import String, Integer, Float, ForeignKey, JSON
from typing import Optional
from backend.models.base import Base, TimestampMixin


class Policy(Base, TimestampMixin):
    """Insurance policy model"""

    __tablename__ = "policies"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    policy_number: Mapped[str] = mapped_column(
        String(50), unique=True, nullable=False, index=True
    )
    customer_id: Mapped[Optional[int]] = mapped_column(
        Integer,
        ForeignKey("customers.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    customer_name: Mapped[str] = mapped_column(String(255), nullable=False)

    # Policy details
    policy_type: Mapped[str] = mapped_column(
        String(50), nullable=False, index=True
    )  # Auto, Home, Life, Health
    status: Mapped[str] = mapped_column(
        String(32), default="active", index=True
    )  # active, expired, cancelled
    premium: Mapped[str] = mapped_column(
        String(50), nullable=False
    )  # e.g., "$1,200/yr"
    coverage_amount: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Dates
    start_date: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    end_date: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    renewal_date: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)

    # Additional metadata (stored in DB column "metadata" but avoid reserved
    # attribute name on the SQLAlchemy declarative base)
    policy_metadata: Mapped[Optional[dict]] = mapped_column(
        "metadata", JSON, nullable=True
    )
