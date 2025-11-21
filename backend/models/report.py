from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import String, Integer, JSON, Text
from typing import Optional
from backend.models.base import Base, TimestampMixin

class Report(Base, TimestampMixin):
    """Report model for generated reports"""
    __tablename__ = "reports"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    report_number: Mapped[str] = mapped_column(String(50), unique=True, nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    
    # Report details
    report_type: Mapped[str] = mapped_column(String(100), nullable=False, index=True)  # Sales, Pipeline, Performance, etc.
    period: Mapped[str] = mapped_column(String(100), nullable=False)  # This Month, Q1 2024, etc.
    format: Mapped[str] = mapped_column(String(50), default="PDF")  # PDF, Excel, CSV, JSON
    status: Mapped[str] = mapped_column(String(32), default="completed", index=True)  # pending, completed, failed
    
    # Dates
    generated_date: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    
    # Report data
    data: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    file_path: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    metadata: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

