from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import String, Integer, Boolean, DateTime, JSON
from typing import Optional
from models.base import Base, TimestampMixin

class User(Base, TimestampMixin):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    username: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    role: Mapped[str] = mapped_column(String(32), default="agent")
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    permissions: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    # Security/Account state
    failed_login_attempts: Mapped[int] = mapped_column(Integer, default=0)
    locked_until: Mapped[Optional[DateTime]] = mapped_column(DateTime(timezone=True), nullable=True)
    last_login: Mapped[Optional[DateTime]] = mapped_column(DateTime(timezone=True), nullable=True)

    # Optional MFA fields
    mfa_enabled: Mapped[bool] = mapped_column(Boolean, default=False)
    mfa_secret: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)

