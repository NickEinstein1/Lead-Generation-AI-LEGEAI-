from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import String, Integer, JSON
from models.base import Base, TimestampMixin

class Lead(Base, TimestampMixin):
    __tablename__ = "leads"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    idempotency_key: Mapped[str | None] = mapped_column(String(64), unique=True, index=True, nullable=True)
    channel: Mapped[str | None] = mapped_column(String(32))
    source: Mapped[str | None] = mapped_column(String(64))
    product_interest: Mapped[str | None] = mapped_column(String(64))
    # Using JSON for flexible contact info and metadata
    contact_info: Mapped[dict | None] = mapped_column(JSON)
    consent: Mapped[dict | None] = mapped_column(JSON)
    metadata: Mapped[dict | None] = mapped_column(JSON)

