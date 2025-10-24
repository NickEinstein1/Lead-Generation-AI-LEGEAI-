from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import Integer, String, JSON, ForeignKey, DateTime, func
from models.base import Base, TimestampMixin


class Document(Base, TimestampMixin):
    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    lead_id: Mapped[int] = mapped_column(Integer, ForeignKey("leads.id", ondelete="CASCADE"), index=True, nullable=False)
    title: Mapped[str] = mapped_column(String(255))
    status: Mapped[str] = mapped_column(String(32), default="pending")  # draft|pending|signed|declined|voided
    provider: Mapped[str | None] = mapped_column(String(64), default="internal")
    provider_request_id: Mapped[str | None] = mapped_column(String(255))
    signing_url: Mapped[str | None] = mapped_column(String(512))
    metadata: Mapped[dict | None] = mapped_column(JSON)
    signed_at: Mapped[DateTime | None] = mapped_column(DateTime(timezone=True))

