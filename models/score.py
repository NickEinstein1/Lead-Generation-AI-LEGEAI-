from datetime import datetime
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import Integer, Float, String, JSON, ForeignKey, DateTime, func
from models.base import Base

class Score(Base):
    __tablename__ = "scores"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    lead_id: Mapped[int] = mapped_column(Integer, ForeignKey("leads.id", ondelete="CASCADE"), index=True, nullable=False)
    score: Mapped[float | None] = mapped_column(Float)
    band: Mapped[str | None] = mapped_column(String(32))
    explanation: Mapped[dict | None] = mapped_column(JSON)
    model_version: Mapped[str | None] = mapped_column(String(64))
    features: Mapped[dict | None] = mapped_column(JSON)
    scored_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

