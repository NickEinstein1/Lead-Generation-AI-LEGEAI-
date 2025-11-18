"""
File Document Management Models
Separate from e-signature documents - this is for general file management
"""
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy import Integer, String, JSON, ForeignKey, DateTime, BigInteger, Text, Index
from backend.models.base import Base, TimestampMixin
from typing import Optional
from datetime import datetime


class FileDocument(Base, TimestampMixin):
    """
    General file/document storage model
    Supports: PDF, Word (docx), Excel (xlsx, xls), CSV, and other file types
    """
    __tablename__ = "file_documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    
    # File Information
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    original_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    file_type: Mapped[str] = mapped_column(String(50), nullable=False)  # pdf, docx, xlsx, csv, etc.
    mime_type: Mapped[str] = mapped_column(String(100), nullable=False)
    file_size: Mapped[int] = mapped_column(BigInteger, nullable=False)  # in bytes
    file_path: Mapped[str] = mapped_column(String(512), nullable=False)  # storage path
    
    # Document Metadata
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    category: Mapped[str] = mapped_column(String(100), nullable=False, index=True)  # policy, claim, customer_data, etc.
    tags: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)  # flexible tagging system
    
    # Relationships
    lead_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("leads.id", ondelete="SET NULL"), index=True, nullable=True)
    uploaded_by: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("users.id", ondelete="SET NULL"), index=True, nullable=True)
    
    # Status and Versioning
    status: Mapped[str] = mapped_column(String(32), default="active", index=True)  # active, archived, deleted
    version: Mapped[int] = mapped_column(Integer, default=1)
    parent_document_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("file_documents.id", ondelete="SET NULL"), nullable=True)
    
    # Access Control
    is_public: Mapped[bool] = mapped_column(default=False)
    access_level: Mapped[str] = mapped_column(String(32), default="private")  # private, team, public

    # Additional Metadata
    extra_metadata: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    
    # Timestamps for document lifecycle
    last_accessed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    archived_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    deleted_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        Index('ix_file_documents_category_status', 'category', 'status'),
        Index('ix_file_documents_file_type', 'file_type'),
        Index('ix_file_documents_uploaded_by', 'uploaded_by'),
    )


class DocumentCategory(Base, TimestampMixin):
    """
    Document categories for organization
    """
    __tablename__ = "document_categories"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False, unique=True)
    display_name: Mapped[str] = mapped_column(String(100), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    icon: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)  # emoji or icon name
    color: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)  # hex color
    parent_category_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("document_categories.id", ondelete="SET NULL"), nullable=True)
    sort_order: Mapped[int] = mapped_column(Integer, default=0)
    is_active: Mapped[bool] = mapped_column(default=True)
    extra_metadata: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)


class DocumentShare(Base, TimestampMixin):
    """
    Document sharing and permissions
    """
    __tablename__ = "document_shares"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    document_id: Mapped[int] = mapped_column(Integer, ForeignKey("file_documents.id", ondelete="CASCADE"), nullable=False, index=True)
    shared_with_user_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=True, index=True)
    shared_by_user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    
    # Permissions
    can_view: Mapped[bool] = mapped_column(default=True)
    can_download: Mapped[bool] = mapped_column(default=True)
    can_edit: Mapped[bool] = mapped_column(default=False)
    can_delete: Mapped[bool] = mapped_column(default=False)
    can_share: Mapped[bool] = mapped_column(default=False)
    
    # Share link (for public sharing)
    share_token: Mapped[Optional[str]] = mapped_column(String(255), unique=True, nullable=True)
    expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    
    # Tracking
    access_count: Mapped[int] = mapped_column(Integer, default=0)
    last_accessed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    
    __table_args__ = (
        Index('ix_document_shares_token', 'share_token'),
        Index('ix_document_shares_document_user', 'document_id', 'shared_with_user_id'),
    )


class DocumentActivity(Base, TimestampMixin):
    """
    Audit trail for document activities
    """
    __tablename__ = "document_activities"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    document_id: Mapped[int] = mapped_column(Integer, ForeignKey("file_documents.id", ondelete="CASCADE"), nullable=False, index=True)
    user_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)
    
    activity_type: Mapped[str] = mapped_column(String(50), nullable=False, index=True)  # upload, view, download, edit, delete, share
    activity_details: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    ip_address: Mapped[Optional[str]] = mapped_column(String(45), nullable=True)
    user_agent: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    
    __table_args__ = (
        Index('ix_document_activities_type_created', 'activity_type', 'created_at'),
        Index('ix_document_activities_document_created', 'document_id', 'created_at'),
    )

