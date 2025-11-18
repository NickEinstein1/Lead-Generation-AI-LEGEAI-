"""
File Document Management API
Handles upload, download, and management of various file types (PDF, Word, Excel, CSV)
"""
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Query, Request
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, or_, desc
from backend.database.connection import session_dep
from backend.models.file_document import FileDocument, DocumentCategory, DocumentShare, DocumentActivity
from datetime import datetime, timezone
import os
import uuid
import secrets
import mimetypes
from pathlib import Path
import aiofiles

router = APIRouter(tags=["file-documents"])

USE_DB = os.getenv("USE_DB", "false").lower() == "true"
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "uploads/documents"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Allowed file types
ALLOWED_EXTENSIONS = {
    'pdf': 'application/pdf',
    'doc': 'application/msword',
    'docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
    'xls': 'application/vnd.ms-excel',
    'xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    'csv': 'text/csv',
    'txt': 'text/plain',
    'png': 'image/png',
    'jpg': 'image/jpeg',
    'jpeg': 'image/jpeg',
}

MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB


class FileDocumentResponse(BaseModel):
    id: int
    filename: str
    original_filename: str
    file_type: str
    mime_type: str
    file_size: int
    title: str
    description: Optional[str]
    category: str
    tags: Optional[Dict[str, Any]]
    lead_id: Optional[int]
    uploaded_by: Optional[int]
    status: str
    version: int
    is_public: bool
    access_level: str
    created_at: datetime
    updated_at: datetime
    last_accessed_at: Optional[datetime]


class UploadDocumentRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    category: str = Field(..., min_length=1, max_length=100)
    tags: Optional[Dict[str, Any]] = None
    lead_id: Optional[int] = None
    access_level: str = Field(default="private", pattern="^(private|team|public)$")


class DocumentCategoryResponse(BaseModel):
    id: int
    name: str
    display_name: str
    description: Optional[str]
    icon: Optional[str]
    color: Optional[str]
    sort_order: int
    is_active: bool


def validate_file(file: UploadFile) -> tuple[bool, Optional[str]]:
    """Validate uploaded file"""
    # Check file extension
    if not file.filename:
        return False, "Filename is required"
    
    ext = file.filename.rsplit('.', 1)[-1].lower() if '.' in file.filename else ''
    if ext not in ALLOWED_EXTENSIONS:
        return False, f"File type .{ext} not allowed. Allowed types: {', '.join(ALLOWED_EXTENSIONS.keys())}"
    
    return True, None


async def save_uploaded_file(file: UploadFile, category: str) -> tuple[str, str, int]:
    """
    Save uploaded file to disk
    Returns: (file_path, filename, file_size)
    """
    # Generate unique filename
    ext = file.filename.rsplit('.', 1)[-1].lower() if '.' in file.filename else ''
    unique_filename = f"{uuid.uuid4()}.{ext}"
    
    # Create category subdirectory
    category_dir = UPLOAD_DIR / category
    category_dir.mkdir(parents=True, exist_ok=True)
    
    file_path = category_dir / unique_filename
    
    # Save file
    file_size = 0
    async with aiofiles.open(file_path, 'wb') as f:
        while chunk := await file.read(8192):  # Read in 8KB chunks
            file_size += len(chunk)
            if file_size > MAX_FILE_SIZE:
                # Clean up partial file
                await f.close()
                file_path.unlink(missing_ok=True)
                raise HTTPException(status_code=413, detail=f"File too large. Maximum size: {MAX_FILE_SIZE / 1024 / 1024}MB")
            await f.write(chunk)
    
    return str(file_path), unique_filename, file_size


async def log_activity(
    session: AsyncSession,
    document_id: int,
    user_id: Optional[int],
    activity_type: str,
    details: Optional[Dict[str, Any]] = None,
    request: Optional[Request] = None
):
    """Log document activity"""
    if not USE_DB:
        return

    activity = DocumentActivity(
        document_id=document_id,
        user_id=user_id,
        activity_type=activity_type,
        activity_details=details,
        ip_address=request.client.host if request else None,
        user_agent=request.headers.get("user-agent") if request else None
    )
    session.add(activity)
    await session.commit()


# ============================================================================
# API Endpoints
# ============================================================================

@router.get("/categories", response_model=List[DocumentCategoryResponse])
async def list_categories(
    session: AsyncSession = Depends(session_dep),
    active_only: bool = Query(True)
):
    """Get all document categories"""
    if not USE_DB:
        # Return default categories for in-memory mode
        return [
            {"id": 1, "name": "policies", "display_name": "Insurance Policies", "description": "Insurance policy documents", "icon": "📋", "color": "#3B82F6", "sort_order": 1, "is_active": True},
            {"id": 2, "name": "claims", "display_name": "Claims", "description": "Insurance claim documents", "icon": "📝", "color": "#EF4444", "sort_order": 2, "is_active": True},
            {"id": 3, "name": "customer_data", "display_name": "Customer Data", "description": "Customer information and data files", "icon": "👥", "color": "#10B981", "sort_order": 3, "is_active": True},
            {"id": 4, "name": "contracts", "display_name": "Contracts", "description": "Legal contracts and agreements", "icon": "📄", "color": "#8B5CF6", "sort_order": 4, "is_active": True},
            {"id": 5, "name": "reports", "display_name": "Reports", "description": "Business reports and analytics", "icon": "📊", "color": "#F59E0B", "sort_order": 5, "is_active": True},
            {"id": 6, "name": "correspondence", "display_name": "Correspondence", "description": "Email and letter correspondence", "icon": "✉️", "color": "#6366F1", "sort_order": 6, "is_active": True},
            {"id": 7, "name": "financial", "display_name": "Financial Documents", "description": "Financial statements and records", "icon": "💰", "color": "#059669", "sort_order": 7, "is_active": True},
            {"id": 8, "name": "other", "display_name": "Other", "description": "Miscellaneous documents", "icon": "📁", "color": "#6B7280", "sort_order": 99, "is_active": True},
        ]

    query = select(DocumentCategory)
    if active_only:
        query = query.where(DocumentCategory.is_active == True)
    query = query.order_by(DocumentCategory.sort_order, DocumentCategory.display_name)

    result = await session.execute(query)
    categories = result.scalars().all()
    return categories


@router.post("/upload", response_model=FileDocumentResponse)
async def upload_document(
    file: UploadFile = File(...),
    title: str = Form(...),
    category: str = Form(...),
    description: Optional[str] = Form(None),
    tags: Optional[str] = Form(None),  # JSON string
    lead_id: Optional[int] = Form(None),
    access_level: str = Form("private"),
    user_id: Optional[int] = Form(None),
    session: AsyncSession = Depends(session_dep),
    request: Request = None
):
    """Upload a new document"""

    # Validate file
    is_valid, error_msg = validate_file(file)
    if not is_valid:
        raise HTTPException(status_code=400, detail=error_msg)

    # Save file to disk
    try:
        file_path, filename, file_size = await save_uploaded_file(file, category)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

    # Get file type and mime type
    ext = file.filename.rsplit('.', 1)[-1].lower() if '.' in file.filename else ''
    mime_type = ALLOWED_EXTENSIONS.get(ext, 'application/octet-stream')

    # Parse tags if provided
    import json
    parsed_tags = None
    if tags:
        try:
            parsed_tags = json.loads(tags)
        except:
            pass

    if not USE_DB:
        # In-memory mode - return mock response
        return {
            "id": 1,
            "filename": filename,
            "original_filename": file.filename,
            "file_type": ext,
            "mime_type": mime_type,
            "file_size": file_size,
            "title": title,
            "description": description,
            "category": category,
            "tags": parsed_tags,
            "lead_id": lead_id,
            "uploaded_by": user_id,
            "status": "active",
            "version": 1,
            "is_public": False,
            "access_level": access_level,
            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc),
            "last_accessed_at": None
        }

    # Create database record
    doc = FileDocument(
        filename=filename,
        original_filename=file.filename,
        file_type=ext,
        mime_type=mime_type,
        file_size=file_size,
        file_path=file_path,
        title=title,
        description=description,
        category=category,
        tags=parsed_tags,
        lead_id=lead_id,
        uploaded_by=user_id,
        status="active",
        version=1,
        is_public=(access_level == "public"),
        access_level=access_level
    )

    session.add(doc)
    await session.commit()
    await session.refresh(doc)

    # Log activity
    await log_activity(session, doc.id, user_id, "upload", {"filename": file.filename}, request)

    return doc


@router.get("/documents", response_model=Dict[str, Any])
async def list_documents(
    session: AsyncSession = Depends(session_dep),
    category: Optional[str] = Query(None),
    status: str = Query("active"),
    lead_id: Optional[int] = Query(None),
    search: Optional[str] = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    sort_by: str = Query("created_at"),
    sort_order: str = Query("desc")
):
    """List documents with filtering and pagination"""

    if not USE_DB:
        # Return mock data for in-memory mode
        return {
            "items": [],
            "total": 0,
            "page": page,
            "page_size": page_size,
            "total_pages": 0
        }

    # Build query
    query = select(FileDocument).where(FileDocument.status == status)

    if category:
        query = query.where(FileDocument.category == category)

    if lead_id:
        query = query.where(FileDocument.lead_id == lead_id)

    if search:
        search_pattern = f"%{search}%"
        query = query.where(
            or_(
                FileDocument.title.ilike(search_pattern),
                FileDocument.description.ilike(search_pattern),
                FileDocument.original_filename.ilike(search_pattern)
            )
        )

    # Get total count
    count_query = select(func.count()).select_from(query.subquery())
    total = (await session.execute(count_query)).scalar()

    # Apply sorting
    sort_column = getattr(FileDocument, sort_by, FileDocument.created_at)
    if sort_order == "desc":
        query = query.order_by(desc(sort_column))
    else:
        query = query.order_by(sort_column)

    # Apply pagination
    offset = (page - 1) * page_size
    query = query.offset(offset).limit(page_size)

    result = await session.execute(query)
    documents = result.scalars().all()

    return {
        "items": documents,
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": (total + page_size - 1) // page_size
    }


@router.get("/documents/{document_id}", response_model=FileDocumentResponse)
async def get_document(
    document_id: int,
    session: AsyncSession = Depends(session_dep),
    user_id: Optional[int] = Query(None),
    request: Request = None
):
    """Get document details"""

    if not USE_DB:
        raise HTTPException(status_code=404, detail="Document not found")

    result = await session.execute(
        select(FileDocument).where(FileDocument.id == document_id)
    )
    doc = result.scalar_one_or_none()

    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    # Update last accessed time
    doc.last_accessed_at = datetime.now(timezone.utc)
    await session.commit()

    # Log activity
    await log_activity(session, doc.id, user_id, "view", None, request)

    return doc


@router.get("/documents/{document_id}/download")
async def download_document(
    document_id: int,
    session: AsyncSession = Depends(session_dep),
    user_id: Optional[int] = Query(None),
    request: Request = None
):
    """Download a document"""

    if not USE_DB:
        raise HTTPException(status_code=404, detail="Document not found")

    result = await session.execute(
        select(FileDocument).where(FileDocument.id == document_id)
    )
    doc = result.scalar_one_or_none()

    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    file_path = Path(doc.file_path)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found on disk")

    # Update last accessed time
    doc.last_accessed_at = datetime.now(timezone.utc)
    await session.commit()

    # Log activity
    await log_activity(session, doc.id, user_id, "download", None, request)

    return FileResponse(
        path=file_path,
        filename=doc.original_filename,
        media_type=doc.mime_type
    )


@router.delete("/documents/{document_id}")
async def delete_document(
    document_id: int,
    session: AsyncSession = Depends(session_dep),
    user_id: Optional[int] = Query(None),
    permanent: bool = Query(False),
    request: Request = None
):
    """Delete a document (soft delete by default)"""

    if not USE_DB:
        return {"status": "success", "message": "Document deleted"}

    result = await session.execute(
        select(FileDocument).where(FileDocument.id == document_id)
    )
    doc = result.scalar_one_or_none()

    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    if permanent:
        # Permanent delete - remove file and database record
        file_path = Path(doc.file_path)
        if file_path.exists():
            file_path.unlink()

        await session.delete(doc)
        await log_activity(session, doc.id, user_id, "delete_permanent", None, request)
    else:
        # Soft delete
        doc.status = "deleted"
        doc.deleted_at = datetime.now(timezone.utc)
        await log_activity(session, doc.id, user_id, "delete", None, request)

    await session.commit()

    return {"status": "success", "message": "Document deleted"}


@router.get("/stats")
async def get_document_stats(
    session: AsyncSession = Depends(session_dep)
):
    """Get document statistics"""

    if not USE_DB:
        return {
            "total_documents": 0,
            "total_size": 0,
            "by_category": {},
            "by_type": {},
            "recent_uploads": 0
        }

    # Total documents
    total = (await session.execute(
        select(func.count()).select_from(FileDocument).where(FileDocument.status == "active")
    )).scalar()

    # Total size
    total_size = (await session.execute(
        select(func.sum(FileDocument.file_size)).where(FileDocument.status == "active")
    )).scalar() or 0

    # By category
    category_stats = (await session.execute(
        select(FileDocument.category, func.count(), func.sum(FileDocument.file_size))
        .where(FileDocument.status == "active")
        .group_by(FileDocument.category)
    )).all()

    by_category = {cat: {"count": count, "size": size or 0} for cat, count, size in category_stats}

    # By file type
    type_stats = (await session.execute(
        select(FileDocument.file_type, func.count())
        .where(FileDocument.status == "active")
        .group_by(FileDocument.file_type)
    )).all()

    by_type = {ftype: count for ftype, count in type_stats}

    # Recent uploads (last 7 days)
    from datetime import timedelta, timezone
    seven_days_ago = datetime.now(timezone.utc) - timedelta(days=7)
    recent = (await session.execute(
        select(func.count()).select_from(FileDocument)
        .where(and_(FileDocument.status == "active", FileDocument.created_at >= seven_days_ago))
    )).scalar()

    return {
        "total_documents": total,
        "total_size": total_size,
        "by_category": by_category,
        "by_type": by_type,
        "recent_uploads": recent
    }

