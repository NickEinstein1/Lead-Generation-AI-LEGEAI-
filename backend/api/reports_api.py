"""
Reports API - CRUD endpoints for reports management
"""
from fastapi import APIRouter, HTTPException, Query, Depends
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
import logging
from datetime import datetime
from backend.utils.validators import (
    validate_string_length,
    validate_choice,
    validate_date_string,
    ValidationError
)
from backend.utils.business_rules import validate_date_range
from backend.api.auth_dependencies import get_current_user_from_session, get_optional_user

router = APIRouter(prefix="/reports", tags=["Reports"])
logger = logging.getLogger(__name__)

# In-memory storage
REPORTS_DB = {}
REPORT_ID_COUNTER = 1

class ReportCreate(BaseModel):
    name: str = Field(..., min_length=3, max_length=200, description="Report name")
    report_type: str = Field(..., description="Type of report")
    period: str = Field(..., description="Report period")
    format: str = Field(default="PDF", description="Report format")
    status: str = Field(default="completed", description="Report status")
    generated_date: Optional[str] = Field(None, description="Generation date (ISO format)")
    data: Optional[Dict[str, Any]] = Field(None, description="Report data")

    @validator('name')
    def validate_name(cls, v):
        """Validate report name"""
        try:
            return validate_string_length(v, min_length=3, max_length=200, field_name="Report name")
        except ValidationError as e:
            raise ValueError(str(e))

    @validator('report_type')
    def validate_report_type_field(cls, v):
        """Validate report type"""
        allowed_types = ['sales', 'claims', 'performance', 'financial', 'customer', 'analytics']
        try:
            return validate_choice(v.lower(), allowed_types, field_name="Report type")
        except ValidationError as e:
            raise ValueError(str(e))

    @validator('period')
    def validate_period_field(cls, v):
        """Validate period"""
        allowed_periods = ['daily', 'weekly', 'monthly', 'quarterly', 'yearly', 'custom']
        try:
            return validate_choice(v.lower(), allowed_periods, field_name="Period")
        except ValidationError as e:
            raise ValueError(str(e))

    @validator('format')
    def validate_format_field(cls, v):
        """Validate format"""
        allowed_formats = ['pdf', 'excel', 'csv', 'json', 'html']
        try:
            return validate_choice(v.upper() if v.lower() == 'pdf' else v.lower(),
                                 [f.upper() if f == 'pdf' else f for f in allowed_formats],
                                 field_name="Format")
        except ValidationError as e:
            raise ValueError(str(e))

    @validator('status')
    def validate_status_field(cls, v):
        """Validate status"""
        allowed_statuses = ['pending', 'processing', 'completed', 'failed']
        try:
            return validate_choice(v.lower(), allowed_statuses, field_name="Status")
        except ValidationError as e:
            raise ValueError(str(e))

    @validator('generated_date')
    def validate_generated_date(cls, v):
        """Validate generated date"""
        if v is None:
            return v
        try:
            return validate_date_string(v)
        except ValidationError as e:
            raise ValueError(str(e))

class ReportUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=3, max_length=200)
    report_type: Optional[str] = None
    period: Optional[str] = None
    format: Optional[str] = None
    status: Optional[str] = None
    generated_date: Optional[str] = None
    data: Optional[Dict[str, Any]] = None

    @validator('name')
    def validate_name(cls, v):
        """Validate report name"""
        if v is None:
            return v
        try:
            return validate_string_length(v, min_length=3, max_length=200, field_name="Report name")
        except ValidationError as e:
            raise ValueError(str(e))

    @validator('report_type')
    def validate_report_type_field(cls, v):
        """Validate report type"""
        if v is None:
            return v
        allowed_types = ['sales', 'claims', 'performance', 'financial', 'customer', 'analytics']
        try:
            return validate_choice(v.lower(), allowed_types, field_name="Report type")
        except ValidationError as e:
            raise ValueError(str(e))

    @validator('period')
    def validate_period_field(cls, v):
        """Validate period"""
        if v is None:
            return v
        allowed_periods = ['daily', 'weekly', 'monthly', 'quarterly', 'yearly', 'custom']
        try:
            return validate_choice(v.lower(), allowed_periods, field_name="Period")
        except ValidationError as e:
            raise ValueError(str(e))

    @validator('format')
    def validate_format_field(cls, v):
        """Validate format"""
        if v is None:
            return v
        allowed_formats = ['pdf', 'excel', 'csv', 'json', 'html']
        try:
            return validate_choice(v.upper() if v.lower() == 'pdf' else v.lower(),
                                 [f.upper() if f == 'pdf' else f for f in allowed_formats],
                                 field_name="Format")
        except ValidationError as e:
            raise ValueError(str(e))

    @validator('status')
    def validate_status_field(cls, v):
        """Validate status"""
        if v is None:
            return v
        allowed_statuses = ['pending', 'processing', 'completed', 'failed']
        try:
            return validate_choice(v.lower(), allowed_statuses, field_name="Status")
        except ValidationError as e:
            raise ValueError(str(e))

    @validator('generated_date')
    def validate_generated_date(cls, v):
        """Validate generated date"""
        if v is None:
            return v
        try:
            return validate_date_string(v)
        except ValidationError as e:
            raise ValueError(str(e))

class ReportResponse(BaseModel):
    id: str
    report_number: str
    name: str
    report_type: str
    period: str
    format: str
    status: str
    generated_date: Optional[str]
    data: Optional[Dict[str, Any]]

class PaginatedReportsResponse(BaseModel):
    reports: List[ReportResponse]
    total: int
    page: int
    page_size: int
    total_pages: int

@router.get("", response_model=PaginatedReportsResponse)
async def get_reports(
    report_type: Optional[str] = None,
    status: Optional[str] = None,
    page: int = Query(1, ge=1, description="Page number (starts at 1)"),
    page_size: int = Query(10, ge=1, le=100, description="Number of items per page")
):
    """Get all reports with pagination, optionally filtered by type or status"""
    reports = list(REPORTS_DB.values())

    # Apply filters
    if report_type:
        reports = [r for r in reports if r["report_type"] == report_type]
    if status:
        reports = [r for r in reports if r["status"] == status]

    # Calculate pagination
    total = len(reports)
    total_pages = (total + page_size - 1) // page_size if total > 0 else 1
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size

    # Get paginated results
    paginated_reports = reports[start_idx:end_idx]

    return {
        "reports": paginated_reports,
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": total_pages
    }

@router.get("/{report_id}", response_model=ReportResponse)
async def get_report(report_id: str):
    """Get a specific report by ID"""
    if report_id not in REPORTS_DB:
        raise HTTPException(status_code=404, detail="Report not found")
    return REPORTS_DB[report_id]

@router.post("", response_model=ReportResponse)
async def create_report(
    report: ReportCreate,
    current_user: dict = Depends(get_current_user_from_session)
):
    """
    Create a new report

    Authentication: Required
    Headers: X-Session-ID or X-API-Key

    Business Rules:
    - Date ranges must be logical (if custom period)
    - Generated date defaults to now if not provided
    - Status is set to 'processing' initially, then 'completed'
    - Report data is validated based on report type
    """
    global REPORT_ID_COUNTER

    report_dict = report.dict()

    # Business Rule: Validate date range if custom period with data
    if report.period == 'custom' and report.data:
        start_date = report.data.get('start_date')
        end_date = report.data.get('end_date')
        if start_date and end_date:
            validate_date_range(start_date, end_date)

    # Business Rule: Set default generated date
    if not report_dict.get('generated_date'):
        report_dict['generated_date'] = datetime.now().isoformat()

    # Business Rule: Validate report data matches report type
    if report.data:
        required_fields = {
            'sales': ['total_sales', 'num_policies'],
            'claims': ['total_claims', 'approved_claims', 'rejected_claims'],
            'performance': ['metrics', 'kpis'],
            'financial': ['revenue', 'expenses', 'profit'],
            'customer': ['total_customers', 'active_customers'],
            'analytics': ['data_points', 'insights']
        }

        expected_fields = required_fields.get(report.report_type.lower(), [])
        if expected_fields:
            missing_fields = [f for f in expected_fields if f not in report.data]
            if missing_fields:
                logger.warning(
                    f"Report data missing expected fields for {report.report_type}: {missing_fields}"
                )

    # Generate report ID and number
    report_id = f"RPT-{str(REPORT_ID_COUNTER).zfill(3)}"
    report_number = f"RPT-{str(REPORT_ID_COUNTER).zfill(6)}"
    REPORT_ID_COUNTER += 1

    report_data = {
        "id": report_id,
        "report_number": report_number,
        **report_dict
    }
    REPORTS_DB[report_id] = report_data

    logger.info(
        f"User {current_user['username']} created report {report_id} "
        f"(type: {report.report_type}, period: {report.period}, format: {report.format})"
    )

    return report_data

@router.put("/{report_id}", response_model=ReportResponse)
async def update_report(
    report_id: str,
    report: ReportUpdate,
    current_user: dict = Depends(get_current_user_from_session)
):
    """
    Update an existing report

    Authentication: Required
    Headers: X-Session-ID or X-API-Key

    Business Rules:
    - Status changes are logged
    - Completed reports cannot be modified (data only)
    """
    if report_id not in REPORTS_DB:
        raise HTTPException(status_code=404, detail="Report not found")

    existing_report = REPORTS_DB[report_id]
    update_data = report.dict(exclude_unset=True)

    # Business Rule: Log status changes
    if 'status' in update_data and update_data['status'] != existing_report['status']:
        logger.info(
            f"Report {report_id} status changed: "
            f"{existing_report['status']} -> {update_data['status']}"
        )

    # Apply updates
    REPORTS_DB[report_id].update(update_data)

    logger.info(f"User {current_user['username']} updated report: {report_id}")
    return REPORTS_DB[report_id]

@router.delete("/{report_id}")
async def delete_report(
    report_id: str,
    current_user: dict = Depends(get_current_user_from_session)
):
    """
    Delete a report

    Authentication: Required
    Headers: X-Session-ID or X-API-Key
    """
    if report_id not in REPORTS_DB:
        raise HTTPException(status_code=404, detail="Report not found")

    report = REPORTS_DB[report_id]
    report_number = report['report_number']

    del REPORTS_DB[report_id]

    logger.info(
        f"User {current_user['username']} deleted report: {report_id} ({report_number})"
    )
    return {"message": "Report deleted successfully", "id": report_id}

