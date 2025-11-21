"""
Scheduler API - Meeting scheduling with Zoom, Google Meet, Microsoft Teams integration
"""
from fastapi import APIRouter, HTTPException, Query, Depends
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
import logging
from datetime import datetime, timedelta
from backend.api.auth_dependencies import get_current_user_from_session, get_optional_user
from backend.utils.validators import (
    validate_email,
    validate_string_length,
    validate_choice,
    validate_date_string,
    ValidationError
)

router = APIRouter(prefix="/scheduler", tags=["Scheduler"])
logger = logging.getLogger(__name__)

# In-memory storage
MEETINGS_DB = {}
MEETING_ID_COUNTER = 1

# Meeting platform configurations
MEETING_PLATFORMS = {
    "zoom": {
        "name": "Zoom",
        "icon": "🎥",
        "url_pattern": "https://zoom.us/j/{meeting_id}",
        "requires_auth": True
    },
    "google_meet": {
        "name": "Google Meet",
        "icon": "📹",
        "url_pattern": "https://meet.google.com/{meeting_id}",
        "requires_auth": True
    },
    "microsoft_teams": {
        "name": "Microsoft Teams",
        "icon": "👥",
        "url_pattern": "https://teams.microsoft.com/l/meetup-join/{meeting_id}",
        "requires_auth": True
    },
    "webex": {
        "name": "Cisco Webex",
        "icon": "💼",
        "url_pattern": "https://webex.com/meet/{meeting_id}",
        "requires_auth": True
    },
    "phone": {
        "name": "Phone Call",
        "icon": "📞",
        "url_pattern": "tel:{phone_number}",
        "requires_auth": False
    },
    "in_person": {
        "name": "In Person",
        "icon": "🏢",
        "url_pattern": None,
        "requires_auth": False
    }
}

class MeetingCreate(BaseModel):
    title: str = Field(..., min_length=3, max_length=200, description="Meeting title")
    description: Optional[str] = Field(None, max_length=1000, description="Meeting description")
    platform: str = Field(..., description="Meeting platform (zoom, google_meet, microsoft_teams, webex, phone, in_person)")
    start_time: str = Field(..., description="Meeting start time (ISO format)")
    end_time: str = Field(..., description="Meeting end time (ISO format)")
    attendees: List[str] = Field(default_factory=list, description="List of attendee emails")
    organizer_email: str = Field(..., description="Organizer email")
    location: Optional[str] = Field(None, description="Physical location (for in-person meetings)")
    meeting_url: Optional[str] = Field(None, description="Meeting URL (auto-generated for online platforms)")
    reminder_minutes: Optional[int] = Field(15, description="Reminder time in minutes before meeting")
    notes: Optional[str] = Field(None, description="Additional notes")
    customer_name: Optional[str] = Field(None, description="Associated customer name")
    lead_id: Optional[str] = Field(None, description="Associated lead ID")
    
    @validator('platform')
    def validate_platform(cls, v):
        return validate_choice(v, list(MEETING_PLATFORMS.keys()), "Platform")
    
    @validator('organizer_email')
    def validate_organizer_email(cls, v):
        return validate_email(v)
    
    @validator('attendees')
    def validate_attendee_emails(cls, v):
        return [validate_email(email) for email in v]
    
    @validator('start_time', 'end_time')
    def validate_datetime(cls, v):
        return validate_date_string(v)
    
    @validator('title')
    def validate_title_length(cls, v):
        return validate_string_length(v, min_length=3, max_length=200, field_name="Title")


class MeetingUpdate(BaseModel):
    title: Optional[str] = Field(None, min_length=3, max_length=200)
    description: Optional[str] = Field(None, max_length=1000)
    platform: Optional[str] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    attendees: Optional[List[str]] = None
    location: Optional[str] = None
    meeting_url: Optional[str] = None
    reminder_minutes: Optional[int] = None
    notes: Optional[str] = None
    status: Optional[str] = None
    
    @validator('platform')
    def validate_platform(cls, v):
        if v:
            return validate_choice(v, list(MEETING_PLATFORMS.keys()), "Platform")
        return v
    
    @validator('attendees')
    def validate_attendee_emails(cls, v):
        if v:
            return [validate_email(email) for email in v]
        return v
    
    @validator('start_time', 'end_time')
    def validate_datetime(cls, v):
        if v:
            return validate_date_string(v)
        return v
    
    @validator('status')
    def validate_status(cls, v):
        if v:
            return validate_choice(v, ['scheduled', 'completed', 'cancelled', 'rescheduled'], "Status")
        return v


class MeetingResponse(BaseModel):
    id: str
    title: str
    description: Optional[str]
    platform: str
    platform_name: str
    platform_icon: str
    start_time: str
    end_time: str
    duration_minutes: int
    attendees: List[str]
    organizer_email: str
    location: Optional[str]
    meeting_url: Optional[str]
    reminder_minutes: int
    notes: Optional[str]
    status: str
    customer_name: Optional[str]
    lead_id: Optional[str]
    created_at: str
    updated_at: Optional[str]


# Helper functions
def generate_meeting_url(platform: str, meeting_id: str) -> Optional[str]:
    """Generate meeting URL based on platform"""
    platform_config = MEETING_PLATFORMS.get(platform)
    if not platform_config or not platform_config['url_pattern']:
        return None

    # Generate unique meeting ID based on platform
    if platform == "zoom":
        # Zoom meeting ID format: 123-456-7890
        return platform_config['url_pattern'].format(meeting_id=f"{meeting_id[:3]}-{meeting_id[3:6]}-{meeting_id[6:]}")
    elif platform == "google_meet":
        # Google Meet format: abc-defg-hij
        return platform_config['url_pattern'].format(meeting_id=f"{meeting_id[:3]}-{meeting_id[3:7]}-{meeting_id[7:]}")
    elif platform == "microsoft_teams":
        # Teams format: unique string
        return platform_config['url_pattern'].format(meeting_id=meeting_id)
    elif platform == "webex":
        return platform_config['url_pattern'].format(meeting_id=meeting_id)
    elif platform == "phone":
        return None

    return None


def calculate_duration(start_time: str, end_time: str) -> int:
    """Calculate meeting duration in minutes"""
    try:
        start = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
        end = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
        duration = (end - start).total_seconds() / 60
        return int(duration)
    except Exception:
        return 60  # Default to 60 minutes


# ============================================================================
# CRUD ENDPOINTS
# ============================================================================

@router.post("", response_model=MeetingResponse)
async def create_meeting(
    meeting: MeetingCreate,
    current_user: dict = Depends(get_current_user_from_session)
):
    """
    Create a new meeting

    Authentication: Required
    Headers: X-Session-ID or X-API-Key

    Supports platforms:
    - Zoom
    - Google Meet
    - Microsoft Teams
    - Cisco Webex
    - Phone Call
    - In Person
    """
    global MEETING_ID_COUNTER

    # Validate start time is in the future
    try:
        start_dt = datetime.fromisoformat(meeting.start_time.replace('Z', '+00:00'))
        end_dt = datetime.fromisoformat(meeting.end_time.replace('Z', '+00:00'))

        if start_dt < datetime.now():
            raise HTTPException(status_code=400, detail="Start time must be in the future")

        if end_dt <= start_dt:
            raise HTTPException(status_code=400, detail="End time must be after start time")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid datetime format: {str(e)}")

    # Generate meeting ID
    meeting_id = f"MTG-{str(MEETING_ID_COUNTER).zfill(6)}"
    MEETING_ID_COUNTER += 1

    # Generate meeting URL for online platforms
    meeting_url = meeting.meeting_url
    if not meeting_url and meeting.platform in ['zoom', 'google_meet', 'microsoft_teams', 'webex']:
        meeting_url = generate_meeting_url(meeting.platform, meeting_id.replace('-', ''))

    # Get platform info
    platform_info = MEETING_PLATFORMS[meeting.platform]

    # Calculate duration
    duration = calculate_duration(meeting.start_time, meeting.end_time)

    # Create meeting record
    meeting_data = {
        "id": meeting_id,
        "title": meeting.title,
        "description": meeting.description,
        "platform": meeting.platform,
        "platform_name": platform_info['name'],
        "platform_icon": platform_info['icon'],
        "start_time": meeting.start_time,
        "end_time": meeting.end_time,
        "duration_minutes": duration,
        "attendees": meeting.attendees,
        "organizer_email": meeting.organizer_email,
        "location": meeting.location,
        "meeting_url": meeting_url,
        "reminder_minutes": meeting.reminder_minutes or 15,
        "notes": meeting.notes,
        "status": "scheduled",
        "customer_name": meeting.customer_name,
        "lead_id": meeting.lead_id,
        "created_at": datetime.now().isoformat(),
        "updated_at": None,
        "created_by": current_user['username']
    }

    MEETINGS_DB[meeting_id] = meeting_data

    logger.info(
        f"User {current_user['username']} created meeting: {meeting.title} "
        f"on {meeting.platform} at {meeting.start_time}"
    )

    # Log notification (in production, send actual calendar invites)
    logger.info(
        f"NOTIFICATION: Meeting invitation sent to {len(meeting.attendees)} attendees. "
        f"Platform: {platform_info['name']}, URL: {meeting_url or 'N/A'}"
    )

    return meeting_data


@router.get("", response_model=List[MeetingResponse])
async def get_meetings(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    platform: Optional[str] = Query(None, description="Filter by platform"),
    status: Optional[str] = Query(None, description="Filter by status"),
    start_date: Optional[str] = Query(None, description="Filter meetings from this date"),
    end_date: Optional[str] = Query(None, description="Filter meetings until this date"),
    customer_name: Optional[str] = Query(None, description="Filter by customer name"),
    current_user: dict = Depends(get_optional_user)
):
    """
    Get all meetings with pagination and filters

    Authentication: Optional (read-only)

    Filters:
    - platform: zoom, google_meet, microsoft_teams, webex, phone, in_person
    - status: scheduled, completed, cancelled, rescheduled
    - start_date/end_date: ISO format date range
    - customer_name: Filter by associated customer
    """
    meetings = list(MEETINGS_DB.values())

    # Apply filters
    if platform:
        meetings = [m for m in meetings if m['platform'] == platform]

    if status:
        meetings = [m for m in meetings if m['status'] == status]

    if customer_name:
        meetings = [m for m in meetings if m.get('customer_name', '').lower() == customer_name.lower()]

    if start_date:
        try:
            start_dt = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
            meetings = [
                m for m in meetings
                if datetime.fromisoformat(m['start_time'].replace('Z', '+00:00')) >= start_dt
            ]
        except ValueError:
            pass

    if end_date:
        try:
            end_dt = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
            meetings = [
                m for m in meetings
                if datetime.fromisoformat(m['start_time'].replace('Z', '+00:00')) <= end_dt
            ]
        except ValueError:
            pass

    # Sort by start time (newest first)
    meetings.sort(key=lambda x: x['start_time'], reverse=True)

    # Pagination
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    paginated_meetings = meetings[start_idx:end_idx]

    return paginated_meetings


@router.get("/{meeting_id}", response_model=MeetingResponse)
async def get_meeting(
    meeting_id: str,
    current_user: dict = Depends(get_optional_user)
):
    """
    Get a specific meeting by ID

    Authentication: Optional (read-only)
    """
    if meeting_id not in MEETINGS_DB:
        raise HTTPException(status_code=404, detail="Meeting not found")

    return MEETINGS_DB[meeting_id]


@router.put("/{meeting_id}", response_model=MeetingResponse)
async def update_meeting(
    meeting_id: str,
    meeting_update: MeetingUpdate,
    current_user: dict = Depends(get_current_user_from_session)
):
    """
    Update a meeting

    Authentication: Required
    Headers: X-Session-ID or X-API-Key
    """
    if meeting_id not in MEETINGS_DB:
        raise HTTPException(status_code=404, detail="Meeting not found")

    meeting = MEETINGS_DB[meeting_id]

    # Update fields
    update_data = meeting_update.dict(exclude_unset=True)

    # Validate datetime changes
    if 'start_time' in update_data or 'end_time' in update_data:
        start_time = update_data.get('start_time', meeting['start_time'])
        end_time = update_data.get('end_time', meeting['end_time'])

        try:
            start_dt = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
            end_dt = datetime.fromisoformat(end_time.replace('Z', '+00:00'))

            if end_dt <= start_dt:
                raise HTTPException(status_code=400, detail="End time must be after start time")

            # Recalculate duration
            update_data['duration_minutes'] = calculate_duration(start_time, end_time)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Invalid datetime format: {str(e)}")

    # Update platform info if platform changed
    if 'platform' in update_data:
        platform_info = MEETING_PLATFORMS[update_data['platform']]
        update_data['platform_name'] = platform_info['name']
        update_data['platform_icon'] = platform_info['icon']

        # Regenerate meeting URL if platform changed
        if update_data['platform'] in ['zoom', 'google_meet', 'microsoft_teams', 'webex']:
            update_data['meeting_url'] = generate_meeting_url(
                update_data['platform'],
                meeting_id.replace('-', '')
            )

    # Apply updates
    for key, value in update_data.items():
        meeting[key] = value

    meeting['updated_at'] = datetime.now().isoformat()

    logger.info(
        f"User {current_user['username']} updated meeting: {meeting_id}. "
        f"Changes: {list(update_data.keys())}"
    )

    # Log notification if time changed
    if 'start_time' in update_data or 'end_time' in update_data:
        logger.info(
            f"NOTIFICATION: Meeting rescheduled. New time: {meeting['start_time']}. "
            f"Attendees notified: {len(meeting['attendees'])}"
        )

    return meeting


@router.delete("/{meeting_id}")
async def delete_meeting(
    meeting_id: str,
    current_user: dict = Depends(get_current_user_from_session)
):
    """
    Delete (cancel) a meeting

    Authentication: Required
    Headers: X-Session-ID or X-API-Key
    """
    if meeting_id not in MEETINGS_DB:
        raise HTTPException(status_code=404, detail="Meeting not found")

    meeting = MEETINGS_DB[meeting_id]

    # Mark as cancelled instead of deleting
    MEETINGS_DB[meeting_id]['status'] = 'cancelled'
    MEETINGS_DB[meeting_id]['updated_at'] = datetime.now().isoformat()

    logger.info(
        f"User {current_user['username']} cancelled meeting: {meeting_id} ({meeting['title']})"
    )

    # Log notification
    logger.info(
        f"NOTIFICATION: Meeting cancelled. Attendees notified: {len(meeting['attendees'])}"
    )

    return {"message": "Meeting cancelled successfully", "id": meeting_id}


# ============================================================================
# ADVANCED SCHEDULER ENDPOINTS
# ============================================================================

@router.get("/upcoming/today")
async def get_todays_meetings(
    current_user: dict = Depends(get_optional_user)
):
    """
    Get today's meetings

    Authentication: Optional (read-only)
    """
    today = datetime.now().date()

    todays_meetings = []
    for meeting in MEETINGS_DB.values():
        try:
            meeting_date = datetime.fromisoformat(meeting['start_time'].replace('Z', '+00:00')).date()
            if meeting_date == today and meeting['status'] == 'scheduled':
                todays_meetings.append(meeting)
        except ValueError:
            continue

    # Sort by start time
    todays_meetings.sort(key=lambda x: x['start_time'])

    return {
        "date": today.isoformat(),
        "count": len(todays_meetings),
        "meetings": todays_meetings
    }


@router.get("/upcoming/week")
async def get_this_weeks_meetings(
    current_user: dict = Depends(get_optional_user)
):
    """
    Get this week's meetings

    Authentication: Optional (read-only)
    """
    today = datetime.now()
    week_start = today - timedelta(days=today.weekday())
    week_end = week_start + timedelta(days=7)

    weeks_meetings = []
    for meeting in MEETINGS_DB.values():
        try:
            meeting_dt = datetime.fromisoformat(meeting['start_time'].replace('Z', '+00:00'))
            if week_start <= meeting_dt < week_end and meeting['status'] == 'scheduled':
                weeks_meetings.append(meeting)
        except ValueError:
            continue

    # Sort by start time
    weeks_meetings.sort(key=lambda x: x['start_time'])

    # Group by day
    meetings_by_day = {}
    for meeting in weeks_meetings:
        meeting_date = datetime.fromisoformat(meeting['start_time'].replace('Z', '+00:00')).date()
        day_key = meeting_date.isoformat()

        if day_key not in meetings_by_day:
            meetings_by_day[day_key] = []

        meetings_by_day[day_key].append(meeting)

    return {
        "week_start": week_start.date().isoformat(),
        "week_end": week_end.date().isoformat(),
        "total_count": len(weeks_meetings),
        "meetings_by_day": meetings_by_day
    }


@router.get("/calendar/month")
async def get_calendar_month(
    year: int = Query(..., description="Year (e.g., 2024)"),
    month: int = Query(..., ge=1, le=12, description="Month (1-12)"),
    current_user: dict = Depends(get_optional_user)
):
    """
    Get calendar view for a specific month

    Authentication: Optional (read-only)

    Returns meetings grouped by day for the entire month
    """
    # Get first and last day of month
    from calendar import monthrange

    first_day = datetime(year, month, 1)
    last_day_num = monthrange(year, month)[1]
    last_day = datetime(year, month, last_day_num, 23, 59, 59)

    # Get all meetings in this month
    month_meetings = []
    for meeting in MEETINGS_DB.values():
        try:
            meeting_dt = datetime.fromisoformat(meeting['start_time'].replace('Z', '+00:00'))
            if first_day <= meeting_dt <= last_day:
                month_meetings.append(meeting)
        except ValueError:
            continue

    # Group by day
    meetings_by_day = {}
    for meeting in month_meetings:
        meeting_date = datetime.fromisoformat(meeting['start_time'].replace('Z', '+00:00')).date()
        day_key = meeting_date.isoformat()

        if day_key not in meetings_by_day:
            meetings_by_day[day_key] = {
                "date": day_key,
                "count": 0,
                "meetings": []
            }

        meetings_by_day[day_key]['count'] += 1
        meetings_by_day[day_key]['meetings'].append(meeting)

    # Sort meetings within each day
    for day_data in meetings_by_day.values():
        day_data['meetings'].sort(key=lambda x: x['start_time'])

    return {
        "year": year,
        "month": month,
        "month_name": first_day.strftime("%B"),
        "total_meetings": len(month_meetings),
        "meetings_by_day": meetings_by_day
    }


@router.get("/availability/check")
async def check_availability(
    start_time: str = Query(..., description="Proposed start time (ISO format)"),
    end_time: str = Query(..., description="Proposed end time (ISO format)"),
    organizer_email: Optional[str] = Query(None, description="Check organizer's availability"),
    current_user: dict = Depends(get_optional_user)
):
    """
    Check availability for a time slot

    Authentication: Optional (read-only)

    Returns whether the time slot is available and any conflicts
    """
    try:
        proposed_start = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
        proposed_end = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid datetime format: {str(e)}")

    # Find conflicts
    conflicts = []
    for meeting in MEETINGS_DB.values():
        if meeting['status'] != 'scheduled':
            continue

        # Check if organizer matches (if specified)
        if organizer_email and meeting['organizer_email'].lower() != organizer_email.lower():
            continue

        try:
            meeting_start = datetime.fromisoformat(meeting['start_time'].replace('Z', '+00:00'))
            meeting_end = datetime.fromisoformat(meeting['end_time'].replace('Z', '+00:00'))

            # Check for overlap
            if (proposed_start < meeting_end and proposed_end > meeting_start):
                conflicts.append({
                    "meeting_id": meeting['id'],
                    "title": meeting['title'],
                    "start_time": meeting['start_time'],
                    "end_time": meeting['end_time'],
                    "platform": meeting['platform_name']
                })
        except ValueError:
            continue

    is_available = len(conflicts) == 0

    return {
        "available": is_available,
        "proposed_start": start_time,
        "proposed_end": end_time,
        "conflicts": conflicts,
        "conflict_count": len(conflicts),
        "message": "Time slot is available" if is_available else f"Found {len(conflicts)} conflict(s)"
    }


@router.get("/statistics/summary")
async def get_meeting_statistics(
    start_date: Optional[str] = Query(None, description="Start date for statistics"),
    end_date: Optional[str] = Query(None, description="End date for statistics"),
    current_user: dict = Depends(get_optional_user)
):
    """
    Get meeting statistics and analytics

    Authentication: Optional (read-only)

    Returns:
    - Total meetings by status
    - Meetings by platform
    - Average meeting duration
    - Busiest days
    """
    meetings = list(MEETINGS_DB.values())

    # Apply date filters
    if start_date:
        try:
            start_dt = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
            meetings = [
                m for m in meetings
                if datetime.fromisoformat(m['start_time'].replace('Z', '+00:00')) >= start_dt
            ]
        except ValueError:
            pass

    if end_date:
        try:
            end_dt = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
            meetings = [
                m for m in meetings
                if datetime.fromisoformat(m['start_time'].replace('Z', '+00:00')) <= end_dt
            ]
        except ValueError:
            pass

    # Calculate statistics
    total_meetings = len(meetings)

    # By status
    by_status = {}
    for meeting in meetings:
        status = meeting['status']
        by_status[status] = by_status.get(status, 0) + 1

    # By platform
    by_platform = {}
    for meeting in meetings:
        platform = meeting['platform_name']
        by_platform[platform] = by_platform.get(platform, 0) + 1

    # Average duration
    total_duration = sum(m['duration_minutes'] for m in meetings)
    avg_duration = total_duration / total_meetings if total_meetings > 0 else 0

    # Busiest days
    by_day = {}
    for meeting in meetings:
        try:
            meeting_date = datetime.fromisoformat(meeting['start_time'].replace('Z', '+00:00')).date()
            day_key = meeting_date.isoformat()
            by_day[day_key] = by_day.get(day_key, 0) + 1
        except ValueError:
            continue

    # Get top 5 busiest days
    busiest_days = sorted(by_day.items(), key=lambda x: x[1], reverse=True)[:5]

    return {
        "total_meetings": total_meetings,
        "by_status": by_status,
        "by_platform": by_platform,
        "average_duration_minutes": round(avg_duration, 1),
        "total_duration_hours": round(total_duration / 60, 1),
        "busiest_days": [{"date": day, "count": count} for day, count in busiest_days],
        "date_range": {
            "start": start_date,
            "end": end_date
        }
    }


@router.post("/{meeting_id}/complete")
async def mark_meeting_complete(
    meeting_id: str,
    notes: Optional[str] = Query(None, description="Meeting notes/summary"),
    current_user: dict = Depends(get_current_user_from_session)
):
    """
    Mark a meeting as completed

    Authentication: Required
    Headers: X-Session-ID or X-API-Key
    """
    if meeting_id not in MEETINGS_DB:
        raise HTTPException(status_code=404, detail="Meeting not found")

    meeting = MEETINGS_DB[meeting_id]

    MEETINGS_DB[meeting_id]['status'] = 'completed'
    MEETINGS_DB[meeting_id]['completed_at'] = datetime.now().isoformat()
    MEETINGS_DB[meeting_id]['completed_by'] = current_user['username']

    if notes:
        MEETINGS_DB[meeting_id]['completion_notes'] = notes

    logger.info(
        f"User {current_user['username']} marked meeting {meeting_id} as completed"
    )

    return {
        "message": "Meeting marked as completed",
        "meeting_id": meeting_id,
        "title": meeting['title'],
        "status": "completed",
        "completed_at": MEETINGS_DB[meeting_id]['completed_at']
    }


@router.get("/platforms/list")
async def get_available_platforms(
    current_user: dict = Depends(get_optional_user)
):
    """
    Get list of available meeting platforms

    Authentication: Optional (read-only)
    """
    platforms = []
    for key, config in MEETING_PLATFORMS.items():
        platforms.append({
            "id": key,
            "name": config['name'],
            "icon": config['icon'],
            "requires_auth": config['requires_auth'],
            "type": "online" if key in ['zoom', 'google_meet', 'microsoft_teams', 'webex'] else "offline"
        })

    return {
        "platforms": platforms,
        "total": len(platforms)
    }

