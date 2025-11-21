"""
Validation utilities for Lead Generation AI
Provides reusable validators for common data types
"""
import re
from datetime import datetime, date
from typing import Optional
from pydantic import validator


class ValidationError(Exception):
    """Custom validation error"""
    pass


def validate_email(email: str) -> str:
    """
    Validate email format
    Returns cleaned email or raises ValidationError
    """
    if not email:
        raise ValidationError("Email is required")
    
    email = email.strip().lower()
    
    # RFC 5322 simplified email regex
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    if not re.match(pattern, email):
        raise ValidationError(f"Invalid email format: {email}")
    
    return email


def validate_phone(phone: str) -> str:
    """
    Validate phone number format
    Accepts: +1234567890, 123-456-7890, (123) 456-7890, etc.
    Returns cleaned phone number (digits only with optional +)
    """
    if not phone:
        raise ValidationError("Phone number is required")
    
    # Remove common separators
    cleaned = re.sub(r'[\s\-\(\)\.]+', '', phone.strip())
    
    # Check if it matches valid phone pattern
    # Allows optional + at start, then 9-15 digits
    if not re.match(r'^\+?\d{9,15}$', cleaned):
        raise ValidationError(f"Invalid phone number format: {phone}")
    
    return cleaned


def validate_positive_number(value: float, field_name: str = "Value") -> float:
    """Validate that a number is positive"""
    if value < 0:
        raise ValidationError(f"{field_name} must be positive, got {value}")
    return value


def validate_non_negative_number(value: float, field_name: str = "Value") -> float:
    """Validate that a number is non-negative (>= 0)"""
    if value < 0:
        raise ValidationError(f"{field_name} cannot be negative, got {value}")
    return value


def validate_date_string(date_str: str, field_name: str = "Date") -> str:
    """
    Validate ISO date string format (YYYY-MM-DD)
    Returns the date string if valid
    """
    if not date_str:
        return date_str
    
    try:
        datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        return date_str
    except ValueError:
        raise ValidationError(f"{field_name} must be in ISO format (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS)")


def validate_date_not_future(date_str: str, field_name: str = "Date") -> str:
    """Validate that a date is not in the future"""
    if not date_str:
        return date_str
    
    try:
        date_obj = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        if date_obj > datetime.now():
            raise ValidationError(f"{field_name} cannot be in the future")
        return date_str
    except ValueError:
        raise ValidationError(f"{field_name} must be in ISO format")


def validate_string_length(value: str, min_length: int = 1, max_length: int = 255, field_name: str = "Field") -> str:
    """Validate string length"""
    if not value or not value.strip():
        raise ValidationError(f"{field_name} cannot be empty")
    
    value = value.strip()
    
    if len(value) < min_length:
        raise ValidationError(f"{field_name} must be at least {min_length} characters")
    
    if len(value) > max_length:
        raise ValidationError(f"{field_name} must be at most {max_length} characters")
    
    return value


def validate_choice(value: str, choices: list, field_name: str = "Field") -> str:
    """Validate that value is in allowed choices"""
    if value not in choices:
        raise ValidationError(f"{field_name} must be one of {choices}, got '{value}'")
    return value


def validate_amount_range(amount: float, min_amount: float = 0, max_amount: float = 1000000, field_name: str = "Amount") -> float:
    """Validate that amount is within acceptable range"""
    if amount < min_amount:
        raise ValidationError(f"{field_name} must be at least ${min_amount}")
    
    if amount > max_amount:
        raise ValidationError(f"{field_name} cannot exceed ${max_amount}")
    
    return amount


def validate_percentage(value: float, field_name: str = "Percentage") -> float:
    """Validate that value is a valid percentage (0-100)"""
    if value < 0 or value > 100:
        raise ValidationError(f"{field_name} must be between 0 and 100, got {value}")
    return value


def validate_url(url: str, field_name: str = "URL") -> str:
    """Validate URL format"""
    if not url:
        return url
    
    url = url.strip()
    pattern = r'^https?://[^\s/$.?#].[^\s]*$'
    
    if not re.match(pattern, url, re.IGNORECASE):
        raise ValidationError(f"{field_name} must be a valid URL")
    
    return url

