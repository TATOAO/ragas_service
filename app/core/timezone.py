from datetime import datetime, timezone, timedelta
from typing import Optional

# Define the +8 timezone offset
UTC_PLUS_8 = timezone(timedelta(hours=8))

def get_current_time_utc_plus_8() -> datetime:
    """Get current time in UTC+8 timezone"""
    return datetime.now(UTC_PLUS_8)

def get_current_time_utc() -> datetime:
    """Get current time in UTC timezone"""
    return datetime.now(timezone.utc)

def convert_to_utc_plus_8(dt: datetime) -> datetime:
    """Convert a datetime to UTC+8 timezone"""
    if dt.tzinfo is None:
        # If naive datetime, assume it's UTC
        dt = dt.replace(tzinfo=timezone.utc)
    
    # Convert to UTC+8
    return dt.astimezone(UTC_PLUS_8)

def convert_from_utc_plus_8(dt: datetime) -> datetime:
    """Convert a UTC+8 datetime to UTC timezone"""
    if dt.tzinfo is None:
        # If naive datetime, assume it's UTC+8
        dt = dt.replace(tzinfo=UTC_PLUS_8)
    
    # Convert to UTC
    return dt.astimezone(timezone.utc)

def format_datetime_utc_plus_8(dt: datetime, format_str: str = "%Y-%m-%d %H:%M:%S") -> str:
    """Format datetime in UTC+8 timezone"""
    utc_plus_8_dt = convert_to_utc_plus_8(dt)
    return utc_plus_8_dt.strftime(format_str)