from sqlmodel import SQLModel
from pydantic import ConfigDict, field_serializer
from datetime import datetime
from typing import Any
from app.core.timezone import convert_to_utc_plus_8


class TimezoneAwareModel(SQLModel):
    """Base model that automatically converts datetime fields to UTC+8 timezone"""
    
    model_config = ConfigDict(
        from_attributes=True,
        json_encoders={
            datetime: lambda v: convert_to_utc_plus_8(v).isoformat() if v else None
        }
    )
    
    @field_serializer('*')
    def serialize_datetime_fields(self, value: Any, info: Any) -> Any:
        """Automatically serialize datetime fields to UTC+8 timezone"""
        if isinstance(value, datetime):
            return convert_to_utc_plus_8(value).isoformat()
        return value
