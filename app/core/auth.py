from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from datetime import datetime, timedelta
from typing import Optional
import logging

from app.core.config import settings
from app.core.exceptions import AuthenticationError

logger = logging.getLogger(__name__)

security = HTTPBearer()


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create JWT access token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    return encoded_jwt


def verify_token(token: str) -> dict:
    """Verify JWT token"""
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        return payload
    except JWTError as e:
        logger.error(f"JWT verification failed: {e}")
        raise AuthenticationError("Invalid token")


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Get current authenticated user"""
    try:
        payload = verify_token(credentials.credentials)
        user_id: str = payload.get("sub")
        if user_id is None:
            raise AuthenticationError("Invalid token payload")
        return {"user_id": user_id, "payload": payload}
    except Exception as e:
        raise AuthenticationError(str(e))


# For now, we'll use a simple API key authentication
# In production, you'd want to implement proper user management
API_KEYS = {
    "test-api-key": "test-user",
    "admin-api-key": "admin-user"
}


async def get_current_user_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Get current user using API key authentication"""
    api_key = credentials.credentials
    
    if api_key not in API_KEYS:
        raise AuthenticationError("Invalid API key")
    
    return {"user_id": API_KEYS[api_key], "api_key": api_key}
