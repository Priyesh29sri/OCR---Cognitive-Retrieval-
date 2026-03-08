"""
Authentication Schemas
Pydantic models for request/response validation
"""
from pydantic import BaseModel, EmailStr, Field
from typing import Optional
from datetime import datetime


# ===== User Registration =====
class UserRegister(BaseModel):
    """User registration request"""
    email: EmailStr
    password: str = Field(..., min_length=8, description="Password must be at least 8 characters")
    full_name: Optional[str] = None


class UserLogin(BaseModel):
    """User login request"""
    email: EmailStr
    password: str


# ===== Token Models =====
class Token(BaseModel):
    """JWT token response"""
    access_token: str
    token_type: str = "bearer"
    expires_in: int  # Seconds until expiration


class TokenData(BaseModel):
    """Data stored in JWT token"""
    user_id: Optional[int] = None
    email: Optional[str] = None


# ===== User Response Models =====
class UserResponse(BaseModel):
    """User information response"""
    id: int
    email: str
    full_name: Optional[str] = None
    is_active: bool
    created_at: datetime
    
    class Config:
        from_attributes = True  # Pydantic v2 (was orm_mode in v1)


class UserProfile(BaseModel):
    """Extended user profile with statistics"""
    id: int
    email: str
    full_name: Optional[str] = None
    is_active: bool
    created_at: datetime
    last_login: Optional[datetime] = None
    total_documents: int = 0
    total_conversations: int = 0
    
    class Config:
        from_attributes = True


# ===== Password Change =====
class PasswordChange(BaseModel):
    """Password change request"""
    current_password: str
    new_password: str = Field(..., min_length=8)


# ===== Error Response =====
class ErrorResponse(BaseModel):
    """Standard error response"""
    detail: str
