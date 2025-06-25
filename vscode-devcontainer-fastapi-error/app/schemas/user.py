"""ユーザー関連のスキーマ."""
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, EmailStr


class UserBase(BaseModel):
    """ユーザーベーススキーマ."""
    name: str
    email: EmailStr


class UserCreate(UserBase):
    """ユーザー作成スキーマ."""
    pass


class UserUpdate(BaseModel):
    """ユーザー更新スキーマ."""
    name: Optional[str] = None
    email: Optional[EmailStr] = None


class User(UserBase):
    """ユーザーデータベーススキーマ."""
    id: int
    created_at: datetime
    
    class Config:
        from_attributes = True


class UserResponse(User):
    """ユーザーレスポンススキーマ."""
    pass
