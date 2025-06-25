"""ユーザーモデル."""

from sqlalchemy import Column, DateTime, Integer, String, func

from app.db.database import Base


class User(Base):
    """ユーザーテーブルモデル."""
    
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now(), nullable=False)
