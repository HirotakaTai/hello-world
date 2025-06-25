"""ユーザー関連のエンドポイント."""
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from databases import Database

from app.db.database import get_database
from app.schemas.user import UserCreate, UserResponse

router = APIRouter()


@router.get("/", response_model=list[UserResponse])
async def read_users(
    skip: int = 0,
    limit: int = 100,
    db: Database = Depends(get_database),
) -> Any:
    """ユーザー一覧取得."""
    query = "SELECT id, name, email, created_at FROM users LIMIT :limit OFFSET :skip"
    users = await db.fetch_all(query=query, values={"limit": limit, "skip": skip})
    return users


@router.post("/", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def create_user(
    user: UserCreate,
    db: Database = Depends(get_database),
) -> Any:
    """ユーザー作成."""
    # メールアドレスの重複チェック
    query = "SELECT id FROM users WHERE email = :email"
    existing_user = await db.fetch_one(query=query, values={"email": user.email})
    
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered",
        )
    
    # ユーザー作成
    query = """
        INSERT INTO users (name, email)
        VALUES (:name, :email)
        RETURNING id, name, email, created_at
    """
    new_user = await db.fetch_one(
        query=query,
        values={"name": user.name, "email": user.email},
    )
    
    return new_user


@router.get("/{user_id}", response_model=UserResponse)
async def read_user(
    user_id: int,
    db: Database = Depends(get_database),
) -> Any:
    """ユーザー詳細取得."""
    query = "SELECT id, name, email, created_at FROM users WHERE id = :user_id"
    user = await db.fetch_one(query=query, values={"user_id": user_id})
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )
    
    return user
