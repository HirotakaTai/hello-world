"""データベース接続設定."""
from databases import Database
from sqlalchemy import MetaData, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from app.core.config import settings

# SQLAlchemy設定
engine = create_engine(
    settings.DATABASE_URL.replace("+asyncpg", ""),  # 同期用
    pool_pre_ping=True,
    echo=settings.ENVIRONMENT == "development",
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()
metadata = MetaData()

# 非同期データベース接続
database = Database(settings.DATABASE_URL)


async def get_database() -> Database:
    """データベース接続を取得."""
    return database


def get_db():
    """同期データベースセッションを取得."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
