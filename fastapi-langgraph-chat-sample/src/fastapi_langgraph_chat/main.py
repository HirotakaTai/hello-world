"""FastAPI メインアプリケーション."""

import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
import os

from .routes import router
from .logging_config import setup_logging

# 環境変数を読み込み
load_dotenv()

# ロギング設定を初期化
setup_logging()
logger = logging.getLogger(__name__)

# FastAPIアプリケーションを作成
app = FastAPI(
    title="FastAPI LangGraph Chat",
    description="LangGraphを使用したAIエージェントチャットボット",
    version="0.1.0",
)

logger.info("FastAPIアプリケーションが初期化されました")

# CORS設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 開発環境用。本番環境では適切に設定してください
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 静的ファイルの配信設定
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

# ルーターを追加
app.include_router(router, prefix="/api")


@app.get("/health")
async def health_check():
    """ヘルスチェックエンドポイント."""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "fastapi_langgraph_chat.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
