"""FastAPIルーティング定義."""

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse

from .agent import ChatAgent
from .models import ChatRequest, ChatResponse

logger = logging.getLogger(__name__)

# ルーターを作成
router = APIRouter()


@lru_cache(maxsize=1)
def get_chat_agent() -> ChatAgent:
    """チャットエージェントのシングルトンインスタンスを取得（スレッドセーフ）."""
    logger.info("新しいChatAgentインスタンスを作成します")
    return ChatAgent()


def clear_chat_agent_cache() -> None:
    """テスト用: エージェントキャッシュをクリア."""
    get_chat_agent.cache_clear()
    logger.info("ChatAgentキャッシュをクリアしました")


def get_chat_agent_cache_info() -> str:
    """デバッグ用: キャッシュ統計を取得."""
    info = get_chat_agent.cache_info()
    return f"hits={info.hits}, misses={info.misses}, maxsize={info.maxsize}, currsize={info.currsize}"


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest) -> ChatResponse:
    """チャットエンドポイント."""
    agent = get_chat_agent()
    
    try:
        logger.info(
            "チャットリクエスト受信: conversation_id=%s", 
            request.conversation_id
        )
        
        # OpenAI APIキーの確認
        if not os.getenv("OPENAI_API_KEY"):
            logger.error("OpenAI APIキーが設定されていません")
            raise HTTPException(
                status_code=500, detail="OpenAI APIキーが設定されていません"
            )

        # エージェントでチャット処理を実行
        result = await agent.chat(
            message=request.message,
            conversation_id=request.conversation_id,
            context=request.context,
        )

        logger.info(
            "チャットレスポンス生成完了: conversation_id=%s", 
            result["conversation_id"]
        )
        return ChatResponse(**result)

    except Exception as e:
        logger.exception("チャット処理中にエラー: %r", e)
        raise HTTPException(
            status_code=500, detail=f"チャット処理中にエラーが発生しました: {e!r}"
        ) from e


@router.get("/", response_class=HTMLResponse)
async def serve_chat_page() -> HTMLResponse:
    """チャットページを提供."""
    html_path = Path("static/index.html")
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text(encoding="utf-8"))

    return HTMLResponse(
        content="<h1>チャットページが見つかりません</h1><p>static/index.htmlを作成してください。</p>",
        status_code=404,
    )


@router.get("/conversations/{conversation_id}")
async def get_conversation(conversation_id: str) -> dict[str, Any]:
    """会話履歴を取得(将来の実装用)."""
    # 実際の実装では、データベースから会話履歴を取得
    return {
        "conversation_id": conversation_id,
        "messages": [],
        "message": "会話履歴機能は未実装です",
    }


@router.get("/debug/agent-cache")
async def debug_agent_cache() -> dict[str, Any]:
    """デバッグ用: エージェントキャッシュ情報を取得."""
    cache_info = get_chat_agent_cache_info()
    logger.info("エージェントキャッシュ情報: %s", cache_info)
    return {
        "cache_info": cache_info,
        "agent_id": id(get_chat_agent()),
        "message": "エージェントキャッシュ統計"
    }
