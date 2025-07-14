"""データモデル定義."""

from pydantic import BaseModel
from typing import Optional, List, Dict, Any


class ChatMessage(BaseModel):
    """チャットメッセージモデル."""

    role: str  # "user" or "assistant"
    content: str
    timestamp: Optional[str] = None


class ChatRequest(BaseModel):
    """チャットリクエストモデル."""

    message: str
    conversation_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = None


class ChatResponse(BaseModel):
    """チャットレスポンスモデル."""

    response: str
    conversation_id: str
    timestamp: str
    metadata: Optional[Dict[str, Any]] = None


class AgentState(BaseModel):
    """エージェントの状態モデル."""

    messages: List[ChatMessage]
    conversation_id: str
    context: Optional[Dict[str, Any]] = None
    step_count: int = 0
