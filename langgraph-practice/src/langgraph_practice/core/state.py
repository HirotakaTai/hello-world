"""LangGraph用の状態定義

TypedDictを使用したタイプセーフな状態管理
"""
from typing import Dict, List, Any, Optional, TypedDict
from enum import Enum


class ConversationRole(str, Enum):
    """会話の役割定義"""
    HUMAN = "human"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class Message(TypedDict):
    """メッセージの型定義"""
    role: ConversationRole
    content: str
    timestamp: Optional[str]
    metadata: Optional[Dict[str, Any]]


class AgentState(TypedDict):
    """エージェントの基本状態"""
    messages: List[Message]
    current_step: str
    iteration_count: int
    error_count: int
    last_error: Optional[str]
    metadata: Dict[str, Any]


class ChatState(AgentState):
    """チャットボット用の拡張状態"""
    user_input: str
    ai_response: str
    conversation_id: str
    session_active: bool


class MultiAgentState(AgentState):
    """マルチエージェント用状態"""
    active_agents: List[str]
    agent_responses: Dict[str, Any]
    coordination_status: str
    final_result: Optional[str]


class StreamingState(TypedDict):
    """ストリーミング処理用状態"""
    stream_buffer: List[str]
    streaming_active: bool
    chunks_processed: int
    total_chunks: Optional[int]


class ErrorRecoveryState(TypedDict):
    """エラー回復用状態"""
    retry_count: int
    max_retries: int
    recovery_strategy: str
    fallback_enabled: bool
    recovery_data: Optional[Dict[str, Any]]


# 複合状態の例
class CompleteState(ChatState, StreamingState, ErrorRecoveryState):
    """すべての機能を含む完全な状態定義"""
    pass