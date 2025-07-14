"""LangGraphエージェント実装."""

import os
import logging
from typing import Dict, Any, List
from datetime import datetime
import uuid

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_core.rate_limiters import InMemoryRateLimiter

from .models import AgentState, ChatMessage

logger = logging.getLogger(__name__)


class ChatAgent:
    """チャットエージェントクラス."""

    def __init__(self):
        """エージェントを初期化."""
        logger.info("ChatAgentを初期化中...")
        
        # 基本的なレート制限を設定
        rate_limiter = InMemoryRateLimiter(
            requests_per_second=10,  # 秒間10リクエスト
            check_every_n_seconds=0.1,
            max_bucket_size=10
        )
        
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo", 
            temperature=0.7, 
            api_key=os.getenv("OPENAI_API_KEY"),
            rate_limiter=rate_limiter,
            max_retries=2
        )
        self.graph = self._create_graph()
        logger.info("ChatAgentの初期化が完了しました")

    def _create_graph(self) -> StateGraph:
        """LangGraphのグラフを作成."""
        workflow = StateGraph(AgentState)

        # ノードを追加
        workflow.add_node("process_message", self._process_message)
        workflow.add_node("generate_response", self._generate_response)

        # エッジを追加
        workflow.set_entry_point("process_message")
        workflow.add_edge("process_message", "generate_response")
        workflow.add_edge("generate_response", END)

        return workflow.compile()

    def _process_message(self, state: AgentState) -> Dict[str, Any]:
        """メッセージを処理."""
        # ステップカウントを増加
        state.step_count += 1

        # ここで必要に応じて前処理を実行
        # 例: 入力の検証、コンテキストの更新など

        return {
            "messages": state.messages,
            "conversation_id": state.conversation_id,
            "context": state.context,
            "step_count": state.step_count,
        }

    async def _generate_response(self, state: AgentState) -> Dict[str, Any]:
        """AIレスポンスを生成."""
        # メッセージ履歴をLangChainメッセージに変換
        langchain_messages = []
        for msg in state.messages:
            if msg.role == "user":
                langchain_messages.append(HumanMessage(content=msg.content))
            elif msg.role == "assistant":
                langchain_messages.append(AIMessage(content=msg.content))

        # AIレスポンスを生成（非同期）
        try:
            response = await self.llm.ainvoke(langchain_messages)
        except Exception as e:
            logger.error("LLM呼び出しエラー: %r", e)
            raise e

        # レスポンスメッセージを追加
        assistant_message = ChatMessage(
            role="assistant",
            content=response.content,
            timestamp=datetime.now().isoformat(),
        )
        state.messages.append(assistant_message)

        return {
            "messages": state.messages,
            "conversation_id": state.conversation_id,
            "context": state.context,
            "step_count": state.step_count,
        }

    async def chat(
        self, message: str, conversation_id: str | None = None, context: Dict[str, Any] | None = None
    ) -> Dict[str, Any]:
        """チャット処理を実行."""
        # 会話IDを生成または使用
        if conversation_id is None:
            conversation_id = str(uuid.uuid4())

        logger.debug("チャット処理開始: %s", conversation_id)
        
        # ユーザーメッセージを作成
        user_message = ChatMessage(
            role="user", content=message, timestamp=datetime.now().isoformat()
        )

        # 初期状態を作成
        initial_state = AgentState(
            messages=[user_message],
            conversation_id=conversation_id,
            context=context or {},
            step_count=0,
        )

        # グラフを実行
        result = await self.graph.ainvoke(initial_state)

        # レスポンスを取得
        assistant_message = result["messages"][-1]

        logger.debug("チャット処理完了: %s", conversation_id)
        
        return {
            "response": assistant_message.content,
            "conversation_id": conversation_id,
            "timestamp": assistant_message.timestamp,
            "metadata": {
                "step_count": result["step_count"],
                "context": result["context"],
            },
        }
