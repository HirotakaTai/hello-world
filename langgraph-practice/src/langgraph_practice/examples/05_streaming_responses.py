#!/usr/bin/env python3
"""
LangGraph例5: ストリーミングレスポンス

リアルタイムストリーミング表示の実装とバッファリング
"""
import asyncio
import time
import random
from typing import AsyncGenerator, Dict, Any

from langgraph import StateGraph
from langgraph.checkpoint.sqlite import SqliteSaver

from ..core.state import ChatState, Message, ConversationRole
from ..streaming.stream_processor import (
    StreamingChatDisplay, 
    StreamProcessor,
    StreamChunk
)


def create_initial_state(user_input: str) -> ChatState:
    """初期状態の作成"""
    return {
        "messages": [],
        "current_step": "start",
        "iteration_count": 0,
        "error_count": 0,
        "last_error": None,
        "metadata": {},
        "user_input": user_input,
        "ai_response": "",
        "conversation_id": "streaming_example",
        "session_active": True,
    }


async def simulate_llm_stream(prompt: str) -> AsyncGenerator[str, None]:
    """LLMストリーミングレスポンスのシミュレーション"""
    # 実際のLLMの代わりにシミュレーション
    responses = {
        "こんにちは": [
            "こんにちは！", "今日は", "とても", "良い", "天気", "ですね。",
            "何か", "お手伝い", "できる", "ことは", "ありますか？"
        ],
        "langgraph": [
            "LangGraph", "は", "複雑な", "AI", "ワークフロー", "を", "構築", "するための",
            "強力な", "フレームワーク", "です。", "ステート", "グラフ", "を", "使用", "して",
            "エージェント", "の", "動作", "を", "制御", "できます。"
        ],
        "ストリーミング": [
            "ストリーミング", "処理", "により、", "ユーザー", "は", "AI", "の", "回答", "を",
            "リアルタイム", "で", "確認", "できます。", "これにより", "より", "自然", "な",
            "対話", "体験", "が", "可能", "に", "なります。"
        ]
    }
    
    # キーワードマッチング
    words = None
    for keyword, word_list in responses.items():
        if keyword in prompt.lower():
            words = word_list
            break
    
    if words is None:
        words = ["申し訳", "ありません", "が、", "その", "内容", "について", "詳しく", 
                "説明", "できません。", "他の", "質問", "はありますか？"]
    
    # 各単語を順次yield（ストリーミングをシミュレート）
    for word in words:
        # ランダムな遅延でよりリアルに
        await asyncio.sleep(random.uniform(0.1, 0.3))
        yield word


def user_input_node(state: ChatState) -> Dict[str, Any]:
    """ユーザー入力処理ノード"""
    print(f"\n👤 ユーザー: {state['user_input']}")
    
    user_message: Message = {
        "role": ConversationRole.HUMAN,
        "content": state["user_input"],
        "timestamp": None,
        "metadata": None,
    }
    
    return {
        "messages": state["messages"] + [user_message],
        "current_step": "streaming",
        "iteration_count": state["iteration_count"] + 1,
    }


async def streaming_ai_response_node(state: ChatState) -> Dict[str, Any]:
    """ストリーミングAI応答ノード"""
    print(f"\n🔄 ストリーミング処理を開始...")
    
    # ストリーミング表示クラスを初期化
    chat_display = StreamingChatDisplay()
    
    # LLMストリームのシミュレーション
    response_stream = simulate_llm_stream(state["user_input"])
    
    # ストリーミング表示実行
    full_response = await chat_display.display_streaming_response(
        response_stream,
        title="🤖 AI応答（ストリーミング）"
    )
    
    # 統計情報表示
    chat_display.display_statistics()
    
    # AI応答メッセージを追加
    ai_message: Message = {
        "role": ConversationRole.ASSISTANT,
        "content": full_response,
        "timestamp": None,
        "metadata": {
            "streaming": True,
            "response_length": len(full_response),
        }
    }
    
    return {
        "messages": state["messages"] + [ai_message],
        "ai_response": full_response,
        "current_step": "completed",
    }


def should_continue(state: ChatState) -> str:
    """継続判定関数"""
    return "end"


async def demonstrate_buffer_management():
    """バッファ管理のデモ"""
    print("\n" + "="*50)
    print("🔧 ストリームバッファ管理のデモ")
    print("="*50)
    
    processor = StreamProcessor(buffer_size=5)  # 小さなバッファサイズ
    
    async def test_stream():
        """テスト用ストリーム"""
        test_chunks = [f"チャンク{i} " for i in range(10)]
        for chunk in test_chunks:
            yield chunk
            await asyncio.sleep(0.1)
    
    # ストリーム処理実行
    result = await processor.process_stream(
        test_stream(),
        display_progress=True
    )
    
    print(f"\n📊 処理結果:")
    print(f"  - 完全な内容: {result}")
    print(f"  - バッファ内容: {processor.get_buffer_content()}")
    
    stats = processor.get_chunk_statistics()
    print(f"  - 統計: {stats}")
    
    # バッファクリア
    processor.clear_buffer()
    print(f"  - バッファクリア後: '{processor.get_buffer_content()}'")


async def main():
    """メイン実行関数"""
    print("=" * 50)
    print("🎯 例5: ストリーミングレスポンス")
    print("リアルタイム表示とバッファリング")
    print("=" * 50)
    
    # チェックポイント設定
    checkpointer = SqliteSaver.from_conn_string(":memory:")
    
    # グラフの構築
    workflow = StateGraph(ChatState)
    
    # ノードの追加
    workflow.add_node("user_input", user_input_node)
    workflow.add_node("streaming_response", streaming_ai_response_node)
    
    # エッジの設定
    workflow.add_edge("user_input", "streaming_response")
    workflow.add_conditional_edges(
        "streaming_response",
        should_continue,
        {"end": "__end__"}
    )
    
    # エントリーポイント設定
    workflow.set_entry_point("user_input")
    
    # グラフのコンパイル
    app = workflow.compile(checkpointer=checkpointer)
    
    # 実行例
    test_inputs = [
        "こんにちは",
        "LangGraphについて教えて",
        "ストリーミング処理の仕組みは？"
    ]
    
    for i, user_input in enumerate(test_inputs, 1):
        print(f"\n{'='*20} 実行例 {i} {'='*20}")
        
        initial_state = create_initial_state(user_input)
        thread_id = f"streaming_thread_{i}"
        config = {"configurable": {"thread_id": thread_id}}
        
        # 処理時間計測
        start_time = time.time()
        result = await app.ainvoke(initial_state, config)
        end_time = time.time()
        
        print(f"\n⏱️  処理時間: {end_time - start_time:.2f}秒")
        print(f"📈 最終状態: {result['current_step']}")
    
    # バッファ管理のデモ
    await demonstrate_buffer_management()
    
    print(f"\n✅ ストリーミングレスポンスの例が完了しました")


if __name__ == "__main__":
    asyncio.run(main())