#!/usr/bin/env python3
"""
LangGraph基本例1: 基本状態グラフ

TypedDict状態定義とシンプルなノード実装の学習
"""
import asyncio
from typing import Dict, Any

from langgraph import StateGraph
from langgraph.checkpoint.sqlite import SqliteSaver

from ..core.state import ChatState, Message, ConversationRole


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
        "conversation_id": "example_01",
        "session_active": True,
    }


def user_input_node(state: ChatState) -> Dict[str, Any]:
    """ユーザー入力処理ノード"""
    print(f"👤 ユーザー: {state['user_input']}")
    
    # メッセージをリストに追加
    user_message: Message = {
        "role": ConversationRole.HUMAN,
        "content": state["user_input"],
        "timestamp": None,
        "metadata": None,
    }
    
    return {
        "messages": state["messages"] + [user_message],
        "current_step": "processing",
        "iteration_count": state["iteration_count"] + 1,
    }


def ai_response_node(state: ChatState) -> Dict[str, Any]:
    """AI応答生成ノード（モックレスポンス）"""
    # シンプルなルールベース応答
    user_message = state["user_input"].lower()
    
    if "こんにちは" in user_message or "hello" in user_message:
        response = "こんにちは！どのようにお手伝いできますか？"
    elif "ありがとう" in user_message or "thank" in user_message:
        response = "どういたしまして！他にご質問はありますか？"
    elif "さようなら" in user_message or "bye" in user_message:
        response = "さようなら！また何かありましたらお声をかけてください。"
    else:
        response = f"「{state['user_input']}」について理解しました。詳しく教えていただけますか？"
    
    print(f"🤖 AI: {response}")
    
    # AI応答メッセージを追加
    ai_message: Message = {
        "role": ConversationRole.ASSISTANT,
        "content": response,
        "timestamp": None,
        "metadata": None,
    }
    
    return {
        "messages": state["messages"] + [ai_message],
        "ai_response": response,
        "current_step": "completed",
    }


def should_continue(state: ChatState) -> str:
    """継続判定関数"""
    return "end"  # この例では常に終了


async def main():
    """メイン実行関数"""
    print("=" * 50)
    print("🎯 例1: 基本状態グラフ")
    print("TypedDict状態とシンプルなノード処理")
    print("=" * 50)
    
    # チェックポイント設定
    checkpointer = SqliteSaver.from_conn_string(":memory:")
    
    # グラフの構築
    workflow = StateGraph(ChatState)
    
    # ノードの追加
    workflow.add_node("user_input", user_input_node)
    workflow.add_node("ai_response", ai_response_node)
    
    # エッジの設定
    workflow.add_edge("user_input", "ai_response")
    workflow.add_conditional_edges(
        "ai_response",
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
        "ありがとうございました"
    ]
    
    for i, user_input in enumerate(test_inputs, 1):
        print(f"\n--- 実行例 {i} ---")
        
        initial_state = create_initial_state(user_input)
        
        # スレッドID（各実行で異なる）
        thread_id = f"thread_{i}"
        config = {"configurable": {"thread_id": thread_id}}
        
        # グラフ実行
        result = await app.ainvoke(initial_state, config)
        
        print(f"📊 実行結果:")
        print(f"  - 現在のステップ: {result['current_step']}")
        print(f"  - 反復回数: {result['iteration_count']}")
        print(f"  - メッセージ数: {len(result['messages'])}")
        print(f"  - セッション状態: {'アクティブ' if result['session_active'] else '非アクティブ'}")
    
    print(f"\n✅ 基本状態グラフの例が完了しました")
    
    # チェックポイントの確認
    print(f"\n📋 チェックポイント情報:")
    for i in range(1, len(test_inputs) + 1):
        thread_id = f"thread_{i}"
        config = {"configurable": {"thread_id": thread_id}}
        
        # 最新のチェックポイントを取得
        checkpoint = await app.aget_state(config)
        if checkpoint:
            print(f"  Thread {i}: ステップ={checkpoint.values.get('current_step', 'N/A')}")


if __name__ == "__main__":
    asyncio.run(main())