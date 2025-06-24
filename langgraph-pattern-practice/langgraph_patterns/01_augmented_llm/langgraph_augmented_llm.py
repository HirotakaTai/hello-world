#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LangGraph版 Augmented LLM Pattern
LangGraphを使用してツールを持つLLMエージェントを構築するパターン
"""

import datetime
import math
from typing import Annotated
from typing import Any
from typing import Dict
from typing import List
from typing import TypedDict

from dotenv import load_dotenv
from langchain.tools import BaseTool
from langchain_core.messages import AIMessage
from langchain_core.messages import HumanMessage
from langchain_core.messages import ToolMessage
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.pydantic_v1 import Field

# LangChain関連のインポート
from langchain_openai import ChatOpenAI
from langgraph.graph import END

# LangGraph関連のインポート
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolExecutor
from langgraph.prebuilt import ToolInvocation

# ===== 環境変数の読み込み =====
load_dotenv()

# ===== カスタムツールクラスの定義 =====


class CalculatorInput(BaseModel):
    """計算機ツールの入力スキーマ"""

    expression: str = Field(description="計算する数式（例: '2 + 3 * 4'）")


class Calculator(BaseTool):
    """安全な数式計算を行うツール"""

    name = "calculator"
    description = "数学計算を実行します。四則演算、べき乗、平方根などが使用できます。"
    args_schema = CalculatorInput

    def _run(self, expression: str) -> str:
        """計算を実行する内部メソッド"""
        try:
            # 安全な数学関数のみを許可
            allowed_names = {
                k: v for k, v in math.__dict__.items() if not k.startswith("__")
            }
            allowed_names.update({"abs": abs, "round": round})

            # eval()を安全に使用（実際のプロダクションでは ast.literal_eval等を推奨）
            result = eval(expression, {"__builtins__": {}}, allowed_names)
            return f"計算結果: {expression} = {result}"
        except Exception as e:
            return f"計算エラー: {str(e)}"


class TimeInput(BaseModel):
    """時刻取得ツールの入力スキーマ"""

    timezone: str = Field(default="Asia/Tokyo", description="タイムゾーン")


class CurrentTime(BaseTool):
    """現在時刻を取得するツール"""

    name = "current_time"
    description = "現在の日時を取得します。"
    args_schema = TimeInput

    def _run(self, timezone: str = "Asia/Tokyo") -> str:
        """現在時刻を取得する内部メソッド"""
        now = datetime.datetime.now()
        return f"現在の日時: {now.strftime('%Y年%m月%d日 %H時%M分%S秒')}"


class WeatherInput(BaseModel):
    """天気ツールの入力スキーマ"""

    location: str = Field(description="場所（例: '東京'）")


class Weather(BaseTool):
    """天気情報を取得するツール（モックデータ）"""

    name = "weather"
    description = "指定した場所の天気情報を取得します。"
    args_schema = WeatherInput

    def _run(self, location: str) -> str:
        """天気情報を取得する内部メソッド（モックデータを返す）"""
        # 実際の実装では、天気APIを呼び出します
        mock_weather = {
            "東京": "晴れ、気温25度",
            "大阪": "曇り、気温23度",
            "北海道": "雪、気温-5度",
            "沖縄": "晴れ、気温28度",
        }

        weather_info = mock_weather.get(
            location, f"{location}の天気情報は見つかりませんでした"
        )
        return f"{location}の天気: {weather_info}"


# ===== LangGraphの状態定義 =====


class GraphState(TypedDict):
    """LangGraphで使用する状態の定義"""

    messages: Annotated[List[HumanMessage | AIMessage | ToolMessage], add_messages]
    # メッセージのリストを管理。add_messagesで自動的にメッセージが追加される


# ===== LangGraphベースのAugmented LLMクラス =====


class LangGraphAugmentedLLM:
    """LangGraphを使用したツール付きLLMエージェント"""

    def __init__(self):
        """エージェントの初期化"""
        print("🤖 LangGraph版 Augmented LLMエージェントを初期化中...")

        # OpenAI ChatLLMモデルを初期化
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1, verbose=True)

        # 利用可能なツールを定義
        self.tools = [
            Calculator(),  # 計算機ツール
            CurrentTime(),  # 現在時刻取得ツール
            Weather(),  # 天気情報取得ツール
        ]

        # ツール実行器を作成
        self.tool_executor = ToolExecutor(self.tools)

        # LLMにツール情報をバインド
        self.llm_with_tools = self.llm.bind_tools(self.tools)

        # LangGraphワークフローを構築
        self.graph = self._build_graph()

        print("✅ エージェントの初期化が完了しました！")

    def _build_graph(self) -> StateGraph:
        """LangGraphワークフローを構築するメソッド"""
        print("🔧 LangGraphワークフローを構築中...")

        # StateGraphを作成
        workflow = StateGraph(GraphState)

        # ノード（処理ステップ）を追加
        workflow.add_node("agent", self._call_model)  # LLMを呼び出すノード
        workflow.add_node("action", self._call_tool)  # ツールを実行するノード

        # エントリーポイントを設定
        workflow.set_entry_point("agent")

        # 条件分岐を追加
        workflow.add_conditional_edges(
            "agent",  # 分岐元のノード
            self._should_continue,  # 条件判定関数
            {
                "continue": "action",  # ツール実行が必要な場合
                "end": END,  # 処理完了の場合
            },
        )

        # ツール実行後は必ずエージェントに戻る
        workflow.add_edge("action", "agent")

        # グラフをコンパイルして実行可能な形に変換
        return workflow.compile()

    def _call_model(self, state: GraphState) -> Dict[str, Any]:
        """LLMモデルを呼び出すノード処理"""
        print("🧠 LLMを呼び出し中...")
        messages = state["messages"]

        # ツール付きLLMを呼び出し
        response = self.llm_with_tools.invoke(messages)

        # 新しいメッセージを状態に追加して返す
        return {"messages": [response]}

    def _call_tool(self, state: GraphState) -> Dict[str, Any]:
        """ツールを実行するノード処理"""
        print("🔧 ツールを実行中...")
        messages = state["messages"]

        # 最後のメッセージ（AIの応答）を取得
        last_message = messages[-1]

        # ツール呼び出し情報を取得
        tool_calls = last_message.tool_calls

        # 各ツール呼び出しを実行
        tool_messages = []
        for tool_call in tool_calls:
            print(f"⚡ ツール '{tool_call['name']}' を実行: {tool_call['args']}")

            # ツール実行
            action = ToolInvocation(
                tool=tool_call["name"], tool_input=tool_call["args"]
            )
            result = self.tool_executor.invoke(action)

            # ツール実行結果をメッセージとして作成
            tool_message = ToolMessage(
                content=str(result), tool_call_id=tool_call["id"]
            )
            tool_messages.append(tool_message)

            print(f"📋 ツール実行結果: {result}")

        return {"messages": tool_messages}

    def _should_continue(self, state: GraphState) -> str:
        """処理を続行するかどうかを判定する条件分岐関数"""
        messages = state["messages"]
        last_message = messages[-1]

        # 最後のメッセージがツール呼び出しを含む場合は続行
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "continue"
        else:
            return "end"

    def chat(self, user_input: str) -> str:
        """ユーザー入力に対してチャット応答を生成"""
        print(f"\n💬 ユーザー: {user_input}")
        print("-" * 50)

        # 初期状態を作成
        initial_state = {"messages": [HumanMessage(content=user_input)]}

        # グラフを実行
        result = self.graph.invoke(initial_state)

        # 最終的なAI応答を取得
        final_message = result["messages"][-1]
        ai_response = final_message.content

        print("-" * 50)
        print(f"🤖 AI: {ai_response}")

        return ai_response


# ===== デモ用のメイン関数 =====


def main():
    """LangGraph版 Augmented LLMのデモンストレーション"""
    print("=" * 60)
    print("🚀 LangGraph版 Augmented LLM Pattern デモ")
    print("=" * 60)
    print("このデモでは、LangGraphを使用してツール付きLLMエージェントを実装します。")
    print("利用可能なツール: 計算機、現在時刻取得、天気情報取得")
    print()

    try:
        # Augmented LLMエージェントを初期化
        agent = LangGraphAugmentedLLM()

        # デモ用の質問リスト
        demo_questions = [
            "2の8乗を計算してください",
            "現在の時刻を教えてください",
            "東京の天気はどうですか？",
            "25 * 4 + 10を計算して、その結果と現在時刻を教えてください",
        ]

        print("\n📚 デモ用質問への応答:")
        print("=" * 60)

        for i, question in enumerate(demo_questions, 1):
            print(f"\n【質問 {i}】")
            agent.chat(question)
            print()

        # 対話モードの開始
        print("\n" + "=" * 60)
        print("💬 対話モード開始 (終了するには 'quit' と入力)")
        print("=" * 60)

        while True:
            try:
                user_input = input("\n🙋 あなた: ").strip()

                if user_input.lower() in ["quit", "exit", "終了", "q"]:
                    print("👋 対話を終了します。")
                    break

                if not user_input:
                    print("⚠️  質問を入力してください。")
                    continue

                # エージェントに質問を送信
                agent.chat(user_input)

            except KeyboardInterrupt:
                print("\n\n👋 対話を終了します。")
                break

    except Exception as e:
        print(f"❌ エラーが発生しました: {e}")
        print("💡 OpenAI APIキーが正しく設定されているか確認してください。")


if __name__ == "__main__":
    main()
