"""
LangGraph v0.4.8を使用した四則演算システム（修正版）

GraphRecursionErrorを解決するための修正版です。

必要なライブラリのインストール:
pip install langgraph langchain langchain-openai openai pydantic typing-extensions
"""

import operator
import re
from typing import Annotated
from typing import Any
from typing import Dict
from typing import List
from typing import Literal
from typing import TypedDict

from langchain_openai import ChatOpenAI
from langgraph.errors import GraphRecursionError
from langgraph.graph import END
from langgraph.graph import START
from langgraph.graph import StateGraph
from typing_extensions import TypedDict


# デモ用のダミーLLM
class DummyLLM:
    """デモ用のダミーLLM（実際のOpenAI APIの代替）"""

    def invoke(self, messages):
        human_content = ""
        for msg in messages:
            if hasattr(msg, "content"):
                human_content = msg.content
                break

        if "計算ステップを抽出" in human_content:
            return type(
                "obj",
                (object,),
                {
                    "content": '[{"operation": "add", "value": 3}, {"operation": "multiply", "value": 2}]'
                },
            )()
        elif "残りのステップ" in human_content:
            return type("obj", (object,), {"content": "いいえ、計算は完了しました。"})()
        else:
            return type("obj", (object,), {"content": "デモ用レスポンス"})()


# LLMの初期化
llm = ChatOpenAI(model="gpt-3.5-turbo")


class CalculatorState(TypedDict):
    """
    LangGraphで使用するState（状態）の定義
    """

    user_input: str
    current_value: float
    remaining_steps: Annotated[List[Dict], operator.add]
    calculation_history: Annotated[List[str], operator.add]
    final_result: str
    is_complete: bool
    step_count: int  # 追加：ステップカウンター


def parse_input_node(state: CalculatorState) -> Dict[str, Any]:
    """
    ノード1: 入力解析ノード
    """
    print(f"🔍 入力解析中: {state['user_input']}")

    # デモ用の簡単な解析
    steps = []
    input_text = state["user_input"].lower()

    # 数値の抽出
    numbers = re.findall(r"\d+", state["user_input"])
    initial_value = float(numbers[0]) if numbers else 10.0

    # パターンマッチングによるステップ抽出
    if "足し" in input_text and "掛け" in input_text:
        if len(numbers) >= 3:
            steps = [
                {"operation": "add", "value": float(numbers[1])},
                {"operation": "multiply", "value": float(numbers[2])},
            ]
    elif "割り" in input_text and "足し" in input_text:
        if len(numbers) >= 3:
            steps = [
                {"operation": "divide", "value": float(numbers[1])},
                {"operation": "add", "value": float(numbers[2])},
            ]

    print(f"📊 抽出されたステップ: {steps}")
    print(f"🔢 初期値: {initial_value}")

    return {
        "current_value": initial_value,
        "remaining_steps": steps,
        "calculation_history": [f"初期値: {initial_value}"],
        "step_count": 0,
    }


def route_calculation_node(
    state: CalculatorState,
) -> Literal[
    "add_node", "subtract_node", "multiply_node", "divide_node", "output_node"
]:
    """
    ノード2: 計算振り分けノード（修正版）

    無限ループを防ぐため、終了条件を明確化
    """
    print(
        f"🔄 ルーティング中 - 残りステップ数: {len(state.get('remaining_steps', []))}"
    )
    print(f"📊 現在のステップカウント: {state.get('step_count', 0)}")

    # 終了条件の確認
    if not state.get("remaining_steps") or len(state.get("remaining_steps", [])) == 0:
        print("✅ 全ての計算が完了しました")
        return "output_node"

    # 安全性チェック：最大ステップ数制限
    if state.get("step_count", 0) >= 10:
        print("⚠️ 最大ステップ数に達しました")
        return "output_node"

    next_step = state["remaining_steps"][0]
    operation = next_step["operation"]

    print(f"🔄 次の演算: {operation}")

    # 演算の種類に応じて適切なノードにルーティング
    if operation == "add":
        return "add_node"
    elif operation == "subtract":
        return "subtract_node"
    elif operation == "multiply":
        return "multiply_node"
    elif operation == "divide":
        return "divide_node"
    else:
        return "output_node"


def add_node(state: CalculatorState) -> Dict[str, Any]:
    """
    ノード3-1: 加算ノード（修正版）
    """
    print("➕ 加算ノード実行中")

    if not state.get("remaining_steps") or len(state.get("remaining_steps", [])) == 0:
        return {"is_complete": True}

    step = state["remaining_steps"][0]
    value_to_add = step["value"]
    new_value = state["current_value"] + value_to_add

    print(f"➕ 加算: {state['current_value']} + {value_to_add} = {new_value}")

    # 実行済みのステップを削除
    remaining_steps = state["remaining_steps"][1:]
    history_entry = f"{state['current_value']} + {value_to_add} = {new_value}"

    return {
        "current_value": new_value,
        "remaining_steps": remaining_steps,
        "calculation_history": [history_entry],
        "step_count": state.get("step_count", 0) + 1,
    }


def subtract_node(state: CalculatorState) -> Dict[str, Any]:
    """
    ノード3-2: 減算ノード（修正版）
    """
    print("➖ 減算ノード実行中")

    if not state.get("remaining_steps") or len(state.get("remaining_steps", [])) == 0:
        return {"is_complete": True}

    step = state["remaining_steps"][0]
    value_to_subtract = step["value"]
    new_value = state["current_value"] - value_to_subtract

    print(f"➖ 減算: {state['current_value']} - {value_to_subtract} = {new_value}")

    remaining_steps = state["remaining_steps"][1:]
    history_entry = f"{state['current_value']} - {value_to_subtract} = {new_value}"

    return {
        "current_value": new_value,
        "remaining_steps": remaining_steps,
        "calculation_history": [history_entry],
        "step_count": state.get("step_count", 0) + 1,
    }


def multiply_node(state: CalculatorState) -> Dict[str, Any]:
    """
    ノード3-3: 乗算ノード（修正版）
    """
    print("✖️ 乗算ノード実行中")

    if not state.get("remaining_steps") or len(state.get("remaining_steps", [])) == 0:
        return {"is_complete": True}

    step = state["remaining_steps"][0]
    value_to_multiply = step["value"]
    new_value = state["current_value"] * value_to_multiply

    print(f"✖️ 乗算: {state['current_value']} × {value_to_multiply} = {new_value}")

    remaining_steps = state["remaining_steps"][1:]
    history_entry = f"{state['current_value']} × {value_to_multiply} = {new_value}"

    return {
        "current_value": new_value,
        "remaining_steps": remaining_steps,
        "calculation_history": [history_entry],
        "step_count": state.get("step_count", 0) + 1,
    }


def divide_node(state: CalculatorState) -> Dict[str, Any]:
    """
    ノード3-4: 除算ノード（修正版）
    """
    print("➗ 除算ノード実行中")

    if not state.get("remaining_steps") or len(state.get("remaining_steps", [])) == 0:
        return {"is_complete": True}

    step = state["remaining_steps"][0]
    value_to_divide = step["value"]

    if value_to_divide == 0:
        print("❌ エラー: ゼロで割ることはできません")
        return {
            "final_result": "エラー: ゼロで割ることはできません",
            "is_complete": True,
            "remaining_steps": [],
        }

    new_value = state["current_value"] / value_to_divide

    print(f"➗ 除算: {state['current_value']} ÷ {value_to_divide} = {new_value}")

    remaining_steps = state["remaining_steps"][1:]
    history_entry = f"{state['current_value']} ÷ {value_to_divide} = {new_value}"

    return {
        "current_value": new_value,
        "remaining_steps": remaining_steps,
        "calculation_history": [history_entry],
        "step_count": state.get("step_count", 0) + 1,
    }


def output_node(state: CalculatorState) -> Dict[str, Any]:
    """
    ノード4: 出力ノード（修正版）
    """
    print("📋 計算結果をまとめています...")

    history_text = "\n".join(state.get("calculation_history", []))

    final_result = f"""
🧮 計算完了！

📊 計算過程:
{history_text}

🎯 最終結果: {state["current_value"]}
"""

    print(final_result)

    return {"final_result": final_result, "is_complete": True}


def create_calculator_graph() -> StateGraph:
    """
    LangGraphの作成関数（修正版）

    無限ループを防ぐため、エッジの構造を修正
    """
    print("🏗️ LangGraphを構築中...")

    graph_builder = StateGraph(CalculatorState)

    # ノードの追加
    graph_builder.add_node("parse_input", parse_input_node)
    graph_builder.add_node("add_node", add_node)
    graph_builder.add_node("subtract_node", subtract_node)
    graph_builder.add_node("multiply_node", multiply_node)
    graph_builder.add_node("divide_node", divide_node)
    graph_builder.add_node("output_node", output_node)

    # エッジの定義（修正版）
    graph_builder.add_edge(START, "parse_input")

    # 入力解析後の条件付きエッジ
    graph_builder.add_conditional_edges(
        "parse_input",
        route_calculation_node,
        {
            "add_node": "add_node",
            "subtract_node": "subtract_node",
            "multiply_node": "multiply_node",
            "divide_node": "divide_node",
            "output_node": "output_node",
        },
    )

    # 各演算ノードから条件付きエッジ（修正版）
    for operation_node in ["add_node", "subtract_node", "multiply_node", "divide_node"]:
        graph_builder.add_conditional_edges(
            operation_node,
            route_calculation_node,
            {
                "add_node": "add_node",
                "subtract_node": "subtract_node",
                "multiply_node": "multiply_node",
                "divide_node": "divide_node",
                "output_node": "output_node",
            },
        )

    # 出力ノードから終了
    graph_builder.add_edge("output_node", END)

    compiled_graph = graph_builder.compile()
    print("✅ LangGraph構築完了！")
    return compiled_graph


def run_calculator(user_input: str) -> str:
    """
    計算システムの実行関数（修正版）

    recursion_limitを設定し、GraphRecursionErrorをキャッチ
    """
    print(f"🚀 計算システム開始: {user_input}")
    print("=" * 50)

    calculator_graph = create_calculator_graph()

    initial_state = {
        "user_input": user_input,
        "current_value": 0.0,
        "remaining_steps": [],
        "calculation_history": [],
        "final_result": "",
        "is_complete": False,
        "step_count": 0,
    }

    try:
        # recursion_limitを設定（検索結果[6][10]による）
        config = {"recursion_limit": 50}
        final_state = calculator_graph.invoke(initial_state, config)

        print("=" * 50)
        print("🎉 計算システム完了！")
        return final_state["final_result"]

    except GraphRecursionError as e:
        print(f"⚠️ 再帰制限に達しました: {e}")
        return "エラー: 計算が複雑すぎるため、処理を中断しました。"
    except Exception as e:
        print(f"❌ エラーが発生しました: {e}")
        return f"エラー: {str(e)}"


# 使用例とテスト実行
if __name__ == "__main__":
    print("🧮 LangGraph 四則演算システム デモ（修正版）")
    print("=" * 60)

    # テスト例1: 加算と乗算
    print("\n📝 テスト例1: 加算と乗算")
    try:
        result1 = run_calculator("5に3を足して、その結果に2を掛けてください")
        print("結果:", result1)
    except Exception as e:
        print(f"エラー: {e}")

    print("\n" + "─" * 40)

    # テスト例2: 除算と加算
    print("\n📝 テスト例2: 除算と加算")
    try:
        result2 = run_calculator("10を5で割って、それに2を足してください")
        print("結果:", result2)
    except Exception as e:
        print(f"エラー: {e}")

"""
🔧 修正点の説明：

1. **recursion_limitの設定**[6][10]:
   - invoke()時にconfigで{"recursion_limit": 50}を設定
   - GraphRecursionErrorをキャッチして適切に処理

2. **無限ループの防止**:
   - step_countを追加してステップ数を追跡
   - 最大ステップ数制限を設けて安全性を確保
   - 終了条件の明確化

3. **デバッグ情報の追加**:
   - 各ノードで詳細なログ出力
   - 状態の変化を追跡可能

4. **エラーハンドリングの強化**:
   - try-except文でGraphRecursionErrorを捕捉
   - 適切なエラーメッセージを表示

5. **ルーティング論理の改善**:
   - 終了条件をより明確に定義
   - remaining_stepsの存在確認を強化
"""
