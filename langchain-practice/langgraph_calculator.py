"""
LangGraph v0.4.8 + LLM を使用した自然言語対応四則演算システム

このコードは以下の機能を提供します：
- 自然言語での計算指示の理解
- 四則演算（足し算、引き算、掛け算、割り算）の個別ノード
- 繰り返し処理ノード
- 計算履歴記録ノード
- 計算途中経過の出力機能
- LLMによる結果の自然言語での説明
"""

# 必要なライブラリのインストール例
# pip install langgraph==0.4.8 langchain langchain-openai pydantic typing python-dotenv

import json
import os
import re
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import TypedDict

from dotenv import load_dotenv
from langchain.schema import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END
from langgraph.graph import StateGraph

# 環境変数を読み込み（OpenAI APIキーなど）
load_dotenv()

# =============================================================================
# ステート（状態）の定義
# =============================================================================


class CalculatorState(TypedDict):
    """
    計算システム全体で共有される状態を定義

    Attributes:
        user_input: ユーザーからの自然言語入力
        current_value: 現在の計算結果
        operand: 演算対象の数値
        operation: 実行する演算の種類
        history: 計算履歴のリスト
        repeat_count: 繰り返し回数
        current_repeat: 現在の繰り返し回数
        intermediate_results: 途中経過の結果
        messages: システムメッセージ
        llm_response: LLMからの応答
        parsed_instruction: 解析された計算指示
        conversation_history: 会話履歴
        error_message: エラーメッセージ
    """

    user_input: str
    current_value: float
    operand: float
    operation: str
    history: List[Dict[str, Any]]
    repeat_count: int
    current_repeat: int
    intermediate_results: List[float]
    messages: List[str]
    llm_response: str
    parsed_instruction: Dict[str, Any]
    conversation_history: List[Dict[str, str]]
    error_message: Optional[str]


# =============================================================================
# LLM設定
# =============================================================================


def get_llm():
    """
    OpenAI GPTモデルを初期化
    環境変数 OPENAI_API_KEY が必要
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("⚠️  OPENAI_API_KEY が設定されていません。")
        print("   環境変数を設定するか、.envファイルを作成してください。")
        print("   デモ用のモックLLMを使用します。")
        return None

    return ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1, api_key=api_key)


# =============================================================================
# モックLLM（APIキーがない場合のフォールバック）
# =============================================================================


class MockLLM:
    """
    APIキーがない場合のためのモックLLM
    """

    def invoke(self, messages):
        """簡単な自然言語解析をシミュレート"""
        content = messages[0].content.lower() if messages else ""

        # 基本的なパターンマッチング
        if (
            "足し" in content
            or "加算" in content
            or "プラス" in content
            or "+" in content
        ):
            operation = "add"
        elif (
            "引き" in content
            or "減算" in content
            or "マイナス" in content
            or "-" in content
        ):
            operation = "subtract"
        elif "掛け" in content or "乗算" in content or "×" in content or "*" in content:
            operation = "multiply"
        elif "割り" in content or "除算" in content or "÷" in content or "/" in content:
            operation = "divide"
        else:
            operation = "add"

        # 数値を抽出
        numbers = re.findall(r"\d+(?:\.\d+)?", content)

        mock_response = {
            "operation": operation,
            "initial_value": float(numbers[0]) if len(numbers) > 0 else 10.0,
            "operand": float(numbers[1]) if len(numbers) > 1 else 5.0,
            "repeat_count": int(float(numbers[2])) if len(numbers) > 2 else 1,
            "explanation": f"モックLLM: {operation}演算を実行します",
        }

        class MockMessage:
            def __init__(self, content):
                self.content = json.dumps(mock_response, ensure_ascii=False)

        return MockMessage(json.dumps(mock_response, ensure_ascii=False))


# =============================================================================
# LLM関連のノード関数
# =============================================================================


def natural_language_parser_node(state: CalculatorState) -> CalculatorState:
    """
    自然言語入力を解析して計算指示を抽出するノード

    Args:
        state: 現在の計算状態

    Returns:
        更新された計算状態（解析結果を含む）
    """
    llm = get_llm() or MockLLM()

    # LLMに送信するプロンプト
    prompt = f"""
    以下のユーザーの自然言語入力を解析して、計算指示をJSONで返してください。

    ユーザー入力: "{state["user_input"]}"

    以下のJSONフォーマットで回答してください：
    {{
        "operation": "add|subtract|multiply|divide",
        "initial_value": 数値,
        "operand": 数値,
        "repeat_count": 繰り返し回数（整数）,
        "explanation": "何を計算するかの説明"
    }}

    例：
    - "10に5を3回足して" → {{"operation": "add", "initial_value": 10, "operand": 5, "repeat_count": 3, "explanation": "10に5を3回足し算します"}}
    - "20から3を2回引いて" → {{"operation": "subtract", "initial_value": 20, "operand": 3, "repeat_count": 2, "explanation": "20から3を2回引き算します"}}
    - "7に4を掛ける" → {{"operation": "multiply", "initial_value": 7, "operand": 4, "repeat_count": 1, "explanation": "7に4を掛け算します"}}
    
    指示が不明確な場合は、妥当なデフォルト値を使用してください。
    """

    try:
        # LLMに問い合わせ
        messages = [HumanMessage(content=prompt)]
        response = llm.invoke(messages)

        # JSONレスポンスを解析
        response_text = response.content

        # JSON部分を抽出（LLMの応答に余計なテキストが含まれる場合があるため）
        json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group()
            parsed_instruction = json.loads(json_str)
        else:
            # JSONが見つからない場合のフォールバック
            parsed_instruction = {
                "operation": "add",
                "initial_value": 10.0,
                "operand": 5.0,
                "repeat_count": 1,
                "explanation": "デフォルトの足し算を実行します",
            }

        # 状態を更新
        new_state = state.copy()
        new_state["parsed_instruction"] = parsed_instruction
        new_state["current_value"] = parsed_instruction["initial_value"]
        new_state["operand"] = parsed_instruction["operand"]
        new_state["operation"] = parsed_instruction["operation"]
        new_state["repeat_count"] = parsed_instruction["repeat_count"]
        new_state["messages"] = state["messages"] + [
            f"🤖 LLM解析結果: {parsed_instruction['explanation']}"
        ]
        new_state["conversation_history"] = state["conversation_history"] + [
            {"role": "user", "content": state["user_input"]},
            {
                "role": "assistant",
                "content": f"解析完了: {parsed_instruction['explanation']}",
            },
        ]

    except Exception as e:
        # エラーハンドリング
        new_state = state.copy()
        new_state["error_message"] = f"自然言語解析エラー: {str(e)}"
        new_state["messages"] = state["messages"] + [f"❌ 解析エラー: {str(e)}"]

        # デフォルト値を設定
        new_state["current_value"] = 10.0
        new_state["operand"] = 5.0
        new_state["operation"] = "add"
        new_state["repeat_count"] = 1

    return new_state


def result_explainer_node(state: CalculatorState) -> CalculatorState:
    """
    計算結果を自然言語で説明するノード

    Args:
        state: 現在の計算状態

    Returns:
        更新された計算状態（LLMによる説明を含む）
    """
    llm = get_llm() or MockLLM()

    # 計算結果の要約を作成
    summary = {
        "initial_request": state.get("user_input", ""),
        "operation_performed": state.get("operation", ""),
        "initial_value": state.get("parsed_instruction", {}).get("initial_value", 0),
        "operand": state.get("operand", 0),
        "repeat_count": state.get("repeat_count", 1),
        "final_result": state.get("current_value", 0),
        "calculation_steps": len(state.get("history", [])),
        "intermediate_results": state.get("intermediate_results", []),
    }

    # LLMに送信するプロンプト
    prompt = f"""
    以下の計算結果を、わかりやすい日本語で説明してください：

    ユーザーの依頼: "{summary["initial_request"]}"
    実行した演算: {summary["operation_performed"]}
    初期値: {summary["initial_value"]}
    演算対象: {summary["operand"]}
    繰り返し回数: {summary["repeat_count"]}
    最終結果: {summary["final_result"]}
    計算ステップ数: {summary["calculation_steps"]}
    途中経過: {summary["intermediate_results"]}

    以下の形式で自然な日本語で説明してください：
    1. ユーザーの依頼の要約
    2. 実行した計算の説明
    3. 最終結果
    4. 計算過程の特徴や注目点（あれば）

    親しみやすく、わかりやすい説明を心がけてください。
    """

    try:
        # LLMに問い合わせ
        messages = [HumanMessage(content=prompt)]
        response = llm.invoke(messages)

        llm_explanation = response.content

        # 状態を更新
        new_state = state.copy()
        new_state["llm_response"] = llm_explanation
        new_state["messages"] = state["messages"] + [
            "🤖 LLMによる結果説明:",
            llm_explanation,
        ]
        new_state["conversation_history"] = state["conversation_history"] + [
            {"role": "assistant", "content": llm_explanation}
        ]

    except Exception as e:
        # エラーハンドリング
        new_state = state.copy()
        new_state["llm_response"] = f"結果の説明中にエラーが発生しました: {str(e)}"
        new_state["messages"] = state["messages"] + [f"❌ 説明生成エラー: {str(e)}"]

    return new_state


# =============================================================================
# 既存の計算ノード関数（前回と同じ）
# =============================================================================


def addition_node(state: CalculatorState) -> CalculatorState:
    """足し算を行うノード"""
    print(f"🔢 足し算実行: {state['current_value']} + {state['operand']}")
    result = state["current_value"] + state["operand"]

    history_entry = {
        "operation": "addition",
        "operand1": state["current_value"],
        "operand2": state["operand"],
        "result": result,
        "timestamp": f"計算 {len(state['history']) + 1}",
    }

    new_state = state.copy()
    new_state["current_value"] = result
    new_state["history"] = state["history"] + [history_entry]
    new_state["intermediate_results"] = state["intermediate_results"] + [result]
    new_state["messages"] = state["messages"] + [
        f"➕ 足し算実行: {state['current_value']} + {state['operand']} = {result}"
    ]

    print(f"✅ 足し算完了: 結果 = {result}")
    return new_state


def subtraction_node(state: CalculatorState) -> CalculatorState:
    """引き算を行うノード"""
    print(f"🔢 引き算実行: {state['current_value']} - {state['operand']}")
    result = state["current_value"] - state["operand"]

    history_entry = {
        "operation": "subtraction",
        "operand1": state["current_value"],
        "operand2": state["operand"],
        "result": result,
        "timestamp": f"計算 {len(state['history']) + 1}",
    }

    new_state = state.copy()
    new_state["current_value"] = result
    new_state["history"] = state["history"] + [history_entry]
    new_state["intermediate_results"] = state["intermediate_results"] + [result]
    new_state["messages"] = state["messages"] + [
        f"➖ 引き算実行: {state['current_value']} - {state['operand']} = {result}"
    ]

    print(f"✅ 引き算完了: 結果 = {result}")
    return new_state


def multiplication_node(state: CalculatorState) -> CalculatorState:
    """掛け算を行うノード"""
    result = state["current_value"] * state["operand"]

    history_entry = {
        "operation": "multiplication",
        "operand1": state["current_value"],
        "operand2": state["operand"],
        "result": result,
        "timestamp": f"計算 {len(state['history']) + 1}",
    }

    new_state = state.copy()
    new_state["current_value"] = result
    new_state["history"] = state["history"] + [history_entry]
    new_state["intermediate_results"] = state["intermediate_results"] + [result]
    new_state["messages"] = state["messages"] + [
        f"✖️ 掛け算実行: {state['current_value']} × {state['operand']} = {result}"
    ]

    return new_state


def division_node(state: CalculatorState) -> CalculatorState:
    """割り算を行うノード（ゼロ除算エラーハンドリング付き）"""
    if state["operand"] == 0:
        new_state = state.copy()
        new_state["error_message"] = "ゼロ除算エラー"
        new_state["messages"] = state["messages"] + [
            "❌ エラー: ゼロで割ることはできません"
        ]
        return new_state

    result = state["current_value"] / state["operand"]

    history_entry = {
        "operation": "division",
        "operand1": state["current_value"],
        "operand2": state["operand"],
        "result": result,
        "timestamp": f"計算 {len(state['history']) + 1}",
    }

    new_state = state.copy()
    new_state["current_value"] = result
    new_state["history"] = state["history"] + [history_entry]
    new_state["intermediate_results"] = state["intermediate_results"] + [result]
    new_state["messages"] = state["messages"] + [
        f"➗ 割り算実行: {state['current_value']} ÷ {state['operand']} = {result}"
    ]

    return new_state


def repeat_controller_node(state: CalculatorState) -> CalculatorState:
    """繰り返し処理を制御するノード"""
    new_state = state.copy()
    current_repeat = state["current_repeat"] + 1
    new_state["current_repeat"] = current_repeat
    new_state["messages"] = state["messages"] + [
        f"🔄 繰り返し {current_repeat}/{state['repeat_count']} 回目を実行中"
    ]

    # デバッグ用メッセージを追加
    debug_msg = f"📊 デバッグ: current_repeat={current_repeat}, repeat_count={state['repeat_count']}"
    new_state["messages"] = new_state["messages"] + [debug_msg]

    return new_state


def history_recorder_node(state: CalculatorState) -> CalculatorState:
    """計算履歴を整理・記録するノード"""
    new_state = state.copy()

    if state["history"]:
        summary_message = f"📊 計算履歴: {len(state['history'])} 回の演算を実行しました"
        new_state["messages"] = state["messages"] + [summary_message]

        if state["intermediate_results"]:
            max_val = max(state["intermediate_results"])
            min_val = min(state["intermediate_results"])
            avg_val = sum(state["intermediate_results"]) / len(
                state["intermediate_results"]
            )

            stats_message = f"📈 統計情報 - 最大値: {max_val}, 最小値: {min_val}, 平均値: {avg_val:.2f}"
            new_state["messages"] = new_state["messages"] + [stats_message]

    return new_state


def output_progress_node(state: CalculatorState) -> CalculatorState:
    """計算の途中経過を出力するノード"""
    new_state = state.copy()

    progress_message = f"⏳ 進捗: 現在値 = {state['current_value']}, 実行済み演算数 = {len(state['history'])}"
    new_state["messages"] = state["messages"] + [progress_message]

    if state["intermediate_results"]:
        results_str = " → ".join([str(r) for r in state["intermediate_results"]])
        results_message = f"🔄 計算経過: {results_str}"
        new_state["messages"] = new_state["messages"] + [results_message]

    return new_state


# =============================================================================
# 条件分岐関数
# =============================================================================


def should_continue_repeat(state: CalculatorState) -> str:
    """繰り返し処理を続けるかどうかを判定"""
    if state.get("error_message"):
        return "result_explainer"  # エラーがある場合は説明へ

    current_repeat = state["current_repeat"]
    repeat_count = state["repeat_count"]

    # デバッグ情報
    print(f"🔍 判定中: current_repeat={current_repeat}, repeat_count={repeat_count}")
    print(f"🔍 実行済み演算数: {len(state.get('history', []))}")

    # 実行済み演算数で判定する方法に変更
    executed_operations = len(state.get("history", []))

    if executed_operations < repeat_count:
        print("→ output_progress (演算実行)")
        return "output_progress"  # まだ繰り返す（演算を実行）
    else:
        print("→ history_recorder (履歴記録)")
        return "history_recorder"  # 繰り返し終了、履歴記録へ


def route_operation(state: CalculatorState) -> str:
    """演算の種類に応じて適切なノードにルーティング"""
    if state.get("error_message"):
        return "result_explainer"  # エラーがある場合は説明へ

    operation = state["operation"]
    print(f"🔀 演算ルーティング: operation={operation}")

    operation_map = {
        "add": "addition",
        "subtract": "subtraction",
        "multiply": "multiplication",
        "divide": "division",
    }

    target = operation_map.get(operation, "output_progress")
    print(f"→ {target} ノードに移行")
    return target


# =============================================================================
# LangGraphの構築（LLMノードを含む）
# =============================================================================


def create_nlp_calculator_graph():
    """
    自然言語処理機能付き計算システムのLangGraphを作成

    Returns:
        設定済みのStateGraph
    """
    graph = StateGraph(CalculatorState)

    # 既存のノードを追加
    graph.add_node("natural_language_parser", natural_language_parser_node)
    graph.add_node("addition", addition_node)
    graph.add_node("subtraction", subtraction_node)
    graph.add_node("multiplication", multiplication_node)
    graph.add_node("division", division_node)
    graph.add_node("repeat_controller", repeat_controller_node)
    graph.add_node("history_recorder", history_recorder_node)
    graph.add_node("output_progress", output_progress_node)
    graph.add_node("result_explainer", result_explainer_node)

    # エントリーポイントを自然言語解析に設定
    graph.set_entry_point("natural_language_parser")

    # フローの構築
    graph.add_edge("natural_language_parser", "repeat_controller")

    # 条件分岐エッジ
    graph.add_conditional_edges(
        "repeat_controller",
        should_continue_repeat,
        {
            "output_progress": "output_progress",  # 演算を実行
            "history_recorder": "history_recorder",  # 履歴記録
            "result_explainer": "result_explainer",  # エラー時の説明
        },
    )

    graph.add_conditional_edges(
        "output_progress",
        route_operation,
        {
            "addition": "addition",
            "subtraction": "subtraction",
            "multiplication": "multiplication",
            "division": "division",
            "result_explainer": "result_explainer",
        },
    )

    # 各演算ノードから繰り返し制御に戻る
    graph.add_edge("addition", "repeat_controller")
    graph.add_edge("subtraction", "repeat_controller")
    graph.add_edge("multiplication", "repeat_controller")
    graph.add_edge("division", "repeat_controller")

    # 履歴記録から結果説明へ
    graph.add_edge("history_recorder", "result_explainer")

    # 結果説明から終了
    graph.add_edge("result_explainer", END)

    return graph.compile()


# =============================================================================
# 対話型インターフェース
# =============================================================================


def create_interactive_calculator():
    """
    対話型の自然言語計算システム
    """
    print("🤖 自然言語対応計算システムへようこそ！")
    print("=" * 60)
    print("💡 使用例:")
    print("  - '10に5を3回足して'")
    print("  - '20から3を引いて'")
    print("  - '7と4を掛けて'")
    print("  - '100を5で2回割って'")
    print("  - 'quit' で終了")
    print("=" * 60)

    calculator_graph = create_nlp_calculator_graph()

    while True:
        try:
            # ユーザー入力を取得
            user_input = input("\n🗣️  計算の指示を入力してください: ").strip()

            if user_input.lower() in ["quit", "exit", "終了", "q"]:
                print("👋 ありがとうございました！")
                break

            if not user_input:
                print("⚠️  何か入力してください。")
                continue

            # 初期状態を設定
            initial_state: CalculatorState = {
                "user_input": user_input,
                "current_value": 0.0,
                "operand": 0.0,
                "operation": "",
                "history": [],
                "repeat_count": 1,
                "current_repeat": 0,
                "intermediate_results": [],
                "messages": ["🚀 計算を開始します"],
                "llm_response": "",
                "parsed_instruction": {},
                "conversation_history": [],
                "error_message": None,
            }

            print("\n⚙️  処理中...")

            # グラフを実行
            result = calculator_graph.invoke(initial_state)

            # 結果を表示
            print("\n" + "=" * 50)
            print("📋 実行結果:")
            print("=" * 50)

            # メッセージを表示
            for i, message in enumerate(result["messages"], 1):
                if message.startswith("🤖 LLMによる結果説明:"):
                    continue  # LLM説明は別途表示
                print(f"{i:2d}. {message}")

            # LLMの説明を表示
            if result.get("llm_response"):
                print("\n🤖 **AI による結果説明:**")
                print("-" * 30)
                print(result["llm_response"])

            # 詳細情報（オプション）
            show_details = input("\n📊 詳細な計算履歴を表示しますか？ (y/n): ").lower()
            if show_details in ["y", "yes", "はい"]:
                print(f"\n📈 最終結果: {result['current_value']}")
                print(f"🔢 実行された演算回数: {len(result['history'])}")

                if result["history"]:
                    print("\n📝 詳細な計算履歴:")
                    for i, entry in enumerate(result["history"], 1):
                        print(
                            f"  {i:2d}. {entry['operation']}: {entry['operand1']} → {entry['result']}"
                        )

                if result["intermediate_results"]:
                    print(
                        f"\n🎯 計算過程: {' → '.join(map(str, result['intermediate_results']))}"
                    )

        except KeyboardInterrupt:
            print("\n👋 終了します。")
            break
        except Exception as e:
            print(f"\n❌ エラーが発生しました: {e}")
            print("もう一度お試しください。")


# =============================================================================
# バッチ実行例
# =============================================================================


def run_batch_examples():
    """
    複数の自然言語例をバッチで実行
    """
    print("🧪 自然言語計算システム バッチテスト")
    print("=" * 60)

    calculator_graph = create_nlp_calculator_graph()

    # テスト用の自然言語入力例
    test_cases = [
        "10に5を3回足して",
        "20から3を2回引いて",
        "7と4を掛けて",
        "100を5で割って",
        "2に3を4回掛けて",
    ]

    for i, test_input in enumerate(test_cases, 1):
        print(f"\n--- テストケース {i} ---")
        print(f"入力: 「{test_input}」")

        initial_state: CalculatorState = {
            "user_input": test_input,
            "current_value": 0.0,
            "operand": 0.0,
            "operation": "",
            "history": [],
            "repeat_count": 1,
            "current_repeat": 0,
            "intermediate_results": [],
            "messages": [],
            "llm_response": "",
            "parsed_instruction": {},
            "conversation_history": [],
            "error_message": None,
        }

        try:
            result = calculator_graph.invoke(initial_state)
            print(f"結果: {result['current_value']}")
            if result.get("parsed_instruction", {}).get("explanation"):
                print(f"解析: {result['parsed_instruction']['explanation']}")
            if result.get("llm_response"):
                print(f"AI説明: {result['llm_response'][:100]}...")  # 最初の100文字
        except Exception as e:
            print(f"エラー: {e}")


# =============================================================================
# メイン実行部分
# =============================================================================

if __name__ == "__main__":
    import sys

    print("🌟 LangGraph + LLM 自然言語対応計算システム")
    print("=" * 60)

    # 実行モードを選択
    if len(sys.argv) > 1 and sys.argv[1] == "batch":
        # バッチモード
        run_batch_examples()
    else:
        # 対話モード
        print(
            "💡 OPENAI_API_KEY を環境変数に設定すると、より高精度な自然言語処理が利用できます。"
        )
        print("   設定されていない場合は、簡易的なモック処理を使用します。")
        print()

        create_interactive_calculator()

# =============================================================================
# 環境設定用の.envファイル例
# =============================================================================

"""
.envファイルの例（プロジェクトルートに配置）:

OPENAI_API_KEY=your_openai_api_key_here

使用方法:
1. OpenAI APIキーを取得: https://platform.openai.com/api-keys
2. プロジェクトルートに .env ファイルを作成
3. 上記の内容でAPIキーを設定
4. このスクリプトを実行

注意: .env ファイルは .gitignore に追加してください
"""

# =============================================================================
# 📚 新機能の解説
# =============================================================================

"""
🆕 追加された自然言語処理機能:

1. natural_language_parser_node:
   - ユーザーの自然言語入力をLLMで解析
   - 計算指示を構造化データに変換
   - JSONフォーマットで演算種類、数値、繰り返し回数を抽出

2. result_explainer_node:
   - 計算結果をLLMが自然言語で説明
   - ユーザーにとって理解しやすい形で結果を提示
   - 計算過程の特徴や注目点も含めて説明

3. 対話型インターフェース:
   - ユーザーとの自然な対話を実現
   - リアルタイムでの自然言語入力処理
   - 詳細情報の表示オプション

4. エラーハンドリング:
   - API キーが未設定の場合のモックLLM
   - 自然言語解析失敗時のフォールバック
   - ゼロ除算などの計算エラーの適切な処理

5. 会話履歴の管理:
   - ユーザーとAIの対話履歴を記録
   - 文脈を考慮した応答の生成

🚀 使用方法:
1. 対話モード: python langgraph_calculator_with_llm.py
2. バッチモード: python langgraph_calculator_with_llm.py batch

💡 自然言語入力例:
- "10に5を3回足して"
- "20から3を引いて" 
- "7と4を掛けて"
- "100を5で2回割って"
- "2の3乗を計算して"（掛け算の繰り返しとして解釈）
"""
