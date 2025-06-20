"""
LangGraph v0.4.8を使用した四則演算計算システム

必要なライブラリのインストール:
pip install langgraph==0.4.8 langchain==0.1.8 langchain-openai==0.0.5 pydantic typing
"""

# 必要なライブラリのインポート
import json
from typing import Any
from typing import Dict
from typing import List
from typing import Literal
from typing import TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage
from langchain_core.messages import HumanMessage
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph

load_dotenv()

# ------------- 状態クラスの定義 -------------


class CalcState(TypedDict):
    """
    計算グラフの状態を管理するクラス

    LangGraphでは、グラフの状態を TypedDict で管理するのが一般的です。
    各フィールドは計算プロセス中に必要な情報を保持します。
    """

    # ユーザーからの入力プロンプト
    prompt: str

    # 抽出された計算ステップのリスト
    steps: List[Dict[str, Any]]

    # 現在処理中のステップのインデックス
    current_step_index: int

    # 現在の計算結果（途中経過）
    current_result: float

    # 計算が完了したかどうかのフラグ
    is_finished: bool

    # メッセージの履歴（処理の過程を保存）
    messages: List[Any]


# ------------- ノード関数の定義 -------------


def extract_calculation_steps(state: CalcState) -> CalcState:
    """
    入力プロンプトから計算ステップを抽出するノード

    このノードはLLMを使って自然言語から計算手順を構造化されたデータに変換します。
    """
    # LLMの初期化（GPT-3.5-turboを使用）
    llm = ChatOpenAI(model="gpt-3.5-turbo")

    # LLMに渡すシステムプロンプトを作成
    system_prompt = """
    あなたは計算ステップを抽出するAIアシスタントです。
    ユーザーの自然言語による計算要求から、実行すべき計算ステップを順番に抽出してください。
    
    出力は以下の形式のJSONリストにしてください：
    [
        {
            "operation": "add"（足し算）、"subtract"（引き算）、"multiply"（掛け算）、"divide"（割り算）のいずれか,
            "value1": 最初の数値（最初のステップでは入力値、それ以降は前回の計算結果を表す "result" を使用）,
            "value2": 二番目の数値
        },
        ...
    ]
    
    例：
    入力: "5に3を足して、その結果に2を掛けてください"
    出力: [{"operation": "add", "value1": 5, "value2": 3}, {"operation": "multiply", "value1": "result", "value2": 2}]
    
    入力: "10を2で割って、その後に5を引いてください"
    出力: [{"operation": "divide", "value1": 10, "value2": 2}, {"operation": "subtract", "value1": "result", "value2": 5}]
    
    必ず有効なJSONのみを返してください。
    """

    # LLMに送信するメッセージを作成
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=state["prompt"]),
    ]

    # LLMを呼び出して計算ステップを抽出
    response = llm.invoke(messages)

    # LLMの応答からJSON部分を抽出して解析
    try:
        # 応答テキストからJSONを抽出
        response_text = response.content
        # JSON文字列を解析してPythonオブジェクトに変換
        steps = json.loads(response_text)
    except Exception as e:
        # JSON解析に失敗した場合は、エラーメッセージを表示して空のステップリストを設定
        print(f"ステップの解析に失敗しました: {e}")
        steps = []

    # 状態を更新して返す
    new_state = state.copy()
    new_state["steps"] = steps
    new_state["current_step_index"] = 0
    new_state["current_result"] = 0
    new_state["is_finished"] = False
    new_state["messages"] = state.get("messages", []) + [
        AIMessage(
            content=f"計算ステップを抽出しました:\n{json.dumps(steps, indent=2, ensure_ascii=False)}"
        )
    ]

    return new_state


def router(
    state: CalcState,
) -> Literal[
    "add", "subtract", "multiply", "divide", "check_completion", "output_result"
]:
    """
    次に実行すべきノードを決定するルーターノード

    このノードは現在の状態に基づいて、次に実行すべきノードを決定します。
    LangGraphでは、このようなルーター関数を使って条件分岐を実現します。
    """
    # 計算が終了している場合は結果出力ノードへ
    if state["is_finished"]:
        return "output_result"

    # すべてのステップを処理済みの場合は完了チェックノードへ
    if state["current_step_index"] >= len(state["steps"]):
        return "check_completion"

    # 現在のステップを取得
    current_step = state["steps"][state["current_step_index"]]
    operation = current_step["operation"]

    # 演算子に応じて適切なノードへルーティング
    if operation == "add":
        return "add"
    elif operation == "subtract":
        return "subtract"
    elif operation == "multiply":
        return "multiply"
    elif operation == "divide":
        return "divide"
    else:
        # 未知の演算子の場合は完了チェックノードへ
        return "check_completion"


def add_operation(state: CalcState) -> CalcState:
    """
    加算を行うノード

    このノードは2つの数値を受け取り、足し算を実行します。
    """
    new_state = state.copy()
    current_step = state["steps"][state["current_step_index"]]

    # value1が"result"の場合は、前回の計算結果を使用
    value1 = (
        state["current_result"]
        if current_step["value1"] == "result"
        else float(current_step["value1"])
    )
    value2 = float(current_step["value2"])

    # 加算を実行
    result = value1 + value2

    # 状態を更新
    new_state["current_result"] = result
    new_state["current_step_index"] += 1
    new_state["messages"] = state["messages"] + [
        AIMessage(
            content=f"計算ステップ {state['current_step_index'] + 1}: {value1} + {value2} = {result}"
        )
    ]

    return new_state


def subtract_operation(state: CalcState) -> CalcState:
    """
    減算を行うノード

    このノードは2つの数値を受け取り、引き算を実行します。
    """
    new_state = state.copy()
    current_step = state["steps"][state["current_step_index"]]

    # value1が"result"の場合は、前回の計算結果を使用
    value1 = (
        state["current_result"]
        if current_step["value1"] == "result"
        else float(current_step["value1"])
    )
    value2 = float(current_step["value2"])

    # 減算を実行
    result = value1 - value2

    # 状態を更新
    new_state["current_result"] = result
    new_state["current_step_index"] += 1
    new_state["messages"] = state["messages"] + [
        AIMessage(
            content=f"計算ステップ {state['current_step_index'] + 1}: {value1} - {value2} = {result}"
        )
    ]

    return new_state


def multiply_operation(state: CalcState) -> CalcState:
    """
    乗算を行うノード

    このノードは2つの数値を受け取り、掛け算を実行します。
    """
    new_state = state.copy()
    current_step = state["steps"][state["current_step_index"]]

    # value1が"result"の場合は、前回の計算結果を使用
    value1 = (
        state["current_result"]
        if current_step["value1"] == "result"
        else float(current_step["value1"])
    )
    value2 = float(current_step["value2"])

    # 乗算を実行
    result = value1 * value2

    # 状態を更新
    new_state["current_result"] = result
    new_state["current_step_index"] += 1
    new_state["messages"] = state["messages"] + [
        AIMessage(
            content=f"計算ステップ {state['current_step_index'] + 1}: {value1} × {value2} = {result}"
        )
    ]

    return new_state


def divide_operation(state: CalcState) -> CalcState:
    """
    除算を行うノード

    このノードは2つの数値を受け取り、割り算を実行します。
    ゼロ除算のエラー処理も含まれています。
    """
    new_state = state.copy()
    current_step = state["steps"][state["current_step_index"]]

    # value1が"result"の場合は、前回の計算結果を使用
    value1 = (
        state["current_result"]
        if current_step["value1"] == "result"
        else float(current_step["value1"])
    )
    value2 = float(current_step["value2"])

    # ゼロ除算のチェック
    if value2 == 0:
        new_state["messages"] = state["messages"] + [
            AIMessage(content="エラー: ゼロで割ることはできません。")
        ]
        new_state["is_finished"] = True
        return new_state

    # 除算を実行
    result = value1 / value2

    # 状態を更新
    new_state["current_result"] = result
    new_state["current_step_index"] += 1
    new_state["messages"] = state["messages"] + [
        AIMessage(
            content=f"計算ステップ {state['current_step_index'] + 1}: {value1} ÷ {value2} = {result}"
        )
    ]

    return new_state


def check_completion(state: CalcState) -> CalcState:
    """
    計算ステップが完了したかを確認するノード

    このノードは、すべての計算ステップが完了したかどうかを確認し、
    完了していれば is_finished フラグを True に設定します。
    """
    new_state = state.copy()

    # すべてのステップが完了しているかチェック
    if state["current_step_index"] >= len(state["steps"]):
        new_state["is_finished"] = True
        new_state["messages"] = state["messages"] + [
            AIMessage(
                content=f"計算が完了しました。最終結果: {state['current_result']}"
            )
        ]

    return new_state


def output_result(state: CalcState) -> CalcState:
    """
    最終結果を出力するノード

    このノードは最終的な計算結果を出力します。
    LangGraphのフローでは、最後に呼び出されるノードです。
    """
    # すでにメッセージに結果が含まれているので、状態をそのまま返す
    return state


# ------------- グラフの構築 -------------


def build_calculator_graph():
    """
    計算グラフを構築する関数

    この関数はLangGraphの核心部分で、ノード間の接続やフローを定義します。
    """
    # 状態を管理するグラフの初期化
    # CalcStateという型を指定して、グラフ内で扱う状態の型を定義します
    graph = StateGraph(CalcState)

    # ノードの追加
    # add_nodeメソッドで各処理をグラフのノードとして登録します
    graph.add_node("extract_steps", extract_calculation_steps)
    graph.add_node("add", add_operation)
    graph.add_node("subtract", subtract_operation)
    graph.add_node("multiply", multiply_operation)
    graph.add_node("divide", divide_operation)
    graph.add_node("check_completion", check_completion)
    graph.add_node("output_result", output_result)

    # エッジ（ノード間の接続）の設定
    # router関数の戻り値に基づいて次のノードを決定するため、
    # routerノードからの接続を条件付きエッジとして定義します
    graph.add_conditional_edges(
        # 条件分岐の起点となるノード
        "extract_steps",
        # 条件分岐を行う関数
        router,
        # 条件分岐の接続先
        {
            "add": "add",
            "subtract": "subtract",
            "multiply": "multiply",
            "divide": "divide",
            "check_completion": "check_completion",
            "output_result": "output_result",
        },
    )

    # 各演算ノードからrouterノードへの接続
    # 演算が完了したら、再びrouterを通じて次のステップへ進みます
    graph.add_conditional_edges(
        "add",
        router,
        {
            "add": "add",
            "subtract": "subtract",
            "multiply": "multiply",
            "divide": "divide",
            "check_completion": "check_completion",
            "output_result": "output_result",
        },
    )

    graph.add_conditional_edges(
        "subtract",
        router,
        {
            "add": "add",
            "subtract": "subtract",
            "multiply": "multiply",
            "divide": "divide",
            "check_completion": "check_completion",
            "output_result": "output_result",
        },
    )

    graph.add_conditional_edges(
        "multiply",
        router,
        {
            "add": "add",
            "subtract": "subtract",
            "multiply": "multiply",
            "divide": "divide",
            "check_completion": "check_completion",
            "output_result": "output_result",
        },
    )

    graph.add_conditional_edges(
        "divide",
        router,
        {
            "add": "add",
            "subtract": "subtract",
            "multiply": "multiply",
            "divide": "divide",
            "check_completion": "check_completion",
            "output_result": "output_result",
        },
    )

    graph.add_conditional_edges(
        "check_completion",
        router,
        {
            "add": "add",
            "subtract": "subtract",
            "multiply": "multiply",
            "divide": "divide",
            "check_completion": "check_completion",
            "output_result": "output_result",
        },
    )

    # 開始ノードと終了ノードの設定
    # エントリーポイント（最初に実行されるノード）の設定
    graph.set_entry_point("extract_steps")

    # グラフのコンパイル（実行可能な形式に変換）
    return graph.compile()


# ------------- メイン実行部分 -------------


def main():
    """
    メイン関数

    このシステムを対話的に実行するためのメイン関数です。
    """
    # 計算グラフの構築
    calculator = build_calculator_graph()

    print(calculator.get_graph(xray=True).draw_mermaid())

    print("==== 四則演算計算機へようこそ！ ====")
    print("例: '5に3を足して、その結果に2を掛けてください'")
    print("終了するには 'exit' と入力してください")
    print("=====================================")

    while True:
        # ユーザー入力の受け取り
        user_input = input("\n計算したい内容を入力してください: ")

        # 終了コマンドのチェック
        if user_input.lower() in ["exit", "quit", "終了"]:
            print("計算機を終了します。ありがとうございました。")
            break

        # 初期状態の設定
        initial_state = {
            "prompt": user_input,
            "steps": [],
            "current_step_index": 0,
            "current_result": 0,
            "is_finished": False,
            "messages": [],
        }

        try:
            # グラフの実行
            result = calculator.invoke(initial_state)

            # 結果の表示
            print("\n--- 計算過程 ---")
            for message in result["messages"]:
                if hasattr(message, "content"):
                    print(message.content)

            # 最終結果の強調表示
            if result["is_finished"]:
                print("\n============================")
                print(f"最終結果: {result['current_result']}")
                print("============================")
        except Exception as e:
            print(f"エラーが発生しました: {e}")


# スクリプトが直接実行された場合に main() 関数を呼び出す
if __name__ == "__main__":
    main()
