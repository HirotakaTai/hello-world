#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LangGraph版 Agents Pattern
LangGraphを使用して自律的に判断・行動するエージェントを実装するパターン
"""

import datetime
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import TypedDict

from dotenv import load_dotenv
from langchain.tools import BaseTool
from langchain_core.messages import HumanMessage
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.pydantic_v1 import Field

# LangChain関連のインポート
from langchain_openai import ChatOpenAI
from langgraph.graph import END

# LangGraph関連のインポート
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolExecutor
from langgraph.prebuilt import ToolInvocation

# ===== 環境変数の読み込み =====
load_dotenv()

# ===== ツールクラスの定義 =====


class WebSearchInput(BaseModel):
    """Web検索ツールの入力スキーマ"""

    query: str = Field(description="検索クエリ")
    max_results: int = Field(default=5, description="最大検索結果数")


class WebSearchTool(BaseTool):
    """Web検索ツール（モック実装）"""

    name = "web_search"
    description = "Webから情報を検索します。最新の情報やニュースを取得できます。"
    args_schema = WebSearchInput

    def _run(self, query: str, max_results: int = 5) -> str:
        """Web検索を実行（モック実装）"""
        # 実際の実装では、Google Search APIやSerpAPIなどを使用
        mock_results = [
            f"【検索結果1】{query}に関する最新情報が発見されました。詳細な分析が利用可能です。",
            f"【検索結果2】{query}の専門家による解説記事が公開されています。",
            f"【検索結果3】{query}に関する最新の統計データが更新されました。",
            f"【検索結果4】{query}の実践事例とベストプラクティスが紹介されています。",
            f"【検索結果5】{query}に関する今後の展望と予測が議論されています。",
        ]

        results = mock_results[:max_results]
        return f"Web検索結果 (クエリ: '{query}'):\n" + "\n".join(results)


class FileOperationInput(BaseModel):
    """ファイル操作ツールの入力スキーマ"""

    operation: str = Field(description="操作タイプ (read, write, append)")
    filename: str = Field(description="ファイル名")
    content: Optional[str] = Field(
        default="", description="書き込み内容 (write/append時)"
    )


class FileOperationTool(BaseTool):
    """ファイル操作ツール"""

    name = "file_operation"
    description = "ファイルの読み書きを行います。テキストファイルの読み取り、作成、追記が可能です。"
    args_schema = FileOperationInput

    def _run(self, operation: str, filename: str, content: str = "") -> str:
        """ファイル操作を実行"""
        try:
            if operation == "read":
                try:
                    with open(filename, "r", encoding="utf-8") as f:
                        file_content = f.read()
                    return f"ファイル '{filename}' の内容:\n{file_content}"
                except FileNotFoundError:
                    return f"ファイル '{filename}' が見つかりません。"

            elif operation == "write":
                with open(filename, "w", encoding="utf-8") as f:
                    f.write(content)
                return f"ファイル '{filename}' に内容を書き込みました。"

            elif operation == "append":
                with open(filename, "a", encoding="utf-8") as f:
                    f.write(content)
                return f"ファイル '{filename}' に内容を追記しました。"

            else:
                return f"サポートされていない操作: {operation}"

        except Exception as e:
            return f"ファイル操作エラー: {str(e)}"


class DataAnalysisInput(BaseModel):
    """データ分析ツールの入力スキーマ"""

    data: str = Field(description="分析するデータ（CSV形式やJSON形式）")
    analysis_type: str = Field(description="分析タイプ (summary, trend, correlation)")


class DataAnalysisTool(BaseTool):
    """データ分析ツール"""

    name = "data_analysis"
    description = (
        "データの分析を行います。統計サマリー、トレンド分析、相関分析が可能です。"
    )
    args_schema = DataAnalysisInput

    def _run(self, data: str, analysis_type: str) -> str:
        """データ分析を実行（簡易実装）"""
        try:
            # 実際の実装では、pandasやnumpyを使用してより詳細な分析を行う
            if analysis_type == "summary":
                return f"データサマリー分析結果:\n- データポイント数: 約{len(data.split())}個\n- データ型: テキスト形式\n- 基本統計: 平均値、中央値、標準偏差を計算"
            elif analysis_type == "trend":
                return "トレンド分析結果:\n- 時系列データのトレンド: 上昇傾向\n- 季節性: 検出されました\n- 異常値: 3個検出"
            elif analysis_type == "correlation":
                return "相関分析結果:\n- 強い正の相関: 変数A-B間 (r=0.85)\n- 弱い負の相関: 変数C-D間 (r=-0.32)\n- 統計的有意性: p<0.05"
            else:
                return f"サポートされていない分析タイプ: {analysis_type}"
        except Exception as e:
            return f"データ分析エラー: {str(e)}"


class TaskPlanningInput(BaseModel):
    """タスク計画ツールの入力スキーマ"""

    goal: str = Field(description="達成したい目標")
    resources: str = Field(description="利用可能なリソース")
    constraints: str = Field(description="制約条件")


class TaskPlanningTool(BaseTool):
    """タスク計画ツール"""

    name = "task_planning"
    description = "目標達成のためのタスク計画を作成します。リソースと制約を考慮した実行可能な計画を生成します。"
    args_schema = TaskPlanningInput

    def _run(self, goal: str, resources: str, constraints: str) -> str:
        """タスク計画を作成"""
        plan = f"""
タスク計画 - 目標: {goal}

【Phase 1: 準備段階】
1. 要件定義と現状分析
2. リソース確保と配分
3. リスク評価と対策立案

【Phase 2: 実行段階】
4. 優先度の高いタスクから実行
5. 進捗モニタリングと調整
6. 品質チェックとレビュー

【Phase 3: 完了段階】
7. 最終検証とテスト
8. ドキュメント作成
9. 成果報告と評価

利用可能リソース: {resources}
制約条件: {constraints}

推定期間: 4-6週間
成功指標: 目標達成率90%以上
        """
        return plan.strip()


# ===== 状態定義 =====


class AgentState(TypedDict):
    """エージェントワークフローの状態定義"""

    user_goal: str  # ユーザーの目標
    current_plan: str  # 現在の計画
    completed_actions: List[str]  # 完了したアクション
    gathered_information: List[str]  # 収集した情報
    analysis_results: List[str]  # 分析結果
    next_action: str  # 次のアクション
    final_result: str  # 最終結果
    step_count: int  # ステップ数
    max_steps: int  # 最大ステップ数
    execution_log: List[str]  # 实行ログ


# ===== LangGraphベースのAgentsクラス =====


class LangGraphAgents:
    """LangGraphを使用した自律エージェントシステム"""

    def __init__(self):
        """エージェントシステムの初期化"""
        print("🤖 LangGraph版 Agentsシステムを初期化中...")

        # OpenAI ChatLLMモデルを初期化
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3, verbose=True)

        # 利用可能なツールを定義
        self.tools = [
            WebSearchTool(),
            FileOperationTool(),
            DataAnalysisTool(),
            TaskPlanningTool(),
        ]

        # ツール実行器を作成
        self.tool_executor = ToolExecutor(self.tools)

        # LLMにツール情報をバインド
        self.llm_with_tools = self.llm.bind_tools(self.tools)

        # プロンプトテンプレートを設定
        self._setup_prompts()

        # LangGraphワークフローを構築
        self.graph = self._build_graph()

        print("✅ 自律エージェントシステムの初期化が完了しました！")

    def _setup_prompts(self):
        """エージェント用のプロンプトを設定"""

        self.agent_system_prompt = """あなたは高度な自律エージェントです。ユーザーの目標を達成するために、以下の能力を持っています:

利用可能なツール:
1. web_search: Web検索で最新情報を取得
2. file_operation: ファイルの読み書き
3. data_analysis: データ分析と統計処理
4. task_planning: タスク計画の作成

エージェントとしての行動指針:
1. ユーザーの目標を正確に理解する
2. 目標達成に必要な情報を収集する
3. 適切なツールを選択して実行する
4. 結果を分析して次のアクションを決定する
5. 段階的に目標に近づく
6. 最終的に具体的で実用的な結果を提供する

現在の状況:
- ユーザー目標: {user_goal}
- 完了したアクション: {completed_actions}
- 収集した情報: {gathered_information}
- 分析結果: {analysis_results}
- ステップ数: {step_count}/{max_steps}

次に取るべき最適なアクションを判断し、必要に応じてツールを使用してください。
目標達成に向けて具体的で実用的な行動を取ってください。"""

    def _build_graph(self) -> StateGraph:
        """エージェントワークフローを構築"""
        print("🔧 エージェントワークフローを構築中...")

        workflow = StateGraph(AgentState)

        # ノード（処理ステップ）を追加
        workflow.add_node("plan_initial", self._plan_initial_action)  # 初期計画
        workflow.add_node("decide_action", self._decide_next_action)  # アクション決定
        workflow.add_node("execute_action", self._execute_action)  # アクション実行
        workflow.add_node("analyze_progress", self._analyze_progress)  # 進捗分析
        workflow.add_node("finalize_result", self._finalize_result)  # 結果確定

        # エントリーポイントを設定
        workflow.set_entry_point("plan_initial")

        # 経路を設定
        workflow.add_edge("plan_initial", "decide_action")
        workflow.add_edge("execute_action", "analyze_progress")

        # 条件分岐を設定
        workflow.add_conditional_edges(
            "decide_action",
            self._should_execute_action,
            {"execute": "execute_action", "finalize": "finalize_result"},
        )

        workflow.add_conditional_edges(
            "analyze_progress",
            self._should_continue,
            {"continue": "decide_action", "finish": "finalize_result"},
        )

        workflow.add_edge("finalize_result", END)

        return workflow.compile()

    def _plan_initial_action(self, state: AgentState) -> Dict[str, Any]:
        """初期アクション計画ステップ"""
        print("📋 ステップ1: 初期アクション計画を作成中...")

        user_goal = state["user_goal"]

        # 初期計画を作成
        initial_plan = f"""
目標: {user_goal}

初期計画:
1. 目標の詳細分析と要件整理
2. 必要な情報とリソースの特定
3. 段階的なアプローチの策定
4. 実行可能なアクションの優先順位付け
5. 成功指標の設定
        """.strip()

        # 実行ログを更新
        log_entry = f"[{datetime.datetime.now().strftime('%H:%M:%S')}] 初期計画作成完了"
        execution_log = state.get("execution_log", [])
        execution_log.append(log_entry)

        print("✅ 初期計画作成完了")

        return {
            "current_plan": initial_plan,
            "step_count": 1,
            "execution_log": execution_log,
        }

    def _decide_next_action(self, state: AgentState) -> Dict[str, Any]:
        """次のアクション決定ステップ"""
        print(f"🤔 ステップ{state['step_count']}: 次のアクションを決定中...")

        # エージェントプロンプトを生成
        prompt = self.agent_system_prompt.format(
            user_goal=state["user_goal"],
            completed_actions=", ".join(state.get("completed_actions", [])),
            gathered_information=", ".join(state.get("gathered_information", [])),
            analysis_results=", ".join(state.get("analysis_results", [])),
            step_count=state["step_count"],
            max_steps=state["max_steps"],
        )

        # LLMを呼び出してアクションを決定
        response = self.llm_with_tools.invoke([HumanMessage(content=prompt)])

        # ツール呼び出しがあるかチェック
        if hasattr(response, "tool_calls") and response.tool_calls:
            next_action = f"ツール実行: {response.tool_calls[0]['name']}"
        else:
            next_action = "分析・検討"

        # 実行ログを更新
        log_entry = f"[{datetime.datetime.now().strftime('%H:%M:%S')}] 次のアクション決定: {next_action}"
        execution_log = state["execution_log"]
        execution_log.append(log_entry)

        print(f"✅ 次のアクション決定: {next_action}")

        return {
            "next_action": next_action,
            "execution_log": execution_log,
            "_llm_response": response,  # 内部使用のため
        }

    def _should_execute_action(self, state: AgentState) -> str:
        """アクション実行の判定（条件分岐関数）"""
        # ステップ数が上限に達した場合は終了
        if state["step_count"] >= state["max_steps"]:
            return "finalize"

        # LLM応答にツール呼び出しがある場合は実行
        llm_response = state.get("_llm_response")
        if (
            llm_response
            and hasattr(llm_response, "tool_calls")
            and llm_response.tool_calls
        ):
            return "execute"
        else:
            return "finalize"

    def _execute_action(self, state: AgentState) -> Dict[str, Any]:
        """アクション実行ステップ"""
        print(f"⚡ ステップ{state['step_count']}: アクションを実行中...")

        llm_response = state.get("_llm_response")

        if not (
            llm_response
            and hasattr(llm_response, "tool_calls")
            and llm_response.tool_calls
        ):
            return {"execution_log": state["execution_log"]}

        # ツール実行結果を収集
        execution_results = []

        for tool_call in llm_response.tool_calls:
            print(f"🔧 ツール '{tool_call['name']}' を実行中...")

            try:
                # ツール実行
                action = ToolInvocation(
                    tool=tool_call["name"], tool_input=tool_call["args"]
                )
                result = self.tool_executor.invoke(action)
                execution_results.append(f"{tool_call['name']}: {str(result)}")

                print(f"✅ ツール実行完了: {tool_call['name']}")

            except Exception as e:
                error_result = f"{tool_call['name']}: エラー - {str(e)}"
                execution_results.append(error_result)
                print(f"❌ ツール実行エラー: {e}")

        # 完了したアクションを更新
        completed_actions = state.get("completed_actions", [])
        completed_actions.extend(execution_results)

        # 情報と分析結果を分類して保存
        gathered_information = state.get("gathered_information", [])
        analysis_results = state.get("analysis_results", [])

        for result in execution_results:
            if "web_search" in result or "file_operation" in result:
                gathered_information.append(result)
            elif "data_analysis" in result or "task_planning" in result:
                analysis_results.append(result)

        # 実行ログを更新
        log_entry = f"[{datetime.datetime.now().strftime('%H:%M:%S')}] アクション実行完了: {len(execution_results)}個"
        execution_log = state["execution_log"]
        execution_log.append(log_entry)

        print(f"✅ アクション実行完了: {len(execution_results)}個の結果")

        return {
            "completed_actions": completed_actions,
            "gathered_information": gathered_information,
            "analysis_results": analysis_results,
            "execution_log": execution_log,
        }

    def _analyze_progress(self, state: AgentState) -> Dict[str, Any]:
        """進捗分析ステップ"""
        print(f"📊 ステップ{state['step_count']}: 進捗を分析中...")

        # ステップ数を増加
        step_count = state["step_count"] + 1

        # 進捗分析（簡易実装）
        completed_actions = state.get("completed_actions", [])
        gathered_information = state.get("gathered_information", [])
        analysis_results = state.get("analysis_results", [])

        progress_summary = f"""
進捗サマリー (ステップ {step_count - 1}):
- 完了アクション: {len(completed_actions)}個
- 収集情報: {len(gathered_information)}個
- 分析結果: {len(analysis_results)}個
        """.strip()

        # 実行ログを更新
        log_entry = f"[{datetime.datetime.now().strftime('%H:%M:%S')}] 進捗分析完了: ステップ{step_count - 1}"
        execution_log = state["execution_log"]
        execution_log.append(log_entry)

        print(f"✅ 進捗分析完了: ステップ{step_count - 1}")

        return {"step_count": step_count, "execution_log": execution_log}

    def _should_continue(self, state: AgentState) -> str:
        """継続判定（条件分岐関数）"""
        step_count = state["step_count"]
        max_steps = state["max_steps"]

        # 最大ステップ数に達した場合は終了
        if step_count >= max_steps:
            return "finish"

        # まだ実行可能なアクションがある場合は継続
        completed_actions = state.get("completed_actions", [])
        if len(completed_actions) < 3:  # 最低3つのアクションを実行
            return "continue"

        return "finish"

    def _finalize_result(self, state: AgentState) -> Dict[str, Any]:
        """結果確定ステップ"""
        print("🎯 最終ステップ: 結果を確定中...")

        user_goal = state["user_goal"]
        completed_actions = state.get("completed_actions", [])
        gathered_information = state.get("gathered_information", [])
        analysis_results = state.get("analysis_results", [])

        # 最終結果をまとめる
        final_result = f"""
【目標達成レポート】

目標: {user_goal}

実行サマリー:
- 総ステップ数: {state["step_count"]}
- 完了アクション: {len(completed_actions)}個
- 収集情報: {len(gathered_information)}個
- 分析結果: {len(analysis_results)}個

【主要な成果】
{chr(10).join([f"- {action}" for action in completed_actions[-3:]])}

【収集した情報】
{chr(10).join([f"- {info}" for info in gathered_information[-2:]])}

【分析結果】
{chr(10).join([f"- {result}" for result in analysis_results[-2:]])}

【結論】
エージェントは自律的に{len(completed_actions)}個のアクションを実行し、
目標達成に向けて段階的に進展しました。収集した情報と分析結果により、
実用的で具体的な成果を提供できました。
        """.strip()

        # 実行ログを更新
        log_entry = f"[{datetime.datetime.now().strftime('%H:%M:%S')}] 最終結果確定完了"
        execution_log = state["execution_log"]
        execution_log.append(log_entry)

        print("✅ 最終結果確定完了")

        return {"final_result": final_result, "execution_log": execution_log}

    def execute_goal(self, user_goal: str, max_steps: int = 5) -> Dict[str, Any]:
        """目標実行のメイン関数"""
        print("🚀 エージェント実行開始")
        print(f"目標: {user_goal}")
        print(f"最大ステップ数: {max_steps}")
        print("-" * 60)

        # 初期状態を設定
        initial_state = {
            "user_goal": user_goal,
            "current_plan": "",
            "completed_actions": [],
            "gathered_information": [],
            "analysis_results": [],
            "next_action": "",
            "final_result": "",
            "step_count": 0,
            "max_steps": max_steps,
            "execution_log": [],
        }

        # ワークフローを実行
        start_time = datetime.datetime.now()
        result = self.graph.invoke(initial_state)
        end_time = datetime.datetime.now()

        # 実行時間を計算
        execution_time = (end_time - start_time).total_seconds()

        print("-" * 60)
        print(f"🎉 エージェント実行完了！ (実行時間: {execution_time:.2f}秒)")

        return {
            "user_goal": result["user_goal"],
            "final_result": result["final_result"],
            "completed_actions": result["completed_actions"],
            "gathered_information": result["gathered_information"],
            "analysis_results": result["analysis_results"],
            "step_count": result["step_count"],
            "execution_log": result["execution_log"],
            "execution_time": execution_time,
        }


# ===== デモ用のメイン関数 =====


def main():
    """LangGraph版 Agentsのデモンストレーション"""
    print("=" * 60)
    print("🤖 LangGraph版 Agents Pattern デモ")
    print("=" * 60)
    print(
        "このデモでは、LangGraphを使用して自律的に判断・行動するエージェントを実装します。"
    )
    print("利用可能ツール: Web検索、ファイル操作、データ分析、タスク計画")
    print()

    try:
        # Agentsシステムを初期化
        agent_system = LangGraphAgents()

        # デモ用の目標リスト
        demo_goals = [
            "競合他社分析レポートを作成して、市場での自社の位置づけを明確にしてください。",
            "新製品のマーケティング戦略を立案し、実行計画を作成してください。",
            "チーム生産性向上のための改善提案書を作成してください。",
        ]

        print("📚 デモ用目標の実行:")
        print("=" * 60)

        for i, goal in enumerate(demo_goals, 1):
            print(f"\n【目標 {i}】")

            # エージェントを実行
            result = agent_system.execute_goal(goal, max_steps=4)

            # 結果の表示
            print("\n📊 実行結果:")
            print(f"実行ステップ数: {result['step_count']}")
            print(f"完了アクション: {len(result['completed_actions'])}")
            print(f"実行時間: {result['execution_time']:.2f}秒")

            print("\n📝 最終結果概要:")
            result_preview = (
                result["final_result"][:400] + "..."
                if len(result["final_result"]) > 400
                else result["final_result"]
            )
            print(result_preview)

            # 詳細表示の確認
            show_details = input("\n詳細を表示しますか？ (y/n): ").lower().strip()
            if show_details == "y":
                print("\n" + "=" * 50)
                print("📋 詳細結果")
                print("=" * 50)

                print("\n🎯 完了したアクション:")
                for j, action in enumerate(result["completed_actions"], 1):
                    print(f"{j}. {action}")

                print("\n📊 収集した情報:")
                for j, info in enumerate(result["gathered_information"], 1):
                    print(f"{j}. {info}")

                print("\n🔍 分析結果:")
                for j, analysis in enumerate(result["analysis_results"], 1):
                    print(f"{j}. {analysis}")

                print("\n📝 最終結果:")
                print("-" * 30)
                print(result["final_result"])

                print("\n📊 実行ログ:")
                for log_entry in result["execution_log"]:
                    print(f"  {log_entry}")

            print()

        # カスタム目標実行モード
        print("\n" + "=" * 60)
        print("💬 カスタム目標実行モード (終了するには 'quit' と入力)")
        print("=" * 60)

        while True:
            try:
                user_goal = input("\n🎯 達成したい目標を入力してください: ").strip()

                if user_goal.lower() in ["quit", "exit", "終了", "q"]:
                    print("👋 エージェント実行を終了します。")
                    break

                if not user_goal:
                    print("⚠️  目標を入力してください。")
                    continue

                # 最大ステップ数を取得
                try:
                    max_steps = int(
                        input("最大ステップ数 (デフォルト: 5): ").strip() or "5"
                    )
                except ValueError:
                    print("⚠️  無効な値です。デフォルト値5を使用します。")
                    max_steps = 5

                # カスタム目標を実行
                result = agent_system.execute_goal(user_goal, max_steps)

                # 結果の表示
                print("\n🎉 目標実行完了！")
                print(f"実行ステップ: {result['step_count']}")
                print(f"完了アクション: {len(result['completed_actions'])}")
                print(f"実行時間: {result['execution_time']:.2f}秒")

                print("\n📊 最終結果:")
                print("-" * 40)
                print(result["final_result"])

            except KeyboardInterrupt:
                print("\n\n👋 エージェント実行を終了します。")
                break

    except Exception as e:
        print(f"❌ エラーが発生しました: {e}")
        print("💡 OpenAI APIキーが正しく設定されているか確認してください。")


if __name__ == "__main__":
    main()
