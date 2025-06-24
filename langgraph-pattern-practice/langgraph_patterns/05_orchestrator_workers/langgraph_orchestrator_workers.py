#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LangGraph版 Orchestrator-Workers Pattern
LangGraphを使用してオーケストレーターが複数のワーカーを管理し、
動的にタスクを分解・配布するパターン
"""

import datetime
from typing import Any
from typing import Dict
from typing import List
from typing import Literal
from typing import TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.pydantic_v1 import Field

# LangChain関連のインポート
from langchain_openai import ChatOpenAI
from langgraph.graph import END

# LangGraph関連のインポート
from langgraph.graph import StateGraph

# ===== 環境変数の読み込み =====
load_dotenv()

# ===== データクラスの定義 =====


class Task(BaseModel):
    """タスクの定義"""

    id: str = Field(description="タスクID")
    type: Literal["research", "analysis", "writing", "coding", "review"] = Field(
        description="タスクタイプ"
    )
    title: str = Field(description="タスクタイトル")
    description: str = Field(description="タスクの詳細説明")
    priority: int = Field(description="優先度（1-5、5が最高）", ge=1, le=5)
    dependencies: List[str] = Field(default=[], description="依存するタスクIDのリスト")
    assigned_worker: str = Field(default="", description="割り当てられたワーカー")
    status: Literal["pending", "in_progress", "completed", "failed"] = Field(
        default="pending", description="タスク状態"
    )
    result: str = Field(default="", description="タスク結果")


class OrchestratorState(TypedDict):
    """オーケストレーターワークフローの状態定義"""

    user_request: str  # ユーザーリクエスト
    project_plan: str  # プロジェクト計画
    tasks: List[Task]  # タスクリスト
    completed_tasks: List[Task]  # 完了タスクリスト
    current_task: Task  # 現在処理中のタスク
    final_report: str  # 最終レポート
    execution_log: List[str]  # 実行ログ


# ===== LangGraphベースのOrchestrator-Workersクラス =====


class LangGraphOrchestratorWorkers:
    """LangGraphを使用したオーケストレーター・ワーカーシステム"""

    def __init__(self):
        """システムの初期化"""
        print("🎭 LangGraph版 Orchestrator-Workersシステムを初期化中...")

        # OpenAI ChatLLMモデルを初期化
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3, verbose=True)

        # プロンプトテンプレートを設定
        self._setup_prompts()

        # ワーカーの専門分野を定義
        self.workers = {
            "researcher": "調査・情報収集の専門家",
            "analyst": "データ分析・洞察抽出の専門家",
            "writer": "文書作成・編集の専門家",
            "coder": "プログラミング・技術実装の専門家",
            "reviewer": "品質保証・レビューの専門家",
        }

        # LangGraphワークフローを構築
        self.graph = self._build_graph()

        print("✅ オーケストレーター・ワーカーシステムの初期化が完了しました！")

    def _setup_prompts(self):
        """各処理用のプロンプトテンプレートを設定"""

        # プロジェクト計画作成プロンプト
        self.planning_prompt = ChatPromptTemplate.from_template(
            """あなたは経験豊富なプロジェクトマネージャーです。以下のユーザーリクエストを分析し、詳細なプロジェクト計画を作成してください。

ユーザーリクエスト:
{user_request}

プロジェクト計画に含めるべき内容:
1. プロジェクトの目標と範囲
2. 主要な成果物
3. 必要なリソースとスキル
4. 推定期間とマイルストーン
5. リスクと対策

詳細なプロジェクト計画を作成してください。
"""
        )

        # タスク分解プロンプト
        self.task_decomposition_prompt = ChatPromptTemplate.from_template(
            """あなたは優秀なプロジェクトマネージャーです。以下のプロジェクト計画を基に、実行可能なタスクに分解してください。

プロジェクト計画:
{project_plan}

利用可能なワーカー:
- researcher: 調査・情報収集の専門家
- analyst: データ分析・洞察抽出の専門家
- writer: 文書作成・編集の専門家
- coder: プログラミング・技術実装の専門家
- reviewer: 品質保証・レビューの専門家

以下のJSON形式でタスクリストを作成してください:
[
  {{
    "id": "task_001",
    "type": "research",
    "title": "タスクタイトル",
    "description": "タスクの詳細説明",
    "priority": 5,
    "dependencies": [],
    "assigned_worker": "researcher"
  }}
]

タスクは実行可能な単位に分割し、依存関係を適切に設定してください。
"""
        )

        # 研究者ワーカープロンプト
        self.researcher_prompt = ChatPromptTemplate.from_template(
            """あなたは経験豊富な研究者です。以下のタスクを実行してください。

タスク: {task_title}
詳細: {task_description}

研究・調査時の注意点:
1. 信頼できる情報源を使用
2. 多角的な視点から情報を収集
3. 最新の動向やトレンドを含める
4. 事実と意見を明確に区別
5. 情報の出典を明記

調査結果を詳細にまとめてください。
"""
        )

        # 分析者ワーカープロンプト
        self.analyst_prompt = ChatPromptTemplate.from_template(
            """あなたは優秀なデータ分析者です。以下のタスクを実行してください。

タスク: {task_title}
詳細: {task_description}

分析時の注意点:
1. データの信頼性を評価
2. 適切な分析手法を選択
3. パターンや傾向を特定
4. 意味のある洞察を抽出
5. 結論を明確に示す

分析結果と洞察を詳細にまとめてください。
"""
        )

        # ライターワーカープロンプト
        self.writer_prompt = ChatPromptTemplate.from_template(
            """あなたは経験豊富なライターです。以下のタスクを実行してください。

タスク: {task_title}
詳細: {task_description}

執筆時の注意点:
1. 読者にとって分かりやすい構成
2. 論理的な文章の流れ
3. 適切な語彙と文体の選択
4. 具体例や事例を含める
5. 魅力的で読みやすい文章

高品質な文書を作成してください。
"""
        )

        # コーダーワーカープロンプト
        self.coder_prompt = ChatPromptTemplate.from_template(
            """あなたは熟練したプログラマーです。以下のタスクを実行してください。

タスク: {task_title}
詳細: {task_description}

プログラミング時の注意点:
1. 清潔で読みやすいコード
2. 適切なコメントの追加
3. エラーハンドリングの実装
4. 効率的なアルゴリズムの選択
5. テスト可能な設計

高品質なコードと説明を提供してください。
"""
        )

        # レビューワーワーカープロンプト
        self.reviewer_prompt = ChatPromptTemplate.from_template(
            """あなたは厳格な品質保証担当者です。以下のタスクを実行してください。

タスク: {task_title}
詳細: {task_description}

レビュー時の注意点:
1. 品質基準との適合性
2. 完全性と正確性
3. 一貫性と整合性
4. 改善可能な点の特定
5. 具体的な改善提案

詳細なレビュー結果と改善提案を提供してください。
"""
        )

        # 最終レポート作成プロンプト
        self.final_report_prompt = ChatPromptTemplate.from_template(
            """あなたは優秀なプロジェクトマネージャーです。以下の完了したタスクの結果を統合し、包括的な最終レポートを作成してください。

プロジェクト計画:
{project_plan}

完了したタスクの結果:
{completed_tasks_summary}

最終レポートに含めるべき内容:
1. プロジェクトの概要と目標
2. 実行されたタスクとその結果
3. 主要な成果と発見
4. 課題と解決策
5. 今後の推奨事項
6. 結論

包括的で実用的な最終レポートを作成してください。
"""
        )

    def _build_graph(self) -> StateGraph:
        """オーケストレーター・ワーカーワークフローを構築"""
        print("🔧 オーケストレーター・ワーカーワークフローを構築中...")

        workflow = StateGraph(OrchestratorState)

        # ノード（処理ステップ）を追加
        workflow.add_node("plan_project", self._plan_project)  # プロジェクト計画
        workflow.add_node("decompose_tasks", self._decompose_tasks)  # タスク分解
        workflow.add_node("assign_task", self._assign_next_task)  # タスク割り当て
        workflow.add_node("execute_task", self._execute_task)  # タスク実行
        workflow.add_node(
            "create_final_report", self._create_final_report
        )  # 最終レポート作成

        # エントリーポイントを設定
        workflow.set_entry_point("plan_project")

        # 経路を設定
        workflow.add_edge("plan_project", "decompose_tasks")
        workflow.add_edge("decompose_tasks", "assign_task")

        # 条件分岐：タスクが残っているかチェック
        workflow.add_conditional_edges(
            "assign_task",
            self._check_remaining_tasks,
            {"execute": "execute_task", "finish": "create_final_report"},
        )

        # タスク実行後は次のタスク割り当てに戻る
        workflow.add_edge("execute_task", "assign_task")
        workflow.add_edge("create_final_report", END)

        return workflow.compile()

    def _plan_project(self, state: OrchestratorState) -> Dict[str, Any]:
        """プロジェクト計画作成ステップ"""
        print("📋 ステップ1: プロジェクト計画を作成中...")

        user_request = state["user_request"]

        # プロジェクト計画プロンプトを生成
        prompt = self.planning_prompt.format(user_request=user_request)

        # LLMを呼び出し
        response = self.llm.invoke([HumanMessage(content=prompt)])
        project_plan = response.content

        # 実行ログを更新
        log_entry = (
            f"[{datetime.datetime.now().strftime('%H:%M:%S')}] プロジェクト計画作成完了"
        )
        execution_log = state.get("execution_log", [])
        execution_log.append(log_entry)

        print(f"✅ プロジェクト計画作成完了: {len(project_plan)}文字")

        return {"project_plan": project_plan, "execution_log": execution_log}

    def _decompose_tasks(self, state: OrchestratorState) -> Dict[str, Any]:
        """タスク分解ステップ"""
        print("🔨 ステップ2: タスクを分解中...")

        project_plan = state["project_plan"]

        # タスク分解プロンプトを生成
        prompt = self.task_decomposition_prompt.format(project_plan=project_plan)

        # LLMを呼び出し
        response = self.llm.invoke([HumanMessage(content=prompt)])
        tasks_json = response.content

        # JSONからタスクリストを解析（簡易実装）
        try:
            import json
            import re

            # JSON部分を抽出
            json_match = re.search(r"\[.*\]", tasks_json, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                tasks_data = json.loads(json_str)

                tasks = []
                for task_data in tasks_data:
                    task = Task(
                        id=task_data["id"],
                        type=task_data["type"],
                        title=task_data["title"],
                        description=task_data["description"],
                        priority=task_data["priority"],
                        dependencies=task_data.get("dependencies", []),
                        assigned_worker=task_data["assigned_worker"],
                    )
                    tasks.append(task)
            else:
                # JSONが見つからない場合のフォールバック
                tasks = [
                    Task(
                        id="fallback_task",
                        type="research",
                        title="プロジェクト調査",
                        description="プロジェクトに関する基本的な調査を実行",
                        priority=3,
                        assigned_worker="researcher",
                    )
                ]

        except Exception as e:
            print(f"⚠️  タスク解析エラー: {e}")
            # エラー時のフォールバックタスク
            tasks = [
                Task(
                    id="error_fallback",
                    type="analysis",
                    title="要件分析",
                    description="ユーザー要件の詳細分析",
                    priority=3,
                    assigned_worker="analyst",
                )
            ]

        # 実行ログを更新
        log_entry = f"[{datetime.datetime.now().strftime('%H:%M:%S')}] タスク分解完了: {len(tasks)}個のタスク"
        execution_log = state["execution_log"]
        execution_log.append(log_entry)

        print(f"✅ タスク分解完了: {len(tasks)}個のタスクを生成")

        return {"tasks": tasks, "completed_tasks": [], "execution_log": execution_log}

    def _assign_next_task(self, state: OrchestratorState) -> Dict[str, Any]:
        """次のタスクを割り当て"""
        print("🎯 ステップ3: 次のタスクを割り当て中...")

        tasks = state["tasks"]
        completed_tasks = state.get("completed_tasks", [])
        completed_task_ids = {task.id for task in completed_tasks}

        # 実行可能なタスクを探す（依存関係をチェック）
        available_tasks = []
        for task in tasks:
            if task.status == "pending":
                # 依存関係がすべて完了しているかチェック
                if all(dep_id in completed_task_ids for dep_id in task.dependencies):
                    available_tasks.append(task)

        if available_tasks:
            # 優先度の高いタスクを選択
            current_task = max(available_tasks, key=lambda t: t.priority)
            current_task.status = "in_progress"

            # 実行ログを更新
            log_entry = f"[{datetime.datetime.now().strftime('%H:%M:%S')}] タスク割り当て: {current_task.title} -> {current_task.assigned_worker}"
            execution_log = state["execution_log"]
            execution_log.append(log_entry)

            print(
                f"✅ タスク割り当て完了: {current_task.title} -> {current_task.assigned_worker}"
            )

            return {
                "current_task": current_task,
                "tasks": tasks,
                "execution_log": execution_log,
            }
        else:
            # 実行可能なタスクがない場合
            log_entry = f"[{datetime.datetime.now().strftime('%H:%M:%S')}] 全タスク完了"
            execution_log = state["execution_log"]
            execution_log.append(log_entry)

            print("✅ 全タスクが完了しました")

            return {"current_task": None, "execution_log": execution_log}

    def _check_remaining_tasks(self, state: OrchestratorState) -> str:
        """残りタスクの確認（条件分岐関数）"""
        current_task = state.get("current_task")
        return "execute" if current_task else "finish"

    def _execute_task(self, state: OrchestratorState) -> Dict[str, Any]:
        """タスク実行ステップ"""
        current_task = state["current_task"]
        print(f"⚡ ステップ4: タスクを実行中 - {current_task.title}")

        # ワーカータイプに応じたプロンプトを選択
        worker_prompts = {
            "researcher": self.researcher_prompt,
            "analyst": self.analyst_prompt,
            "writer": self.writer_prompt,
            "coder": self.coder_prompt,
            "reviewer": self.reviewer_prompt,
        }

        prompt_template = worker_prompts.get(
            current_task.assigned_worker, self.researcher_prompt
        )

        # プロンプトを生成
        prompt = prompt_template.format(
            task_title=current_task.title, task_description=current_task.description
        )

        # ワーカー（LLM）を呼び出し
        response = self.llm.invoke([HumanMessage(content=prompt)])
        task_result = response.content

        # タスクを完了状態に更新
        current_task.status = "completed"
        current_task.result = task_result

        # 完了タスクリストに追加
        completed_tasks = state.get("completed_tasks", [])
        completed_tasks.append(current_task)

        # タスクリストの状態を更新
        tasks = state["tasks"]
        for i, task in enumerate(tasks):
            if task.id == current_task.id:
                tasks[i] = current_task
                break

        # 実行ログを更新
        log_entry = f"[{datetime.datetime.now().strftime('%H:%M:%S')}] タスク実行完了: {current_task.title}"
        execution_log = state["execution_log"]
        execution_log.append(log_entry)

        print(f"✅ タスク実行完了: {current_task.title}")

        return {
            "tasks": tasks,
            "completed_tasks": completed_tasks,
            "execution_log": execution_log,
        }

    def _create_final_report(self, state: OrchestratorState) -> Dict[str, Any]:
        """最終レポート作成ステップ"""
        print("📊 ステップ5: 最終レポートを作成中...")

        project_plan = state["project_plan"]
        completed_tasks = state["completed_tasks"]

        # 完了タスクの結果をまとめる
        completed_tasks_summary = "\n".join(
            [
                f"【{task.title}】（{task.assigned_worker}担当）\n{task.result}\n"
                for task in completed_tasks
            ]
        )

        # 最終レポートプロンプトを生成
        prompt = self.final_report_prompt.format(
            project_plan=project_plan, completed_tasks_summary=completed_tasks_summary
        )

        # LLMを呼び出し
        response = self.llm.invoke([HumanMessage(content=prompt)])
        final_report = response.content

        # 実行ログを更新
        log_entry = (
            f"[{datetime.datetime.now().strftime('%H:%M:%S')}] 最終レポート作成完了"
        )
        execution_log = state["execution_log"]
        execution_log.append(log_entry)

        print(f"✅ 最終レポート作成完了: {len(final_report)}文字")

        return {"final_report": final_report, "execution_log": execution_log}

    def execute_project(self, user_request: str) -> Dict[str, Any]:
        """プロジェクト実行のメイン関数"""
        print("🚀 プロジェクト実行開始")
        print(f"ユーザーリクエスト: {user_request}")
        print("-" * 60)

        # 初期状態を設定
        initial_state = {
            "user_request": user_request,
            "project_plan": "",
            "tasks": [],
            "completed_tasks": [],
            "current_task": None,
            "final_report": "",
            "execution_log": [],
        }

        # ワークフローを実行
        start_time = datetime.datetime.now()
        result = self.graph.invoke(initial_state)
        end_time = datetime.datetime.now()

        # 実行時間を計算
        execution_time = (end_time - start_time).total_seconds()

        print("-" * 60)
        print(f"🎉 プロジェクト実行完了！ (実行時間: {execution_time:.2f}秒)")

        return {
            "user_request": result["user_request"],
            "project_plan": result["project_plan"],
            "tasks": result["tasks"],
            "completed_tasks": result["completed_tasks"],
            "final_report": result["final_report"],
            "execution_log": result["execution_log"],
            "execution_time": execution_time,
        }


# ===== デモ用のメイン関数 =====


def main():
    """LangGraph版 Orchestrator-Workersのデモンストレーション"""
    print("=" * 60)
    print("🎭 LangGraph版 Orchestrator-Workers Pattern デモ")
    print("=" * 60)
    print(
        "このデモでは、LangGraphを使用してオーケストレーターが複数のワーカーを管理します。"
    )
    print("ワーカー: 研究者、分析者、ライター、コーダー、レビュアー")
    print()

    try:
        # Orchestrator-Workersシステムを初期化
        orchestrator = LangGraphOrchestratorWorkers()

        # デモ用のプロジェクトリクエスト
        demo_requests = [
            "オンライン学習プラットフォームの企画書を作成してください。ターゲット、機能、技術スタック、ビジネスモデルを含めてください。",
            "企業向けのAIチャットボット導入提案書を作成してください。効果、実装方法、コストを分析してください。",
        ]

        print("📚 デモ用プロジェクトの実行:")
        print("=" * 60)

        for i, request in enumerate(demo_requests, 1):
            print(f"\n【プロジェクト {i}】")

            # プロジェクトを実行
            result = orchestrator.execute_project(request)

            # 結果の表示
            print("\n📊 実行結果:")
            print(f"実行タスク数: {len(result['completed_tasks'])}")
            print(f"実行時間: {result['execution_time']:.2f}秒")

            print("\n📋 実行されたタスク:")
            for task in result["completed_tasks"]:
                print(f"  - {task.title} ({task.assigned_worker})")

            print("\n📝 最終レポート概要:")
            report_preview = (
                result["final_report"][:300] + "..."
                if len(result["final_report"]) > 300
                else result["final_report"]
            )
            print(report_preview)

            # 詳細表示の確認
            show_details = input("\n詳細を表示しますか？ (y/n): ").lower().strip()
            if show_details == "y":
                print("\n" + "=" * 50)
                print("📋 詳細結果")
                print("=" * 50)

                print("\n📋 プロジェクト計画:")
                print("-" * 30)
                print(result["project_plan"])

                print("\n⚡ タスク実行結果:")
                print("-" * 30)
                for task in result["completed_tasks"]:
                    print(f"\n【{task.title}】({task.assigned_worker}担当)")
                    print(task.result)

                print("\n📊 最終レポート:")
                print("-" * 30)
                print(result["final_report"])

                print("\n📊 実行ログ:")
                print("-" * 30)
                for log_entry in result["execution_log"]:
                    print(f"  {log_entry}")

            print()

        # カスタムプロジェクトモード
        print("\n" + "=" * 60)
        print("💬 カスタムプロジェクトモード (終了するには 'quit' と入力)")
        print("=" * 60)

        while True:
            try:
                user_request = input(
                    "\n🎯 プロジェクトの要件を入力してください: "
                ).strip()

                if user_request.lower() in ["quit", "exit", "終了", "q"]:
                    print("👋 プロジェクト実行を終了します。")
                    break

                if not user_request:
                    print("⚠️  プロジェクト要件を入力してください。")
                    continue

                # カスタムプロジェクトを実行
                result = orchestrator.execute_project(user_request)

                # 結果の表示
                print(
                    f"\n🎉 プロジェクト完了！ (実行時間: {result['execution_time']:.2f}秒)"
                )
                print(f"📋 実行タスク: {len(result['completed_tasks'])}個")

                print("\n📊 最終レポート:")
                print("-" * 40)
                print(result["final_report"])

            except KeyboardInterrupt:
                print("\n\n👋 プロジェクト実行を終了します。")
                break

    except Exception as e:
        print(f"❌ エラーが発生しました: {e}")
        print("💡 OpenAI APIキーが正しく設定されているか確認してください。")


if __name__ == "__main__":
    main()
