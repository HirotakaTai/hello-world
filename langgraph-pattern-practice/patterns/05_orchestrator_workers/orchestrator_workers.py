"""
Orchestrator-workers パターン
============================

このパターンは、中央のオーケストレーターが動的にタスクを分解し、
複数のワーカーに配布して処理させる方法です。

特徴：
- オーケストレーターがタスクの複雑さを分析
- 動的にサブタスクを生成
- 各ワーカーが専門化されたタスクを実行
- 結果を統合して最終的な成果物を作成

例：
- 複数ファイルの変更が必要なコーディングタスク
- 複数ソースからの情報収集と分析
- 複雑なレポート作成（データ収集、分析、執筆を分担）

このパターンの利点：
- 複雑なタスクの動的な分解
- 各ワーカーの専門性を活用
- スケーラブルな処理
"""

import json
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
from datetime import datetime
from typing import Any
from typing import Dict
from typing import List

from dotenv import load_dotenv
from langchain.schema import HumanMessage
from langchain.schema import SystemMessage
from langchain_openai import ChatOpenAI

# ===== 環境変数の読み込み =====
load_dotenv()


class WorkerType:
    """
    ワーカーの種類を定義するクラス
    """

    RESEARCHER = "researcher"  # 調査専門ワーカー
    ANALYZER = "analyzer"  # 分析専門ワーカー
    WRITER = "writer"  # 執筆専門ワーカー
    CODER = "coder"  # コーディング専門ワーカー
    REVIEWER = "reviewer"  # レビュー専門ワーカー
    DATA_PROCESSOR = "data_processor"  # データ処理専門ワーカー
    TRANSLATOR = "translator"  # 翻訳専門ワーカー


class Task:
    """
    タスクを表現するデータクラス
    """

    def __init__(
        self,
        task_id: str,
        task_type: str,
        description: str,
        worker_type: str,
        priority: int = 1,
        dependencies: List[str] = None,
    ):
        self.task_id = task_id
        self.task_type = task_type
        self.description = description
        self.worker_type = worker_type
        self.priority = priority
        self.dependencies = dependencies or []
        self.status = "pending"  # pending, running, completed, failed
        self.result = None
        self.start_time = None
        self.end_time = None
        self.worker_id = None

    def to_dict(self) -> Dict[str, Any]:
        """タスクを辞書形式で返す"""
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "description": self.description,
            "worker_type": self.worker_type,
            "priority": self.priority,
            "dependencies": self.dependencies,
            "status": self.status,
            "result": self.result,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "worker_id": self.worker_id,
        }


class Worker:
    """
    ワーカークラス：特定の種類のタスクを実行する
    """

    def __init__(self, worker_id: str, worker_type: str, llm: ChatOpenAI):
        self.worker_id = worker_id
        self.worker_type = worker_type
        self.llm = llm
        self.completed_tasks = 0

        # ===== ワーカータイプ別のシステムプロンプト =====
        self.system_prompts = {
            WorkerType.RESEARCHER: """
            あなたは専門的な調査研究者です。
            与えられたトピックについて包括的な調査を行い、
            信頼できる情報を収集・整理してください。
            """,
            WorkerType.ANALYZER: """
            あなたはデータ分析の専門家です。
            提供された情報やデータを分析し、
            パターン、トレンド、重要な洞察を特定してください。
            """,
            WorkerType.WRITER: """
            あなたは優秀なライターです。
            与えられた情報を基に、読みやすく魅力的な
            文章を作成してください。
            """,
            WorkerType.CODER: """
            あなたは経験豊富なプログラマーです。
            要求に応じて、高品質で保守性の高い
            コードを作成してください。
            """,
            WorkerType.REVIEWER: """
            あなたは品質保証の専門家です。
            提供されたコンテンツを詳細にレビューし、
            改善点や問題点を特定してください。
            """,
            WorkerType.DATA_PROCESSOR: """
            あなたはデータ処理の専門家です。
            データの変換、整理、構造化を行い、
            使いやすい形式で提供してください。
            """,
            WorkerType.TRANSLATOR: """
            あなたは多言語翻訳の専門家です。
            正確で自然な翻訳を提供し、
            文脈とニュアンスを適切に伝えてください。
            """,
        }

    def execute_task(
        self, task: Task, context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        タスクを実行する

        Args:
            task (Task): 実行するタスク
            context (Dict[str, Any]): 実行コンテキスト（他のタスクの結果など）

        Returns:
            Dict[str, Any]: 実行結果
        """

        print(f"🔧 Worker {self.worker_id} がタスク {task.task_id} を開始")

        task.status = "running"
        task.start_time = datetime.now()
        task.worker_id = self.worker_id

        try:
            # ===== システムプロンプトを取得 =====
            system_prompt = self.system_prompts.get(
                self.worker_type,
                "あなたは汎用的なタスク実行者です。与えられたタスクを最善を尽くして実行してください。",
            )

            # ===== コンテキスト情報を含むユーザープロンプトを作成 =====
            user_prompt = f"""
            以下のタスクを実行してください：
            
            【タスク内容】
            {task.description}
            
            【タスクタイプ】
            {task.task_type}
            """

            # ===== コンテキスト情報があれば追加 =====
            if context:
                context_str = ""
                for key, value in context.items():
                    if isinstance(value, dict) and "result" in value:
                        context_str += f"\n【{key}の結果】\n{value['result']}\n"
                    else:
                        context_str += f"\n【{key}】\n{str(value)}\n"

                if context_str:
                    user_prompt += f"\n\n【関連情報・前段階の結果】{context_str}"

            user_prompt += """
            
            専門知識を活用して、高品質な結果を提供してください。
            結果は具体的で実用的なものにしてください。
            """

            # ===== LLMを呼び出して実行 =====
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ]

            response = self.llm.invoke(messages)

            # ===== 実行完了 =====
            task.status = "completed"
            task.end_time = datetime.now()
            task.result = response.content
            self.completed_tasks += 1

            execution_time = (task.end_time - task.start_time).total_seconds()
            print(
                f"✅ Worker {self.worker_id} がタスク {task.task_id} を完了 ({execution_time:.2f}秒)"
            )

            return {
                "success": True,
                "task_id": task.task_id,
                "result": response.content,
                "execution_time": execution_time,
                "worker_id": self.worker_id,
            }

        except Exception as e:
            # ===== エラー処理 =====
            task.status = "failed"
            task.end_time = datetime.now()
            task.result = f"エラー: {str(e)}"

            execution_time = (
                (task.end_time - task.start_time).total_seconds()
                if task.start_time
                else 0
            )
            print(
                f"❌ Worker {self.worker_id} のタスク {task.task_id} が失敗: {str(e)}"
            )

            return {
                "success": False,
                "task_id": task.task_id,
                "error": str(e),
                "execution_time": execution_time,
                "worker_id": self.worker_id,
            }


class Orchestrator:
    """
    オーケストレーター：タスクを分解し、ワーカーに配布・管理する
    """

    def __init__(self, max_workers: int = 4):
        # ===== ChatOpenAI モデルの初期化 =====
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)

        # ===== ワーカー管理 =====
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.workers = {}
        self.task_queue = []
        self.completed_tasks = {}
        self.execution_log = []

        # ===== 利用可能なワーカータイプ =====
        self.available_worker_types = [
            WorkerType.RESEARCHER,
            WorkerType.ANALYZER,
            WorkerType.WRITER,
            WorkerType.CODER,
            WorkerType.REVIEWER,
            WorkerType.DATA_PROCESSOR,
            WorkerType.TRANSLATOR,
        ]

    def _create_worker(self, worker_type: str) -> Worker:
        """
        新しいワーカーを作成

        Args:
            worker_type (str): ワーカーの種類

        Returns:
            Worker: 作成されたワーカー
        """
        worker_id = f"{worker_type}_{len([w for w in self.workers.values() if w.worker_type == worker_type]) + 1}"
        return Worker(worker_id, worker_type, self.llm)

    def _get_or_create_worker(self, worker_type: str) -> Worker:
        """
        ワーカーを取得または作成

        Args:
            worker_type (str): ワーカーの種類

        Returns:
            Worker: ワーカーインスタンス
        """
        # ===== 指定されたタイプのワーカーを探す =====
        for worker in self.workers.values():
            if worker.worker_type == worker_type:
                return worker

        # ===== 新しいワーカーを作成 =====
        worker = self._create_worker(worker_type)
        self.workers[worker.worker_id] = worker
        return worker

    def decompose_complex_task(
        self, main_task: str, task_type: str = "general"
    ) -> List[Task]:
        """
        複雑なタスクをサブタスクに分解

        Args:
            main_task (str): メインタスク
            task_type (str): タスクの種類

        Returns:
            List[Task]: 分解されたサブタスクのリスト
        """

        print(f"🧠 オーケストレーターがタスクを分解中: {main_task[:50]}...")

        # ===== タスク分解のためのプロンプト =====
        decomposition_prompt = f"""
        あなたはプロジェクト管理の専門家です。
        以下の複雑なタスクを、実行可能なサブタスクに分解してください。
        
        【メインタスク】
        {main_task}
        
        【タスクタイプ】
        {task_type}
        
        【利用可能なワーカータイプ】
        - researcher: 調査・情報収集
        - analyzer: データ分析・パターン発見
        - writer: 文章作成・編集
        - coder: プログラミング・技術実装
        - reviewer: 品質チェック・レビュー
        - data_processor: データ処理・変換
        - translator: 翻訳・多言語対応
        
        以下のJSON形式でサブタスクを定義してください：
        
        {{
            "subtasks": [
                {{
                    "task_id": "task_1",
                    "task_type": "research",
                    "description": "具体的なタスクの説明",
                    "worker_type": "researcher",
                    "priority": 1,
                    "dependencies": []
                }},
                {{
                    "task_id": "task_2",
                    "task_type": "analysis", 
                    "description": "具体的なタスクの説明",
                    "worker_type": "analyzer",
                    "priority": 2,
                    "dependencies": ["task_1"]
                }}
            ]
        }}
        
        注意事項：
        1. 各サブタスクは独立して実行可能である必要があります
        2. 依存関係がある場合は dependencies で指定してください
        3. priority は実行順序を示します（数字が小さいほど優先）
        4. 適切なworker_typeを選択してください
        
        JSONのみを返してください。余分な説明は不要です。
        """

        try:
            response = self.llm.invoke([HumanMessage(content=decomposition_prompt)])
            response_text = response.content.strip()

            # ===== JSONの抽出（マークダウンコードブロックがある場合） =====
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                response_text = response_text[json_start:json_end].strip()
            elif "```" in response_text:
                json_start = response_text.find("```") + 3
                json_end = response_text.rfind("```")
                response_text = response_text[json_start:json_end].strip()

            # ===== JSONをパース =====
            task_data = json.loads(response_text)

            # ===== Taskオブジェクトを作成 =====
            tasks = []
            for subtask_data in task_data.get("subtasks", []):
                task = Task(
                    task_id=subtask_data["task_id"],
                    task_type=subtask_data["task_type"],
                    description=subtask_data["description"],
                    worker_type=subtask_data["worker_type"],
                    priority=subtask_data.get("priority", 1),
                    dependencies=subtask_data.get("dependencies", []),
                )
                tasks.append(task)

            print(f"📋 {len(tasks)}個のサブタスクに分解完了")
            for task in tasks:
                deps = (
                    f" (依存: {', '.join(task.dependencies)})"
                    if task.dependencies
                    else ""
                )
                print(f"  - {task.task_id}: {task.description[:40]}...{deps}")

            return tasks

        except (json.JSONDecodeError, KeyError) as e:
            print(f"⚠️ タスク分解に失敗: {str(e)}")
            # ===== フォールバック: シンプルなタスク作成 =====
            return [
                Task(
                    task_id="fallback_task",
                    task_type=task_type,
                    description=main_task,
                    worker_type=WorkerType.RESEARCHER,
                    priority=1,
                )
            ]

    def _check_dependencies(self, task: Task) -> bool:
        """
        タスクの依存関係がすべて完了しているかチェック

        Args:
            task (Task): チェックするタスク

        Returns:
            bool: 依存関係が満たされているかどうか
        """
        for dep_id in task.dependencies:
            if dep_id not in self.completed_tasks:
                return False
            if self.completed_tasks[dep_id]["success"] is False:
                return False
        return True

    def _get_dependency_context(self, task: Task) -> Dict[str, Any]:
        """
        タスクの依存タスクからコンテキストを構築

        Args:
            task (Task): 対象タスク

        Returns:
            Dict[str, Any]: コンテキスト情報
        """
        context = {}
        for dep_id in task.dependencies:
            if dep_id in self.completed_tasks:
                context[dep_id] = self.completed_tasks[dep_id]
        return context

    def execute_workflow(
        self, main_task: str, task_type: str = "general"
    ) -> Dict[str, Any]:
        """
        ワークフロー全体を実行

        Args:
            main_task (str): メインタスク
            task_type (str): タスクの種類

        Returns:
            Dict[str, Any]: 実行結果
        """

        start_time = time.time()
        print(f"🚀 ワークフロー実行開始: {main_task[:50]}...")

        # ===== ステップ1: タスクを分解 =====
        tasks = self.decompose_complex_task(main_task, task_type)
        self.task_queue = tasks.copy()

        # ===== ステップ2: タスクを優先度順にソート =====
        self.task_queue.sort(key=lambda t: t.priority)

        # ===== ステップ3: タスクを並列実行 =====
        futures = {}
        completed_count = 0
        total_tasks = len(tasks)

        print(f"📊 {total_tasks}個のタスクを実行中...")

        while completed_count < total_tasks:
            # ===== 実行可能なタスクを特定 =====
            ready_tasks = [
                task
                for task in self.task_queue
                if task.status == "pending" and self._check_dependencies(task)
            ]

            # ===== ワーカーに任務を割り当て =====
            for task in ready_tasks:
                if len(futures) < self.max_workers:
                    worker = self._get_or_create_worker(task.worker_type)
                    context = self._get_dependency_context(task)

                    future = self.executor.submit(worker.execute_task, task, context)
                    futures[future] = task

                    task.status = "running"
                    print(f"🔄 タスク {task.task_id} を {worker.worker_id} に割り当て")

            # ===== 完了したタスクを処理 =====
            if futures:
                for future in as_completed(futures, timeout=1):
                    task = futures[future]
                    result = future.result()

                    self.completed_tasks[task.task_id] = result
                    completed_count += 1

                    print(
                        f"✅ タスク {task.task_id} 完了 ({completed_count}/{total_tasks})"
                    )

                    # ===== futuresから削除 =====
                    del futures[future]
                    break

            # ===== 無限ループ防止 =====
            if not futures and not ready_tasks:
                remaining_tasks = [t for t in tasks if t.status == "pending"]
                if remaining_tasks:
                    print(
                        f"⚠️ 依存関係のエラーで{len(remaining_tasks)}個のタスクが実行できません"
                    )
                    for task in remaining_tasks:
                        task.status = "failed"
                        task.result = "依存関係が満たされませんでした"
                        self.completed_tasks[task.task_id] = {
                            "success": False,
                            "error": "依存関係エラー",
                            "task_id": task.task_id,
                        }
                        completed_count += 1
                break

        # ===== ステップ4: 結果を統合 =====
        print("🔄 結果統合中...")

        successful_results = [
            result
            for result in self.completed_tasks.values()
            if result.get("success", False)
        ]

        if successful_results:
            integration_result = self._integrate_results(
                main_task, successful_results, tasks
            )
        else:
            integration_result = "すべてのタスクが失敗しました"

        execution_time = time.time() - start_time

        # ===== 実行ログに記録 =====
        self.execution_log.append(
            {
                "main_task": main_task,
                "task_type": task_type,
                "total_tasks": total_tasks,
                "successful_tasks": len(successful_results),
                "execution_time": execution_time,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
        )

        print(f"🎉 ワークフロー完了 (実行時間: {execution_time:.2f}秒)")

        return {
            "main_task": main_task,
            "total_tasks": total_tasks,
            "successful_tasks": len(successful_results),
            "failed_tasks": total_tasks - len(successful_results),
            "task_details": [task.to_dict() for task in tasks],
            "task_results": self.completed_tasks,
            "integrated_result": integration_result,
            "execution_time": execution_time,
        }

    def _integrate_results(
        self,
        main_task: str,
        successful_results: List[Dict[str, Any]],
        tasks: List[Task],
    ) -> str:
        """
        複数のタスク結果を統合

        Args:
            main_task (str): メインタスク
            successful_results (List[Dict[str, Any]]): 成功したタスクの結果
            tasks (List[Task]): 全タスクリスト

        Returns:
            str: 統合された結果
        """

        integration_prompt = f"""
        あなたは結果統合の専門家です。
        複数のサブタスクの結果を統合し、元のメインタスクに対する
        包括的で一貫した最終成果物を作成してください。
        
        【元のメインタスク】
        {main_task}
        
        【サブタスクの結果】
        """

        for i, result in enumerate(successful_results, 1):
            task_id = result.get("task_id", f"task_{i}")
            task_result = result.get("result", "結果なし")
            integration_prompt += f"\n\n【{task_id}の結果】\n{task_result}"

        integration_prompt += """
        
        これらの結果を統合して、以下の要件を満たす最終成果物を作成してください：
        
        1. 元のメインタスクの要求に完全に応える内容
        2. 各サブタスクの結果を論理的に組み合わせた構成
        3. 一貫性があり、読みやすい形式
        4. 実用的で価値ある情報の提供
        5. 必要に応じて要約や結論を含める
        
        統合された最終成果物を提供してください。
        """

        try:
            response = self.llm.invoke([HumanMessage(content=integration_prompt)])
            return response.content
        except Exception as e:
            return f"結果統合中にエラーが発生しました: {str(e)}"

    def get_workflow_statistics(self) -> Dict[str, Any]:
        """
        ワークフロー実行統計を取得

        Returns:
            Dict[str, Any]: 統計情報
        """

        if not self.execution_log:
            return {"total_workflows": 0}

        total_workflows = len(self.execution_log)
        total_tasks = sum(log["total_tasks"] for log in self.execution_log)
        successful_tasks = sum(log["successful_tasks"] for log in self.execution_log)
        total_time = sum(log["execution_time"] for log in self.execution_log)

        # ===== ワーカー統計 =====
        worker_stats = {}
        for worker in self.workers.values():
            worker_stats[worker.worker_id] = {
                "type": worker.worker_type,
                "completed_tasks": worker.completed_tasks,
            }

        return {
            "total_workflows": total_workflows,
            "total_tasks": total_tasks,
            "successful_tasks": successful_tasks,
            "success_rate": (successful_tasks / total_tasks * 100)
            if total_tasks > 0
            else 0,
            "total_execution_time": total_time,
            "average_workflow_time": total_time / total_workflows,
            "active_workers": len(self.workers),
            "worker_statistics": worker_stats,
        }

    def cleanup(self):
        """
        リソースをクリーンアップ
        """
        self.executor.shutdown(wait=True)
        print("🧹 オーケストレーターリソースをクリーンアップしました")


# ===== 使用例 =====
def main():
    """
    Orchestrator-workersパターンのデモンストレーション
    """
    print("=== Orchestrator-workers パターンのデモ ===\n")

    # ===== オーケストレーターのインスタンスを作成 =====
    orchestrator = Orchestrator(max_workers=4)

    try:
        # ===== デモ1: 市場調査レポート作成 =====
        print("📊 デモ1: 市場調査レポート作成ワークフロー")
        print("-" * 60)

        market_research_task = """
        「日本の電動車（EV）市場の現状と将来予測」に関する包括的な市場調査レポートを作成してください。
        
        レポートには以下の要素を含める必要があります：
        - 現在の市場規模と主要プレイヤー
        - 消費者の動向と購買要因
        - 政府政策と規制の影響
        - 技術トレンドとイノベーション
        - 将来の成長予測と機会
        - 競合分析と市場シェア
        - 投資家向けの推奨事項
        
        プロフェッショナルなビジネスレポート形式で作成してください。
        """

        result1 = orchestrator.execute_workflow(
            main_task=market_research_task, task_type="market_research"
        )

        print("\n📈 市場調査レポート作成結果:")
        print(f"- 総タスク数: {result1['total_tasks']}")
        print(f"- 成功タスク数: {result1['successful_tasks']}")
        print(f"- 実行時間: {result1['execution_time']:.2f}秒")
        print(f"\n最終レポート:\n{result1['integrated_result'][:300]}...\n")

        # ===== デモ2: ソフトウェア開発プロジェクト =====
        print("\n💻 デモ2: ソフトウェア開発プロジェクトワークフロー")
        print("-" * 60)

        software_project_task = """
        Pythonを使用してタスク管理アプリケーションを開発してください。
        
        要件：
        - RESTful API（FastAPIまたはFlask使用）
        - タスクのCRUD操作（作成、読取、更新、削除）
        - ユーザー認証システム
        - データベース統合（SQLite）
        - 基本的なフロントエンド（HTML/CSS/JavaScript）
        - APIドキュメント
        - テストケース
        - デプロイメント手順
        
        プロダクションレディなアプリケーションとして開発してください。
        """

        result2 = orchestrator.execute_workflow(
            main_task=software_project_task, task_type="software_development"
        )

        print("\n🛠️ ソフトウェア開発プロジェクト結果:")
        print(f"- 総タスク数: {result2['total_tasks']}")
        print(f"- 成功タスク数: {result2['successful_tasks']}")
        print(f"- 実行時間: {result2['execution_time']:.2f}秒")
        print(f"\n開発成果物:\n{result2['integrated_result'][:300]}...\n")

        # ===== デモ3: 多言語コンテンツ作成 =====
        print("\n🌍 デモ3: 多言語コンテンツ作成ワークフロー")
        print("-" * 60)

        multilingual_content_task = """
        企業のグローバル展開のための多言語マーケティングコンテンツを作成してください。
        
        要件：
        - 製品紹介記事（日本語原文）
        - 英語、中国語、スペイン語への翻訳
        - 各言語圏の文化的コンテキストを考慮した内容調整
        - SEO最適化されたコンテンツ
        - ソーシャルメディア用の短縮版
        - 各国の規制や法的要件のチェック
        - マーケティング効果の予測分析
        
        グローバルマーケティング戦略として統合的にまとめてください。
        """

        result3 = orchestrator.execute_workflow(
            main_task=multilingual_content_task, task_type="content_creation"
        )

        print("\n🌐 多言語コンテンツ作成結果:")
        print(f"- 総タスク数: {result3['total_tasks']}")
        print(f"- 成功タスク数: {result3['successful_tasks']}")
        print(f"- 実行時間: {result3['execution_time']:.2f}秒")
        print(f"\nグローバルコンテンツ:\n{result3['integrated_result'][:300]}...\n")

        # ===== 統計情報の表示 =====
        print("📊 ワークフロー実行統計")
        print("-" * 40)
        stats = orchestrator.get_workflow_statistics()
        print(f"総ワークフロー数: {stats['total_workflows']}")
        print(f"総タスク数: {stats['total_tasks']}")
        print(f"成功タスク数: {stats['successful_tasks']}")
        print(f"成功率: {stats['success_rate']:.1f}%")
        print(f"総実行時間: {stats['total_execution_time']:.2f}秒")
        print(f"平均ワークフロー時間: {stats['average_workflow_time']:.2f}秒")
        print(f"アクティブワーカー数: {stats['active_workers']}")

        print("\nワーカー統計:")
        for worker_id, worker_data in stats["worker_statistics"].items():
            print(
                f"  - {worker_id} ({worker_data['type']}): {worker_data['completed_tasks']}タスク完了"
            )

    finally:
        # ===== リソースのクリーンアップ =====
        orchestrator.cleanup()


if __name__ == "__main__":
    main()
