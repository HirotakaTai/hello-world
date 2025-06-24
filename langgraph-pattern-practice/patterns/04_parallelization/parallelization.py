"""
Parallelization パターン
========================

このパターンは、タスクを並列で実行することで効率性を向上させる方法です。

2つの主要なバリエーション：
1. Sectioning（セクショニング）: タスクを独立したサブタスクに分割し、並列実行
2. Voting（投票）: 同じタスクを複数回実行し、結果を集約

例：
- セクショニング: 長い文書を複数の部分に分けて同時に要約
- 投票: 同じ質問に対して複数の回答を生成し、最良の回答を選択

このパターンの利点：
- 処理時間の短縮
- より高い精度と信頼性
- 複数の視点からの分析が可能
"""

import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any
from typing import Dict

from dotenv import load_dotenv
from langchain.schema import HumanMessage
from langchain.schema import SystemMessage
from langchain_openai import ChatOpenAI

# ===== 環境変数の読み込み =====
load_dotenv()


class ParallelizationSystem:
    """
    並列化パターンの実装クラス
    セクショニングと投票の両方のパターンを実装
    """

    def __init__(self, max_workers: int = 3):
        # ===== ChatOpenAI モデルの初期化 =====
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

        # ===== 並列処理用のスレッドプール =====
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

        # ===== 実行ログを保存するリスト =====
        self.execution_log = []

    def _log_execution(
        self,
        pattern_type: str,
        task_description: str,
        execution_time: float,
        results: Any,
    ):
        """
        実行をログに記録

        Args:
            pattern_type (str): パターンの種類（sectioning/voting）
            task_description (str): タスクの説明
            execution_time (float): 実行時間（秒）
            results (Any): 実行結果
        """
        self.execution_log.append(
            {
                "pattern": pattern_type,
                "task": task_description,
                "execution_time": execution_time,
                "results_count": len(results) if isinstance(results, list) else 1,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
        )

    def _call_llm_sync(
        self, system_prompt: str, user_prompt: str, task_id: str = ""
    ) -> Dict[str, Any]:
        """
        LLMを同期的に呼び出す（並列処理用）

        Args:
            system_prompt (str): システムプロンプト
            user_prompt (str): ユーザープロンプト
            task_id (str): タスクID（ログ用）

        Returns:
            Dict[str, Any]: 実行結果
        """
        start_time = time.time()

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        try:
            response = self.llm.invoke(messages)
            execution_time = time.time() - start_time

            return {
                "task_id": task_id,
                "success": True,
                "response": response.content,
                "execution_time": execution_time,
                "error": None,
            }
        except Exception as e:
            execution_time = time.time() - start_time
            return {
                "task_id": task_id,
                "success": False,
                "response": None,
                "execution_time": execution_time,
                "error": str(e),
            }

    def sectioning_document_analysis(
        self, document: str, analysis_type: str = "要約"
    ) -> Dict[str, Any]:
        """
        セクショニングパターン: 文書を複数の部分に分割して並列分析

        Args:
            document (str): 分析する文書
            analysis_type (str): 分析の種類（要約、分析、翻訳など）

        Returns:
            Dict[str, Any]: 分析結果
        """

        start_time = time.time()
        print(f"📄 セクショニング分析開始: {analysis_type}")

        # ===== 文書を複数のセクションに分割 =====
        # 簡単な分割方法：段落ごとに分割
        paragraphs = [p.strip() for p in document.split("\n\n") if p.strip()]

        # ===== セクションが少ない場合は文字数で分割 =====
        if len(paragraphs) < 3:
            # 文字数による分割
            section_size = len(document) // 3
            sections = [
                document[i : i + section_size]
                for i in range(0, len(document), section_size)
            ]
        else:
            sections = paragraphs

        print(f"📊 文書を{len(sections)}個のセクションに分割")

        # ===== 各セクション用のプロンプトを準備 =====
        tasks = []

        system_prompt = f"""
        あなたは文書分析の専門家です。
        与えられた文書のセクションに対して、{analysis_type}を行ってください。
        """

        for i, section in enumerate(sections):
            user_prompt = f"""
            以下は文書のセクション{i + 1}です。このセクションに対して{analysis_type}を行ってください：
            
            【セクション{i + 1}】
            {section}
            
            このセクションの{analysis_type}を明確で簡潔に提供してください。
            """

            tasks.append((system_prompt, user_prompt, f"section_{i + 1}"))

        # ===== 並列実行 =====
        print("⚡ 並列処理実行中...")
        futures = [
            self.executor.submit(self._call_llm_sync, sys_prompt, user_prompt, task_id)
            for sys_prompt, user_prompt, task_id in tasks
        ]

        # ===== 結果を収集 =====
        section_results = []
        for future in futures:
            result = future.result()
            section_results.append(result)

            if result["success"]:
                print(f"✅ {result['task_id']} 完了 ({result['execution_time']:.2f}秒)")
            else:
                print(f"❌ {result['task_id']} エラー: {result['error']}")

        # ===== 結果を統合 =====
        print("🔄 結果統合中...")

        successful_results = [r for r in section_results if r["success"]]

        if successful_results:
            # ===== 統合プロンプト =====
            integration_system_prompt = f"""
            あなたは文書統合の専門家です。
            複数のセクションの{analysis_type}結果を統合し、文書全体の一貫した{analysis_type}を作成してください。
            """

            section_summaries = "\n\n".join(
                [
                    f"【セクション{i + 1}の{analysis_type}】\n{result['response']}"
                    for i, result in enumerate(successful_results)
                ]
            )

            integration_user_prompt = f"""
            以下は文書の各セクションの{analysis_type}結果です。
            これらを統合して、文書全体の包括的な{analysis_type}を作成してください：
            
            {section_summaries}
            
            統合された{analysis_type}は、論理的で一貫性があり、元の文書の主要なポイントをすべて含むようにしてください。
            """

            integration_result = self._call_llm_sync(
                integration_system_prompt, integration_user_prompt, "integration"
            )

            final_result = (
                integration_result["response"]
                if integration_result["success"]
                else "統合に失敗しました"
            )
        else:
            final_result = "すべてのセクション処理に失敗しました"

        execution_time = time.time() - start_time

        # ===== ログに記録 =====
        self._log_execution(
            "sectioning", f"文書{analysis_type}", execution_time, section_results
        )

        print(f"🎉 セクショニング分析完了 (総実行時間: {execution_time:.2f}秒)")

        return {
            "pattern": "sectioning",
            "analysis_type": analysis_type,
            "sections_count": len(sections),
            "successful_sections": len(successful_results),
            "section_results": section_results,
            "integrated_result": final_result,
            "total_execution_time": execution_time,
        }

    def voting_quality_assessment(
        self, content: str, assessment_criteria: str, num_votes: int = 3
    ) -> Dict[str, Any]:
        """
        投票パターン: 同じタスクを複数回実行し、結果を評価して最良の結果を選択

        Args:
            content (str): 評価するコンテンツ
            assessment_criteria (str): 評価基準
            num_votes (int): 投票数（並列実行数）

        Returns:
            Dict[str, Any]: 評価結果
        """

        start_time = time.time()
        print(f"🗳️ 投票評価開始: {assessment_criteria} ({num_votes}票)")

        # ===== 異なる視点からの評価プロンプトを準備 =====
        voting_prompts = [
            {
                "perspective": "厳格な評価者",
                "system_prompt": f"""
                あなたは厳格で批判的な評価者です。
                コンテンツを{assessment_criteria}の観点から厳しく評価してください。
                問題点や改善点を積極的に指摘してください。
                """,
                "task_id": "strict_evaluator",
            },
            {
                "perspective": "建設的な評価者",
                "system_prompt": f"""
                あなたは建設的でバランスの取れた評価者です。
                コンテンツを{assessment_criteria}の観点から公平に評価し、
                良い点と改善点の両方を指摘してください。
                """,
                "task_id": "balanced_evaluator",
            },
            {
                "perspective": "専門家評価者",
                "system_prompt": f"""
                あなたはその分野の専門家として評価を行います。
                専門的な知識と経験に基づいて、{assessment_criteria}の観点から
                深い洞察を提供してください。
                """,
                "task_id": "expert_evaluator",
            },
            {
                "perspective": "ユーザー視点評価者",
                "system_prompt": f"""
                あなたは一般ユーザーの視点から評価を行います。
                {assessment_criteria}について、使いやすさや理解しやすさを
                重視して評価してください。
                """,
                "task_id": "user_evaluator",
            },
            {
                "perspective": "革新性評価者",
                "system_prompt": f"""
                あなたは革新性と創造性を重視する評価者です。
                {assessment_criteria}の観点から、独創性や新しいアプローチに
                注目して評価してください。
                """,
                "task_id": "innovation_evaluator",
            },
        ]

        # ===== 指定された数の評価者を選択 =====
        selected_evaluators = voting_prompts[:num_votes]

        # ===== 各評価者のタスクを準備 =====
        tasks = []
        for evaluator in selected_evaluators:
            user_prompt = f"""
            以下のコンテンツを{assessment_criteria}の観点から評価してください：
            
            【評価対象コンテンツ】
            {content}
            
            評価結果は以下の形式で提供してください：
            1. 総合評価（1-10点）
            2. 良い点
            3. 改善点
            4. 具体的な推奨事項
            
            あなたの視点（{evaluator["perspective"]}）から率直な評価をお願いします。
            """

            tasks.append(
                (evaluator["system_prompt"], user_prompt, evaluator["task_id"])
            )

        # ===== 並列実行 =====
        print("⚡ 複数評価者による並列評価実行中...")
        futures = [
            self.executor.submit(self._call_llm_sync, sys_prompt, user_prompt, task_id)
            for sys_prompt, user_prompt, task_id in tasks
        ]

        # ===== 結果を収集 =====
        evaluation_results = []
        for future in futures:
            result = future.result()
            evaluation_results.append(result)

            if result["success"]:
                print(
                    f"✅ {result['task_id']} 評価完了 ({result['execution_time']:.2f}秒)"
                )
            else:
                print(f"❌ {result['task_id']} エラー: {result['error']}")

        # ===== 投票結果を集約 =====
        print("📊 投票結果集約中...")

        successful_evaluations = [r for r in evaluation_results if r["success"]]

        if successful_evaluations:
            # ===== 集約プロンプト =====
            aggregation_system_prompt = f"""
            あなたは評価結果の集約専門家です。
            複数の評価者による{assessment_criteria}の評価結果を分析し、
            総合的な評価と最終的な推奨事項を提供してください。
            """

            all_evaluations = "\n\n".join(
                [
                    f"【{result['task_id']}の評価】\n{result['response']}"
                    for result in successful_evaluations
                ]
            )

            aggregation_user_prompt = f"""
            以下は複数の評価者による評価結果です。
            これらを総合して、最終的な評価レポートを作成してください：
            
            {all_evaluations}
            
            最終レポートには以下を含めてください：
            1. 総合評価スコア（各評価者のスコアの平均など）
            2. 共通して指摘された良い点
            3. 共通して指摘された改善点
            4. 評価者間の意見の相違点
            5. 最終的な推奨事項
            """

            aggregation_result = self._call_llm_sync(
                aggregation_system_prompt, aggregation_user_prompt, "final_aggregation"
            )

            final_assessment = (
                aggregation_result["response"]
                if aggregation_result["success"]
                else "集約に失敗しました"
            )
        else:
            final_assessment = "すべての評価に失敗しました"

        execution_time = time.time() - start_time

        # ===== ログに記録 =====
        self._log_execution(
            "voting", f"{assessment_criteria}評価", execution_time, evaluation_results
        )

        print(f"🎉 投票評価完了 (総実行時間: {execution_time:.2f}秒)")

        return {
            "pattern": "voting",
            "assessment_criteria": assessment_criteria,
            "num_evaluators": len(selected_evaluators),
            "successful_evaluations": len(successful_evaluations),
            "individual_evaluations": evaluation_results,
            "final_assessment": final_assessment,
            "total_execution_time": execution_time,
        }

    def parallel_code_review(self, code: str, num_reviewers: int = 3) -> Dict[str, Any]:
        """
        並列コードレビュー: 複数の視点からコードを同時にレビュー

        Args:
            code (str): レビューするコード
            num_reviewers (int): レビュアー数

        Returns:
            Dict[str, Any]: レビュー結果
        """

        start_time = time.time()
        print(f"👥 並列コードレビュー開始 ({num_reviewers}人のレビュアー)")

        # ===== 異なる専門性を持つレビュアーを定義 =====
        reviewers = [
            {
                "name": "セキュリティ専門家",
                "system_prompt": """
                あなたはセキュリティ専門家です。
                コードをセキュリティの観点から詳細にレビューしてください。
                脆弱性、セキュリティホール、危険な実装パターンを特定してください。
                """,
                "task_id": "security_reviewer",
            },
            {
                "name": "パフォーマンス専門家",
                "system_prompt": """
                あなたはパフォーマンス最適化の専門家です。
                コードの実行効率、メモリ使用量、アルゴリズムの効率性を
                重点的にレビューしてください。
                """,
                "task_id": "performance_reviewer",
            },
            {
                "name": "コード品質専門家",
                "system_prompt": """
                あなたはコード品質とメンテナビリティの専門家です。
                可読性、拡張性、保守性、設計パターンの適用について
                詳細にレビューしてください。
                """,
                "task_id": "quality_reviewer",
            },
            {
                "name": "バグ検出専門家",
                "system_prompt": """
                あなたはバグ検出の専門家です。
                論理エラー、エッジケースの処理不備、例外処理の問題など
                潜在的なバグを特定してください。
                """,
                "task_id": "bug_reviewer",
            },
            {
                "name": "ベストプラクティス専門家",
                "system_prompt": """
                あなたはプログラミングベストプラクティスの専門家です。
                コーディング規約、命名規則、構造化の適切さを
                評価してください。
                """,
                "task_id": "practices_reviewer",
            },
        ]

        # ===== 指定された数のレビュアーを選択 =====
        selected_reviewers = reviewers[:num_reviewers]

        # ===== 各レビュアーのタスクを準備 =====
        tasks = []
        for reviewer in selected_reviewers:
            user_prompt = f"""
            以下のコードを{reviewer["name"]}の視点からレビューしてください：
            
            【レビュー対象コード】
            ```
            {code}
            ```
            
            レビュー結果は以下の形式で提供してください：
            1. 総合評価（1-10点）
            2. 発見された問題点（優先度付き）
            3. 良い点・推奨できる点
            4. 具体的な改善提案
            5. 修正が必要な箇所の指摘
            
            専門分野の観点から詳細で実用的なレビューをお願いします。
            """

            tasks.append((reviewer["system_prompt"], user_prompt, reviewer["task_id"]))

        # ===== 並列実行 =====
        print("⚡ 複数レビュアーによる並列レビュー実行中...")
        futures = [
            self.executor.submit(self._call_llm_sync, sys_prompt, user_prompt, task_id)
            for sys_prompt, user_prompt, task_id in tasks
        ]

        # ===== 結果を収集 =====
        review_results = []
        for future in futures:
            result = future.result()
            review_results.append(result)

            if result["success"]:
                print(
                    f"✅ {result['task_id']} レビュー完了 ({result['execution_time']:.2f}秒)"
                )
            else:
                print(f"❌ {result['task_id']} エラー: {result['error']}")

        # ===== レビュー結果を統合 =====
        print("📋 レビュー結果統合中...")

        successful_reviews = [r for r in review_results if r["success"]]

        if successful_reviews:
            # ===== 統合レビューレポート作成 =====
            integration_system_prompt = """
            あなたはコードレビューの統合専門家です。
            複数の専門家によるレビュー結果を統合し、
            総合的なコードレビューレポートを作成してください。
            """

            all_reviews = "\n\n".join(
                [
                    f"【{result['task_id']}のレビュー】\n{result['response']}"
                    for result in successful_reviews
                ]
            )

            integration_user_prompt = f"""
            以下は複数の専門家によるコードレビュー結果です。
            これらを統合して、包括的なレビューレポートを作成してください：
            
            {all_reviews}
            
            統合レポートには以下を含めてください：
            1. 総合評価と概要
            2. 最優先で修正すべき問題
            3. セキュリティ、パフォーマンス、品質の各観点からの主要な指摘事項
            4. 良い点・評価できる実装
            5. 段階的な改善ロードマップ
            6. 次のレビューまでのアクションアイテム
            """

            integration_result = self._call_llm_sync(
                integration_system_prompt, integration_user_prompt, "review_integration"
            )

            final_review = (
                integration_result["response"]
                if integration_result["success"]
                else "統合に失敗しました"
            )
        else:
            final_review = "すべてのレビューに失敗しました"

        execution_time = time.time() - start_time

        # ===== ログに記録 =====
        self._log_execution("voting", "コードレビュー", execution_time, review_results)

        print(f"🎉 並列コードレビュー完了 (総実行時間: {execution_time:.2f}秒)")

        return {
            "pattern": "parallel_code_review",
            "num_reviewers": len(selected_reviewers),
            "successful_reviews": len(successful_reviews),
            "individual_reviews": review_results,
            "integrated_review": final_review,
            "total_execution_time": execution_time,
        }

    def get_performance_statistics(self) -> Dict[str, Any]:
        """
        パフォーマンス統計を取得

        Returns:
            Dict[str, Any]: 実行統計
        """

        if not self.execution_log:
            return {"total_executions": 0}

        # ===== パターン別統計 =====
        pattern_stats = {}
        total_time = 0

        for log_entry in self.execution_log:
            pattern = log_entry["pattern"]
            exec_time = log_entry["execution_time"]

            if pattern not in pattern_stats:
                pattern_stats[pattern] = {"count": 0, "total_time": 0, "avg_time": 0}

            pattern_stats[pattern]["count"] += 1
            pattern_stats[pattern]["total_time"] += exec_time
            total_time += exec_time

        # ===== 平均時間を計算 =====
        for pattern_data in pattern_stats.values():
            pattern_data["avg_time"] = (
                pattern_data["total_time"] / pattern_data["count"]
            )

        return {
            "total_executions": len(self.execution_log),
            "total_time": total_time,
            "pattern_statistics": pattern_stats,
            "average_execution_time": total_time / len(self.execution_log),
        }

    def cleanup(self):
        """
        リソースをクリーンアップ
        """
        self.executor.shutdown(wait=True)
        print("🧹 リソースをクリーンアップしました")


# ===== 使用例 =====
def main():
    """
    Parallelizationパターンのデモンストレーション
    """
    print("=== Parallelization パターンのデモ ===\n")

    # ===== 並列化システムのインスタンスを作成 =====
    parallel_system = ParallelizationSystem(max_workers=4)

    try:
        # ===== デモ1: セクショニング - 文書要約 =====
        print("📄 デモ1: セクショニング - 長文書の並列要約")
        print("-" * 60)

        sample_document = """
        人工知能（AI）の発展は、現代社会に革命的な変化をもたらしています。
        機械学習、深層学習、自然言語処理などの技術が急速に進歩し、
        私たちの日常生活からビジネス、研究まで、あらゆる分野に影響を与えています。
        
        特に注目すべきは、生成AIの登場です。ChatGPTやGPT-4などの大規模言語モデルは、
        人間のような自然な対話能力を示し、文章作成、翻訳、プログラミング支援など、
        多様なタスクを高精度で実行できるようになりました。
        これにより、知識労働者の働き方が大きく変化することが予想されます。
        
        一方で、AIの急速な発展は新たな課題も生み出しています。
        雇用への影響、プライバシーの問題、AIの倫理的な使用、
        偽情報の拡散リスクなど、社会全体で取り組むべき問題が山積しています。
        
        教育分野では、AIを活用した個別指導システムや、
        学習者の理解度に応じたカスタマイズされた教材の提供が可能になりました。
        しかし、教育者の役割の変化や、学習者の批判的思考力の育成などの
        新たな課題にも対応する必要があります。
        
        医療分野においても、AIは診断支援、薬物発見、個別化医療などで
        大きな貢献を果たしています。画像診断においては、
        熟練した医師と同等またはそれ以上の精度を示すAIシステムも登場しています。
        
        今後、AIと人間が協働する社会を実現するためには、
        技術的な発展だけでなく、法的枠組みの整備、
        倫理的ガイドラインの策定、そして人々のAIリテラシーの向上が不可欠です。
        """

        sectioning_result = parallel_system.sectioning_document_analysis(
            document=sample_document, analysis_type="要約"
        )

        print("\n📊 セクショニング結果:")
        print(f"- 処理セクション数: {sectioning_result['sections_count']}")
        print(f"- 成功セクション数: {sectioning_result['successful_sections']}")
        print(f"- 実行時間: {sectioning_result['total_execution_time']:.2f}秒")
        print(f"\n統合要約:\n{sectioning_result['integrated_result'][:200]}...\n")

        # ===== デモ2: 投票 - 品質評価 =====
        print("🗳️ デモ2: 投票 - 複数評価者による品質評価")
        print("-" * 60)

        sample_content = """
        【プロダクト提案書】
        
        革新的なAI搭載学習アプリ「StudyMate」
        
        概要：
        StudyMateは、個々の学習者に最適化されたAI学習支援アプリです。
        機械学習アルゴリズムを使用して学習者の弱点を特定し、
        パーソナライズされた学習プランを提供します。
        
        主な機能：
        1. 適応的学習システム - 学習者の理解度に応じて難易度を調整
        2. インタラクティブなQ&A - AI チューターとの対話型学習
        3. 進捗追跡とアナリティクス - 詳細な学習データの可視化
        4. ゲーミフィケーション要素 - ポイント制度とバッジシステム
        
        ターゲット市場：
        - 中学生・高校生（主要ターゲット）
        - 大学受験生
        - 資格試験受験者
        
        収益モデル：
        - フリーミアム（基本機能無料、高度機能有料）
        - 月額サブスクリプション
        - 企業向けライセンス
        """

        voting_result = parallel_system.voting_quality_assessment(
            content=sample_content,
            assessment_criteria="ビジネス提案書の品質",
            num_votes=4,
        )

        print("\n📊 投票評価結果:")
        print(f"- 評価者数: {voting_result['num_evaluators']}")
        print(f"- 成功評価数: {voting_result['successful_evaluations']}")
        print(f"- 実行時間: {voting_result['total_execution_time']:.2f}秒")
        print(f"\n最終評価:\n{voting_result['final_assessment'][:200]}...\n")

        # ===== デモ3: 並列コードレビュー =====
        print("👥 デモ3: 複数専門家による並列コードレビュー")
        print("-" * 60)

        sample_code = """
def user_login(username, password):
    # ユーザー認証システム
    import sqlite3
    
    # データベース接続
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    
    # SQL クエリ実行
    query = f"SELECT * FROM users WHERE username = '{username}' AND password = '{password}'"
    cursor.execute(query)
    result = cursor.fetchone()
    
    if result:
        # ログイン成功
        session_token = username + "_" + str(datetime.now())
        return {"status": "success", "token": session_token}
    else:
        # ログイン失敗
        return {"status": "failed"}
    
    conn.close()

def get_user_data(user_id):
    # ユーザーデータ取得
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    
    query = f"SELECT * FROM user_data WHERE id = {user_id}"
    result = cursor.execute(query).fetchall()
    
    return result
        """

        code_review_result = parallel_system.parallel_code_review(
            code=sample_code, num_reviewers=3
        )

        print("\n📊 コードレビュー結果:")
        print(f"- レビュアー数: {code_review_result['num_reviewers']}")
        print(f"- 成功レビュー数: {code_review_result['successful_reviews']}")
        print(f"- 実行時間: {code_review_result['total_execution_time']:.2f}秒")
        print(f"\n統合レビュー:\n{code_review_result['integrated_review'][:300]}...\n")

        # ===== パフォーマンス統計の表示 =====
        print("📈 パフォーマンス統計")
        print("-" * 30)
        stats = parallel_system.get_performance_statistics()
        print(f"総実行回数: {stats['total_executions']}")
        print(f"総実行時間: {stats['total_time']:.2f}秒")
        print(f"平均実行時間: {stats['average_execution_time']:.2f}秒")

        print("\nパターン別統計:")
        for pattern, data in stats["pattern_statistics"].items():
            print(f"  - {pattern}: {data['count']}回, 平均{data['avg_time']:.2f}秒")

    finally:
        # ===== リソースのクリーンアップ =====
        parallel_system.cleanup()


if __name__ == "__main__":
    main()
