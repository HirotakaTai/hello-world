"""
Evaluator-optimizer パターン
===========================

このパターンは、一つのLLMが回答を生成し、別のLLMがその回答を評価して
フィードバックを提供し、元のLLMが改善版を作成するループを実装します。

特徴：
- Generator（生成者）: 初期回答を作成
- Evaluator（評価者）: 回答の品質を評価し、改善点を特定
- Optimizer（最適化者）: 評価を基に改善版を作成
- 反復プロセス: 満足のいく品質になるまで繰り返し

例：
- 文学翻訳の品質向上
- 複雑な検索タスクでの情報精度向上
- コードの品質改善
- 文章の推敲と改善

このパターンの利点：
- 反復的な品質向上
- 客観的な評価による改善
- 人間の推敲プロセスの模倣
- 高品質な最終成果物の獲得
"""

import time
from datetime import datetime
from typing import Any
from typing import Dict

from dotenv import load_dotenv
from langchain.schema import HumanMessage
from langchain.schema import SystemMessage
from langchain_openai import ChatOpenAI

# ===== 環境変数の読み込み =====
load_dotenv()


class EvaluationCriteria:
    """
    評価基準を定義するクラス
    """

    # ===== 翻訳品質評価基準 =====
    TRANSLATION_QUALITY = {
        "accuracy": "原文の意味の正確性",
        "fluency": "自然さと読みやすさ",
        "completeness": "内容の完全性",
        "cultural_adaptation": "文化的適応性",
        "terminology": "専門用語の適切性",
    }

    # ===== 文章品質評価基準 =====
    WRITING_QUALITY = {
        "clarity": "明確性と理解しやすさ",
        "coherence": "論理的な一貫性",
        "engagement": "読者の関心を引く度合い",
        "grammar": "文法的正確性",
        "style": "文体の適切性",
    }

    # ===== コード品質評価基準 =====
    CODE_QUALITY = {
        "functionality": "機能の正確性",
        "readability": "可読性",
        "efficiency": "効率性",
        "maintainability": "保守性",
        "security": "セキュリティ",
    }

    # ===== 研究レポート評価基準 =====
    RESEARCH_QUALITY = {
        "accuracy": "情報の正確性",
        "completeness": "包括性",
        "methodology": "調査方法の適切性",
        "analysis": "分析の深さ",
        "presentation": "プレゼンテーションの質",
    }


class Generator:
    """
    コンテンツ生成を担当するクラス
    """

    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=0.7,  # 創造的な生成のため適度なランダム性
        )
        self.generation_count = 0

    def generate_initial_response(self, task: str, context: str = "") -> Dict[str, Any]:
        """
        初期回答を生成

        Args:
            task (str): 実行するタスク
            context (str): 追加のコンテキスト情報

        Returns:
            Dict[str, Any]: 生成結果
        """

        start_time = time.time()
        self.generation_count += 1

        print("✍️ Generator: 初期回答を生成中...")

        system_prompt = """
        あなたは高品質なコンテンツ生成の専門家です。
        与えられたタスクに対して、最善を尽くして回答を作成してください。
        初回生成なので、可能な限り包括的で高品質な回答を心がけてください。
        """

        user_prompt = f"""
        以下のタスクを実行してください：
        
        【タスク】
        {task}
        """

        if context:
            user_prompt += f"\n\n【追加コンテキスト】\n{context}"

        user_prompt += """
        
        高品質で完全な回答を提供してください。
        """

        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ]

            response = self.llm.invoke(messages)
            execution_time = time.time() - start_time

            print(f"✅ Generator: 初期回答生成完了 ({execution_time:.2f}秒)")

            return {
                "success": True,
                "content": response.content,
                "generation_type": "initial",
                "generation_number": self.generation_count,
                "execution_time": execution_time,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }

        except Exception as e:
            execution_time = time.time() - start_time
            print(f"❌ Generator: 初期回答生成エラー: {str(e)}")

            return {
                "success": False,
                "error": str(e),
                "generation_type": "initial",
                "generation_number": self.generation_count,
                "execution_time": execution_time,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }

    def generate_improved_response(
        self,
        original_task: str,
        previous_response: str,
        evaluation_feedback: str,
        context: str = "",
    ) -> Dict[str, Any]:
        """
        評価フィードバックを基に改善された回答を生成

        Args:
            original_task (str): 元のタスク
            previous_response (str): 前の回答
            evaluation_feedback (str): 評価フィードバック
            context (str): 追加のコンテキスト情報

        Returns:
            Dict[str, Any]: 改善された生成結果
        """

        start_time = time.time()
        self.generation_count += 1

        print(f"🔄 Generator: 改善回答を生成中（{self.generation_count}回目）...")

        system_prompt = """
        あなたは継続的改善の専門家です。
        評価者からのフィードバックを基に、前回の回答を改善してください。
        指摘された問題点を修正し、より高品質な回答を作成してください。
        """

        user_prompt = f"""
        以下の情報を基に、改善された回答を作成してください：
        
        【元のタスク】
        {original_task}
        
        【前回の回答】
        {previous_response}
        
        【評価者からのフィードバック】
        {evaluation_feedback}
        """

        if context:
            user_prompt += f"\n\n【追加コンテキスト】\n{context}"

        user_prompt += """
        
        評価者の指摘を真摯に受け止め、具体的な改善を行った回答を提供してください。
        前回の良い部分は維持しつつ、問題点を修正してください。
        """

        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ]

            response = self.llm.invoke(messages)
            execution_time = time.time() - start_time

            print(f"✅ Generator: 改善回答生成完了 ({execution_time:.2f}秒)")

            return {
                "success": True,
                "content": response.content,
                "generation_type": "improved",
                "generation_number": self.generation_count,
                "execution_time": execution_time,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }

        except Exception as e:
            execution_time = time.time() - start_time
            print(f"❌ Generator: 改善回答生成エラー: {str(e)}")

            return {
                "success": False,
                "error": str(e),
                "generation_type": "improved",
                "generation_number": self.generation_count,
                "execution_time": execution_time,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }


class Evaluator:
    """
    コンテンツ評価を担当するクラス
    """

    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=0.2,  # 評価の一貫性を保つため低温度
        )
        self.evaluation_count = 0

    def evaluate_response(
        self,
        task: str,
        response: str,
        criteria: Dict[str, str],
        quality_threshold: float = 7.0,
    ) -> Dict[str, Any]:
        """
        回答を評価

        Args:
            task (str): 元のタスク
            response (str): 評価する回答
            criteria (Dict[str, str]): 評価基準
            quality_threshold (float): 品質閾値（10点満点）

        Returns:
            Dict[str, Any]: 評価結果
        """

        start_time = time.time()
        self.evaluation_count += 1

        print(f"🔍 Evaluator: 回答を評価中（{self.evaluation_count}回目）...")

        # ===== 評価基準を文字列化 =====
        criteria_text = "\n".join(
            [f"- {key}: {description}" for key, description in criteria.items()]
        )

        system_prompt = f"""
        あなたは厳格で公正な評価者です。
        以下の評価基準に従って、回答の品質を客観的に評価してください。
        
        【評価基準】
        {criteria_text}
        
        各基準について1-10点で評価し、建設的なフィードバックを提供してください。
        """

        user_prompt = f"""
        以下のタスクと回答を評価してください：
        
        【元のタスク】
        {task}
        
        【評価対象の回答】
        {response}
        
        以下の形式で評価結果を提供してください：
        
        ## 総合評価
        - 総合スコア: X/10点
        - 品質レベル: [優秀/良好/普通/要改善/不十分]
        
        ## 詳細評価
        {criteria_text}
        
        各項目について：
        - スコア: X/10点
        - コメント: 具体的な評価理由
        
        ## 良い点
        - 評価できる点を具体的に列挙
        
        ## 改善点
        - 具体的な改善提案を列挙
        - 優先度の高い改善点から順に記載
        
        ## 次回への提案
        - より良い回答にするための具体的なアドバイス
        
        厳格かつ建設的な評価をお願いします。
        """

        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ]

            response_obj = self.llm.invoke(messages)
            execution_time = time.time() - start_time

            # ===== スコアを抽出（簡易的な方法） =====
            evaluation_text = response_obj.content
            overall_score = self._extract_overall_score(evaluation_text)
            needs_improvement = overall_score < quality_threshold

            print(f"📊 Evaluator: 評価完了 - スコア: {overall_score}/10")
            if needs_improvement:
                print(f"⚠️ 品質閾値({quality_threshold})を下回っているため改善が必要")
            else:
                print(f"✅ 品質閾値({quality_threshold})を満たしています")

            return {
                "success": True,
                "evaluation_text": evaluation_text,
                "overall_score": overall_score,
                "needs_improvement": needs_improvement,
                "quality_threshold": quality_threshold,
                "evaluation_number": self.evaluation_count,
                "execution_time": execution_time,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }

        except Exception as e:
            execution_time = time.time() - start_time
            print(f"❌ Evaluator: 評価エラー: {str(e)}")

            return {
                "success": False,
                "error": str(e),
                "evaluation_number": self.evaluation_count,
                "execution_time": execution_time,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }

    def _extract_overall_score(self, evaluation_text: str) -> float:
        """
        評価テキストから総合スコアを抽出

        Args:
            evaluation_text (str): 評価テキスト

        Returns:
            float: 抽出されたスコア（抽出できない場合は5.0）
        """

        import re

        # ===== 「総合スコア: X/10点」のパターンを探す =====
        score_patterns = [
            r"総合スコア[：:]\s*(\d+(?:\.\d+)?)[/／]10",
            r"総合[：:].*?(\d+(?:\.\d+)?)[/／]10",
            r"(\d+(?:\.\d+)?)[/／]10点",
            r"スコア[：:]\s*(\d+(?:\.\d+)?)",
        ]

        for pattern in score_patterns:
            match = re.search(pattern, evaluation_text)
            if match:
                try:
                    score = float(match.group(1))
                    return min(max(score, 0), 10)  # 0-10の範囲に制限
                except ValueError:
                    continue

        # ===== スコアが見つからない場合のデフォルト値 =====
        print("⚠️ 総合スコアを抽出できませんでした。デフォルト値 5.0 を使用します。")
        return 5.0


class EvaluatorOptimizer:
    """
    評価者・最適化者パターンのメインクラス
    """

    def __init__(self, max_iterations: int = 3, quality_threshold: float = 7.0):
        self.generator = Generator()
        self.evaluator = Evaluator()
        self.max_iterations = max_iterations
        self.quality_threshold = quality_threshold
        self.optimization_log = []

    def optimize_response(
        self, task: str, evaluation_criteria: Dict[str, str], context: str = ""
    ) -> Dict[str, Any]:
        """
        評価・最適化ループを実行

        Args:
            task (str): 実行するタスク
            evaluation_criteria (Dict[str, str]): 評価基準
            context (str): 追加のコンテキスト情報

        Returns:
            Dict[str, Any]: 最適化結果
        """

        start_time = time.time()
        print("🚀 評価・最適化プロセス開始")
        print(
            f"📋 最大反復回数: {self.max_iterations}, 品質閾値: {self.quality_threshold}/10"
        )
        print("-" * 60)

        iterations = []
        current_response = None
        best_response = None
        best_score = 0

        for iteration in range(1, self.max_iterations + 1):
            print(f"\n🔄 反復 {iteration}/{self.max_iterations}")
            print("-" * 30)

            iteration_start = time.time()

            # ===== ステップ1: 回答生成 =====
            if iteration == 1:
                # ===== 初回生成 =====
                generation_result = self.generator.generate_initial_response(
                    task, context
                )
            else:
                # ===== 改善生成 =====
                previous_evaluation = iterations[-1]["evaluation"]
                generation_result = self.generator.generate_improved_response(
                    task,
                    current_response,
                    previous_evaluation["evaluation_text"],
                    context,
                )

            if not generation_result["success"]:
                print(f"❌ 反復 {iteration}: 生成失敗")
                break

            current_response = generation_result["content"]

            # ===== ステップ2: 評価 =====
            evaluation_result = self.evaluator.evaluate_response(
                task, current_response, evaluation_criteria, self.quality_threshold
            )

            if not evaluation_result["success"]:
                print(f"❌ 反復 {iteration}: 評価失敗")
                break

            current_score = evaluation_result["overall_score"]

            # ===== 最良回答の更新 =====
            if current_score > best_score:
                best_response = current_response
                best_score = current_score
                print(f"🏆 新しい最良回答を記録: {best_score}/10")

            # ===== 反復結果を記録 =====
            iteration_time = time.time() - iteration_start
            iteration_data = {
                "iteration": iteration,
                "generation": generation_result,
                "evaluation": evaluation_result,
                "response": current_response,
                "score": current_score,
                "is_best": current_score == best_score,
                "iteration_time": iteration_time,
            }
            iterations.append(iteration_data)

            print(
                f"📊 反復 {iteration} 完了 - スコア: {current_score}/10 ({iteration_time:.2f}秒)"
            )

            # ===== 品質閾値に達した場合は終了 =====
            if not evaluation_result["needs_improvement"]:
                print(
                    f"🎉 品質閾値 {self.quality_threshold}/10 に達しました！最適化完了"
                )
                break

        total_time = time.time() - start_time

        # ===== 最終結果をまとめる =====
        result = {
            "task": task,
            "context": context,
            "evaluation_criteria": evaluation_criteria,
            "max_iterations": self.max_iterations,
            "quality_threshold": self.quality_threshold,
            "completed_iterations": len(iterations),
            "iterations": iterations,
            "best_response": best_response,
            "best_score": best_score,
            "final_response": current_response,
            "final_score": iterations[-1]["score"] if iterations else 0,
            "improvement_achieved": best_score > iterations[0]["score"]
            if iterations
            else False,
            "threshold_reached": best_score >= self.quality_threshold,
            "total_execution_time": total_time,
        }

        # ===== ログに記録 =====
        self.optimization_log.append(
            {
                "task_summary": task[:100] + "..." if len(task) > 100 else task,
                "completed_iterations": len(iterations),
                "best_score": best_score,
                "improvement": best_score - iterations[0]["score"] if iterations else 0,
                "threshold_reached": best_score >= self.quality_threshold,
                "execution_time": total_time,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
        )

        print("\n🎯 最適化プロセス完了")
        print(
            f"📊 最終結果: {best_score}/10 (改善度: +{best_score - iterations[0]['score']:.1f})"
        )
        print(f"⏱️ 総実行時間: {total_time:.2f}秒")

        return result

    def translate_and_optimize(
        self, text: str, target_language: str = "英語", domain: str = "一般"
    ) -> Dict[str, Any]:
        """
        翻訳品質最適化の専用メソッド

        Args:
            text (str): 翻訳する文章
            target_language (str): 翻訳先言語
            domain (str): 分野（一般、技術、文学、ビジネスなど）

        Returns:
            Dict[str, Any]: 翻訳最適化結果
        """

        task = f"""
        以下の文章を{target_language}に翻訳してください。
        
        【翻訳対象文章】
        {text}
        
        【翻訳要件】
        - 分野: {domain}
        - 原文の意味とニュアンスを正確に伝える
        - {target_language}として自然で読みやすい表現
        - 専門用語は適切に翻訳
        - 文化的コンテキストを考慮
        """

        context = f"""
        翻訳分野: {domain}
        対象言語: {target_language}
        原文の言語: 日本語
        品質重視の高精度翻訳を求めています。
        """

        return self.optimize_response(
            task=task,
            evaluation_criteria=EvaluationCriteria.TRANSLATION_QUALITY,
            context=context,
        )

    def write_and_optimize(
        self, topic: str, content_type: str = "記事", target_audience: str = "一般読者"
    ) -> Dict[str, Any]:
        """
        文章作成品質最適化の専用メソッド

        Args:
            topic (str): 執筆トピック
            content_type (str): コンテンツタイプ（記事、レポート、エッセイなど）
            target_audience (str): ターゲット読者

        Returns:
            Dict[str, Any]: 文章最適化結果
        """

        task = f"""
        以下のトピックについて{content_type}を執筆してください。
        
        【トピック】
        {topic}
        
        【執筆要件】
        - コンテンツタイプ: {content_type}
        - ターゲット読者: {target_audience}
        - 読みやすく魅力的な文章
        - 論理的な構成
        - 読者の関心を引く内容
        - 適切な長さ（800-1200文字程度）
        """

        context = f"""
        コンテンツタイプ: {content_type}
        ターゲット読者: {target_audience}
        高品質で魅力的な文章を求めています。
        """

        return self.optimize_response(
            task=task,
            evaluation_criteria=EvaluationCriteria.WRITING_QUALITY,
            context=context,
        )

    def get_optimization_statistics(self) -> Dict[str, Any]:
        """
        最適化統計を取得

        Returns:
            Dict[str, Any]: 統計情報
        """

        if not self.optimization_log:
            return {"total_optimizations": 0}

        total_optimizations = len(self.optimization_log)
        total_iterations = sum(
            log["completed_iterations"] for log in self.optimization_log
        )
        successful_optimizations = sum(
            1 for log in self.optimization_log if log["threshold_reached"]
        )
        total_improvement = sum(log["improvement"] for log in self.optimization_log)
        total_time = sum(log["execution_time"] for log in self.optimization_log)

        avg_iterations = total_iterations / total_optimizations
        avg_improvement = total_improvement / total_optimizations
        avg_time = total_time / total_optimizations
        success_rate = (successful_optimizations / total_optimizations) * 100

        return {
            "total_optimizations": total_optimizations,
            "total_iterations": total_iterations,
            "successful_optimizations": successful_optimizations,
            "success_rate": success_rate,
            "average_iterations": avg_iterations,
            "average_improvement": avg_improvement,
            "average_execution_time": avg_time,
            "total_execution_time": total_time,
        }


# ===== 使用例 =====
def main():
    """
    Evaluator-optimizerパターンのデモンストレーション
    """
    print("=== Evaluator-optimizer パターンのデモ ===\n")

    # ===== 評価・最適化システムのインスタンスを作成 =====
    optimizer = EvaluatorOptimizer(max_iterations=3, quality_threshold=7.5)

    # ===== デモ1: 翻訳品質の最適化 =====
    print("🌐 デモ1: 翻訳品質の最適化")
    print("=" * 50)

    japanese_text = """
    人工知能の発展により、我々の社会は根本的な変革を迎えている。
    機械学習技術の進歩は、従来人間の専売特許とされていた認知的タスクを
    自動化することを可能にし、労働市場に大きな影響を与えている。
    しかし、この技術革新は単なる効率化にとどまらず、
    人間と機械の新たな協働関係を築く機会でもある。
    重要なのは、技術の発展を恐れるのではなく、
    いかにして人間の創造性と機械の計算能力を
    最適に組み合わせるかを考えることである。
    """

    translation_result = optimizer.translate_and_optimize(
        text=japanese_text, target_language="英語", domain="技術・学術"
    )

    print("\n📊 翻訳最適化結果:")
    print(f"- 反復回数: {translation_result['completed_iterations']}")
    print(f"- 最高スコア: {translation_result['best_score']}/10")
    print(
        f"- 改善度: +{translation_result['best_score'] - translation_result['iterations'][0]['score']:.1f}点"
    )
    print(
        f"- 閾値達成: {'はい' if translation_result['threshold_reached'] else 'いいえ'}"
    )
    print(f"- 実行時間: {translation_result['total_execution_time']:.2f}秒")

    print("\n🏆 最良の翻訳:")
    print(
        translation_result["best_response"][:300] + "..."
        if len(translation_result["best_response"]) > 300
        else translation_result["best_response"]
    )

    # ===== デモ2: 文章作成の最適化 =====
    print("\n\n✍️ デモ2: 文章作成の最適化")
    print("=" * 50)

    writing_result = optimizer.write_and_optimize(
        topic="リモートワークが企業文化に与える影響と今後の展望",
        content_type="ビジネス記事",
        target_audience="企業経営者・人事担当者",
    )

    print("\n📊 文章最適化結果:")
    print(f"- 反復回数: {writing_result['completed_iterations']}")
    print(f"- 最高スコア: {writing_result['best_score']}/10")
    print(
        f"- 改善度: +{writing_result['best_score'] - writing_result['iterations'][0]['score']:.1f}点"
    )
    print(f"- 閾値達成: {'はい' if writing_result['threshold_reached'] else 'いいえ'}")
    print(f"- 実行時間: {writing_result['total_execution_time']:.2f}秒")

    print("\n🏆 最良の記事:")
    print(
        writing_result["best_response"][:400] + "..."
        if len(writing_result["best_response"]) > 400
        else writing_result["best_response"]
    )

    # ===== デモ3: 一般的なタスクの最適化 =====
    print("\n\n🔬 デモ3: 研究レポートの最適化")
    print("=" * 50)

    research_task = """
    「日本におけるデジタルトランスフォーメーション（DX）の現状と課題」について、
    以下の要素を含む包括的な研究レポートを作成してください：
    
    1. DXの定義と重要性
    2. 日本企業のDX導入状況（統計データを含む）
    3. 主要な成功事例の分析
    4. 導入における主な障害と課題
    5. 政府の政策と支援制度
    6. 国際比較（特にアメリカ、ヨーロッパとの比較）
    7. 今後の展望と提言
    
    レポートは学術的な厳密性を保ちつつ、実務者にも理解しやすい内容にしてください。
    """

    research_result = optimizer.optimize_response(
        task=research_task,
        evaluation_criteria=EvaluationCriteria.RESEARCH_QUALITY,
        context="学術論文レベルの質を求める包括的な研究レポート",
    )

    print("\n📊 研究レポート最適化結果:")
    print(f"- 反復回数: {research_result['completed_iterations']}")
    print(f"- 最高スコア: {research_result['best_score']}/10")
    print(
        f"- 改善度: +{research_result['best_score'] - research_result['iterations'][0]['score']:.1f}点"
    )
    print(f"- 閾値達成: {'はい' if research_result['threshold_reached'] else 'いいえ'}")
    print(f"- 実行時間: {research_result['total_execution_time']:.2f}秒")

    print("\n🏆 最良のレポート:")
    print(
        research_result["best_response"][:500] + "..."
        if len(research_result["best_response"]) > 500
        else research_result["best_response"]
    )

    # ===== 統計情報の表示 =====
    print("\n\n📈 最適化統計")
    print("=" * 30)
    stats = optimizer.get_optimization_statistics()
    print(f"総最適化回数: {stats['total_optimizations']}")
    print(f"総反復回数: {stats['total_iterations']}")
    print(f"成功した最適化: {stats['successful_optimizations']}")
    print(f"成功率: {stats['success_rate']:.1f}%")
    print(f"平均反復回数: {stats['average_iterations']:.1f}")
    print(f"平均改善度: +{stats['average_improvement']:.1f}点")
    print(f"平均実行時間: {stats['average_execution_time']:.2f}秒")
    print(f"総実行時間: {stats['total_execution_time']:.2f}秒")


if __name__ == "__main__":
    main()
