#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LangGraph版 Evaluator-Optimizer Pattern
LangGraphを使用して生成→評価→改善のループを実装し、
品質を段階的に向上させるパターン
"""

import datetime
from typing import Any
from typing import Dict
from typing import List
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


class EvaluationResult(BaseModel):
    """評価結果のデータクラス"""

    score: float = Field(description="評価スコア (0.0-10.0)", ge=0.0, le=10.0)
    strengths: List[str] = Field(description="良い点のリスト")
    weaknesses: List[str] = Field(description="改善が必要な点のリスト")
    suggestions: List[str] = Field(description="具体的な改善提案")
    overall_feedback: str = Field(description="総合的なフィードバック")


class OptimizationState(TypedDict):
    """最適化ワークフローの状態定義"""

    task_description: str  # タスクの説明
    current_content: str  # 現在のコンテンツ
    evaluation_history: List[EvaluationResult]  # 評価履歴
    iteration_count: int  # 反復回数
    max_iterations: int  # 最大反復回数
    target_score: float  # 目標スコア
    final_content: str  # 最終コンテンツ
    optimization_type: str  # 最適化タイプ
    execution_log: List[str]  # 実行ログ


# ===== LangGraphベースのEvaluator-Optimizerクラス =====


class LangGraphEvaluatorOptimizer:
    """LangGraphを使用した評価・最適化システム"""

    def __init__(self):
        """システムの初期化"""
        print("⚖️ LangGraph版 Evaluator-Optimizerシステムを初期化中...")

        # OpenAI ChatLLMモデルを初期化
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7, verbose=True)

        # プロンプトテンプレートを設定
        self._setup_prompts()

        # 各最適化タイプ用のLangGraphワークフローを構築
        self.writing_graph = self._build_writing_optimization_graph()
        self.translation_graph = self._build_translation_optimization_graph()
        self.code_graph = self._build_code_optimization_graph()

        print("✅ 評価・最適化システムの初期化が完了しました！")

    def _setup_prompts(self):
        """各処理用のプロンプトテンプレートを設定"""

        # 初期コンテンツ生成プロンプト
        self.initial_generation_prompts = {
            "writing": ChatPromptTemplate.from_template(
                """以下のタスクに基づいて、初期版の文章を作成してください。

タスク: {task_description}

執筆時の注意点:
1. 読者にとって分かりやすい内容
2. 論理的な構成
3. 適切な例や具体例の使用
4. 魅力的で読みやすい文体

初期版の文章を作成してください。
"""
            ),
            "translation": ChatPromptTemplate.from_template(
                """以下のタスクに基づいて、初期版の翻訳を作成してください。

タスク: {task_description}

翻訳時の注意点:
1. 原文の意味を正確に伝える
2. 自然な日本語表現
3. 適切な敬語や語調の使用
4. 文化的な配慮

初期版の翻訳を作成してください。
"""
            ),
            "code": ChatPromptTemplate.from_template(
                """以下のタスクに基づいて、初期版のコードを作成してください。

タスク: {task_description}

プログラミング時の注意点:
1. 清潔で読みやすいコード
2. 適切なコメントの追加
3. 効率的なアルゴリズム
4. エラーハンドリング

初期版のコードを作成してください。
"""
            ),
        }

        # 評価プロンプト
        self.evaluation_prompts = {
            "writing": ChatPromptTemplate.from_template(
                """あなたは経験豊富な文章評価者です。以下の文章を評価してください。

タスク: {task_description}

評価対象の文章:
{content}

評価基準:
1. 内容の正確性と完全性 (0-2点)
2. 構成の論理性と分かりやすさ (0-2点)
3. 文章の読みやすさと文体 (0-2点)
4. 例や具体例の適切さ (0-2点)
5. 読者への価値提供 (0-2点)

以下のJSON形式で評価結果を返してください:
{{
    "score": 8.5,
    "strengths": ["良い点1", "良い点2"],
    "weaknesses": ["改善点1", "改善点2"],
    "suggestions": ["改善提案1", "改善提案2"],
    "overall_feedback": "総合的なフィードバック"
}}
"""
            ),
            "translation": ChatPromptTemplate.from_template(
                """あなたは経験豊富な翻訳評価者です。以下の翻訳を評価してください。

タスク: {task_description}

評価対象の翻訳:
{content}

評価基準:
1. 原文の意味の正確性 (0-2点)
2. 日本語の自然さ (0-2点)
3. 語調と文体の適切さ (0-2点)
4. 文化的配慮 (0-2点)
5. 読みやすさ (0-2点)

以下のJSON形式で評価結果を返してください:
{{
    "score": 8.5,
    "strengths": ["良い点1", "良い点2"],
    "weaknesses": ["改善点1", "改善点2"],
    "suggestions": ["改善提案1", "改善提案2"],
    "overall_feedback": "総合的なフィードバック"
}}
"""
            ),
            "code": ChatPromptTemplate.from_template(
                """あなたは経験豊富なコードレビュアーです。以下のコードを評価してください。

タスク: {task_description}

評価対象のコード:
{content}

評価基準:
1. 機能の正確性と完全性 (0-2点)
2. コードの可読性 (0-2点)
3. 効率性とパフォーマンス (0-2点)
4. エラーハンドリング (0-2点)
5. ベストプラクティスの遵守 (0-2点)

以下のJSON形式で評価結果を返してください:
{{
    "score": 8.5,
    "strengths": ["良い点1", "良い点2"],
    "weaknesses": ["改善点1", "改善点2"],
    "suggestions": ["改善提案1", "改善提案2"],
    "overall_feedback": "総合的なフィードバック"
}}
"""
            ),
        }

        # 改善プロンプト
        self.improvement_prompts = {
            "writing": ChatPromptTemplate.from_template(
                """あなたは優秀な文章改善者です。以下の評価結果を基に文章を改善してください。

タスク: {task_description}

現在の文章:
{current_content}

評価結果:
- スコア: {score}/10
- 改善が必要な点: {weaknesses}
- 改善提案: {suggestions}
- フィードバック: {overall_feedback}

上記の評価を基に、より良い文章に改善してください。
"""
            ),
            "translation": ChatPromptTemplate.from_template(
                """あなたは優秀な翻訳改善者です。以下の評価結果を基に翻訳を改善してください。

タスク: {task_description}

現在の翻訳:
{current_content}

評価結果:
- スコア: {score}/10
- 改善が必要な点: {weaknesses}
- 改善提案: {suggestions}
- フィードバック: {overall_feedback}

上記の評価を基に、より良い翻訳に改善してください。
"""
            ),
            "code": ChatPromptTemplate.from_template(
                """あなたは優秀なコード改善者です。以下の評価結果を基にコードを改善してください。

タスク: {task_description}

現在のコード:
{current_content}

評価結果:
- スコア: {score}/10
- 改善が必要な点: {weaknesses}
- 改善提案: {suggestions}
- フィードバック: {overall_feedback}

上記の評価を基に、より良いコードに改善してください。
"""
            ),
        }

    def _build_writing_optimization_graph(self) -> StateGraph:
        """文章最適化ワークフローを構築"""
        print("🔧 文章最適化ワークフローを構築中...")

        workflow = StateGraph(OptimizationState)

        # ノードを追加
        workflow.add_node("generate_initial", self._generate_initial_content)
        workflow.add_node("evaluate", self._evaluate_content)
        workflow.add_node("improve", self._improve_content)
        workflow.add_node("finalize", self._finalize_content)

        # エントリーポイントと経路設定
        workflow.set_entry_point("generate_initial")
        workflow.add_edge("generate_initial", "evaluate")

        # 条件分岐: 改善が必要かチェック
        workflow.add_conditional_edges(
            "evaluate",
            self._should_continue_optimization,
            {"improve": "improve", "finalize": "finalize"},
        )

        workflow.add_edge("improve", "evaluate")
        workflow.add_edge("finalize", END)

        return workflow.compile()

    def _build_translation_optimization_graph(self) -> StateGraph:
        """翻訳最適化ワークフローを構築"""
        print("🔧 翻訳最適化ワークフローを構築中...")

        workflow = StateGraph(OptimizationState)

        # ノードを追加（文章最適化と同じ構造）
        workflow.add_node("generate_initial", self._generate_initial_content)
        workflow.add_node("evaluate", self._evaluate_content)
        workflow.add_node("improve", self._improve_content)
        workflow.add_node("finalize", self._finalize_content)

        # エントリーポイントと経路設定
        workflow.set_entry_point("generate_initial")
        workflow.add_edge("generate_initial", "evaluate")

        workflow.add_conditional_edges(
            "evaluate",
            self._should_continue_optimization,
            {"improve": "improve", "finalize": "finalize"},
        )

        workflow.add_edge("improve", "evaluate")
        workflow.add_edge("finalize", END)

        return workflow.compile()

    def _build_code_optimization_graph(self) -> StateGraph:
        """コード最適化ワークフローを構築"""
        print("🔧 コード最適化ワークフローを構築中...")

        workflow = StateGraph(OptimizationState)

        # ノードを追加（文章最適化と同じ構造）
        workflow.add_node("generate_initial", self._generate_initial_content)
        workflow.add_node("evaluate", self._evaluate_content)
        workflow.add_node("improve", self._improve_content)
        workflow.add_node("finalize", self._finalize_content)

        # エントリーポイントと経路設定
        workflow.set_entry_point("generate_initial")
        workflow.add_edge("generate_initial", "evaluate")

        workflow.add_conditional_edges(
            "evaluate",
            self._should_continue_optimization,
            {"improve": "improve", "finalize": "finalize"},
        )

        workflow.add_edge("improve", "evaluate")
        workflow.add_edge("finalize", END)

        return workflow.compile()

    def _generate_initial_content(self, state: OptimizationState) -> Dict[str, Any]:
        """初期コンテンツ生成ステップ"""
        print("✨ ステップ1: 初期コンテンツを生成中...")

        task_description = state["task_description"]
        optimization_type = state["optimization_type"]

        # 最適化タイプに応じたプロンプトを選択
        prompt_template = self.initial_generation_prompts[optimization_type]
        prompt = prompt_template.format(task_description=task_description)

        # LLMを呼び出し
        response = self.llm.invoke([HumanMessage(content=prompt)])
        current_content = response.content

        # 実行ログを更新
        log_entry = (
            f"[{datetime.datetime.now().strftime('%H:%M:%S')}] 初期コンテンツ生成完了"
        )
        execution_log = state.get("execution_log", [])
        execution_log.append(log_entry)

        print(f"✅ 初期コンテンツ生成完了: {len(current_content)}文字")

        return {"current_content": current_content, "execution_log": execution_log}

    def _evaluate_content(self, state: OptimizationState) -> Dict[str, Any]:
        """コンテンツ評価ステップ"""
        print("⚖️ ステップ2: コンテンツを評価中...")

        task_description = state["task_description"]
        current_content = state["current_content"]
        optimization_type = state["optimization_type"]

        # 最適化タイプに応じた評価プロンプトを選択
        prompt_template = self.evaluation_prompts[optimization_type]
        prompt = prompt_template.format(
            task_description=task_description, content=current_content
        )

        # LLMを呼び出し
        response = self.llm.invoke([HumanMessage(content=prompt)])
        evaluation_text = response.content

        # 評価結果を解析（簡易実装）
        try:
            import json
            import re

            # JSON部分を抽出
            json_match = re.search(r"\{.*\}", evaluation_text, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                eval_data = json.loads(json_str)

                evaluation = EvaluationResult(
                    score=eval_data["score"],
                    strengths=eval_data["strengths"],
                    weaknesses=eval_data["weaknesses"],
                    suggestions=eval_data["suggestions"],
                    overall_feedback=eval_data["overall_feedback"],
                )
            else:
                # JSONが見つからない場合のフォールバック
                evaluation = EvaluationResult(
                    score=7.0,
                    strengths=["基本的な要求を満たしている"],
                    weaknesses=["具体的な評価を取得できませんでした"],
                    suggestions=["再評価を実行してください"],
                    overall_feedback="評価の解析に失敗しました",
                )

        except Exception as e:
            print(f"⚠️  評価結果の解析エラー: {e}")
            evaluation = EvaluationResult(
                score=6.0,
                strengths=["基本的な機能は実装されている"],
                weaknesses=["評価解析でエラーが発生"],
                suggestions=["手動での確認が必要"],
                overall_feedback=f"解析エラー: {str(e)}",
            )

        # 評価履歴を更新
        evaluation_history = state.get("evaluation_history", [])
        evaluation_history.append(evaluation)

        # 反復回数を更新
        iteration_count = state.get("iteration_count", 0) + 1

        # 実行ログを更新
        log_entry = f"[{datetime.datetime.now().strftime('%H:%M:%S')}] 評価完了 (反復{iteration_count}): スコア {evaluation.score}/10"
        execution_log = state["execution_log"]
        execution_log.append(log_entry)

        print(f"✅ 評価完了 (反復{iteration_count}): スコア {evaluation.score}/10")

        return {
            "evaluation_history": evaluation_history,
            "iteration_count": iteration_count,
            "execution_log": execution_log,
        }

    def _should_continue_optimization(self, state: OptimizationState) -> str:
        """最適化を続行するかどうかの判定（条件分岐関数）"""
        current_evaluation = state["evaluation_history"][-1]
        iteration_count = state["iteration_count"]
        max_iterations = state["max_iterations"]
        target_score = state["target_score"]

        # 目標スコアに達した、または最大反復回数に達した場合は終了
        if (
            current_evaluation.score >= target_score
            or iteration_count >= max_iterations
        ):
            return "finalize"
        else:
            return "improve"

    def _improve_content(self, state: OptimizationState) -> Dict[str, Any]:
        """コンテンツ改善ステップ"""
        print("🔧 ステップ3: コンテンツを改善中...")

        task_description = state["task_description"]
        current_content = state["current_content"]
        optimization_type = state["optimization_type"]
        current_evaluation = state["evaluation_history"][-1]

        # 最適化タイプに応じた改善プロンプトを選択
        prompt_template = self.improvement_prompts[optimization_type]
        prompt = prompt_template.format(
            task_description=task_description,
            current_content=current_content,
            score=current_evaluation.score,
            weaknesses=", ".join(current_evaluation.weaknesses),
            suggestions=", ".join(current_evaluation.suggestions),
            overall_feedback=current_evaluation.overall_feedback,
        )

        # LLMを呼び出し
        response = self.llm.invoke([HumanMessage(content=prompt)])
        improved_content = response.content

        # 実行ログを更新
        log_entry = f"[{datetime.datetime.now().strftime('%H:%M:%S')}] 改善完了 (反復{state['iteration_count']})"
        execution_log = state["execution_log"]
        execution_log.append(log_entry)

        print(
            f"✅ 改善完了 (反復{state['iteration_count']}): {len(improved_content)}文字"
        )

        return {"current_content": improved_content, "execution_log": execution_log}

    def _finalize_content(self, state: OptimizationState) -> Dict[str, Any]:
        """最終化ステップ"""
        print("🎯 ステップ4: 最終化処理中...")

        final_content = state["current_content"]
        final_evaluation = state["evaluation_history"][-1]

        # 実行ログを更新
        log_entry = f"[{datetime.datetime.now().strftime('%H:%M:%S')}] 最適化完了: 最終スコア {final_evaluation.score}/10"
        execution_log = state["execution_log"]
        execution_log.append(log_entry)

        print(f"✅ 最適化完了: 最終スコア {final_evaluation.score}/10")

        return {"final_content": final_content, "execution_log": execution_log}

    def optimize_writing(
        self, task_description: str, target_score: float = 8.5, max_iterations: int = 3
    ) -> Dict[str, Any]:
        """文章最適化の実行"""
        return self._execute_optimization(
            self.writing_graph,
            task_description,
            "writing",
            target_score,
            max_iterations,
        )

    def optimize_translation(
        self, task_description: str, target_score: float = 8.5, max_iterations: int = 3
    ) -> Dict[str, Any]:
        """翻訳最適化の実行"""
        return self._execute_optimization(
            self.translation_graph,
            task_description,
            "translation",
            target_score,
            max_iterations,
        )

    def optimize_code(
        self, task_description: str, target_score: float = 8.5, max_iterations: int = 3
    ) -> Dict[str, Any]:
        """コード最適化の実行"""
        return self._execute_optimization(
            self.code_graph, task_description, "code", target_score, max_iterations
        )

    def _execute_optimization(
        self,
        graph: StateGraph,
        task_description: str,
        optimization_type: str,
        target_score: float,
        max_iterations: int,
    ) -> Dict[str, Any]:
        """最適化実行の共通処理"""
        print(f"⚖️ {optimization_type}最適化開始")
        print(f"タスク: {task_description}")
        print(f"目標スコア: {target_score}/10, 最大反復: {max_iterations}回")
        print("-" * 60)

        # 初期状態を設定
        initial_state = {
            "task_description": task_description,
            "current_content": "",
            "evaluation_history": [],
            "iteration_count": 0,
            "max_iterations": max_iterations,
            "target_score": target_score,
            "final_content": "",
            "optimization_type": optimization_type,
            "execution_log": [],
        }

        # ワークフローを実行
        start_time = datetime.datetime.now()
        result = graph.invoke(initial_state)
        end_time = datetime.datetime.now()

        # 実行時間を計算
        execution_time = (end_time - start_time).total_seconds()

        print("-" * 60)
        print(f"🎉 {optimization_type}最適化完了！ (実行時間: {execution_time:.2f}秒)")

        return {
            "task_description": result["task_description"],
            "optimization_type": optimization_type,
            "initial_content": result["evaluation_history"][0]
            if result["evaluation_history"]
            else "",
            "final_content": result["final_content"],
            "evaluation_history": result["evaluation_history"],
            "iteration_count": result["iteration_count"],
            "target_score": target_score,
            "execution_log": result["execution_log"],
            "execution_time": execution_time,
        }


# ===== デモ用のメイン関数 =====


def main():
    """LangGraph版 Evaluator-Optimizerのデモンストレーション"""
    print("=" * 60)
    print("⚖️ LangGraph版 Evaluator-Optimizer Pattern デモ")
    print("=" * 60)
    print("このデモでは、LangGraphを使用して生成→評価→改善のループを実装します。")
    print("最適化タイプ: 文章、翻訳、コード")
    print()

    try:
        # Evaluator-Optimizerシステムを初期化
        optimizer = LangGraphEvaluatorOptimizer()

        # デモ用のタスク
        demo_tasks = [
            {
                "type": "writing",
                "task": "効果的なリモートワークの方法について、実用的なアドバイスを含む1000文字程度の記事を作成してください。",
                "method": optimizer.optimize_writing,
            },
            {
                "type": "translation",
                "task": "以下の英文を自然な日本語に翻訳してください: 'The rapid advancement of artificial intelligence is transforming industries across the globe, creating new opportunities while also presenting unprecedented challenges that require careful consideration and strategic planning.'",
                "method": optimizer.optimize_translation,
            },
            {
                "type": "code",
                "task": "ユーザー管理システムのためのPythonクラスを作成してください。ユーザーの追加、削除、検索機能を含み、エラーハンドリングも実装してください。",
                "method": optimizer.optimize_code,
            },
        ]

        print("📚 デモ用最適化タスクの実行:")
        print("=" * 60)

        for i, demo_task in enumerate(demo_tasks, 1):
            print(f"\n【{demo_task['type']}最適化デモ {i}】")

            # 最適化を実行
            result = demo_task["method"](demo_task["task"])

            # 結果の表示
            print("\n📊 最適化結果:")
            print(f"反復回数: {result['iteration_count']}")
            print(f"最終スコア: {result['evaluation_history'][-1].score}/10")
            print(f"実行時間: {result['execution_time']:.2f}秒")

            print("\n📝 最終コンテンツ:")
            content_preview = (
                result["final_content"][:300] + "..."
                if len(result["final_content"]) > 300
                else result["final_content"]
            )
            print(content_preview)

            # 詳細表示の確認
            show_details = input("\n詳細を表示しますか？ (y/n): ").lower().strip()
            if show_details == "y":
                print("\n" + "=" * 50)
                print("📋 詳細結果")
                print("=" * 50)

                print("\n📊 評価履歴:")
                for j, evaluation in enumerate(result["evaluation_history"], 1):
                    print(f"\n--- 反復 {j} ---")
                    print(f"スコア: {evaluation.score}/10")
                    print(f"良い点: {', '.join(evaluation.strengths)}")
                    print(f"改善点: {', '.join(evaluation.weaknesses)}")
                    print(f"提案: {', '.join(evaluation.suggestions)}")
                    print(f"フィードバック: {evaluation.overall_feedback}")

                print("\n📝 最終コンテンツ:")
                print("-" * 30)
                print(result["final_content"])

                print("\n📊 実行ログ:")
                print("-" * 30)
                for log_entry in result["execution_log"]:
                    print(f"  {log_entry}")

            print()

        # カスタム最適化モード
        print("\n" + "=" * 60)
        print("💬 カスタム最適化モード (終了するには 'quit' と入力)")
        print("=" * 60)

        while True:
            try:
                print("\n最適化タイプを選択してください:")
                print("1. 文章最適化")
                print("2. 翻訳最適化")
                print("3. コード最適化")

                choice = input("\n選択 (1-3): ").strip()

                if choice.lower() in ["quit", "exit", "終了", "q"]:
                    print("👋 最適化を終了します。")
                    break

                if choice not in ["1", "2", "3"]:
                    print("⚠️  1-3の数字を選択してください。")
                    continue

                task_description = input("\n最適化タスクを入力してください:\n").strip()

                if not task_description:
                    print("⚠️  タスクを入力してください。")
                    continue

                # 目標スコアと最大反復回数を取得
                try:
                    target_score = float(
                        input("目標スコア (0-10, デフォルト: 8.5): ").strip() or "8.5"
                    )
                    max_iterations = int(
                        input("最大反復回数 (デフォルト: 3): ").strip() or "3"
                    )
                except ValueError:
                    print("⚠️  無効な値です。デフォルト値を使用します。")
                    target_score = 8.5
                    max_iterations = 3

                # 選択に応じて最適化を実行
                if choice == "1":
                    result = optimizer.optimize_writing(
                        task_description, target_score, max_iterations
                    )
                elif choice == "2":
                    result = optimizer.optimize_translation(
                        task_description, target_score, max_iterations
                    )
                else:  # choice == '3'
                    result = optimizer.optimize_code(
                        task_description, target_score, max_iterations
                    )

                # 結果の表示
                final_evaluation = result["evaluation_history"][-1]
                print("\n🎉 最適化完了！")
                print(f"反復回数: {result['iteration_count']}")
                print(f"最終スコア: {final_evaluation.score}/10")
                print(f"実行時間: {result['execution_time']:.2f}秒")

                print("\n📝 最終結果:")
                print("-" * 40)
                print(result["final_content"])

            except KeyboardInterrupt:
                print("\n\n👋 最適化を終了します。")
                break

    except Exception as e:
        print(f"❌ エラーが発生しました: {e}")
        print("💡 OpenAI APIキーが正しく設定されているか確認してください。")


if __name__ == "__main__":
    main()
