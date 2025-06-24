#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LangGraph版 Parallelization Pattern
LangGraphを使用してタスクを並列実行し、効率的な処理を実現するパターン
"""

import asyncio
import datetime
from typing import Any
from typing import Dict
from typing import List
from typing import TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate

# LangChain関連のインポート
from langchain_openai import ChatOpenAI
from langgraph.graph import END

# LangGraph関連のインポート
from langgraph.graph import StateGraph

# ===== 環境変数の読み込み =====
load_dotenv()

# ===== 状態定義 =====


class ParallelizationState(TypedDict):
    """並列処理ワークフローの状態定義"""

    input_text: str  # 入力テキスト
    text_sections: List[str]  # テキストセクション（分割後）
    parallel_results: Dict[str, Any]  # 並列処理結果
    final_summary: str  # 最終サマリー
    execution_log: List[str]  # 実行ログ
    processing_type: str  # 処理タイプ（sectioning, voting, review）


# ===== LangGraphベースのParallelizationクラス =====


class LangGraphParallelization:
    """LangGraphを使用した並列処理システム"""

    def __init__(self):
        """並列処理システムの初期化"""
        print("⚡ LangGraph版 Parallelizationシステムを初期化中...")

        # OpenAI ChatLLMモデルを初期化
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3, verbose=True)

        # プロンプトテンプレートを設定
        self._setup_prompts()

        # 各処理タイプ用のLangGraphワークフローを構築
        self.sectioning_graph = self._build_sectioning_graph()
        self.voting_graph = self._build_voting_graph()
        self.review_graph = self._build_review_graph()

        print("✅ 並列処理システムの初期化が完了しました！")

    def _setup_prompts(self):
        """各処理用のプロンプトテンプレートを設定"""

        # テキスト分割プロンプト
        self.section_split_prompt = ChatPromptTemplate.from_template(
            """以下のテキストを論理的な3つのセクションに分割してください。各セクションは独立して理解できるようにしてください。

入力テキスト:
{input_text}

分割結果を以下の形式で出力してください:
=== セクション1 ===
[セクション1の内容]

=== セクション2 ===
[セクション2の内容]

=== セクション3 ===
[セクション3の内容]
"""
        )

        # セクション要約プロンプト
        self.section_summary_prompt = ChatPromptTemplate.from_template(
            """以下のテキストセクションを簡潔に要約してください。重要なポイントを3-5つの箇条書きで示してください。

セクション内容:
{section_text}

要約結果を箇条書き形式で出力してください。
"""
        )

        # 投票用評価プロンプト
        self.voting_evaluation_prompt = ChatPromptTemplate.from_template(
            """あなたは{evaluator_type}の専門家です。以下の提案を評価してください。

提案内容:
{proposal_text}

評価基準:
{evaluation_criteria}

以下の形式で評価してください:
スコア: [1-10]
評価理由: [評価の詳細な理由]
改善提案: [具体的な改善案があれば]
"""
        )

        # コードレビュープロンプト
        self.code_review_prompt = ChatPromptTemplate.from_template(
            """あなたは{reviewer_type}の専門家です。以下のコードをレビューしてください。

コード:
{code_text}

レビュー観点:
{review_criteria}

以下の形式でレビューしてください:
評価: [Good/Needs Improvement/Critical Issues]
指摘事項: [具体的な問題点]
推奨事項: [改善提案]
"""
        )

        # 統合サマリープロンプト
        self.integration_prompt = ChatPromptTemplate.from_template(
            """以下の並列処理結果を統合して、包括的なサマリーを作成してください。

処理タイプ: {processing_type}
並列処理結果:
{parallel_results}

統合結果として、以下を含むサマリーを作成してください:
1. 全体的な概要
2. 重要な発見や洞察
3. 共通するテーマやパターン
4. 結論と推奨事項
"""
        )

    def _build_sectioning_graph(self) -> StateGraph:
        """セクショニング（文書分割処理）のワークフローを構築"""
        print("🔧 セクショニングワークフローを構築中...")

        workflow = StateGraph(ParallelizationState)

        # ノードを追加
        workflow.add_node("split_text", self._split_text)
        workflow.add_node("summarize_sections", self._summarize_sections_parallel)
        workflow.add_node("integrate_summaries", self._integrate_summaries)

        # エントリーポイントと経路設定
        workflow.set_entry_point("split_text")
        workflow.add_edge("split_text", "summarize_sections")
        workflow.add_edge("summarize_sections", "integrate_summaries")
        workflow.add_edge("integrate_summaries", END)

        return workflow.compile()

    def _build_voting_graph(self) -> StateGraph:
        """投票（複数評価者による評価）のワークフローを構築"""
        print("🔧 投票ワークフローを構築中...")

        workflow = StateGraph(ParallelizationState)

        # ノードを追加
        workflow.add_node("evaluate_parallel", self._evaluate_parallel)
        workflow.add_node("aggregate_votes", self._aggregate_votes)

        # エントリーポイントと経路設定
        workflow.set_entry_point("evaluate_parallel")
        workflow.add_edge("evaluate_parallel", "aggregate_votes")
        workflow.add_edge("aggregate_votes", END)

        return workflow.compile()

    def _build_review_graph(self) -> StateGraph:
        """レビュー（複数レビュアーによるコードレビュー）のワークフローを構築"""
        print("🔧 レビューワークフローを構築中...")

        workflow = StateGraph(ParallelizationState)

        # ノードを追加
        workflow.add_node("review_parallel", self._review_parallel)
        workflow.add_node("consolidate_reviews", self._consolidate_reviews)

        # エントリーポイントと経路設定
        workflow.set_entry_point("review_parallel")
        workflow.add_edge("review_parallel", "consolidate_reviews")
        workflow.add_edge("consolidate_reviews", END)

        return workflow.compile()

    # ===== セクショニング関連の処理メソッド =====

    def _split_text(self, state: ParallelizationState) -> Dict[str, Any]:
        """テキストを複数のセクションに分割"""
        print("✂️  テキスト分割処理中...")

        input_text = state["input_text"]

        # テキスト分割プロンプトを生成
        prompt = self.section_split_prompt.format(input_text=input_text)

        # LLMを呼び出し
        response = self.llm.invoke([HumanMessage(content=prompt)])
        split_result = response.content

        # セクションを抽出（簡易パーサー）
        sections = []
        current_section = ""
        for line in split_result.split("\n"):
            if line.startswith("=== セクション"):
                if current_section.strip():
                    sections.append(current_section.strip())
                current_section = ""
            else:
                current_section += line + "\n"

        if current_section.strip():
            sections.append(current_section.strip())

        # 実行ログを更新
        log_entry = f"[{datetime.datetime.now().strftime('%H:%M:%S')}] テキスト分割完了: {len(sections)}セクション"
        execution_log = state.get("execution_log", [])
        execution_log.append(log_entry)

        print(f"✅ テキスト分割完了: {len(sections)}個のセクション")

        return {"text_sections": sections, "execution_log": execution_log}

    def _summarize_sections_parallel(
        self, state: ParallelizationState
    ) -> Dict[str, Any]:
        """複数のセクションを並列で要約"""
        print("📝 セクション並列要約処理中...")

        sections = state["text_sections"]

        # 各セクションを並列で要約
        async def summarize_section_async(section_idx: int, section_text: str) -> tuple:
            """単一セクションの非同期要約"""
            prompt = self.section_summary_prompt.format(section_text=section_text)
            response = self.llm.invoke([HumanMessage(content=prompt)])
            return section_idx, response.content

        async def process_all_sections():
            """全セクションの並列処理"""
            tasks = [
                summarize_section_async(i, section)
                for i, section in enumerate(sections)
            ]
            return await asyncio.gather(*tasks)

        # 並列処理を実行
        try:
            # 新しいイベントループを作成して実行
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            section_summaries = loop.run_until_complete(process_all_sections())
            loop.close()
        except Exception as e:
            print(f"⚠️  並列処理でエラーが発生、順次処理に切り替え: {e}")
            # フォールバック: 順次処理
            section_summaries = []
            for i, section in enumerate(sections):
                prompt = self.section_summary_prompt.format(section_text=section)
                response = self.llm.invoke([HumanMessage(content=prompt)])
                section_summaries.append((i, response.content))

        # 結果を整理
        parallel_results = {
            f"section_{idx}": summary for idx, summary in section_summaries
        }

        # 実行ログを更新
        log_entry = f"[{datetime.datetime.now().strftime('%H:%M:%S')}] セクション並列要約完了: {len(parallel_results)}件"
        execution_log = state["execution_log"]
        execution_log.append(log_entry)

        print(f"✅ セクション並列要約完了: {len(parallel_results)}件")

        return {"parallel_results": parallel_results, "execution_log": execution_log}

    def _integrate_summaries(self, state: ParallelizationState) -> Dict[str, Any]:
        """要約結果を統合"""
        print("🔗 要約統合処理中...")

        parallel_results = state["parallel_results"]

        # 統合プロンプトを生成
        results_text = "\n".join(
            [
                f"セクション{i}: {summary}"
                for i, summary in enumerate(parallel_results.values())
            ]
        )

        prompt = self.integration_prompt.format(
            processing_type="セクション要約", parallel_results=results_text
        )

        # LLMを呼び出し
        response = self.llm.invoke([HumanMessage(content=prompt)])
        final_summary = response.content

        # 実行ログを更新
        log_entry = f"[{datetime.datetime.now().strftime('%H:%M:%S')}] 要約統合完了"
        execution_log = state["execution_log"]
        execution_log.append(log_entry)

        print("✅ 要約統合完了")

        return {"final_summary": final_summary, "execution_log": execution_log}

    # ===== 投票関連の処理メソッド =====

    def _evaluate_parallel(self, state: ParallelizationState) -> Dict[str, Any]:
        """複数の評価者による並列評価"""
        print("🗳️  並列評価処理中...")

        input_text = state["input_text"]

        # 評価者タイプと評価基準を定義
        evaluators = [
            {"type": "技術専門家", "criteria": "技術的実現性、実装の複雑さ、保守性"},
            {"type": "ビジネス専門家", "criteria": "ビジネス価値、ROI、市場適合性"},
            {
                "type": "ユーザー体験専門家",
                "criteria": "使いやすさ、アクセシビリティ、ユーザーインパクト",
            },
        ]

        # 各評価者による並列評価
        async def evaluate_async(evaluator: dict) -> tuple:
            """単一評価者による非同期評価"""
            prompt = self.voting_evaluation_prompt.format(
                evaluator_type=evaluator["type"],
                proposal_text=input_text,
                evaluation_criteria=evaluator["criteria"],
            )
            response = self.llm.invoke([HumanMessage(content=prompt)])
            return evaluator["type"], response.content

        async def process_all_evaluations():
            """全評価者による並列処理"""
            tasks = [evaluate_async(evaluator) for evaluator in evaluators]
            return await asyncio.gather(*tasks)

        # 並列処理を実行
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            evaluations = loop.run_until_complete(process_all_evaluations())
            loop.close()
        except Exception:
            # フォールバック: 順次処理
            evaluations = []
            for evaluator in evaluators:
                prompt = self.voting_evaluation_prompt.format(
                    evaluator_type=evaluator["type"],
                    proposal_text=input_text,
                    evaluation_criteria=evaluator["criteria"],
                )
                response = self.llm.invoke([HumanMessage(content=prompt)])
                evaluations.append((evaluator["type"], response.content))

        # 結果を整理
        parallel_results = {
            evaluator_type: evaluation for evaluator_type, evaluation in evaluations
        }

        # 実行ログを更新
        log_entry = f"[{datetime.datetime.now().strftime('%H:%M:%S')}] 並列評価完了: {len(parallel_results)}名の評価者"
        execution_log = state.get("execution_log", [])
        execution_log.append(log_entry)

        print(f"✅ 並列評価完了: {len(parallel_results)}名の評価者")

        return {"parallel_results": parallel_results, "execution_log": execution_log}

    def _aggregate_votes(self, state: ParallelizationState) -> Dict[str, Any]:
        """投票結果を集約"""
        print("📊 投票集約処理中...")

        parallel_results = state["parallel_results"]

        # 集約プロンプトを生成
        results_text = "\n".join(
            [
                f"{evaluator}: {evaluation}"
                for evaluator, evaluation in parallel_results.items()
            ]
        )

        prompt = self.integration_prompt.format(
            processing_type="複数評価者による投票", parallel_results=results_text
        )

        # LLMを呼び出し
        response = self.llm.invoke([HumanMessage(content=prompt)])
        final_summary = response.content

        # 実行ログを更新
        log_entry = f"[{datetime.datetime.now().strftime('%H:%M:%S')}] 投票集約完了"
        execution_log = state["execution_log"]
        execution_log.append(log_entry)

        print("✅ 投票集約完了")

        return {"final_summary": final_summary, "execution_log": execution_log}

    # ===== レビュー関連の処理メソッド =====

    def _review_parallel(self, state: ParallelizationState) -> Dict[str, Any]:
        """複数のレビュアーによる並列コードレビュー"""
        print("👥 並列コードレビュー処理中...")

        input_text = state["input_text"]

        # レビュアータイプとレビュー基準を定義
        reviewers = [
            {
                "type": "セキュリティエンジニア",
                "criteria": "セキュリティ脆弱性、認証・認可、データ保護",
            },
            {
                "type": "パフォーマンスエンジニア",
                "criteria": "実行効率、メモリ使用量、スケーラビリティ",
            },
            {
                "type": "コード品質エンジニア",
                "criteria": "可読性、保守性、テスト容易性、設計パターン",
            },
        ]

        # 各レビュアーによる並列レビュー
        async def review_async(reviewer: dict) -> tuple:
            """単一レビュアーによる非同期レビュー"""
            prompt = self.code_review_prompt.format(
                reviewer_type=reviewer["type"],
                code_text=input_text,
                review_criteria=reviewer["criteria"],
            )
            response = self.llm.invoke([HumanMessage(content=prompt)])
            return reviewer["type"], response.content

        async def process_all_reviews():
            """全レビュアーによる並列処理"""
            tasks = [review_async(reviewer) for reviewer in reviewers]
            return await asyncio.gather(*tasks)

        # 並列処理を実行
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            reviews = loop.run_until_complete(process_all_reviews())
            loop.close()
        except Exception:
            # フォールバック: 順次処理
            reviews = []
            for reviewer in reviewers:
                prompt = self.code_review_prompt.format(
                    reviewer_type=reviewer["type"],
                    code_text=input_text,
                    review_criteria=reviewer["criteria"],
                )
                response = self.llm.invoke([HumanMessage(content=prompt)])
                reviews.append((reviewer["type"], response.content))

        # 結果を整理
        parallel_results = {reviewer_type: review for reviewer_type, review in reviews}

        # 実行ログを更新
        log_entry = f"[{datetime.datetime.now().strftime('%H:%M:%S')}] 並列レビュー完了: {len(parallel_results)}名のレビュアー"
        execution_log = state.get("execution_log", [])
        execution_log.append(log_entry)

        print(f"✅ 並列レビュー完了: {len(parallel_results)}名のレビュアー")

        return {"parallel_results": parallel_results, "execution_log": execution_log}

    def _consolidate_reviews(self, state: ParallelizationState) -> Dict[str, Any]:
        """レビュー結果を統合"""
        print("📋 レビュー統合処理中...")

        parallel_results = state["parallel_results"]

        # 統合プロンプトを生成
        results_text = "\n".join(
            [f"{reviewer}: {review}" for reviewer, review in parallel_results.items()]
        )

        prompt = self.integration_prompt.format(
            processing_type="複数レビュアーによるコードレビュー",
            parallel_results=results_text,
        )

        # LLMを呼び出し
        response = self.llm.invoke([HumanMessage(content=prompt)])
        final_summary = response.content

        # 実行ログを更新
        log_entry = f"[{datetime.datetime.now().strftime('%H:%M:%S')}] レビュー統合完了"
        execution_log = state["execution_log"]
        execution_log.append(log_entry)

        print("✅ レビュー統合完了")

        return {"final_summary": final_summary, "execution_log": execution_log}

    # ===== メイン処理メソッド =====

    def process_sectioning(self, input_text: str) -> Dict[str, Any]:
        """セクショニング（文書分割）処理"""
        return self._execute_workflow(
            self.sectioning_graph, input_text, "セクショニング"
        )

    def process_voting(self, input_text: str) -> Dict[str, Any]:
        """投票（複数評価者による評価）処理"""
        return self._execute_workflow(self.voting_graph, input_text, "投票")

    def process_review(self, input_text: str) -> Dict[str, Any]:
        """レビュー（複数レビュアーによるコードレビュー）処理"""
        return self._execute_workflow(self.review_graph, input_text, "レビュー")

    def _execute_workflow(
        self, graph: StateGraph, input_text: str, processing_type: str
    ) -> Dict[str, Any]:
        """ワークフロー実行の共通処理"""
        print(f"⚡ {processing_type}処理開始: {len(input_text)}文字")
        print("-" * 60)

        # 初期状態を設定
        initial_state = {
            "input_text": input_text,
            "text_sections": [],
            "parallel_results": {},
            "final_summary": "",
            "execution_log": [],
            "processing_type": processing_type,
        }

        # ワークフローを実行
        start_time = datetime.datetime.now()
        result = graph.invoke(initial_state)
        end_time = datetime.datetime.now()

        # 実行時間を計算
        execution_time = (end_time - start_time).total_seconds()

        print("-" * 60)
        print(f"🎉 {processing_type}処理完了！ (実行時間: {execution_time:.2f}秒)")

        return {
            "input_text": result["input_text"],
            "processing_type": processing_type,
            "parallel_results": result["parallel_results"],
            "final_summary": result["final_summary"],
            "execution_log": result["execution_log"],
            "execution_time": execution_time,
        }


# ===== デモ用のメイン関数 =====


def main():
    """LangGraph版 Parallelizationのデモンストレーション"""
    print("=" * 60)
    print("⚡ LangGraph版 Parallelization Pattern デモ")
    print("=" * 60)
    print("このデモでは、LangGraphを使用してタスクを並列実行します。")
    print("処理タイプ: セクショニング、投票、コードレビュー")
    print()

    try:
        # Parallelizationシステムを初期化
        processor = LangGraphParallelization()

        # デモ用の入力データ
        demo_data = {
            "sectioning": """
人工知能（AI）は現代社会において急速に発展している技術分野です。機械学習、深層学習、自然言語処理などの技術が統合され、様々な産業分野で革新をもたらしています。

AIの応用分野は広範囲にわたります。医療分野では画像診断や薬物発見、自動車産業では自動運転技術、金融分野では不正検知やアルゴリズム取引などが代表例です。これらの技術は効率性を向上させ、人間の能力を拡張する役割を果たしています。

しかし、AI技術の発展には課題も存在します。プライバシーの保護、雇用への影響、アルゴリズムの偏見、説明可能性などの問題が議論されています。これらの課題に対処するため、AI倫理やガバナンスの確立が重要視されています。将来のAI社会では、技術の利益を最大化しながら、リスクを最小化するバランスの取れたアプローチが求められます。
            """.strip(),
            "voting": """
新しいモバイルアプリケーション「SmartLife」の開発提案です。このアプリは、ユーザーの日常生活を効率化するためのオールインワンソリューションです。主な機能として、スケジュール管理、支出追跡、健康モニタリング、タスク管理を統合し、AIを活用してパーソナライズされた提案を行います。開発期間は6ヶ月、予算は500万円を予定しています。
            """.strip(),
            "review": """
def user_authentication(username, password):
    # ユーザー認証を行う関数
    users_db = {"admin": "password123", "user": "12345"}
    
    if username in users_db:
        if users_db[username] == password:
            print(f"ログイン成功: {username}")
            return True
        else:
            print("パスワードが間違っています")
            return False
    else:
        print("ユーザーが見つかりません")
        return False

def process_user_data(user_id):
    # ユーザーデータを処理する関数
    query = f"SELECT * FROM users WHERE id = {user_id}"
    # データベースクエリを実行（実装は省略）
    return query
            """.strip(),
        }

        # 各処理タイプのデモ実行
        demos = [
            ("セクショニング", "sectioning", processor.process_sectioning),
            ("投票", "voting", processor.process_voting),
            ("コードレビュー", "review", processor.process_review),
        ]

        for demo_name, data_key, process_func in demos:
            print(f"\n【{demo_name}デモ】")
            print("=" * 40)

            # 処理を実行
            result = process_func(demo_data[data_key])

            # 結果の表示
            print("\n📊 処理結果:")
            print(f"処理タイプ: {result['processing_type']}")
            print(f"並列処理数: {len(result['parallel_results'])}")
            print(f"実行時間: {result['execution_time']:.2f}秒")

            print("\n📝 最終サマリー:")
            print(
                result["final_summary"][:300] + "..."
                if len(result["final_summary"]) > 300
                else result["final_summary"]
            )

            # 詳細表示の確認
            show_details = input("\n詳細を表示しますか？ (y/n): ").lower().strip()
            if show_details == "y":
                print("\n" + "=" * 50)
                print("📋 詳細結果")
                print("=" * 50)

                print("\n⚡ 並列処理結果:")
                for key, value in result["parallel_results"].items():
                    print(f"\n--- {key} ---")
                    print(value)

                print("\n📊 実行ログ:")
                for log_entry in result["execution_log"]:
                    print(f"  {log_entry}")

            print()

        # カスタム処理モード
        print("\n" + "=" * 60)
        print("💬 カスタム並列処理モード (終了するには 'quit' と入力)")
        print("=" * 60)

        while True:
            try:
                print("\n処理タイプを選択してください:")
                print("1. セクショニング（文書分割）")
                print("2. 投票（複数評価者による評価）")
                print("3. コードレビュー（複数レビュアー）")

                choice = input("\n選択 (1-3): ").strip()

                if choice.lower() in ["quit", "exit", "終了", "q"]:
                    print("👋 処理を終了します。")
                    break

                if choice not in ["1", "2", "3"]:
                    print("⚠️  1-3の数字を選択してください。")
                    continue

                input_text = input("\n処理したいテキストを入力してください:\n").strip()

                if not input_text:
                    print("⚠️  テキストを入力してください。")
                    continue

                # 選択に応じて処理を実行
                if choice == "1":
                    result = processor.process_sectioning(input_text)
                elif choice == "2":
                    result = processor.process_voting(input_text)
                else:  # choice == '3'
                    result = processor.process_review(input_text)

                # 結果の表示
                print(f"\n🎉 処理完了！ (実行時間: {result['execution_time']:.2f}秒)")
                print("📝 結果サマリー:")
                print("-" * 40)
                print(result["final_summary"])

            except KeyboardInterrupt:
                print("\n\n👋 処理を終了します。")
                break

    except Exception as e:
        print(f"❌ エラーが発生しました: {e}")
        print("💡 OpenAI APIキーが正しく設定されているか確認してください。")


if __name__ == "__main__":
    main()
