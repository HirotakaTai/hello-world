#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LangGraph版 Prompt Chaining Pattern
LangGraphを使用して複数のプロンプトを連鎖させる複雑なワークフローを実装
"""

import datetime
from typing import Annotated
from typing import Any
from typing import Dict
from typing import List
from typing import TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage
from langchain_core.messages import HumanMessage
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate

# LangChain関連のインポート
from langchain_openai import ChatOpenAI
from langgraph.graph import END

# LangGraph関連のインポート
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages

# ===== 環境変数の読み込み =====
load_dotenv()

# ===== LangGraphの状態定義 =====


class BlogCreationState(TypedDict):
    """ブログ作成ワークフローの状態定義"""

    messages: Annotated[List[HumanMessage | AIMessage | SystemMessage], add_messages]
    topic: str  # ブログのトピック
    target_audience: str  # ターゲット読者
    research_notes: str  # 調査結果
    outline: str  # アウトライン
    content_evaluation: str  # コンテンツ評価
    draft_content: str  # 下書き内容
    final_content: str  # 最終コンテンツ
    execution_log: List[str]  # 実行ログ


# ===== LangGraphベースのPrompt Chainingクラス =====


class LangGraphPromptChaining:
    """LangGraphを使用したプロンプトチェーンワークフロー"""

    def __init__(self):
        """ワークフローの初期化"""
        print("🔗 LangGraph版 Prompt Chainingワークフローを初期化中...")

        # OpenAI ChatLLMモデルを初期化
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7, verbose=True)

        # プロンプトテンプレートを定義
        self._setup_prompts()

        # LangGraphワークフローを構築
        self.graph = self._build_graph()

        print("✅ ワークフローの初期化が完了しました！")

    def _setup_prompts(self):
        """各ステップのプロンプトテンプレートを設定"""

        # 1. 調査ステップのプロンプト
        self.research_prompt = ChatPromptTemplate.from_template(
            """あなたは経験豊富な調査員です。以下のトピックについて詳細な調査を行ってください。

トピック: {topic}
ターゲット読者: {target_audience}

調査すべき内容:
1. トピックの基本的な背景情報
2. 最新のトレンドや動向
3. ターゲット読者が知りたがる情報
4. 重要な統計データや事実
5. 関連する専門用語の説明

調査結果を整理して、ブログ記事作成に役立つ形でまとめてください。"""
        )

        # 2. アウトライン作成ステップのプロンプト
        self.outline_prompt = ChatPromptTemplate.from_template(
            """あなたは経験豊富なコンテンツプランナーです。以下の調査結果を基に、ブログ記事のアウトラインを作成してください。

トピック: {topic}
ターゲット読者: {target_audience}

調査結果:
{research_notes}

アウトラインの要件:
1. 読者の関心を引く魅力的なタイトル
2. 論理的な構成（導入・本論・結論）
3. 各セクションの主要ポイント
4. 読者が行動を起こすきっかけとなる結論

詳細なアウトラインを作成してください。"""
        )

        # 3. コンテンツ評価ステップのプロンプト
        self.evaluation_prompt = ChatPromptTemplate.from_template(
            """あなたは経験豊富なコンテンツ評価者です。以下のアウトラインを評価してください。

トピック: {topic}
ターゲット読者: {target_audience}

アウトライン:
{outline}

評価基準:
1. ターゲット読者のニーズとの適合性
2. 論理的な構成の妥当性
3. 内容の独創性と価値
4. 読みやすさと理解しやすさ
5. 実用性と行動喚起力

評価結果と改善提案を具体的に示してください。"""
        )

        # 4. 執筆ステップのプロンプト
        self.writing_prompt = ChatPromptTemplate.from_template(
            """あなたは熟練したブログライターです。以下の情報を基に、魅力的なブログ記事を執筆してください。

トピック: {topic}
ターゲット読者: {target_audience}

アウトライン:
{outline}

評価結果と改善提案:
{content_evaluation}

執筆時の注意点:
1. ターゲット読者に分かりやすい言葉を使用
2. 具体例や体験談を盛り込む
3. 読者の関心を維持する文章構成
4. 実用的な情報を提供
5. 最後に明確な行動喚起を含める

完成したブログ記事を執筆してください。"""
        )

        # 5. 校正ステップのプロンプト
        self.proofreading_prompt = ChatPromptTemplate.from_template(
            """あなたは経験豊富な編集者です。以下のブログ記事を校正してください。

原稿:
{draft_content}

校正チェック項目:
1. 文法・スペルミスの確認
2. 文章の流れと読みやすさ
3. 論理的な一貫性
4. 事実の正確性
5. ターゲット読者への適合性
6. SEO要素の最適化

校正済みの最終版を出力してください。"""
        )

    def _build_graph(self) -> StateGraph:
        """LangGraphワークフローを構築"""
        print("🔧 プロンプトチェーンワークフローを構築中...")

        # StateGraphを作成
        workflow = StateGraph(BlogCreationState)

        # ノード（処理ステップ）を追加
        workflow.add_node("research", self._research_step)  # 調査ステップ
        workflow.add_node("outline", self._outline_step)  # アウトライン作成
        workflow.add_node("evaluate", self._evaluation_step)  # コンテンツ評価
        workflow.add_node("write", self._writing_step)  # 執筆ステップ
        workflow.add_node("proofread", self._proofreading_step)  # 校正ステップ

        # エントリーポイントを設定
        workflow.set_entry_point("research")

        # 線形のワークフローを構築
        workflow.add_edge("research", "outline")
        workflow.add_edge("outline", "evaluate")
        workflow.add_edge("evaluate", "write")
        workflow.add_edge("write", "proofread")
        workflow.add_edge("proofread", END)

        return workflow.compile()

    def _research_step(self, state: BlogCreationState) -> Dict[str, Any]:
        """調査ステップの処理"""
        print("🔍 ステップ1: トピックの調査を実行中...")

        # プロンプトを生成
        prompt = self.research_prompt.format(
            topic=state["topic"], target_audience=state["target_audience"]
        )

        # LLMを呼び出し
        response = self.llm.invoke([HumanMessage(content=prompt)])
        research_notes = response.content

        # 実行ログを更新
        log_entry = f"[{datetime.datetime.now().strftime('%H:%M:%S')}] 調査ステップ完了"
        execution_log = state.get("execution_log", [])
        execution_log.append(log_entry)

        print(f"✅ 調査完了: {len(research_notes)}文字の調査結果を生成")

        return {
            "research_notes": research_notes,
            "execution_log": execution_log,
            "messages": [response],
        }

    def _outline_step(self, state: BlogCreationState) -> Dict[str, Any]:
        """アウトライン作成ステップの処理"""
        print("📝 ステップ2: アウトラインを作成中...")

        # プロンプトを生成
        prompt = self.outline_prompt.format(
            topic=state["topic"],
            target_audience=state["target_audience"],
            research_notes=state["research_notes"],
        )

        # LLMを呼び出し
        response = self.llm.invoke([HumanMessage(content=prompt)])
        outline = response.content

        # 実行ログを更新
        log_entry = (
            f"[{datetime.datetime.now().strftime('%H:%M:%S')}] アウトライン作成完了"
        )
        execution_log = state["execution_log"]
        execution_log.append(log_entry)

        print(f"✅ アウトライン作成完了: {len(outline)}文字のアウトラインを生成")

        return {
            "outline": outline,
            "execution_log": execution_log,
            "messages": state["messages"] + [response],
        }

    def _evaluation_step(self, state: BlogCreationState) -> Dict[str, Any]:
        """コンテンツ評価ステップの処理"""
        print("⚖️ ステップ3: コンテンツを評価中...")

        # プロンプトを生成
        prompt = self.evaluation_prompt.format(
            topic=state["topic"],
            target_audience=state["target_audience"],
            outline=state["outline"],
        )

        # LLMを呼び出し
        response = self.llm.invoke([HumanMessage(content=prompt)])
        content_evaluation = response.content

        # 実行ログを更新
        log_entry = (
            f"[{datetime.datetime.now().strftime('%H:%M:%S')}] コンテンツ評価完了"
        )
        execution_log = state["execution_log"]
        execution_log.append(log_entry)

        print(f"✅ 評価完了: {len(content_evaluation)}文字の評価結果を生成")

        return {
            "content_evaluation": content_evaluation,
            "execution_log": execution_log,
            "messages": state["messages"] + [response],
        }

    def _writing_step(self, state: BlogCreationState) -> Dict[str, Any]:
        """執筆ステップの処理"""
        print("✍️ ステップ4: ブログ記事を執筆中...")

        # プロンプトを生成
        prompt = self.writing_prompt.format(
            topic=state["topic"],
            target_audience=state["target_audience"],
            outline=state["outline"],
            content_evaluation=state["content_evaluation"],
        )

        # LLMを呼び出し
        response = self.llm.invoke([HumanMessage(content=prompt)])
        draft_content = response.content

        # 実行ログを更新
        log_entry = f"[{datetime.datetime.now().strftime('%H:%M:%S')}] 執筆ステップ完了"
        execution_log = state["execution_log"]
        execution_log.append(log_entry)

        print(f"✅ 執筆完了: {len(draft_content)}文字の下書きを生成")

        return {
            "draft_content": draft_content,
            "execution_log": execution_log,
            "messages": state["messages"] + [response],
        }

    def _proofreading_step(self, state: BlogCreationState) -> Dict[str, Any]:
        """校正ステップの処理"""
        print("🔍 ステップ5: 校正を実行中...")

        # プロンプトを生成
        prompt = self.proofreading_prompt.format(draft_content=state["draft_content"])

        # LLMを呼び出し
        response = self.llm.invoke([HumanMessage(content=prompt)])
        final_content = response.content

        # 実行ログを更新
        log_entry = f"[{datetime.datetime.now().strftime('%H:%M:%S')}] 校正ステップ完了"
        execution_log = state["execution_log"]
        execution_log.append(log_entry)

        print(f"✅ 校正完了: {len(final_content)}文字の最終版を生成")

        return {
            "final_content": final_content,
            "execution_log": execution_log,
            "messages": state["messages"] + [response],
        }

    def create_blog_post(self, topic: str, target_audience: str) -> Dict[str, Any]:
        """ブログ記事作成のメイン処理"""
        print("\n📝 ブログ記事作成開始")
        print(f"トピック: {topic}")
        print(f"ターゲット読者: {target_audience}")
        print("-" * 60)

        # 初期状態を設定
        initial_state = {
            "messages": [],
            "topic": topic,
            "target_audience": target_audience,
            "research_notes": "",
            "outline": "",
            "content_evaluation": "",
            "draft_content": "",
            "final_content": "",
            "execution_log": [],
        }

        # ワークフローを実行
        start_time = datetime.datetime.now()
        result = self.graph.invoke(initial_state)
        end_time = datetime.datetime.now()

        # 実行時間を計算
        execution_time = (end_time - start_time).total_seconds()

        print("-" * 60)
        print(f"🎉 ブログ記事作成完了！ (実行時間: {execution_time:.2f}秒)")

        return {
            "topic": result["topic"],
            "target_audience": result["target_audience"],
            "research_notes": result["research_notes"],
            "outline": result["outline"],
            "content_evaluation": result["content_evaluation"],
            "draft_content": result["draft_content"],
            "final_content": result["final_content"],
            "execution_log": result["execution_log"],
            "execution_time": execution_time,
        }


# ===== デモ用のメイン関数 =====


def main():
    """LangGraph版 Prompt Chainingのデモンストレーション"""
    print("=" * 60)
    print("🔗 LangGraph版 Prompt Chaining Pattern デモ")
    print("=" * 60)
    print(
        "このデモでは、LangGraphを使用して複雑なブログ記事作成ワークフローを実装します。"
    )
    print("ワークフロー: 調査 → アウトライン → 評価 → 執筆 → 校正")
    print()

    try:
        # Prompt Chainingワークフローを初期化
        workflow = LangGraphPromptChaining()

        # デモ用のブログ記事作成
        demo_topics = [
            {
                "topic": "AI技術の最新動向と今後の展望",
                "target_audience": "IT業界で働く会社員",
            },
            {
                "topic": "在宅ワークの効率化テクニック",
                "target_audience": "リモートワークをする会社員",
            },
        ]

        for i, demo_params in enumerate(demo_topics, 1):
            print(f"\n【デモ {i}】")
            print("=" * 40)

            # ブログ記事を作成
            result = workflow.create_blog_post(
                topic=demo_params["topic"],
                target_audience=demo_params["target_audience"],
            )

            # 結果の表示
            print("\n📊 実行結果:")
            print("-" * 40)
            print(f"実行時間: {result['execution_time']:.2f}秒")
            print(f"実行ログ: {len(result['execution_log'])}ステップ")

            print("\n📝 生成されたコンテンツ:")
            print("-" * 40)
            print(f"最終記事: {len(result['final_content'])}文字")
            print(f"記事の一部: {result['final_content'][:200]}...")

            # 詳細表示の確認
            show_details = input("\n詳細を表示しますか？ (y/n): ").lower().strip()
            if show_details == "y":
                print("\n" + "=" * 60)
                print("📋 詳細結果")
                print("=" * 60)

                print("\n🔍 調査結果:")
                print("-" * 30)
                print(result["research_notes"])

                print("\n📝 アウトライン:")
                print("-" * 30)
                print(result["outline"])

                print("\n⚖️ 評価結果:")
                print("-" * 30)
                print(result["content_evaluation"])

                print("\n✍️ 最終記事:")
                print("-" * 30)
                print(result["final_content"])

                print("\n📊 実行ログ:")
                print("-" * 30)
                for log_entry in result["execution_log"]:
                    print(f"  {log_entry}")

            print()

        # カスタム記事作成モード
        print("\n" + "=" * 60)
        print("💬 カスタム記事作成モード (終了するには 'quit' と入力)")
        print("=" * 60)

        while True:
            try:
                topic = input("\n📝 記事のトピックを入力してください: ").strip()

                if topic.lower() in ["quit", "exit", "終了", "q"]:
                    print("👋 記事作成を終了します。")
                    break

                if not topic:
                    print("⚠️  トピックを入力してください。")
                    continue

                target_audience = input("🎯 ターゲット読者を入力してください: ").strip()

                if not target_audience:
                    print("⚠️  ターゲット読者を入力してください。")
                    continue

                # カスタム記事を作成
                result = workflow.create_blog_post(topic, target_audience)

                # 結果の表示
                print(
                    f"\n🎉 記事作成完了！ (実行時間: {result['execution_time']:.2f}秒)"
                )
                print(f"📝 記事の一部: {result['final_content'][:300]}...")

                # 詳細表示の確認
                show_details = input("\n詳細を表示しますか？ (y/n): ").lower().strip()
                if show_details == "y":
                    print("\n" + "=" * 60)
                    print("✍️ 完成した記事")
                    print("=" * 60)
                    print(result["final_content"])

            except KeyboardInterrupt:
                print("\n\n👋 記事作成を終了します。")
                break

    except Exception as e:
        print(f"❌ エラーが発生しました: {e}")
        print("💡 OpenAI APIキーが正しく設定されているか確認してください。")


if __name__ == "__main__":
    main()
