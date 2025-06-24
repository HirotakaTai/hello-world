#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LangGraph版 Routing Pattern
LangGraphを使用してクエリを分類し、適切な処理経路に振り分けるパターン
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
from langchain_openai import ChatOpenAI
from langgraph.graph import END
from langgraph.graph import StateGraph
from pydantic import BaseModel
from pydantic import Field

# ===== 環境変数の読み込み =====
load_dotenv()

# ===== データクラスの定義 =====


class RouteClassification(BaseModel):
    """ルーティング分類結果のデータクラス"""

    category: Literal["technical", "billing", "general", "complaint", "unknown"] = (
        Field(description="クエリのカテゴリ分類")
    )
    confidence: float = Field(description="分類の信頼度 (0.0-1.0)", ge=0.0, le=1.0)
    reasoning: str = Field(description="分類の理由")


class RoutingState(TypedDict):
    """ルーティングワークフローの状態定義"""

    user_query: str  # ユーザーの質問
    classification: RouteClassification  # 分類結果
    response: str  # 最終応答
    processing_path: str  # 処理経路
    execution_log: List[str]  # 実行ログ


# ===== LangGraphベースのRoutingクラス =====


class LangGraphRouting:
    """LangGraphを使用したルーティングシステム"""

    def __init__(self):
        """ルーティングシステムの初期化"""
        print("🎯 LangGraph版 Routingシステムを初期化中...")

        # OpenAI ChatLLMモデルを初期化
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.1,  # 分類の一貫性を保つため低めの値
            verbose=True,
        )

        # プロンプトテンプレートを設定
        self._setup_prompts()

        # LangGraphワークフローを構築
        self.graph = self._build_graph()

        print("✅ ルーティングシステムの初期化が完了しました！")

    def _setup_prompts(self):
        """各処理経路のプロンプトテンプレートを設定"""

        # 分類プロンプト
        self.classification_prompt = ChatPromptTemplate.from_template(
            """あなたは顧客サポートクエリの分類専門家です。以下のユーザークエリを適切なカテゴリに分類してください。

ユーザークエリ: {user_query}

分類カテゴリ:
1. technical - 技術的な問題や質問（ログイン問題、機能の使い方、バグ報告など）
2. billing - 料金や請求に関する問題（支払い、プラン変更、返金など）
3. general - 一般的な情報提供（サービス概要、FAQ、営業時間など）
4. complaint - 苦情や不満（サービス品質、対応への不満など）
5. unknown - 上記に該当しない、または分類困難なもの

分類結果を以下のJSON形式で返してください:
{{
    "category": "分類されたカテゴリ",
    "confidence": 0.95,
    "reasoning": "分類の理由"
}}
"""
        )

        # 技術サポート処理プロンプト
        self.technical_prompt = ChatPromptTemplate.from_template(
            """あなたは経験豊富な技術サポートスペシャリストです。以下の技術的な問題に対して、詳細で実用的な解決策を提供してください。

技術的問題: {user_query}

対応時の注意点:
1. 段階的で分かりやすい解決手順を示す
2. 可能性のある原因を整理して説明
3. 予防策や追加のヒントを提供
4. 必要に応じて関連するドキュメントやリソースを案内

技術サポートとして専門的で親切な回答をしてください。
"""
        )

        # 請求サポート処理プロンプト
        self.billing_prompt = ChatPromptTemplate.from_template(
            """あなたは請求・料金サポートの専門家です。以下の請求関連の問題に対して、正確で親切な案内を提供してください。

請求関連の問題: {user_query}

対応時の注意点:
1. 料金体系や支払い方法を明確に説明
2. 具体的な手続きの手順を示す
3. 期限や注意事項を適切に伝える
4. 必要に応じて担当部門への案内を行う

請求サポートとして信頼できる回答をしてください。
"""
        )

        # 一般サポート処理プロンプト
        self.general_prompt = ChatPromptTemplate.from_template(
            """あなたは親切な一般サポート担当者です。以下の一般的な質問に対して、有益で分かりやすい情報を提供してください。

一般的な質問: {user_query}

対応時の注意点:
1. サービスの特徴や利点を分かりやすく説明
2. よくある質問への適切な回答を提供
3. 追加で知っておくべき情報を案内
4. 次のステップや行動を明確に示す

一般サポートとして友好的で有用な回答をしてください。
"""
        )

        # 苦情対応処理プロンプト
        self.complaint_prompt = ChatPromptTemplate.from_template(
            """あなたは経験豊富な苦情対応スペシャリストです。以下の苦情に対して、共感的で建設的な対応を提供してください。

苦情内容: {user_query}

対応時の注意点:
1. お客様の気持ちに共感し、理解を示す
2. 問題を明確に把握し、整理して確認
3. 可能な解決策や改善策を提案
4. 今後の改善への取り組みを説明

苦情対応として誠実で建設的な回答をしてください。
"""
        )

        # 不明カテゴリ処理プロンプト
        self.unknown_prompt = ChatPromptTemplate.from_template(
            """あなたは親切なカスタマーサポート担当者です。以下の質問は明確に分類できませんが、最善を尽くして回答してください。

質問: {user_query}

対応時の注意点:
1. 可能な限り質問の意図を理解しようと努める
2. 関連する情報や代替案を提供
3. より具体的な質問への誘導を行う
4. 適切な担当部門への案内を検討

一般的なサポートとして親切で有用な回答をしてください。
"""
        )

    def _build_graph(self) -> StateGraph:
        """LangGraphルーティングワークフローを構築"""
        print("🔧 ルーティングワークフローを構築中...")

        # StateGraphを作成
        workflow = StateGraph(RoutingState)

        # ノード（処理ステップ）を追加
        workflow.add_node("classify", self._classify_query)  # クエリ分類
        workflow.add_node("technical", self._handle_technical)  # 技術サポート
        workflow.add_node("billing", self._handle_billing)  # 請求サポート
        workflow.add_node("general", self._handle_general)  # 一般サポート
        workflow.add_node("complaint", self._handle_complaint)  # 苦情対応
        workflow.add_node("unknown", self._handle_unknown)  # 不明カテゴリ

        # エントリーポイントを設定
        workflow.set_entry_point("classify")

        # 条件分岐を設定（分類結果に基づく経路選択）
        workflow.add_conditional_edges(
            "classify",  # 分岐元のノード
            self._route_decision,  # 条件判定関数
            {
                "technical": "technical",  # 技術サポート経路
                "billing": "billing",  # 請求サポート経路
                "general": "general",  # 一般サポート経路
                "complaint": "complaint",  # 苦情対応経路
                "unknown": "unknown",  # 不明カテゴリ経路
            },
        )

        # 各専門処理から終了への経路
        workflow.add_edge("technical", END)
        workflow.add_edge("billing", END)
        workflow.add_edge("general", END)
        workflow.add_edge("complaint", END)
        workflow.add_edge("unknown", END)

        return workflow.compile()

    def _classify_query(self, state: RoutingState) -> Dict[str, Any]:
        """クエリ分類ステップの処理"""
        print("🔍 ステップ1: クエリを分類中...")

        user_query = state["user_query"]

        # 分類プロンプトを生成
        prompt = self.classification_prompt.format(user_query=user_query)

        # LLMを呼び出して分類実行
        response = self.llm.invoke([HumanMessage(content=prompt)])
        classification_text = response.content

        # JSON形式の応答をパース（簡易的な実装）
        try:
            import json
            import re

            # JSON部分を抽出
            json_match = re.search(r"\{.*\}", classification_text, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                classification_data = json.loads(json_str)

                classification = RouteClassification(
                    category=classification_data["category"],
                    confidence=classification_data["confidence"],
                    reasoning=classification_data["reasoning"],
                )
            else:
                # JSONが見つからない場合のフォールバック
                classification = RouteClassification(
                    category="unknown",
                    confidence=0.5,
                    reasoning="分類結果のパースに失敗しました",
                )

        except Exception as e:
            print(f"⚠️  分類結果のパースエラー: {e}")
            classification = RouteClassification(
                category="unknown", confidence=0.3, reasoning=f"パースエラー: {str(e)}"
            )

        # 実行ログを更新
        log_entry = f"[{datetime.datetime.now().strftime('%H:%M:%S')}] クエリ分類完了: {classification.category} (信頼度: {classification.confidence})"
        execution_log = state.get("execution_log", [])
        execution_log.append(log_entry)

        print(
            f"✅ 分類完了: {classification.category} (信頼度: {classification.confidence})"
        )

        return {"classification": classification, "execution_log": execution_log}

    def _route_decision(self, state: RoutingState) -> str:
        """ルーティング決定の条件分岐関数"""
        classification = state["classification"]
        return classification.category

    def _handle_technical(self, state: RoutingState) -> Dict[str, Any]:
        """技術サポート経路の処理"""
        print("🔧 技術サポート経路で処理中...")

        # 技術サポートプロンプトを生成
        prompt = self.technical_prompt.format(user_query=state["user_query"])

        # LLMを呼び出し
        response = self.llm.invoke([HumanMessage(content=prompt)])

        # 実行ログを更新
        log_entry = (
            f"[{datetime.datetime.now().strftime('%H:%M:%S')}] 技術サポート処理完了"
        )
        execution_log = state["execution_log"]
        execution_log.append(log_entry)

        print("✅ 技術サポート処理完了")

        return {
            "response": response.content,
            "processing_path": "技術サポート",
            "execution_log": execution_log,
        }

    def _handle_billing(self, state: RoutingState) -> Dict[str, Any]:
        """請求サポート経路の処理"""
        print("💳 請求サポート経路で処理中...")

        # 請求サポートプロンプトを生成
        prompt = self.billing_prompt.format(user_query=state["user_query"])

        # LLMを呼び出し
        response = self.llm.invoke([HumanMessage(content=prompt)])

        # 実行ログを更新
        log_entry = (
            f"[{datetime.datetime.now().strftime('%H:%M:%S')}] 請求サポート処理完了"
        )
        execution_log = state["execution_log"]
        execution_log.append(log_entry)

        print("✅ 請求サポート処理完了")

        return {
            "response": response.content,
            "processing_path": "請求サポート",
            "execution_log": execution_log,
        }

    def _handle_general(self, state: RoutingState) -> Dict[str, Any]:
        """一般サポート経路の処理"""
        print("ℹ️ 一般サポート経路で処理中...")

        # 一般サポートプロンプトを生成
        prompt = self.general_prompt.format(user_query=state["user_query"])

        # LLMを呼び出し
        response = self.llm.invoke([HumanMessage(content=prompt)])

        # 実行ログを更新
        log_entry = (
            f"[{datetime.datetime.now().strftime('%H:%M:%S')}] 一般サポート処理完了"
        )
        execution_log = state["execution_log"]
        execution_log.append(log_entry)

        print("✅ 一般サポート処理完了")

        return {
            "response": response.content,
            "processing_path": "一般サポート",
            "execution_log": execution_log,
        }

    def _handle_complaint(self, state: RoutingState) -> Dict[str, Any]:
        """苦情対応経路の処理"""
        print("😔 苦情対応経路で処理中...")

        # 苦情対応プロンプトを生成
        prompt = self.complaint_prompt.format(user_query=state["user_query"])

        # LLMを呼び出し
        response = self.llm.invoke([HumanMessage(content=prompt)])

        # 実行ログを更新
        log_entry = f"[{datetime.datetime.now().strftime('%H:%M:%S')}] 苦情対応処理完了"
        execution_log = state["execution_log"]
        execution_log.append(log_entry)

        print("✅ 苦情対応処理完了")

        return {
            "response": response.content,
            "processing_path": "苦情対応",
            "execution_log": execution_log,
        }

    def _handle_unknown(self, state: RoutingState) -> Dict[str, Any]:
        """不明カテゴリ経路の処理"""
        print("❓ 不明カテゴリ経路で処理中...")

        # 不明カテゴリプロンプトを生成
        prompt = self.unknown_prompt.format(user_query=state["user_query"])

        # LLMを呼び出し
        response = self.llm.invoke([HumanMessage(content=prompt)])

        # 実行ログを更新
        log_entry = (
            f"[{datetime.datetime.now().strftime('%H:%M:%S')}] 不明カテゴリ処理完了"
        )
        execution_log = state["execution_log"]
        execution_log.append(log_entry)

        print("✅ 不明カテゴリ処理完了")

        return {
            "response": response.content,
            "processing_path": "不明カテゴリ",
            "execution_log": execution_log,
        }

    def process_query(self, user_query: str) -> Dict[str, Any]:
        """クエリ処理のメイン関数"""
        print(f"💬 クエリ処理開始: {user_query}")
        print("-" * 60)

        # 初期状態を設定
        initial_state = {
            "user_query": user_query,
            "classification": None,
            "response": "",
            "processing_path": "",
            "execution_log": [],
        }

        # ワークフローを実行
        start_time = datetime.datetime.now()
        result = self.graph.invoke(initial_state)
        end_time = datetime.datetime.now()

        # 実行時間を計算
        execution_time = (end_time - start_time).total_seconds()

        print("-" * 60)
        print(f"🎉 クエリ処理完了！ (実行時間: {execution_time:.2f}秒)")

        return {
            "user_query": result["user_query"],
            "classification": result["classification"],
            "response": result["response"],
            "processing_path": result["processing_path"],
            "execution_log": result["execution_log"],
            "execution_time": execution_time,
        }


# ===== デモ用のメイン関数 =====


def main():
    """LangGraph版 Routingのデモンストレーション"""
    print("=" * 60)
    print("🎯 LangGraph版 Routing Pattern デモ")
    print("=" * 60)
    print(
        "このデモでは、LangGraphを使用してクエリを分類し、適切な処理経路に振り分けます。"
    )
    print("サポートカテゴリ: 技術サポート、請求サポート、一般サポート、苦情対応、不明")
    print()

    try:
        # Routingシステムを初期化
        router = LangGraphRouting()

        # デモ用のクエリリスト
        demo_queries = [
            "ログインができません。パスワードを入力してもエラーになります。",
            "料金プランを変更したいのですが、手続きを教えてください。",
            "営業時間と連絡先を教えてください。",
            "サービスの品質が悪く、とても不満です。改善してください。",
            "今日の天気はどうですか？",
        ]

        print("📚 デモ用クエリの処理:")
        print("=" * 60)

        for i, query in enumerate(demo_queries, 1):
            print(f"\n【クエリ {i}】")

            # クエリを処理
            result = router.process_query(query)

            # 結果の表示
            print("\n📊 処理結果:")
            print(f"分類: {result['classification'].category}")
            print(f"信頼度: {result['classification'].confidence}")
            print(f"処理経路: {result['processing_path']}")
            print(f"実行時間: {result['execution_time']:.2f}秒")

            print("\n🤖 応答:")
            print(f"{result['response'][:200]}...")

            # 詳細表示の確認
            show_details = input("\n詳細を表示しますか？ (y/n): ").lower().strip()
            if show_details == "y":
                print("\n" + "=" * 50)
                print("📋 詳細結果")
                print("=" * 50)

                print("\n🔍 分類詳細:")
                print(f"  カテゴリ: {result['classification'].category}")
                print(f"  信頼度: {result['classification'].confidence}")
                print(f"  理由: {result['classification'].reasoning}")

                print("\n🤖 完全な応答:")
                print("-" * 30)
                print(result["response"])

                print("\n📊 実行ログ:")
                print("-" * 30)
                for log_entry in result["execution_log"]:
                    print(f"  {log_entry}")

            print()

        # 対話モードの開始
        print("\n" + "=" * 60)
        print("💬 対話モード開始 (終了するには 'quit' と入力)")
        print("=" * 60)

        while True:
            try:
                user_query = input("\n🙋 あなたの質問: ").strip()

                if user_query.lower() in ["quit", "exit", "終了", "q"]:
                    print("👋 対話を終了します。")
                    break

                if not user_query:
                    print("⚠️  質問を入力してください。")
                    continue

                # クエリを処理
                result = router.process_query(user_query)

                # 結果の表示
                print(
                    f"\n📊 分類: {result['classification'].category} (信頼度: {result['classification'].confidence})"
                )
                print(f"🛣️  処理経路: {result['processing_path']}")
                print("\n🤖 サポート担当者からの回答:")
                print("-" * 40)
                print(result["response"])

            except KeyboardInterrupt:
                print("\n\n👋 対話を終了します。")
                break

    except Exception as e:
        print(f"❌ エラーが発生しました: {e}")
        print("💡 OpenAI APIキーが正しく設定されているか確認してください。")


if __name__ == "__main__":
    main()
