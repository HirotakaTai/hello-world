"""
Routing パターン
===============

このパターンは、入力を分類して適切な専門的なワークフローに振り分ける方法です。

例：
- カスタマーサポートでの問い合わせ分類（技術的問題、請求問題、一般的な質問など）
- コンテンツタイプの分類（ニュース、レビュー、チュートリアルなど）
- 難易度による処理の振り分け（簡単な質問→軽量モデル、複雑な質問→高性能モデル）

このパターンの利点：
- 各タスクに特化した処理が可能
- コストと性能の最適化
- より精度の高い結果を得ることができる
"""

from datetime import datetime
from enum import Enum
from typing import Any
from typing import Dict

from dotenv import load_dotenv
from langchain.schema import HumanMessage
from langchain.schema import SystemMessage
from langchain_openai import ChatOpenAI

# ===== 環境変数の読み込み =====
load_dotenv()


class QueryType(Enum):
    """
    問い合わせの種類を定義する列挙型
    """

    TECHNICAL_SUPPORT = "technical_support"  # 技術サポート
    BILLING_INQUIRY = "billing_inquiry"  # 請求に関する問い合わせ
    GENERAL_QUESTION = "general_question"  # 一般的な質問
    PRODUCT_FEEDBACK = "product_feedback"  # 製品フィードバック
    FEATURE_REQUEST = "feature_request"  # 機能リクエスト
    COMPLAINT = "complaint"  # 苦情
    UNKNOWN = "unknown"  # 分類不明


class ContentType(Enum):
    """
    コンテンツの種類を定義する列挙型
    """

    NEWS_ARTICLE = "news_article"  # ニュース記事
    TUTORIAL = "tutorial"  # チュートリアル
    PRODUCT_REVIEW = "product_review"  # 製品レビュー
    OPINION_PIECE = "opinion_piece"  # 意見記事
    TECHNICAL_DOCUMENTATION = "technical_doc"  # 技術文書
    MARKETING_CONTENT = "marketing"  # マーケティングコンテンツ
    UNKNOWN = "unknown"  # 分類不明


class RoutingSystem:
    """
    ルーティングパターンの実装クラス
    入力を分類し、適切なハンドラーに振り分ける
    """

    def __init__(self):
        # ===== ChatOpenAI モデルの初期化 =====
        self.classifier_llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0,  # 分類の一貫性を保つため温度を0に設定
        )

        self.handler_llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.7,  # ハンドラーでは創造的な回答のため適度なランダム性
        )

        # ===== 処理ログを保存するリスト =====
        self.routing_log = []

    def _log_routing(
        self, input_text: str, classification: str, handler_used: str, response: str
    ):
        """
        ルーティング処理をログに記録

        Args:
            input_text (str): 入力テキスト
            classification (str): 分類結果
            handler_used (str): 使用されたハンドラー
            response (str): 応答
        """
        self.routing_log.append(
            {
                "input": input_text,
                "classification": classification,
                "handler": handler_used,
                "response": response,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
        )

    def _classify_customer_query(self, query: str) -> QueryType:
        """
        カスタマーサポートの問い合わせを分類

        Args:
            query (str): 顧客からの問い合わせ

        Returns:
            QueryType: 分類されたクエリタイプ
        """

        classification_prompt = f"""
        あなたはカスタマーサポートの問い合わせ分類の専門家です。
        以下の顧客からの問い合わせを、適切なカテゴリに分類してください。
        
        問い合わせ内容：
        {query}
        
        以下のカテゴリから最も適切なものを1つ選んで、カテゴリ名のみを回答してください：
        
        - technical_support: 技術的な問題や不具合に関する問い合わせ
        - billing_inquiry: 請求、支払い、料金に関する問い合わせ
        - general_question: 製品やサービスに関する一般的な質問
        - product_feedback: 製品に対するフィードバックや感想
        - feature_request: 新機能の要望や改善提案
        - complaint: 苦情や不満の表明
        - unknown: 上記のどれにも当てはまらない場合
        
        回答は必ずカテゴリ名のみを返してください（例: technical_support）
        """

        messages = [SystemMessage(content=classification_prompt)]
        response = self.classifier_llm.invoke(messages)

        # ===== 分類結果をQueryTypeに変換 =====
        try:
            return QueryType(response.content.strip().lower())
        except ValueError:
            print(f"警告: 不明な分類結果 '{response.content}' -> UNKNOWN に設定")
            return QueryType.UNKNOWN

    def _classify_content(self, content: str) -> ContentType:
        """
        コンテンツを分類

        Args:
            content (str): 分類するコンテンツ

        Returns:
            ContentType: 分類されたコンテンツタイプ
        """

        classification_prompt = f"""
        あなたはコンテンツ分類の専門家です。
        以下のコンテンツを、適切なタイプに分類してください。
        
        コンテンツ：
        {content}
        
        以下のタイプから最も適切なものを1つ選んで、タイプ名のみを回答してください：
        
        - news_article: ニュース記事や報道記事
        - tutorial: チュートリアルや手順説明
        - product_review: 製品やサービスのレビュー
        - opinion_piece: 意見記事やコラム
        - technical_doc: 技術文書やドキュメント
        - marketing: マーケティングや宣伝コンテンツ
        - unknown: 上記のどれにも当てはまらない場合
        
        回答は必ずタイプ名のみを返してください（例: news_article）
        """

        messages = [SystemMessage(content=classification_prompt)]
        response = self.classifier_llm.invoke(messages)

        # ===== 分類結果をContentTypeに変換 =====
        try:
            return ContentType(response.content.strip().lower())
        except ValueError:
            print(f"警告: 不明な分類結果 '{response.content}' -> UNKNOWN に設定")
            return ContentType.UNKNOWN

    def _handle_technical_support(self, query: str) -> str:
        """
        技術サポート問い合わせを処理

        Args:
            query (str): 技術的な問い合わせ

        Returns:
            str: 技術サポートの回答
        """

        system_prompt = """
        あなたは技術サポートの専門家です。
        顧客の技術的な問題を解決するため、以下の対応をしてください：
        
        1. 問題を正確に理解する
        2. 可能な原因を特定する
        3. 具体的な解決手順を提供する
        4. 追加のサポートが必要な場合の案内を行う
        
        丁寧で分かりやすい説明を心がけてください。
        """

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"技術的な問題: {query}"),
        ]

        response = self.handler_llm.invoke(messages)
        return response.content

    def _handle_billing_inquiry(self, query: str) -> str:
        """
        請求問い合わせを処理

        Args:
            query (str): 請求に関する問い合わせ

        Returns:
            str: 請求部門の回答
        """

        system_prompt = """
        あなたは請求・経理部門の専門家です。
        顧客の請求に関する問い合わせに対して、以下の対応をしてください：
        
        1. 請求内容を確認する方法を案内
        2. 支払い方法や期限について説明
        3. 請求に関する疑問を解決
        4. 必要に応じて専門部署への転送を案内
        
        正確で信頼できる情報を提供してください。
        """

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"請求に関する問い合わせ: {query}"),
        ]

        response = self.handler_llm.invoke(messages)
        return response.content

    def _handle_general_question(self, query: str) -> str:
        """
        一般的な質問を処理

        Args:
            query (str): 一般的な質問

        Returns:
            str: 一般的な回答
        """

        system_prompt = """
        あなたはカスタマーサービスの担当者です。
        顧客の一般的な質問に対して、以下の対応をしてください：
        
        1. 質問に対する明確で正確な回答を提供
        2. 関連する有用な情報を追加
        3. 必要に応じて詳細情報へのリンクや連絡先を案内
        4. 親しみやすく丁寧な対応を心がける
        
        顧客満足度を重視した回答をしてください。
        """

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"一般的な質問: {query}"),
        ]

        response = self.handler_llm.invoke(messages)
        return response.content

    def _handle_product_feedback(self, query: str) -> str:
        """
        製品フィードバックを処理

        Args:
            query (str): 製品フィードバック

        Returns:
            str: フィードバックへの対応
        """

        system_prompt = """
        あなたは製品管理チームの担当者です。
        顧客からの製品フィードバックに対して、以下の対応をしてください：
        
        1. フィードバックに対する感謝の意を表す
        2. フィードバックの内容を整理し、重要なポイントを確認
        3. 今後の製品改善への活用について説明
        4. 追加の意見があれば聞く姿勢を示す
        
        顧客の声を大切にする姿勢を示してください。
        """

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"製品フィードバック: {query}"),
        ]

        response = self.handler_llm.invoke(messages)
        return response.content

    def _handle_feature_request(self, query: str) -> str:
        """
        機能リクエストを処理

        Args:
            query (str): 機能リクエスト

        Returns:
            str: 機能リクエストへの対応
        """

        system_prompt = """
        あなたは製品開発チームの担当者です。
        顧客からの機能リクエストに対して、以下の対応をしてください：
        
        1. リクエストに対する感謝を表す
        2. 要求された機能の詳細を確認
        3. 実装の可能性や検討プロセスについて説明
        4. 代替案がある場合は提案
        5. 進捗の確認方法を案内
        
        建設的で前向きな回答を心がけてください。
        """

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"機能リクエスト: {query}"),
        ]

        response = self.handler_llm.invoke(messages)
        return response.content

    def _handle_complaint(self, query: str) -> str:
        """
        苦情を処理

        Args:
            query (str): 苦情

        Returns:
            str: 苦情への対応
        """

        system_prompt = """
        あなたは苦情対応の専門家です。
        顧客からの苦情に対して、以下の対応をしてください：
        
        1. 誠実な謝罪と問題の受け止めを表明
        2. 問題の詳細を確認し、原因を理解
        3. 具体的な解決策や改善策を提案
        4. 再発防止への取り組みを説明
        5. 必要に応じて上級者への転送を案内
        
        顧客の気持ちに寄り添い、信頼回復を目指してください。
        """

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"苦情: {query}"),
        ]

        response = self.handler_llm.invoke(messages)
        return response.content

    def _handle_news_article(self, content: str) -> str:
        """
        ニュース記事を処理

        Args:
            content (str): ニュース記事

        Returns:
            str: ニュース記事の要約と分析
        """

        system_prompt = """
        あなたはニュース編集者です。
        ニュース記事に対して以下の処理を行ってください：
        
        1. 記事の要約（3-5行）
        2. 主要なポイントの抽出
        3. 影響や重要性の分析
        4. 関連する背景情報の補足
        
        客観的で正確な分析を提供してください。
        """

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"ニュース記事: {content}"),
        ]

        response = self.handler_llm.invoke(messages)
        return response.content

    def _handle_tutorial(self, content: str) -> str:
        """
        チュートリアルを処理

        Args:
            content (str): チュートリアルコンテンツ

        Returns:
            str: チュートリアルの評価と改善提案
        """

        system_prompt = """
        あなたは教育コンテンツの専門家です。
        チュートリアルに対して以下の評価を行ってください：
        
        1. 内容の明確さと理解しやすさ
        2. 手順の完全性と論理性
        3. 対象読者への適切さ
        4. 改善提案があれば具体的に指摘
        
        建設的なフィードバックを提供してください。
        """

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"チュートリアル: {content}"),
        ]

        response = self.handler_llm.invoke(messages)
        return response.content

    def process_customer_query(self, query: str) -> Dict[str, Any]:
        """
        カスタマーサポートクエリを処理

        Args:
            query (str): 顧客からの問い合わせ

        Returns:
            Dict[str, Any]: 分類結果と処理結果
        """

        print(f"📞 カスタマークエリ処理開始: {query[:50]}...")

        # ===== ステップ1: クエリを分類 =====
        query_type = self._classify_customer_query(query)
        print(f"🏷️ 分類結果: {query_type.value}")

        # ===== ステップ2: 適切なハンドラーで処理 =====
        handler_map = {
            QueryType.TECHNICAL_SUPPORT: self._handle_technical_support,
            QueryType.BILLING_INQUIRY: self._handle_billing_inquiry,
            QueryType.GENERAL_QUESTION: self._handle_general_question,
            QueryType.PRODUCT_FEEDBACK: self._handle_product_feedback,
            QueryType.FEATURE_REQUEST: self._handle_feature_request,
            QueryType.COMPLAINT: self._handle_complaint,
        }

        if query_type in handler_map:
            handler = handler_map[query_type]
            response = handler(query)
            handler_name = handler.__name__
        else:
            # ===== 分類不明の場合は一般的な処理 =====
            response = self._handle_general_question(query)
            handler_name = "_handle_general_question (fallback)"

        # ===== ログに記録 =====
        self._log_routing(query, query_type.value, handler_name, response)

        print(f"✅ 処理完了: {handler_name}")

        return {
            "query": query,
            "classification": query_type.value,
            "handler_used": handler_name,
            "response": response,
        }

    def process_content(self, content: str) -> Dict[str, Any]:
        """
        コンテンツを処理

        Args:
            content (str): 処理するコンテンツ

        Returns:
            Dict[str, Any]: 分類結果と処理結果
        """

        print(f"📄 コンテンツ処理開始: {content[:50]}...")

        # ===== ステップ1: コンテンツを分類 =====
        content_type = self._classify_content(content)
        print(f"🏷️ 分類結果: {content_type.value}")

        # ===== ステップ2: 適切なハンドラーで処理 =====
        handler_map = {
            ContentType.NEWS_ARTICLE: self._handle_news_article,
            ContentType.TUTORIAL: self._handle_tutorial,
        }

        if content_type in handler_map:
            handler = handler_map[content_type]
            response = handler(content)
            handler_name = handler.__name__
        else:
            # ===== その他のタイプは一般的な要約を作成 =====
            response = (
                f"コンテンツタイプ: {content_type.value}\n\n要約: {content[:200]}..."
            )
            handler_name = "general_summary"

        # ===== ログに記録 =====
        self._log_routing(content, content_type.value, handler_name, response)

        print(f"✅ 処理完了: {handler_name}")

        return {
            "content": content,
            "classification": content_type.value,
            "handler_used": handler_name,
            "response": response,
        }

    def get_routing_statistics(self) -> Dict[str, Any]:
        """
        ルーティング統計を取得

        Returns:
            Dict[str, Any]: 統計情報
        """

        if not self.routing_log:
            return {"total_requests": 0}

        # ===== 分類別の統計 =====
        classification_counts = {}
        handler_counts = {}

        for log_entry in self.routing_log:
            classification = log_entry["classification"]
            handler = log_entry["handler"]

            classification_counts[classification] = (
                classification_counts.get(classification, 0) + 1
            )
            handler_counts[handler] = handler_counts.get(handler, 0) + 1

        return {
            "total_requests": len(self.routing_log),
            "classification_distribution": classification_counts,
            "handler_usage": handler_counts,
            "last_processed": self.routing_log[-1]["timestamp"]
            if self.routing_log
            else None,
        }


# ===== 使用例 =====
def main():
    """
    Routingパターンのデモンストレーション
    """
    print("=== Routing パターンのデモ ===\n")

    # ===== ルーティングシステムのインスタンスを作成 =====
    routing_system = RoutingSystem()

    # ===== デモ1: カスタマーサポートクエリのルーティング =====
    print("🏢 デモ1: カスタマーサポートクエリのルーティング")
    print("-" * 60)

    customer_queries = [
        "アプリがクラッシュして使えません。解決方法を教えてください。",
        "今月の請求額が間違っているようです。確認をお願いします。",
        "御社のサービスの利用方法について教えてください。",
        "製品がとても使いやすくて満足しています。",
        "新しい機能として、ダークモードを追加してほしいです。",
        "サポートの対応が悪くて不満です。改善してください。",
    ]

    for i, query in enumerate(customer_queries, 1):
        print(f"\n📝 クエリ {i}:")
        result = routing_system.process_customer_query(query)
        print(f"応答: {result['response'][:100]}...")
        print("-" * 40)

    # ===== デモ2: コンテンツのルーティング =====
    print("\n\n📰 デモ2: コンテンツのルーティング")
    print("-" * 60)

    contents = [
        """
        【速報】新しいAI技術が発表される
        
        本日、大手テクノロジー企業が革新的なAI技術を発表しました。
        この技術により、自然言語処理の精度が大幅に向上することが期待されています。
        業界関係者は「画期的な進歩」と評価しており、今後の展開に注目が集まっています。
        """,
        """
        Pythonでのファイル操作方法
        
        ステップ1: ファイルを開く
        with open("sample.txt", "r") as file:
            content = file.read()
        
        ステップ2: データを処理する
        processed_data = content.upper()
        
        ステップ3: 結果を保存する
        with open("output.txt", "w") as file:
            file.write(processed_data)
        """,
    ]

    for i, content in enumerate(contents, 1):
        print(f"\n📄 コンテンツ {i}:")
        result = routing_system.process_content(content)
        print(f"処理結果: {result['response'][:100]}...")
        print("-" * 40)

    # ===== 統計情報の表示 =====
    print("\n\n📊 ルーティング統計")
    print("-" * 30)
    stats = routing_system.get_routing_statistics()
    print(f"総処理数: {stats['total_requests']}")
    print("分類別分布:")
    for classification, count in stats["classification_distribution"].items():
        print(f"  - {classification}: {count}件")

    print("\nハンドラー使用状況:")
    for handler, count in stats["handler_usage"].items():
        print(f"  - {handler}: {count}回")


if __name__ == "__main__":
    main()
