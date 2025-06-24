"""
Prompt Chaining パターン
======================

このパターンは、複雑なタスクを複数のステップに分割し、
各ステップの出力を次のステップの入力として使用する方法です。

例：
1. 記事のアウトライン作成
2. アウトラインのチェック
3. 最終的な記事の作成

このパターンの利点：
- 複雑なタスクを管理しやすい小さなステップに分割
- 各ステップで品質チェックが可能
- より高精度な結果を得ることができる
"""

from datetime import datetime

from dotenv import load_dotenv
from langchain.schema import HumanMessage
from langchain.schema import SystemMessage
from langchain_openai import ChatOpenAI

# ===== 環境変数の読み込み =====
load_dotenv()


class PromptChaining:
    """
    プロンプトチェーンパターンの実装クラス
    複数のステップを順次実行し、各ステップの出力を次のステップに渡す
    """

    def __init__(self):
        # ===== ChatOpenAI モデルの初期化 =====
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.7,  # 創造的な出力のために適度なランダム性を設定
        )

        # ===== 実行ログを保存するリスト =====
        self.execution_log = []

    def _log_step(self, step_name: str, input_data: str, output_data: str):
        """
        実行ステップをログに記録

        Args:
            step_name (str): ステップ名
            input_data (str): 入力データ
            output_data (str): 出力データ
        """
        self.execution_log.append(
            {
                "step": step_name,
                "input": input_data,
                "output": output_data,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
        )

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """
        LLMを呼び出して応答を取得

        Args:
            system_prompt (str): システムプロンプト（LLMの役割を定義）
            user_prompt (str): ユーザープロンプト（具体的なタスクを指定）

        Returns:
            str: LLMの応答
        """
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        response = self.llm.invoke(messages)
        return response.content

    def create_blog_article(
        self, topic: str, target_audience: str = "一般読者"
    ) -> dict:
        """
        ブログ記事作成のプロンプトチェーン実装

        ステップ1: トピックの分析
        ステップ2: アウトライン作成
        ステップ3: アウトラインの評価・改善
        ステップ4: 記事本文の作成
        ステップ5: 記事の校正・最終チェック

        Args:
            topic (str): ブログ記事のトピック
            target_audience (str): ターゲット読者層

        Returns:
            dict: 各ステップの結果を含む辞書
        """

        print(f"=== ブログ記事作成開始: {topic} ===\n")

        # ===== ステップ1: トピック分析 =====
        print("📊 ステップ1: トピック分析")

        analysis_system_prompt = """
        あなたは優秀なコンテンツ戦略家です。
        与えられたトピックを分析し、読者にとって価値ある情報を特定してください。
        """

        analysis_user_prompt = f"""
        トピック: {topic}
        ターゲット読者: {target_audience}
        
        このトピックについて以下を分析してください：
        1. 読者が知りたい主要なポイント
        2. このトピックの重要性
        3. 読者が抱えている可能性のある疑問
        4. 提供すべき価値のある情報
        
        分析結果を具体的に説明してください。
        """

        topic_analysis = self._call_llm(analysis_system_prompt, analysis_user_prompt)
        self._log_step("トピック分析", analysis_user_prompt, topic_analysis)
        print(f"分析結果:\n{topic_analysis}\n")

        # ===== ステップ2: アウトライン作成 =====
        print("📝 ステップ2: アウトライン作成")

        outline_system_prompt = """
        あなたは経験豊富なライターです。
        トピック分析の結果に基づいて、読みやすく構造化された記事のアウトラインを作成してください。
        """

        outline_user_prompt = f"""
        以下のトピック分析結果に基づいて、ブログ記事のアウトラインを作成してください：
        
        【トピック分析結果】
        {topic_analysis}
        
        アウトラインの要件：
        - 導入、本文（3-5セクション）、結論の構造
        - 各セクションのタイトルと主要なポイント
        - 読者にとって論理的な流れ
        - 各セクションの推定文字数
        
        アウトラインを作成してください。
        """

        outline = self._call_llm(outline_system_prompt, outline_user_prompt)
        self._log_step("アウトライン作成", outline_user_prompt, outline)
        print(f"アウトライン:\n{outline}\n")

        # ===== ステップ3: アウトライン評価・改善 =====
        print("🔍 ステップ3: アウトライン評価・改善")

        evaluation_system_prompt = """
        あなたは記事品質の専門家です。
        アウトラインを評価し、改善点があれば修正版を提案してください。
        """

        evaluation_user_prompt = f"""
        以下のアウトラインを評価し、改善点があれば修正してください：
        
        【現在のアウトライン】
        {outline}
        
        評価基準：
        1. 読者にとっての価値の明確さ
        2. 情報の論理的な流れ
        3. 各セクションのバランス
        4. 読みやすさ
        
        改善点があれば修正版を提案し、変更理由を説明してください。
        問題がなければ「現在のアウトラインは適切です」と回答してください。
        """

        improved_outline = self._call_llm(
            evaluation_system_prompt, evaluation_user_prompt
        )
        self._log_step(
            "アウトライン評価・改善", evaluation_user_prompt, improved_outline
        )
        print(f"評価・改善結果:\n{improved_outline}\n")

        # ===== ステップ4: 記事本文作成 =====
        print("✍️ ステップ4: 記事本文作成")

        writing_system_prompt = """
        あなたは優秀なライターです。
        アウトラインに基づいて、読みやすく魅力的な記事を作成してください。
        """

        # 最終的なアウトラインを決定（改善されたものまたは元のもの）
        final_outline = (
            improved_outline
            if "現在のアウトラインは適切です" not in improved_outline
            else outline
        )

        writing_user_prompt = f"""
        以下のアウトラインに基づいて、ブログ記事の本文を作成してください：
        
        【最終アウトライン】
        {final_outline}
        
        記事作成の要件：
        - 読みやすい文章
        - 具体例や実例の使用
        - 読者の関心を引く導入文
        - 行動を促すような結論
        - 適切な見出し構造
        
        完成した記事を作成してください。
        """

        article_draft = self._call_llm(writing_system_prompt, writing_user_prompt)
        self._log_step("記事本文作成", writing_user_prompt, article_draft)
        print(f"記事草案:\n{article_draft}\n")

        # ===== ステップ5: 記事の校正・最終チェック =====
        print("🔍 ステップ5: 記事の校正・最終チェック")

        proofreading_system_prompt = """
        あなたは校正の専門家です。
        記事を校正し、文法、表現、構成の改善点を特定して修正してください。
        """

        proofreading_user_prompt = f"""
        以下の記事を校正し、必要に応じて修正してください：
        
        【記事草案】
        {article_draft}
        
        校正のポイント：
        1. 文法・表現の正確性
        2. 読みやすさの向上
        3. 論理的な流れの確認
        4. 重複や冗長な表現の削除
        5. タイトルとサブタイトルの適切性
        
        校正した最終版の記事を提供してください。
        """

        final_article = self._call_llm(
            proofreading_system_prompt, proofreading_user_prompt
        )
        self._log_step(
            "記事校正・最終チェック", proofreading_user_prompt, final_article
        )
        print(f"最終版記事:\n{final_article}\n")

        # ===== 結果をまとめて返す =====
        return {
            "topic": topic,
            "target_audience": target_audience,
            "analysis": topic_analysis,
            "outline": outline,
            "improved_outline": improved_outline,
            "draft": article_draft,
            "final_article": final_article,
            "execution_log": self.execution_log.copy(),
        }

    def translate_and_improve(self, text: str, target_language: str = "英語") -> dict:
        """
        翻訳と改善のプロンプトチェーン実装

        ステップ1: 初期翻訳
        ステップ2: 翻訳品質の評価
        ステップ3: 改善された翻訳の作成

        Args:
            text (str): 翻訳する文章
            target_language (str): 翻訳先言語

        Returns:
            dict: 各ステップの結果を含む辞書
        """

        print(f"=== 翻訳・改善チェーン開始: {target_language}への翻訳 ===\n")

        # ===== ステップ1: 初期翻訳 =====
        print("🌐 ステップ1: 初期翻訳")

        translation_system_prompt = f"""
        あなたは優秀な翻訳者です。
        与えられた文章を{target_language}に翻訳してください。
        """

        translation_user_prompt = f"""
        以下の文章を{target_language}に翻訳してください：
        
        【原文】
        {text}
        
        自然で読みやすい翻訳を心がけてください。
        """

        initial_translation = self._call_llm(
            translation_system_prompt, translation_user_prompt
        )
        self._log_step("初期翻訳", translation_user_prompt, initial_translation)
        print(f"初期翻訳:\n{initial_translation}\n")

        # ===== ステップ2: 翻訳品質の評価 =====
        print("📊 ステップ2: 翻訳品質の評価")

        evaluation_system_prompt = """
        あなたは翻訳品質の評価専門家です。
        翻訳の品質を評価し、改善点を特定してください。
        """

        evaluation_user_prompt = f"""
        以下の翻訳を評価してください：
        
        【原文】
        {text}
        
        【翻訳】
        {initial_translation}
        
        評価基準：
        1. 原文の意味の正確性
        2. 自然さ・読みやすさ
        3. 文法的正確性
        4. 適切な表現の使用
        
        改善点があれば具体的に指摘してください。
        """

        evaluation = self._call_llm(evaluation_system_prompt, evaluation_user_prompt)
        self._log_step("翻訳品質評価", evaluation_user_prompt, evaluation)
        print(f"評価結果:\n{evaluation}\n")

        # ===== ステップ3: 改善された翻訳の作成 =====
        print("✨ ステップ3: 改善された翻訳の作成")

        improvement_system_prompt = f"""
        あなたは翻訳改善の専門家です。
        評価結果に基づいて、より良い{target_language}翻訳を作成してください。
        """

        improvement_user_prompt = f"""
        以下の評価結果に基づいて、翻訳を改善してください：
        
        【原文】
        {text}
        
        【初期翻訳】
        {initial_translation}
        
        【評価結果】
        {evaluation}
        
        評価で指摘された問題点を修正し、最高品質の翻訳を作成してください。
        """

        improved_translation = self._call_llm(
            improvement_system_prompt, improvement_user_prompt
        )
        self._log_step("改善された翻訳", improvement_user_prompt, improved_translation)
        print(f"改善された翻訳:\n{improved_translation}\n")

        # ===== 結果をまとめて返す =====
        return {
            "original_text": text,
            "target_language": target_language,
            "initial_translation": initial_translation,
            "evaluation": evaluation,
            "improved_translation": improved_translation,
            "execution_log": self.execution_log.copy(),
        }

    def get_execution_log(self) -> list:
        """
        実行ログを取得

        Returns:
            list: 実行されたステップのログ
        """
        return self.execution_log

    def clear_log(self):
        """
        実行ログをクリア
        """
        self.execution_log = []
        print("実行ログをクリアしました。")


# ===== 使用例 =====
def main():
    """
    Prompt Chainingパターンのデモンストレーション
    """
    print("=== Prompt Chaining パターンのデモ ===\n")

    # ===== プロンプトチェーンのインスタンスを作成 =====
    chain = PromptChaining()

    # ===== デモ1: ブログ記事作成チェーン =====
    print("📝 デモ1: ブログ記事作成チェーン")
    print("-" * 50)

    blog_result = chain.create_blog_article(
        topic="人工知能（AI）の仕事への影響", target_audience="ビジネスパーソン"
    )

    print("\n" + "=" * 50)
    print("ブログ記事作成完了！")
    print("=" * 50)

    # ===== デモ2: 翻訳・改善チェーン =====
    print("\n📝 デモ2: 翻訳・改善チェーン")
    print("-" * 50)

    # 新しいインスタンスを作成（ログを分離するため）
    translation_chain = PromptChaining()

    sample_text = """
    人工知能技術は急速に発展しており、私たちの日常生活や働き方に大きな変化をもたらしています。
    特に、機械学習や深層学習の進歩により、これまで人間にしかできないと考えられていた作業が
    自動化されるようになっています。しかし、この変化は単純に仕事を奪うものではなく、
    新しい価値を創造する機会でもあります。
    """

    translation_result = translation_chain.translate_and_improve(
        text=sample_text, target_language="英語"
    )

    print("\n" + "=" * 50)
    print("翻訳・改善チェーン完了！")
    print("=" * 50)

    # ===== 実行ログの表示 =====
    print("\n📊 実行ログサマリー")
    print("-" * 30)
    print(f"ブログ記事作成: {len(blog_result['execution_log'])} ステップ")
    print(f"翻訳・改善: {len(translation_result['execution_log'])} ステップ")


if __name__ == "__main__":
    main()
