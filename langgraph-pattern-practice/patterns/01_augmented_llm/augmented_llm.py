"""
Augmented LLM パターン
===================

このパターンは、基本的なLLMに以下の機能を追加して拡張したものです：
- 外部ツール（API呼び出し、計算など）の使用
- 検索機能（RAG: Retrieval-Augmented Generation）
- メモリ機能（過去の会話履歴の保持）

これらの拡張により、LLMがより実用的なタスクを実行できるようになります。
"""

from datetime import datetime

from dotenv import load_dotenv
from langchain.agents import AgentType
from langchain.agents import initialize_agent
from langchain.tools import Tool
from langchain_openai import ChatOpenAI

# ===== 環境変数の読み込み =====
load_dotenv()


class AugmentedLLM:
    """
    拡張LLMクラス
    基本的なLLMに外部ツールやメモリ機能を追加
    """

    def __init__(self):
        # ===== ChatOpenAI モデルの初期化 =====
        # temperature=0: 出力の一貫性を保つため、ランダム性を最小限に設定
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

        # ===== 会話履歴を保存するメモリ =====
        self.memory = []

        # ===== 利用可能なツールを定義 =====
        self.tools = self._create_tools()

        # ===== エージェントの初期化 =====
        # エージェントは、LLMがツールを使用する方法を管理
        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,  # ReAct（Reasoning + Acting）パターンを使用
            verbose=True,  # 思考プロセスを表示
        )

    def _create_tools(self) -> list:
        """
        LLMが使用できるツールを作成

        Returns:
            list: 利用可能なツールのリスト
        """

        # ===== 計算ツール =====
        def calculator(expression: str) -> str:
            """
            数学的な計算を実行するツール

            Args:
                expression (str): 計算式（例: "2 + 3 * 4"）

            Returns:
                str: 計算結果
            """
            try:
                # eval()は危険な関数ですが、デモ用として使用
                # 実際のプロダクションでは、より安全な計算ライブラリを使用してください
                result = eval(expression)
                return f"計算結果: {result}"
            except Exception as e:
                return f"計算エラー: {str(e)}"

        # ===== 天気情報取得ツール（模擬） =====
        def get_weather(city: str) -> str:
            """
            指定された都市の天気情報を取得するツール（模擬実装）

            Args:
                city (str): 都市名

            Returns:
                str: 天気情報
            """
            # 実際のAPIを使用する代わりに、サンプルデータを返す
            sample_weather = {
                "東京": "晴れ、気温: 25°C",
                "大阪": "曇り、気温: 23°C",
                "札幌": "雨、気温: 18°C",
            }

            weather = sample_weather.get(
                city, f"{city}の天気情報は取得できませんでした"
            )
            return f"{city}の天気: {weather}"

        # ===== 現在日時取得ツール =====
        def get_current_time() -> str:
            """
            現在の日時を取得するツール

            Returns:
                str: 現在の日時
            """
            now = datetime.now()
            return f"現在の日時: {now.strftime('%Y年%m月%d日 %H:%M:%S')}"

        # ===== ツールをLangChainのTool形式で定義 =====
        tools = [
            Tool(
                name="Calculator",
                func=calculator,
                description="数学的な計算を実行します。計算式を文字列で渡してください。例: '2 + 3 * 4'",
            ),
            Tool(
                name="Weather",
                func=get_weather,
                description="指定された都市の天気情報を取得します。都市名を渡してください。例: '東京'",
            ),
            Tool(
                name="CurrentTime",
                func=get_current_time,
                description="現在の日時を取得します。引数は不要です。",
            ),
        ]

        return tools

    def chat(self, user_input: str) -> str:
        """
        ユーザーの入力に対してAIが応答する

        Args:
            user_input (str): ユーザーの入力

        Returns:
            str: AIの応答
        """

        # ===== 会話履歴にユーザーの入力を追加 =====
        self.memory.append({"role": "user", "content": user_input})

        try:
            # ===== エージェントを使用してAIの応答を生成 =====
            # エージェントはLLMと複数のツールを組み合わせて動作
            response = self.agent.run(user_input)

            # ===== 会話履歴にAIの応答を追加 =====
            self.memory.append({"role": "assistant", "content": response})

            return response

        except Exception as e:
            error_message = f"エラーが発生しました: {str(e)}"
            self.memory.append({"role": "assistant", "content": error_message})
            return error_message

    def get_memory(self) -> list:
        """
        会話履歴を取得

        Returns:
            list: 会話履歴のリスト
        """
        return self.memory

    def clear_memory(self):
        """
        会話履歴をクリア
        """
        self.memory = []
        print("会話履歴をクリアしました。")


# ===== 使用例 =====
def main():
    """
    Augmented LLMパターンのデモンストレーション
    """
    print("=== Augmented LLM パターンのデモ ===\n")

    # ===== 拡張LLMのインスタンスを作成 =====
    augmented_llm = AugmentedLLM()

    # ===== テストケース1: 計算ツールの使用 =====
    print("🧮 計算ツールのテスト")
    response1 = augmented_llm.chat("25 × 4 + 12を計算してください")
    print(f"AI応答: {response1}\n")

    # ===== テストケース2: 天気情報の取得 =====
    print("🌤️ 天気情報ツールのテスト")
    response2 = augmented_llm.chat("東京の天気を教えてください")
    print(f"AI応答: {response2}\n")

    # ===== テストケース3: 現在時刻の取得 =====
    print("⏰ 現在時刻ツールのテスト")
    response3 = augmented_llm.chat("今何時ですか？")
    print(f"AI応答: {response3}\n")

    # ===== テストケース4: 複数ツールの組み合わせ =====
    print("🔄 複数ツールの組み合わせテスト")
    response4 = augmented_llm.chat(
        "現在の時刻を教えて、その後で10 + 20を計算してください"
    )
    print(f"AI応答: {response4}\n")

    # ===== 会話履歴の表示 =====
    print("📝 会話履歴:")
    for i, entry in enumerate(augmented_llm.get_memory(), 1):
        role = "ユーザー" if entry["role"] == "user" else "AI"
        print(f"{i}. {role}: {entry['content']}")


if __name__ == "__main__":
    main()
