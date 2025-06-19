from dotenv import load_dotenv
from langchain.agents import AgentExecutor
from langchain.agents import create_openai_tools_agent
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()


@tool
def add_numbers(x: float, y: float) -> float:
    """二つの数値を足し算します。"""
    return x + y


@tool
def subtract_numbers(x: float, y: float) -> float:
    """二つの数値を引き算します（x - y）。"""
    return x - y


@tool
def multiply_numbers(x: float, y: float) -> float:
    """二つの数値を掛け算します。"""
    return x * y


@tool
def divide_numbers(x: float, y: float) -> float:
    """二つの数値を割り算します（x ÷ y）。ゼロ除算の場合はエラーを返します。"""
    if y == 0:
        raise ValueError("ゼロで割ることはできません")
    return x / y


@tool
def power_numbers(x: float, y: float) -> float:
    """x の y 乗を計算します。"""
    return x**y


def create_calculator_agent():
    """計算エージェントを作成します。"""
    # LLMモデルの初期化
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # 利用可能なツール
    tools = [
        add_numbers,
        subtract_numbers,
        multiply_numbers,
        divide_numbers,
        power_numbers,
    ]

    # プロンプトテンプレート
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """あなたは数学計算の専門家です。
ユーザーからの計算要求を正確に理解し、適切なツールを使用して計算を実行してください。
計算結果は分かりやすく日本語で説明してください。

利用可能なツール:
- add_numbers: 足し算
- subtract_numbers: 引き算
- multiply_numbers: 掛け算
- divide_numbers: 割り算
- power_numbers: べき乗

複数の計算が必要な場合は、段階的に実行してください。""",
            ),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )

    # エージェントの作成
    agent = create_openai_tools_agent(llm, tools, prompt)

    # エージェント実行器の作成
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
    )

    return agent_executor


def main():
    """メイン実行関数"""
    agent = create_calculator_agent()

    print("🤖 計算エージェントが起動しました！")
    print("自然言語で計算をお聞かせください。（'quit'で終了）")
    print("-" * 50)

    while True:
        user_input = input("\n💭 あなた: ")

        if user_input.lower() in ["quit", "exit", "終了", "やめる"]:
            print("👋 ありがとうございました！")
            break

        try:
            # エージェントに問い合わせ
            result = agent.invoke({"input": user_input})
            print(f"\n🤖 エージェント: {result['output']}")
        except Exception as e:
            print(f"\n❌ エラーが発生しました: {e}")


if __name__ == "__main__":
    main()
