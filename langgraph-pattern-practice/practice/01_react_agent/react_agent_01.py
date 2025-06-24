from dotenv import load_dotenv
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

# ===== 環境変数の読み込み =====
load_dotenv()


@tool
def calculator(expression: str) -> float:
    """数式を計算します"""
    return eval(expression)


@tool
def get_weather(city: str) -> str:
    """天気情報を取得します"""
    return f"{city}は晴れです。大阪の天気は？"


tools = [calculator, get_weather]

agent = create_react_agent(
    model="openai:gpt-3.5-turbo",
    tools=tools,
    prompt="あなたは役立つアシスタントです",
)

# ステートフルな実行
result = agent.invoke(
    {"messages": [{"role": "user", "content": "東京の天気と2+3の計算をして"}]}
)


# create_react_agent の実行結果をログ出力
for message in result["messages"]:
    message.pretty_print()
