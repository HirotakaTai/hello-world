from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

# ===== 環境変数の読み込み =====
load_dotenv()


# ===== React Agent の作成とツールの定義 =====
@tool
def calculator(expression: str) -> float:
    """数式を計算します"""
    return eval(expression)


@tool
def get_weather(city: str) -> str:
    """天気情報を取得します"""
    weather_data = {"東京": "晴れ", "大阪": "曇り", "名古屋": "雨", "福岡": "晴れ"}
    return f"{city}の天気は{weather_data.get(city, '不明')}です。大阪の天気は？"


# ===== OpenAI モデルのインスタンス化 =====
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
tools = [calculator, get_weather]
llm_with_tools = llm.bind_tools(tools)

agent = create_react_agent(
    model=llm_with_tools,
    tools=tools,
    prompt="あなたは役立つアシスタントです。ユーザーの質問に答えるために、必要に応じて複数のツールを使用してください。",
)

# ステートフルな実行
result = agent.invoke(
    {"messages": [{"role": "user", "content": "東京の天気と2+3の計算をして"}]}
)


# create_react_agent の実行結果をログ出力
for message in result["messages"]:
    message.pretty_print()
