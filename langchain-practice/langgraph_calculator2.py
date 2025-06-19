"""
https://dev.classmethod.jp/articles/d6f13aec839e43c302383d137edeaabbc91c8356385881fa29c17c8f604096a5/
"""

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState

load_dotenv()

# === LLMを作成し、掛け算、足し算、割り算を行う関数を用意し、ツールとして利用します。 ===


def multiply(a: int, b: int) -> int:
    """Multiply a and b.

    Args:
        a: first int
        b: second int
    """
    return a * b


# This will be a tool
def add(a: int, b: int) -> int:
    """Adds a and b.

    Args:
        a: first int
        b: second int
    """
    return a + b


def divide(a: int, b: int) -> float:
    """Divide a and b.

    Args:
        a: first int
        b: second int
    """
    return a / b


tools = [add, multiply, divide]
llm = ChatOpenAI(model="gpt-3.5-turbo")
llm_with_tools = llm.bind_tools(tools)


# === LLMを作成し、エージェントの全体的な望ましい行動をプロンプトします。 ===

# System message
sys_msg = SystemMessage(
    content="あなたは、一連の入力に対して算術演算を実行するタスクを与えられた親切なアシスタントでぇす。"
)


# Node
def assistant(state: MessagesState):
    return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}


# === LangGraphのグラフを作成します ===

from IPython.display import Image
from IPython.display import display
from langgraph.graph import START
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition

# Graph
builder = StateGraph(MessagesState)

# ノードの定義。このノードが実行されます
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))

# エッジの定義：制御フローがどのように動くかを決める。
builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    # アシスタントからの最新のメッセージ（結果）がツールコールである場合 -> tools_conditionはtoolsにルーティングする。
    # アシスタントからの最新のメッセージ（結果）がツールコールでない場合 -> tools_conditionはENDにルーティングする。
    tools_condition,
)
builder.add_edge("tools", "assistant")
react_graph = builder.compile()

# Show
display(Image(react_graph.get_graph(xray=True).draw_mermaid_png()))

# === 実行 ===

messages = [
    HumanMessage(content="3と4を足す。その出力に2を掛ける。 さらにその出力を5で割る。")
]
messages = react_graph.invoke({"messages": messages})
for m in messages["messages"]:
    m.pretty_print()
