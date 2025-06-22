# 📦 必要なライブラリ（事前にインストール）
# pip install langgraph==0.4.8 langchain openai google-generativeai pydantic

import os
from typing import Optional

import google.generativeai as genai
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from langchain.schema import SystemMessage
from langgraph.graph import END
from langgraph.graph import StateGraph
from pydantic import BaseModel

# 🌐 API キーの設定
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# 🌟 LLM インスタンス
openai_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
genai.configure(api_key=GEMINI_API_KEY)
gemini_llm = genai.GenerativeModel("gemini-pro")


# 🧠 状態定義
class LLMCompareState(BaseModel):
    question: str
    openai_response: Optional[str] = None
    gemini_response: Optional[str] = None
    merged_summary: Optional[str] = None


# 🤖 OpenAI に問い合わせるノード
def ask_openai(state: LLMCompareState) -> LLMCompareState:
    messages = [
        SystemMessage(
            content="あなたは優秀なアシスタントです。以下の命題について丁寧に答えてください。"
        ),
        HumanMessage(content=state.question),
    ]
    response = openai_llm(messages)
    return state.copy(update={"openai_response": response.content.strip()})


# ✨ Gemini に問い合わせるノード
def ask_gemini(state: LLMCompareState) -> LLMCompareState:
    response = gemini_llm.generate_content(state.question)
    return state.copy(update={"gemini_response": response.text.strip()})


# 🔗 回答を統合・要約するノード（OpenAI 使用）
def merge_responses(state: LLMCompareState) -> LLMCompareState:
    messages = [
        SystemMessage(
            content="以下の複数の回答を参考に、矛盾がないように統合された簡潔で正確な回答を出力してください。"
        ),
        HumanMessage(
            content=f"OpenAIの回答: {state.openai_response}\nGeminiの回答: {state.gemini_response}"
        ),
    ]
    summary = openai_llm(messages)
    return state.copy(update={"merged_summary": summary.content.strip()})


# 🖨️ 結果出力ノード
def output_result(state: LLMCompareState) -> LLMCompareState:
    print("\n📝 ユーザーの質問:")
    print(state.question)
    print("\n🤖 OpenAIの回答:")
    print(state.openai_response)
    print("\n✨ Geminiの回答:")
    print(state.gemini_response)
    print("\n🔗 統合要約:")
    print(state.merged_summary)
    return state


# 🛠 グラフ構築
builder = StateGraph(LLMCompareState)

builder.add_node("AskOpenAI", ask_openai)
builder.add_node("AskGemini", ask_gemini)
builder.add_node("Merge", merge_responses)
builder.add_node("Output", output_result)

builder.set_entry_point("FanOut")

# FanOut ノードは実際の処理をしない中継ノード
builder.add_node("FanOut", lambda state: state)
builder.add_edge("FanOut", "AskOpenAI")
builder.add_edge("FanOut", "AskGemini")
builder.add_edge("AskOpenAI", "Merge")
builder.add_edge("AskGemini", "Merge")
builder.add_edge("Merge", "Output")
builder.add_edge("Output", END)

# グラフコンパイル
graph = builder.compile()

# 🚀 CLI 実行
if __name__ == "__main__":
    print("\n🔎 複数LLMによる質問応答と要約（LangGraph + OpenAI + Gemini）")
    question = input("質問を入力してください: ")
    state = LLMCompareState(question=question)
    graph.invoke(state)
    print("\n✅ 完了")
