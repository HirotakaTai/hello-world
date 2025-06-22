from typing import Dict, Any, List, TypedDict
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import StateGraph, END

# .envファイルから環境変数を読み込む
load_dotenv()

# APIキーの確認
api_key = os.environ.get("OPENAI_API_KEY")
print(f"OpenAI API Key available: {api_key is not None and len(api_key) > 0}")

# 状態の型定義
class ChatState(TypedDict):
    messages: List[Dict[str, Any]]
    current_response: str

# チャットモデルの初期化
try:
    print("OpenAIクライアントの初期化を開始")
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",  # 適宜変更してください
        temperature=0.7
    )
    print("OpenAIクライアントの初期化が成功しました")
except Exception as e:
    print(f"OpenAIクライアントの初期化に失敗しました: {str(e)}")
    raise e

# プロンプトテンプレートの定義
prompt = ChatPromptTemplate.from_messages([
    ("system", "あなたは親切で役立つAIアシスタントです。ユーザーの質問に対して丁寧に回答してください。"),
    MessagesPlaceholder(variable_name="messages"),
    ("human", "{input}")
])

# LLMチェーンの定義
chain = prompt | llm

# チャットボットの応答生成関数
def generate_response(state: ChatState) -> ChatState:
    """LLMを使用して応答を生成する"""
    print("LangGraph: 応答生成開始")
    
    try:
        # メッセージ履歴の取得
        messages = state["messages"]
        print(f"LangGraph: メッセージ履歴取得 - 履歴数: {len(messages)}")
        
        # 最新のユーザーメッセージを取得
        last_message = messages[-1].get("content", "") if messages else ""
        print(f"LangGraph: 最新メッセージ - {last_message[:30]}...")
        
        # LLMに入力する形式に変換
        chat_history = []
        for msg in messages[:-1]:  # 最後のメッセージを除く全履歴
            if msg["role"] == "user":
                chat_history.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                chat_history.append(AIMessage(content=msg["content"]))
        print(f"LangGraph: チャット履歴変換完了 - 履歴数: {len(chat_history)}")
        
        # LLMで応答を生成
        print("LangGraph: OpenAI API呼び出し開始")
        response = chain.invoke({
            "messages": chat_history,
            "input": last_message
        })
        print("LangGraph: OpenAI API呼び出し完了")
        
        # 応答を状態に保存
        state["current_response"] = response.content
        print(f"LangGraph: 応答を状態に保存 - 応答: {response.content[:30]}...")
        
        return state
    except Exception as e:
        print(f"LangGraph エラー: {str(e)}")
        state["current_response"] = f"エラーが発生しました: {str(e)}"
        return state
    
    # この部分は上のtryブロック内に移動したので削除
    pass

# グラフの作成
def create_chat_graph():
    """チャットボットのグラフを作成する"""
    # 空の状態グラフを初期化
    workflow = StateGraph(ChatState)
    
    # ノードの追加
    workflow.add_node("generate_response", generate_response)
    
    # エッジの追加（シンプルな直線フロー）
    workflow.set_entry_point("generate_response")
    workflow.add_edge("generate_response", END)
    
    # グラフをコンパイル
    graph = workflow.compile()
    
    return graph

# チャットボットインスタンス生成
chat_graph = create_chat_graph()
