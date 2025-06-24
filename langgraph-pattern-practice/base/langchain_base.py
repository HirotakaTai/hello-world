from dotenv import load_dotenv
from langchain_openai import OpenAI

# ===== 環境変数の読み込み =====
load_dotenv()

# ===== OpenAI モデルのインスタンス化とメッセージの送信 =====
model = OpenAI(model="gpt-3.5-turbo-instruct", temperature=0)

# ===== メッセージの送信と応答の取得 =====
ai_message = model.invoke("こんにちは")
print(ai_message)
