from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import json
import uvicorn
from pathlib import Path
from app.graph.chatbot import chat_graph

# アプリケーションの作成
app = FastAPI(title="LangGraph FastAPI Chatbot")

# 静的ファイルとテンプレートの設定
BASE_DIR = Path(__file__).resolve().parent
print(f"BASE_DIR: {BASE_DIR}")
templates_dir = BASE_DIR / "templates"
static_dir = BASE_DIR / "static"
print(f"Templates directory: {templates_dir}")
print(f"Static directory: {static_dir}")

templates = Jinja2Templates(directory=str(templates_dir))
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# アクティブなWebSocketコネクションを保存する辞書
active_connections = {}

@app.get("/", response_class=HTMLResponse)
async def get_home(request: Request):
    """ホームページを表示する"""
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """WebSocketエンドポイント"""
    print(f"WebSocket接続試行: client_id={client_id}, path={websocket.url.path}")
    
    try:
        await websocket.accept()
        print(f"WebSocket接続確立: client_id={client_id}")
        active_connections[client_id] = websocket
        
        # チャット状態の初期化
        chat_state = {"messages": [], "current_response": ""}
        
        while True:
            # クライアントからメッセージを受信
            print(f"メッセージ待機中: client_id={client_id}")
            data = await websocket.receive_text()
            print(f"メッセージ受信: client_id={client_id}, data={data[:50]}...")
            message = json.loads(data)
            user_message = message.get("message", "")
            
            # メッセージを状態に追加
            chat_state["messages"].append({"role": "user", "content": user_message})
            print(f"LangGraph実行開始: client_id={client_id}, message={user_message}")
            
            try:
                # LangGraphを実行して応答を生成
                result = chat_graph.invoke(chat_state)
                print(f"LangGraph実行成功: client_id={client_id}")
                
                # 応答を取得
                response = result["current_response"]
            except Exception as e:
                print(f"LangGraph実行エラー: client_id={client_id}, error={str(e)}")
                raise e
            
            # 応答をメッセージ履歴に追加
            chat_state["messages"].append({"role": "assistant", "content": response})
            
            # 応答をクライアントに送信
            await websocket.send_json({
                "response": response,
                "status": "success"
            })
            
    except WebSocketDisconnect:
        # 切断された場合、コネクションを削除
        print(f"WebSocket切断: client_id={client_id}")
        if client_id in active_connections:
            del active_connections[client_id]
    except Exception as e:
        # エラーが発生した場合、エラーメッセージを送信
        print(f"WebSocketエラー: client_id={client_id}, error={str(e)}, type={type(e)}")
        try:
            await websocket.send_json({
                "response": f"エラーが発生しました: {str(e)}",
                "status": "error"
            })
        except Exception as send_error:
            print(f"エラー応答送信失敗: {str(send_error)}")

# アプリケーションを直接実行した場合
if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
