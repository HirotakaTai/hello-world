# LangGraph FastAPI チャットボット

LangChainとLangGraphを使用して作られたシンプルなチャットボットアプリケーションです。FastAPIをバックエンドとして使用し、WebSocketによるリアルタイム通信を実現しています。

## 機能

- LangGraphを使用したチャット会話フロー管理
- FastAPIによるWebSocket対応バックエンド
- シンプルでモダンなWebインターフェース
- OpenAI APIを使用したチャット応答生成

## 前提条件

- Python 3.13+
- uv（パッケージマネージャー）

## セットアップ

1. このリポジトリをクローンします:

```bash
git clone <repository-url>
cd langgraph-fastapi-chatbot
```

2. 必要なライブラリをインストールします:

```bash
uv add langchain langchain-openai langgraph fastapi uvicorn python-dotenv jinja2
```

3. `.env`ファイルを作成し、OpenAI APIキーを設定します:

```
OPENAI_API_KEY=your_openai_api_key
```

## 実行方法

以下のコマンドでアプリケーションを起動します:

```bash
python main.py
```

ブラウザで `http://localhost:8000` にアクセスするとチャットボットのインターフェースが表示されます。

## プロジェクト構造

```
langgraph-fastapi-chatbot/
│
├── app/
│   ├── graph/
│   │   └── chatbot.py      # LangGraphのグラフ定義
│   │
│   ├── static/
│   │   ├── css/
│   │   │   └── style.css   # スタイルシート
│   │   └── js/
│   │       └── chat.js     # フロントエンドのJavaScript
│   │
│   ├── templates/
│   │   └── index.html      # メインのHTMLテンプレート
│   │
│   └── main.py             # FastAPIアプリケーション
│
├── .env                     # 環境変数（APIキーなど）
├── main.py                  # エントリーポイント
├── pyproject.toml           # プロジェクト設定
└── README.md                # このファイル
```

## カスタマイズ

- `app/graph/chatbot.py` でLangGraphのフロー定義とプロンプトをカスタマイズできます
- `app/templates/index.html` と `app/static/css/style.css` でUIをカスタマイズできます
- OpenAIのモデルを変更するには `app/graph/chatbot.py` 内の `model` パラメータを変更します
