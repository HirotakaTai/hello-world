# FastAPI LangGraph Chat Sample

LangGraphを使用したAIエージェントチャットボットのFastAPIサンプルアプリケーション

## 概要

このプロジェクトは、以下の技術を組み合わせたAIチャットボットアプリケーションです：

- **FastAPI**: 高性能なPython Webフレームワーク
- **LangGraph**: LangChainベースのエージェント実装フレームワーク
- **OpenAI API**: GPT-3.5-turboを使用したAI応答生成
- **uv**: 高速なPythonパッケージマネージャー

## 技術スタック

### バックエンド
- **FastAPI** 0.116.1+
- **LangGraph** 0.5.2+
- **LangChain OpenAI** 0.3.28+
- **Python** 3.11+
- **uvicorn** (ASGIサーバー)

### フロントエンド
- **HTML/JavaScript** (バニラJS)
- **CSS** (レスポンシブデザイン)

### 開発ツール
- **uv** (パッケージマネージャー)
- **Ruff** (リンター・フォーマッター)
- **Black** (コードフォーマッター)
- **MyPy** (型チェック)

## セットアップ手順

### 1. 前提条件

- Python 3.11以上
- uv パッケージマネージャー
- OpenAI APIキー

### 2. uvのインストール

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 3. プロジェクトのセットアップ

```bash
# リポジトリをクローン
git clone <repository-url>
cd fastapi-langgraph-chat-sample

# 依存関係をインストール
uv sync

# 環境変数ファイルを作成
cp .env.example .env
```

### 4. 環境変数の設定

`.env` ファイルを編集し、OpenAI APIキーを設定してください：

```env
# OpenAI API設定
OPENAI_API_KEY=your_openai_api_key_here

# LangGraph設定
LANGCHAIN_TRACING_V2=false
LANGCHAIN_API_KEY=your_langchain_api_key_here

# アプリケーション設定
DEBUG=true
HOST=0.0.0.0
PORT=8000

# ログ設定
LOG_LEVEL=INFO

# OpenAI API レート制限設定
OPENAI_RATE_LIMIT_REQUESTS_PER_SECOND=10
OPENAI_MAX_RETRIES=3
```

### 5. 開発サーバーの起動

```bash
# 仮想環境をアクティベート
source .venv/bin/activate

# 開発サーバーを起動
uv run uvicorn src.fastapi_langgraph_chat.main:app --reload --host 0.0.0.0 --port 8000
```

## 使用方法

### Webページでのチャット

1. ブラウザで `http://localhost:8000` にアクセス
2. チャット画面が表示されます
3. メッセージを入力して「送信」ボタンをクリック
4. AIからの応答が表示されます

### API仕様

#### チャットエンドポイント

```http
POST /api/chat
Content-Type: application/json

{
  "message": "こんにちは",
  "conversation_id": "optional-conversation-id",
  "context": {}
}
```

#### レスポンス

```json
{
  "response": "こんにちは！何かお手伝いできることはありますか？",
  "conversation_id": "generated-uuid",
  "timestamp": "2024-01-01T12:00:00.000Z",
  "metadata": {
    "step_count": 1,
    "context": {}
  }
}
```

#### その他のエンドポイント

- `GET /health` - ヘルスチェック
- `GET /docs` - API仕様（Swagger UI）
- `GET /redoc` - API仕様（ReDoc）
- `GET /api/debug/agent-cache` - エージェントキャッシュ統計（デバッグ用）

## 開発コマンド

### 基本コマンド

```bash
# 開発サーバー起動
uv run uvicorn src.fastapi_langgraph_chat.main:app --reload

# 依存関係の同期
uv sync

# パッケージの追加
uv add <package-name>

# 開発用パッケージの追加
uv add --dev <package-name>
```

### コード品質管理

```bash
# コードフォーマット
uv run ruff format src/

# リンターチェック
uv run ruff check src/

# 型チェック
uv run mypy src/

# テスト実行
uv run pytest tests/
```

## プロジェクト構造

```
fastapi-langgraph-chat-sample/
├── src/
│   └── fastapi_langgraph_chat/
│       ├── __init__.py
│       ├── main.py          # FastAPIアプリケーション
│       ├── routes.py        # APIルーティング
│       ├── models.py        # データモデル
│       ├── agent.py         # LangGraphエージェント
│       └── logging_config.py # ロギング設定
├── static/
│   └── index.html          # チャットUI
├── tests/                  # テストファイル
├── docs/                   # 技術ドキュメント
│   └── CONCURRENCY_DESIGN.md # 並行処理設計ドキュメント
├── logs/                   # ログファイル出力先
├── .env.example           # 環境変数テンプレート
├── pyproject.toml         # プロジェクト設定
├── uv.lock               # 依存関係ロック
├── CLAUDE.md             # 開発ガイドライン
└── README.md             # このファイル
```

## 主要コンポーネント

### 1. FastAPIアプリケーション (`main.py`)

- FastAPIアプリケーションの初期化
- CORS設定
- 静的ファイル配信設定
- ルーター統合

### 2. LangGraphエージェント (`agent.py`)

- ChatOpenAIを使用したAI応答生成
- LangGraphを使用したエージェント処理フロー
- 会話履歴管理

### 3. APIルーティング (`routes.py`)

- チャットエンドポイント
- HTMLページ配信
- エラーハンドリング

### 4. データモデル (`models.py`)

- Pydanticモデル定義
- リクエスト/レスポンス型定義
- エージェント状態管理

### 5. ロギング設定 (`logging_config.py`)

- シンプルなログ設定
- コンソール・ファイル出力対応
- 日次ローテーション機能
- 環境変数による制御

## 開発時の注意点

### 1. 環境変数

- OpenAI APIキーは必須です
- `.env`ファイルは`.gitignore`に含まれています
- 本番環境では適切な環境変数管理を行ってください

### 2. CORS設定

- 開発環境では全てのオリジンを許可しています
- 本番環境では適切なオリジン設定を行ってください

### 3. エラーハンドリング

- OpenAI APIの制限やエラーに対する適切な処理を実装してください
- レート制限やトークン制限を考慮してください

### 4. ログ設定

- 本番環境では適切なログレベルとログ出力先を設定してください
- 機密情報がログに出力されないよう注意してください
- ログファイルは`logs/`ディレクトリに日次ローテーションで保存されます

### 5. 並行処理

- 複数人同時アクセスに対応済み（@lru_cacheによるスレッドセーフなシングルトン）
- 詳細は[並行処理設計ドキュメント](docs/CONCURRENCY_DESIGN.md)を参照

## トラブルシューティング

### よくある問題

1. **OpenAI APIエラー**
   - APIキーが正しく設定されているか確認
   - API使用量制限を確認
   - インターネット接続を確認

2. **依存関係エラー**
   - `uv sync`を実行して依存関係を更新
   - Python バージョンが3.11以上か確認

3. **ポートエラー**
   - ポート8000が使用中でないか確認
   - 他のポートを使用する場合は環境変数を変更

### デバッグ方法

```bash
# 詳細なログを出力
uv run uvicorn src.fastapi_langgraph_chat.main:app --reload --log-level debug

# 環境変数の確認
uv run python -c "import os; print(os.getenv('OPENAI_API_KEY'))"
```

## コントリビューション

1. このプロジェクトをフォーク
2. 機能ブランチを作成 (`git checkout -b feature/AmazingFeature`)
3. 変更をコミット (`git commit -m 'feat: すごい機能を追加'`)
4. ブランチにプッシュ (`git push origin feature/AmazingFeature`)
5. プルリクエストを作成

## ライセンス

このプロジェクトはMITライセンスの下で公開されています。

## 参考資料

- [FastAPI公式ドキュメント](https://fastapi.tiangolo.com/)
- [LangGraph公式ドキュメント](https://langchain-ai.github.io/langgraph/)
- [OpenAI API リファレンス](https://platform.openai.com/docs/api-reference)
- [uv公式ドキュメント](https://docs.astral.sh/uv/)