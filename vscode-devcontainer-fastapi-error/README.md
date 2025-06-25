# FastAPI Application with PostgreSQL

このプロジェクトは、FastAPIとPostgreSQLを使用したWebアプリケーション開発用のDevContainer環境です。

## 📋 要件

- Docker Desktop
- Visual Studio Code
- Dev Containers拡張機能

## 🚀 クイックスタート

### 1. 環境のセットアップ

```bash
# リポジトリをクローン
git clone <repository-url>
cd fastapi-app

# VS Codeで開く
code .
```

### 2. Dev Containerでの開発環境構築

1. VS Codeで「Dev Containers: Reopen in Container」コマンドを実行
2. コンテナが構築され、自動的に依存関係がインストールされます
3. **初回のみ**: データベースの初期化を実行
   ```bash
   ./scripts/init-db.sh
   ```

### 3. アプリケーションの起動

```bash
# 開発サーバーの起動
./scripts/dev.sh

# または直接コマンド実行
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

アプリケーションは http://localhost:8000 でアクセス可能です。

## 📁 プロジェクト構造

```
.
├── app/                    # メインアプリケーション
│   ├── api/               # API エンドポイント
│   │   └── api_v1/        # API v1
│   │       ├── endpoints/ # 各エンドポイント
│   │       └── api.py     # ルーター統合
│   ├── core/              # 設定とセキュリティ
│   ├── db/                # データベース設定
│   ├── models/            # SQLAlchemy モデル
│   ├── schemas/           # Pydantic スキーマ
│   ├── services/          # ビジネスロジック
│   └── main.py            # FastAPI アプリケーション
├── migrations/            # Alembic マイグレーション
├── tests/                 # テストファイル
├── scripts/               # 便利スクリプト
├── .devcontainer/         # Dev Container設定
├── .env                   # 環境変数（ローカル開発用）
├── pyproject.toml         # プロジェクト設定
└── api_test.http          # REST Client テストファイル
```

## 🛠️ 開発コマンド

### 初回セットアップ（コンテナ起動後に一度だけ実行）

```bash
# データベースの初期化
./scripts/init-db.sh
```

### データベース操作

```bash
# マイグレーション作成
uv run alembic revision --autogenerate -m "マイグレーション名"

# マイグレーション実行
./scripts/migrate.sh
# または
uv run alembic upgrade head

# マイグレーション戻し
uv run alembic downgrade -1
```

### テスト実行

```bash
# テスト実行
./scripts/test.sh
# または
uv run pytest tests/ -v

# カバレッジ付きテスト実行
uv run pytest tests/ -v --cov=app --cov-report=html
```

### コード品質

```bash
# コードフォーマット
./scripts/format.sh
# または
uv run black app/ tests/
uv run ruff check app/ tests/ --fix

# 型チェック
uv run mypy app/

# pre-commitフック設定
uv run pre-commit install
```

## 🐘 データベース設定

### PostgreSQL接続情報

- **ホスト**: localhost
- **ポート**: 5432
- **ユーザー**: postgres
- **パスワード**: postgres
- **データベース**: postgres

### 接続方法

```bash
# psqlでの接続
psql -h localhost -p 5432 -U postgres -d postgres

# VS Code拡張機能でも接続可能
# 推奨拡張機能: PostgreSQL (Chris Kolkman)
```

## 📚 API ドキュメント

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/api/v1/openapi.json

## 🔧 設定

### 環境変数

`.env`ファイルで以下の設定が可能です：

```env
# データベース設定
POSTGRES_SERVER=localhost
POSTGRES_PORT=5432
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
POSTGRES_DB=postgres

# セキュリティ設定
SECRET_KEY=your-secret-key-change-in-production
ACCESS_TOKEN_EXPIRE_MINUTES=30

# CORS設定
BACKEND_CORS_ORIGINS=http://localhost:3000,http://localhost:8080

# 環境設定
ENVIRONMENT=development
```

### VS Code 設定

Dev Containerには以下の拡張機能が自動インストールされます：

- Python
- Black Formatter
- Ruff
- MyPy Type Checker
- Jupyter
- Docker
- REST Client

## 📖 使用技術

### バックエンド
- **FastAPI**: モダンで高速なWebフレームワーク
- **SQLAlchemy**: ORM
- **Alembic**: データベースマイグレーション
- **Pydantic**: データ検証とシリアライゼーション
- **asyncpg**: PostgreSQL非同期ドライバー

### 開発ツール
- **uv**: 高速なPythonパッケージマネージャー
- **Black**: コードフォーマッター
- **Ruff**: リンター
- **MyPy**: 型チェッカー
- **pytest**: テストフレームワーク
- **pre-commit**: Gitフック管理

## 🧪 テスト

### テストファイルの構造

```
tests/
├── __init__.py
├── conftest.py           # テスト設定
├── test_main.py          # メインアプリケーションテスト
└── test_api/             # APIテスト
    └── test_users.py     # ユーザーAPIテスト
```

### テスト実行例

```bash
# 全テスト実行
uv run pytest

# 特定ファイルのテスト実行
uv run pytest tests/test_main.py

# マーカー指定テスト実行
uv run pytest -m "not slow"

# 詳細出力
uv run pytest -v -s
```

## 📝 REST Client テスト

`api_test.http`ファイルを使用してAPIを直接テストできます：

1. VS CodeでREST Client拢張機能を有効化
2. `api_test.http`ファイルを開く
3. リクエストの上にある「Send Request」をクリック

## 🚨 トラブルシューティング

### よくある問題

1. **データベース接続エラー**
   ```bash
   # PostgreSQLコンテナの状態確認
   docker-compose ps
   
   # コンテナ再起動
   docker-compose restart db
   ```

2. **依存関係エラー**
   ```bash
   # 依存関係の再インストール
   uv sync --dev
   ```

3. **マイグレーションエラー**
   ```bash
   # マイグレーション状態確認
   uv run alembic current
   
   # マイグレーション履歴確認
   uv run alembic history
   ```

## 🤝 コントリビューション

1. フォークしてブランチを作成
2. 変更を実装
3. テストを追加・実行
4. コードフォーマットを実行
5. プルリクエストを作成

### コミット前チェックリスト

- [ ] テストが全て通る
- [ ] コードフォーマットが適用されている
- [ ] 型チェックでエラーがない
- [ ] 新機能にはテストが追加されている
- [ ] ドキュメントが更新されている

## 📄 ライセンス

このプロジェクトはMITライセンスの下で公開されています。

## 🆘 サポート

問題や質問がある場合は、GitHubのIssuesを作成してください。