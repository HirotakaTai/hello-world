#!/bin/bash
# データベース初期化スクリプト

echo "データベースの初期化を開始します..."

# PostgreSQLの起動を待つ
echo "PostgreSQLの起動を待っています..."
until pg_isready -h localhost -p 5432 -U postgres; do
    echo "PostgreSQLの起動を待っています..."
    sleep 2
done

echo "PostgreSQL が起動しました！"

# 初期マイグレーションの作成
echo "初期マイグレーションを作成しています..."
uv run alembic revision --autogenerate -m "Initial migration"

# マイグレーションの実行
echo "マイグレーションを実行しています..."
uv run alembic upgrade head

# pre-commitの設定
echo "pre-commitを設定しています..."
uv run pre-commit install

echo "データベースの初期化が完了しました！"
echo ""
echo "アプリケーションを起動する準備ができました："
echo "  ./scripts/dev.sh"
