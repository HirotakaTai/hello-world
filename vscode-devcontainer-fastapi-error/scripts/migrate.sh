#!/bin/bash
# データベースマイグレーション実行スクリプト

echo "データベースマイグレーションを実行しています..."
uv run alembic upgrade head
