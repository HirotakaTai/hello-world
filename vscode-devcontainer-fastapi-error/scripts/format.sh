#!/bin/bash
# コードフォーマット実行スクリプト

echo "コードフォーマットを実行しています..."
uv run black app/ tests/
uv run ruff check app/ tests/ --fix
