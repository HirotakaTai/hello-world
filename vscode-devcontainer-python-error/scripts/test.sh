#!/bin/bash
# テスト実行スクリプト

echo "テストを実行しています..."
uv run pytest tests/ -v --cov=app --cov-report=html
