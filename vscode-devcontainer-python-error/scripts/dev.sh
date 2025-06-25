#!/bin/bash
# 開発サーバー起動スクリプト

echo "FastAPIアプリケーションを起動しています..."
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
