#!/bin/bash
# 初期セットアップスクリプト

echo "初期セットアップを開始します..."

# 依存関係のインストール（ビルド分離を無効化）
echo "依存関係をインストールしています..."
uv sync --dev --no-build-isolation

# スクリプトに実行権限を付与
chmod +x scripts/*.sh

# 環境変数ファイルの確認
if [ ! -f .env ]; then
    echo ".envファイルが作成されました。必要に応じて設定を変更してください。"
else
    echo ".envファイルは既に存在します。"
fi

echo "基本セットアップが完了しました！"
echo ""
echo "次にデータベースのセットアップを行います："
echo "  1. PostgreSQLが起動するまで少し待ってください"
echo "  2. 以下のコマンドでマイグレーションを実行："
echo "     ./scripts/migrate.sh"
echo ""
echo "アプリケーションの起動："
echo "  ./scripts/dev.sh"
echo ""
echo "または以下のVS Codeタスクを使用してください："
echo "  - Ctrl+Shift+P → Tasks: Run Task → Start FastAPI Server"
