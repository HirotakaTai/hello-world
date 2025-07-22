# Hello World

新規プロジェクトの作成サンプルと学習用実験コード置き場。

このリポジトリは、AI・機械学習および開発環境構築に関する様々なプロジェクトを含んでいます。

## 🤖 AI・機械学習プロジェクト

### [langchain-practice](./langchain-practice/)
LangChainとLangGraphを使用した学習プロジェクト。
- 基本的なLLM呼び出し機能
- StateGraphによる計算ワークフロー  
- 自然言語計算エージェント

### [langgraph-fastapi-chatbot](./langgraph-fastapi-chatbot/)
LangGraphとFastAPIを使用したチャットボットアプリケーション。
- WebSocketによるリアルタイム通信
- LangGraphによる会話フロー管理
- シンプルなWebインターフェース

### [langgraph-pattern-practice](./langgraph-pattern-practice/)
Anthropic社のAIエージェント作成パターンの実装サンプル。
- 拡張LLM、プロンプトチェーン、ルーティングなど7つのパターン
- LangGraphを活用した高度な実装
- 評価者・最適化者パターンの実装

## 🛠️ 開発環境プロジェクト

### [vscode-devcontainer-fastapi-error](./vscode-devcontainer-fastapi-error/)
FastAPIとPostgreSQLを使用したWebアプリケーション開発環境。
- DevContainer設定
- Alembicによるデータベースマイグレーション
- テスト・CI/CD環境の構築

### [vscode-devcontainer-python](./vscode-devcontainer-python/)
VSCode DevContainerを使用したPython開発環境。
- Docker本番環境サポート
- GitHub Actionsによる自動CI/CD
- セキュリティスキャンとマルチステージビルド

### [vscode-uv-python-default](./vscode-uv-python-default/)
uvパッケージマネージャーを使用したPython開発環境のシンプルな設定。

### [vscode-default-profile](./vscode-default-profile/)
Visual Studio Codeの既定プロファイル設定。

## 技術スタック

- **AI・機械学習**: LangChain, LangGraph, OpenAI API
- **Webフレームワーク**: FastAPI, WebSocket
- **データベース**: PostgreSQL
- **開発環境**: Docker, VSCode DevContainer
- **パッケージ管理**: uv, Poetry
- **CI/CD**: GitHub Actions

## 使用方法

各プロジェクトの詳細な使用方法については、各ディレクトリ内のREADME.mdを参照してください。