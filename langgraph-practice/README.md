# LangGraph Practice - 高度な学習プロジェクト

LangGraph 0.2系を使用した高度なワークフロー、エラーハンドリング、ストリーミング処理を学習するためのプロジェクトです。

## 🎯 学習目標

- **複雑ワークフロー**: 条件分岐、並列処理、エージェント連携
- **エラーハンドリング**: リトライ戦略、フォールバック、回復処理
- **ストリーミング**: リアルタイムレスポンス表示とバッファリング
- **永続化**: SQLiteチェックポイントによる状態管理

## 🚀 セットアップ

### 1. プロジェクトのクローンと環境構築

```bash
# uvを使用した依存関係のインストール
uv sync

# 開発用依存関係も含める場合
uv sync --extra dev

# 環境変数の設定
cp .env.example .env
# .envファイルを編集してAPIキーを設定
```

### 2. 必須環境変数

```env
OPENAI_API_KEY=your_openai_api_key_here
```

## 📚 学習パス

### フェーズ1: 基礎（LangGraph 0.2系基本機能）

1. **基本状態グラフ** - `examples/01_basic_state_graph.py`
   - TypedDict状態定義
   - 基本ノードとエッジ

2. **チェックポイント永続化** - `examples/02_checkpoint_persistence.py`
   - SQLiteチェックポイント
   - セッション継続

3. **エラー処理とリトライ** - `examples/03_error_recovery.py`
   - 基本的なエラーハンドリング
   - リトライ戦略

### フェーズ2: 中級（0.2系応用機能）

4. **条件分岐エッジ** - `examples/04_conditional_edges.py`
   - 動的ルーティング
   - 条件判定ロジック

5. **ストリーミングレスポンス** - `examples/05_streaming_responses.py`
   - リアルタイム表示
   - Rich活用UI

6. **マルチエージェントとSend API** - `examples/06_multi_agent_send.py`
   - Send API活用
   - エージェント連携

### フェーズ3: 上級（0.2系高度機能）

7. **Human-in-the-loop** - `examples/07_human_in_the_loop.py`
   - 人間介入ポイント
   - インタラクティブワークフロー

## 🛠 使用方法

### 基本的なチャットボット起動

```bash
# 利用可能な例の一覧表示
uv run chat list-examples

# 基本状態グラフの例
uv run chat basic

# 特定の入力で実行
uv run chat basic --input "こんにちは"

# ストリーミングレスポンスの例
uv run chat streaming --input "テスト"

# 対話型チャット
uv run chat interactive
```

### 個別サンプルの直接実行

```bash
# 基本状態グラフの例
uv run python src/langgraph_practice/examples/01_basic_state_graph.py

# ストリーミングチャットの例
uv run python src/langgraph_practice/examples/05_streaming_responses.py
```

### テストの実行

```bash
# 全テスト実行
uv run pytest

# カバレッジ付きテスト
uv run pytest --cov=src/langgraph_practice

# 特定テスト実行
uv run pytest tests/test_core/
```

## 📁 プロジェクト構造

```
langgraph-practice/
├── src/langgraph_practice/
│   ├── core/           # コア機能（状態、エラーハンドリング）
│   ├── agents/         # マルチエージェント実装
│   ├── workflows/      # 複雑ワークフロー
│   ├── streaming/      # ストリーミング処理
│   └── utils/          # ユーティリティ
├── examples/           # 段階的学習サンプル
├── tests/             # テストコード
└── scripts/           # 実行スクリプト
```

## 🔧 技術スタック

- **LangGraph**: 0.2系 (安定版)
- **LangChain**: 0.2系互換
- **チェックポイント**: SQLite永続化
- **UI**: Rich (ターミナル表示)
- **CLI**: Typer
- **テスト**: pytest + pytest-asyncio

## 📖 重要概念

### チェックポイント永続化
```python
from langgraph.checkpoint.sqlite import SqliteSaver

# SQLiteチェックポイント設定
checkpointer = SqliteSaver.from_conn_string("checkpoints.db")
```

### Send API活用
```python
from langgraph.constants import Send

# マップ・リデュースパターン
def route_to_agents(state):
    return [Send("agent_1", {...}), Send("agent_2", {...})]
```

### エラーハンドリング
```python
# チェックポイントからの回復
# 部分実行の継続
# 人間介入ポイント設定
```

## 🤝 貢献

このプロジェクトは学習目的です。改善提案やバグ報告は歓迎します。

## 📄 ライセンス

MIT License