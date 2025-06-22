# LangChain Practice

LangChainとLangGraphを使用した学習プロジェクトです。

## 機能

### 1. 基本的なLLM呼び出し (`main.py`)
OpenAI GPTモデルとの基本的な会話機能

### 2. LangGraphによる計算ワークフロー (`langgraph-practice.py`)
StateGraphを使用した四則演算の繰り返し処理

### 3. 自然言語計算エージェント (`agent_calculator.py`)
自然言語で計算要求を処理できるLLMエージェント

## セットアップ

1. 仮想環境の作成と依存関係のインストール:
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# または .venv\Scripts\activate  # Windows
pip install -e .
```

2. 環境変数の設定:
`.env`ファイルにOpenAI APIキーを設定してください。

## 使用方法

### 基本的なLLM呼び出し
```bash
python main.py
```

### LangGraphワークフロー
```bash
python langgraph-practice.py
```

### 自然言語計算エージェント
```bash
python agent_calculator.py
```

エージェントモードでは以下のような自然言語で計算を依頼できます：
- "10と5を足して"
- "100を3で割った結果を教えて"
- "2の8乗を計算して"
- "複雑な計算: (10 + 5) × 3 - 7"

## 技術スタック

- **LangChain**: LLMアプリケーションフレームワーク
- **LangGraph**: ワークフロー管理
- **OpenAI GPT**: 言語モデル
- **Python 3.13+**: プログラミング言語
