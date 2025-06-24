# Anthropic AIエージェント作成パターン実装

このプロジェクトは、Anthropic社が公開している「[Building effective agents](https://www.anthropic.com/engineering/building-effective-agents)」で紹介されているAIエージェント作成パターンの学習用サンプル実装です。

## 📁 プロジェクト構造

```
langgraph-pattern-practice/
├── base/
│   └── langchain_base.py          # LangChainの基本実装
├── patterns/                      # 各パターンの基本実装
│   ├── 01_augmented_llm/
│   │   └── augmented_llm.py       # 拡張LLMパターン
│   ├── 02_prompt_chaining/
│   │   └── prompt_chaining.py     # プロンプトチェーンパターン
│   ├── 03_routing/
│   │   └── routing.py             # ルーティングパターン
│   ├── 04_parallelization/
│   │   └── parallelization.py     # 並列化パターン
│   ├── 05_orchestrator_workers/
│   │   └── orchestrator_workers.py # オーケストレーター・ワーカーパターン
│   ├── 06_evaluator_optimizer/
│   │   └── evaluator_optimizer.py # 評価者・最適化者パターン
│   └── 07_agents/
│       └── agents.py              # 自律エージェントパターン
├── langgraph_patterns/            # LangGraphを活用した高度な実装
│   ├── 01_augmented_llm/
│   │   └── langgraph_augmented_llm.py
│   ├── 02_prompt_chaining/
│   │   └── langgraph_prompt_chaining.py
│   ├── 03_routing/
│   │   └── langgraph_routing.py
│   ├── 04_parallelization/
│   │   └── langgraph_parallelization.py
│   ├── 05_orchestrator_workers/
│   │   └── langgraph_orchestrator_workers.py
│   ├── 06_evaluator_optimizer/
│   │   └── langgraph_evaluator_optimizer.py
│   └── 07_agents/
│       └── langgraph_agents.py
├── main.py                        # パターン選択式ランチャー
├── pyproject.toml
├── README.md
└── .env                          # 環境変数（OpenAI APIキーなど）
```

## 🚀 セットアップ

### 1. 必要なパッケージのインストール

```bash
# uvを使用する場合
uv pip install langchain langchain-openai langchain-community langgraph python-dotenv requests

# pipを使用する場合
pip install langchain langchain-openai langchain-community langgraph python-dotenv requests
```

### 2. 環境変数の設定

`.env`ファイルを作成し、OpenAI APIキーを設定してください：

```bash
OPENAI_API_KEY=your_openai_api_key_here
```

## 📚 実装パターン

### 🔰 基本実装（patterns/フォルダ）

各パターンの基本的な実装です。LangChainを使用して、理解しやすい形で実装されています。

### 1. Augmented LLM（拡張LLM）
**ファイル**: `patterns/01_augmented_llm/augmented_llm.py`

基本的なLLMに外部ツール、検索機能、メモリ機能を追加したパターン。

**特徴**:
- 計算ツール、天気情報取得、現在時刻取得
- 会話履歴の保持
- エージェントベースのツール使用

**実行方法**:
```bash
python patterns/01_augmented_llm/augmented_llm.py
```

### 2. Prompt Chaining（プロンプトチェーン）
**ファイル**: `patterns/02_prompt_chaining/prompt_chaining.py`

複雑なタスクを複数のステップに分割し、各ステップの出力を次のステップの入力として使用するパターン。

**特徴**:
- ブログ記事作成のワークフロー（分析→アウトライン→評価→執筆→校正）
- 翻訳・改善チェーン
- 実行ログの記録

**実行方法**:
```bash
python patterns/02_prompt_chaining/prompt_chaining.py
```

### 3. Routing（ルーティング）
**ファイル**: `patterns/03_routing/routing.py`

入力を分類して適切な専門的なワークフローに振り分けるパターン。

**特徴**:
- カスタマーサポートクエリの自動分類
- コンテンツタイプの判定
- 専門ハンドラーによる処理

**実行方法**:
```bash
python patterns/03_routing/routing.py
```

### 4. Parallelization（並列化）
**ファイル**: `patterns/04_parallelization/parallelization.py`

タスクを並列で実行することで効率性を向上させるパターン。

**特徴**:
- セクショニング: 文書を分割して並列処理
- 投票: 複数の評価者による並列評価
- 並列コードレビュー

**実行方法**:
```bash
python patterns/04_parallelization/parallelization.py
```

### 5. Orchestrator-workers（オーケストレーター・ワーカー）
**ファイル**: `patterns/05_orchestrator_workers/orchestrator_workers.py`

中央のオーケストレーターが動的にタスクを分解し、複数のワーカーに配布するパターン。

**特徴**:
- 動的なタスク分解
- 専門化されたワーカー（研究者、分析者、ライター、コーダーなど）
- 依存関係の管理

**実行方法**:
```bash
python patterns/05_orchestrator_workers/orchestrator_workers.py
```

### 6. Evaluator-optimizer（評価者・最適化者）
**ファイル**: `patterns/06_evaluator_optimizer/evaluator_optimizer.py`

生成→評価→改善のループを通じて品質を向上させるパターン。

**特徴**:
- 反復的な品質向上
- 専門的な評価基準
- 翻訳品質最適化、文章作成最適化

**実行方法**:
```bash
python patterns/06_evaluator_optimizer/evaluator_optimizer.py
```

### 7. Agents（自律エージェント）
**ファイル**: `patterns/07_agents/agents.py`

LLMが自律的に判断し、ツールを使用して複雑なタスクを実行するパターン。

**特徴**:
- 自律的な意思決定
- 動的なツール選択
- 目標達成まで継続実行

**実行方法**:
```bash
python patterns/07_agents/agents.py
```

## 🌟 LangGraph高度実装（langgraph_patterns/フォルダ）

LangGraphを活用した、より高度で実用的な実装です。複雑なワークフロー管理や状態管理に優れています。

### 1. LangGraph Augmented LLM
**ファイル**: `langgraph_patterns/01_augmented_llm/langgraph_augmented_llm.py`

**LangGraphの特徴を活用**:
- StateGraphによる明確なワークフロー定義
- ツール実行の自動化
- 条件分岐による動的な処理経路選択
- 実行状態の完全な管理

**実行方法**:
```bash
python langgraph_patterns/01_augmented_llm/langgraph_augmented_llm.py
```

### 2. LangGraph Prompt Chaining
**ファイル**: `langgraph_patterns/02_prompt_chaining/langgraph_prompt_chaining.py`

**LangGraphの特徴を活用**:
- 線形ワークフローの明確な定義
- 各ステップ間の状態受け渡し
- 実行ログの自動管理
- エラーハンドリングの改善

**実行方法**:
```bash
python langgraph_patterns/02_prompt_chaining/langgraph_prompt_chaining.py
```

### 3. LangGraph Routing
**ファイル**: `langgraph_patterns/03_routing/langgraph_routing.py`

**LangGraphの特徴を活用**:
- 条件分岐による動的ルーティング
- 複数の処理経路の管理
- 分類結果に基づく自動振り分け
- 各経路での専門的な処理

**実行方法**:
```bash
python langgraph_patterns/03_routing/langgraph_routing.py
```

### 4. LangGraph Parallelization
**ファイル**: `langgraph_patterns/04_parallelization/langgraph_parallelization.py`

**LangGraphの特徴を活用**:
- 複数のワークフローグラフ（セクショニング、投票、レビュー）
- 非同期処理による真の並列実行
- 結果統合の自動化
- 処理タイプ別の最適化

**実行方法**:
```bash
python langgraph_patterns/04_parallelization/langgraph_parallelization.py
```

### 5. LangGraph Orchestrator-Workers
**ファイル**: `langgraph_patterns/05_orchestrator_workers/langgraph_orchestrator_workers.py`

**LangGraphの特徴を活用**:
- 動的なタスク管理とワーカー割り当て
- 依存関係の自動解決
- 条件分岐による継続判定
- 複雑なプロジェクト管理ワークフロー

**実行方法**:
```bash
python langgraph_patterns/05_orchestrator_workers/langgraph_orchestrator_workers.py
```

### 6. LangGraph Evaluator-Optimizer
**ファイル**: `langgraph_patterns/06_evaluator_optimizer/langgraph_evaluator_optimizer.py`

**LangGraphの特徴を活用**:
- 生成→評価→改善のループ制御
- 目標スコア達成までの自動反復
- 複数最適化タイプ（文章、翻訳、コード）
- 評価履歴の完全な追跡

**実行方法**:
```bash
python langgraph_patterns/06_evaluator_optimizer/langgraph_evaluator_optimizer.py
```

### 7. LangGraph Agents
**ファイル**: `langgraph_patterns/07_agents/langgraph_agents.py`

**LangGraphの特徴を活用**:
- 自律的な意思決定フロー
- 複数ツールの統合管理
- 動的なアクション選択と実行
- 進捗監視と継続判定

**実行方法**:
```bash
python langgraph_patterns/07_agents/langgraph_agents.py
```

## 🎯 学習のポイント

### 初心者向け（patterns/フォルダから開始）
1. まず `01_augmented_llm` から始めて、基本的なツール使用を理解
2. `02_prompt_chaining` で段階的なタスク分解を学習
3. `03_routing` で条件分岐の実装を理解

### 中級者向け
1. `04_parallelization` で並列処理の効果を体験
2. `05_orchestrator_workers` で複雑なワークフロー管理を学習
3. `06_evaluator_optimizer` で品質改善プロセスを理解

### 上級者向け（langgraph_patterns/フォルダで実践）
1. LangGraphの状態管理と条件分岐を理解
2. 複雑なワークフローの設計と実装
3. 実用的なエージェントシステムの構築
4. 各パターンを組み合わせたハイブリッドシステムの開発

## 🔧 カスタマイズ

### 新しいツールの追加
`07_agents/agents.py`の`AgentTool`クラスを継承して新しいツールを作成できます：

```python
class CustomTool(AgentTool):
    def __init__(self):
        super().__init__(
            name="custom_tool",
            description="カスタムツールの説明",
            parameters={"param1": {"type": "string", "description": "パラメータの説明"}}
        )
    
    def execute(self, param1: str) -> ToolResult:
        # ツールの実装
        pass
```

### 新しいワーカータイプの追加
`05_orchestrator_workers/orchestrator_workers.py`に新しいワーカータイプを追加：

```python
class WorkerType:
    # 既存のワーカータイプ...
    CUSTOM_WORKER = "custom_worker"
```

### 評価基準のカスタマイズ
`06_evaluator_optimizer/evaluator_optimizer.py`の`EvaluationCriteria`クラスに新しい評価基準を追加できます。

## ⚡ LangGraphの利点

### 1. 明確なワークフロー定義
- StateGraphによる視覚的で理解しやすいフロー
- ノードとエッジによる処理ステップの明確化

### 2. 高度な条件分岐
- 動的な経路選択
- 複雑な条件に基づく処理分岐

### 3. 状態管理
- TypedDictによる型安全な状態管理
- 各ステップ間での情報の確実な受け渡し

### 4. エラーハンドリング
- 各ノードでのエラー処理
- 失敗時の代替経路設定

### 5. 並列処理サポート
- 複数ワークフローの並行実行
- 非同期処理による性能向上

## 📊 パフォーマンス考慮事項

- **API使用量**: 各パターンは複数のLLM呼び出しを行うため、APIコストに注意
- **実行時間**: LangGraphパターンは構造化により若干のオーバーヘッドがありますが、並列化により全体的には高速
- **メモリ使用量**: 状態管理により基本パターンより多くのメモリを使用する場合があります

## 🐛 トラブルシューティング

### よくある問題

1. **APIキーエラー**: `.env`ファイルにOpenAI APIキーが正しく設定されているか確認
2. **パッケージインポートエラー**: `langgraph`パッケージが正しくインストールされているか確認
3. **レート制限エラー**: OpenAI APIのレート制限に達した場合は、しばらく待ってから再実行

### デバッグのヒント

- 各パターンには詳細なログ出力が含まれています
- LangGraphパターンでは実行ログで各ステップの状態を確認可能
- エラー時は実行ログを確認して問題の箇所を特定

## 📖 参考資料

- [Anthropic - Building effective agents](https://www.anthropic.com/engineering/building-effective-agents)
- [LangChain Documentation](https://python.langchain.com/)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)

## 🤝 貢献

このプロジェクトは学習目的で作成されています。改善提案やバグ報告は歓迎します。

## 📄 ライセンス

このプロジェクトはMITライセンスの下で公開されています。
