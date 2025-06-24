# Anthropic AIエージェント作成パターン実装

このプロジェクトは、Anthropic社が公開している「[Building effective agents](https://www.anthropic.com/engineering/building-effective-agents)」で紹介されているAIエージェント作成パターンの学習用サンプル実装です。

## 📁 プロジェクト構造

```
langgraph-pattern-practice/
├── base/
│   └── langchain_base.py          # LangChainの基本実装
├── patterns/                      # 各パターンの実装
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
├── main.py
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

## 🎯 学習のポイント

### 初心者向け
1. まず `01_augmented_llm` から始めて、基本的なツール使用を理解
2. `02_prompt_chaining` で段階的なタスク分解を学習
3. `03_routing` で条件分岐の実装を理解

### 中級者向け
1. `04_parallelization` で並列処理の効果を体験
2. `05_orchestrator_workers` で複雑なワークフロー管理を学習
3. `06_evaluator_optimizer` で品質改善プロセスを理解

### 上級者向け
1. `07_agents` で自律エージェントの実装を学習
2. 各パターンを組み合わせたハイブリッドシステムの構築
3. 独自のツールやワーカーの開発

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

## 📊 パフォーマンス考慮事項

- **API使用量**: 各パターンは複数のLLM呼び出しを行うため、APIコストに注意
- **実行時間**: 並列化パターンは処理時間を短縮しますが、複雑なパターンは時間がかかる場合があります
- **メモリ使用量**: 大量のデータを扱う場合は、メモリ使用量に注意してください

## 🐛 トラブルシューティング

### よくある問題

1. **APIキーエラー**: `.env`ファイルにOpenAI APIキーが正しく設定されているか確認
2. **パッケージインポートエラー**: 必要なパッケージがすべてインストールされているか確認
3. **レート制限エラー**: OpenAI APIのレート制限に達した場合は、しばらく待ってから再実行

### デバッグのヒント

- 各パターンには詳細なログ出力が含まれています
- `verbose=True`オプションを使用してより詳細な情報を取得
- エラー時は実行ログを確認して問題の箇所を特定

## 📖 参考資料

- [Anthropic - Building effective agents](https://www.anthropic.com/engineering/building-effective-agents)
- [LangChain Documentation](https://python.langchain.com/)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)

## 🤝 貢献

このプロジェクトは学習目的で作成されています。改善提案やバグ報告は歓迎します。

## 📄 ライセンス

このプロジェクトはMITライセンスの下で公開されています。
