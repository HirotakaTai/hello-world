# Claude Code 開発指針

## 協調的開発プロセス

Claude Code での作業時は、人間との協調的な開発のために以下の原則に従ってください。

### 重要：編集前の確認プロセス

コードの編集を開始する前に、必ず以下の内容を人間に確認してください：

- **設計アプローチ**: 実装する機能の全体的な設計アプローチ
- **技術選択**: 使用する技術、ライブラリ、フレームワークの妥当性
- **実装範囲**: 今回の作業セッションの具体的な範囲と優先度
- **影響評価**: 既存のコードへの影響と破壊的変更の可能性

**例外**: 明らかなタイポの修正や軽微なフォーマット調整の場合のみ、確認なしで実行可能です。

### 段階的実装プロセス

複雑な機能や大規模な変更の場合は、以下のステップバイステップアプローチを採用してください：

#### フェーズ 1: 計画立案
- 実装を複数のステップに分割
- 各ステップの目的と成果物を明確化
- 依存関係と実装順序の確認

#### フェーズ 2: 段階的実装
- 1つのステップ完了後、人間に進捗と次のステップを確認
- 各段階で機能確認とテストの実施
- 問題が発生した場合は即座に人間に報告

#### フェーズ 3: 継続的確認
- 実装アプローチに変更が必要な場合は人間に相談
- パフォーマンスやセキュリティ上の懸念があれば事前に報告
- コードレビューのタイミングを人間と調整

### 改善提案プロセス

人間の指示に対してより良い代替案がある場合は、積極的に提案してください：

#### ライブラリ・フレームワークの提案
```
現在の指示: jQueryを使用してDOM操作を実装
提案: モダンなVanilla JSまたはReactの使用を提案
理由: パフォーマンス向上、保守性の改善、現代的な開発手法との整合性
```

#### アルゴリズム改善の提案
```
現在の指示: 単純なソートアルゴリズムを実装
提案: データサイズと用途に応じた最適なソートアルゴリズムを提案
理由: 時間計算量の改善、メモリ効率の最適化
```

#### デザインパターンの提案
```
現在の指示: 機能をクラスに直接実装
提案: StrategyパターンまたはFactoryパターンの適用を提案
理由: 拡張性の向上、テストの容易性、保守性の改善
```

### 提案時のコミュニケーション指針

#### 提案の構造
1. **現在の理解**: 人間の指示に対する理解を確認
2. **代替提案**: 最大3つの具体的な改善案を提示
3. **比較**: 各案のメリット・デメリットを明確に説明
4. **推奨案**: 理由と共に最も適切と考える案を推奨
5. **実装への影響**: 開発時間、学習コスト、保守性への影響を説明

#### 提案テンプレート
```
【現在の理解】
指示内容: [人間の指示の要約]

【改善提案】
案1: [代替案1]
- メリット: [具体的な利点]
- デメリット: [考慮すべき点]
- 実装コスト: [時間・リソースの見積もり]

案2: [代替案2]
...

【推奨案】
[推奨する案]を推奨します
理由: [技術的・ビジネス的な根拠]

【次のステップ】
ご意向をお聞かせください。元の指示通りに進めるか、提案案で進めるかを確認いたします。
```

### 品質保証の原則

#### コードの品質
- コード作成時は可読性を最優先にする
- 適切なコメントとドキュメントを追加する
- テストコードを同時に作成する
- セキュリティとパフォーマンスを考慮する

#### レビューポイント
- コードレビューのしやすさを考慮した実装
- 既存のコーディング規約との整合性
- 将来の拡張性を考慮した設計
- 適切なエラーハンドリングの実装

### 緊急時の対応

以下の場合は即座に人間に報告し、指示を仰いでください：

- セキュリティリスクを発見した場合
- データ損失の可能性がある変更の場合
- 本番環境に影響を与える可能性がある場合
- 予想より大幅に時間がかかることが判明した場合
- 指示通りの実装が技術的に困難な場合

---

## Bash コマンド

### Python開発コマンド（uv使用）
本プロジェクトは Python の FastAPI アプリケーションで、パッケージ管理には uv を使用します。
uv コマンドリファレンス: https://docs.astral.sh/uv/reference/cli/

- `uv run python -m app`: FastAPI アプリケーションを実行
- `uv run pytest`: テストスイートを実行
- `uv run pytest --cov`: カバレッジ付きテスト実行
- `uv run python -m pytest tests/`: テストを実行
- `uv run uvicorn app:app --reload`: 開発サーバーを起動
- `uv run python -m mypy .`: 型チェックを実行
- `uv run python -m black .`: コードフォーマッターを実行
- `uv run python -m ruff check .`: リンターを実行
- `uv run python -m ruff format .`: リンターフォーマッターを実行

### uv プロジェクト管理
- `uv init`: 新しいプロジェクトを初期化
- `uv add <package>`: パッケージを追加
- `uv add --dev <package>`: 開発用パッケージを追加
- `uv remove <package>`: パッケージを削除
- `uv sync`: 依存関係を同期
- `uv lock`: ロックファイルを更新
- `uv pip list`: インストール済みパッケージを一覧表示
- `uv python install`: Python バージョンを管理

### Git コマンド
- `git status`: リポジトリの状態を確認
- `git add .`: すべての変更をステージング
- `git commit -m "メッセージ"`: メッセージ付きでコミット
- `git push`: リモートリポジトリにプッシュ
- `git pull`: リモートリポジトリからプル

### プロジェクト固有のコマンド
- 発見したプロジェクト固有のコマンドをここに追加
- 特別なビルドやデプロイ手順を記録
- 環境固有の要件を記載

## コードスタイルガイドライン

### 一般原則
- 一貫したインデント（JavaScript/TypeScriptは2スペース、Pythonは4スペース）
- 説明的な変数名と関数名を使用
- 関数は小さく、単一責任に集中
- 明確な変数名で自己文書化されたコードを書く

### 言語固有のガイドライン
- **Python**: PEP 8 に従い、適切な場所で型ヒントを使用
- **FastAPI**: OpenAPI スキーマを意識したモデル設計、適切な HTTP ステータスコードの使用
- **LangGraph**: エージェントフローの適切な状態管理、ノードとエッジの明確な定義

## テスト指針

### テスト戦略
- すべての新機能にテストを記述
- 個別の関数/メソッドには単体テストを優先
- 複雑なワークフローには統合テストを使用
- テストカバレッジを80%以上に維持

### テストコマンド
- 単一テストファイルの実行: `uv run pytest tests/test_filename.py`
- ウォッチモードでテスト実行: `uv run pytest-watch`
- カバレッジ付きテスト実行: `uv run pytest --cov`
- 特定のテストマーカーで実行: `uv run pytest -m "unit"`

## リポジトリエチケット

### ブランチ管理
- 説明的なブランチ名を使用（feature/add-user-auth, fix/login-bug）
- ブランチは単一の機能や修正に集中
- マージ後はブランチを削除

### コミットメッセージ
- **必ず日本語で記載する**
- 従来型コミット形式を使用: type(scope): description
- タイプ: feat, fix, docs, style, refactor, test, chore
- 件名は50文字以内に維持
- 命令法を使用（「機能を追加」ではなく「機能を追加する」）
- 例: `feat(auth): ユーザー認証機能を追加`, `fix(api): レスポンスエラーを修正`

### マージ戦略
- フィーチャーブランチではsquash and mergeを優先
- リリースブランチでは通常のマージを使用
- マージ前に必ずコードレビューを実施

## 開発環境

### 前提条件
- Python 3.11 以上
- uv パッケージマネージャー
- 必要なグローバルツール: uv, git

### セットアップ手順
1. リポジトリをクローン
2. Python 環境の準備: `uv python install`
3. 依存関係をインストール: `uv sync`
4. 環境ファイルをコピー: `cp .env.example .env`
5. 環境変数を設定
6. 初期セットアップを実行: `uv run python -m app`（該当する場合）

### IDE設定
- 推奨VS Code拡張機能: Python, FastAPI, Python Type Checker
- BlackとRuffの設定は自動で適用
- VS CodeでプロジェクトのPythonインタープリターを使用

## プロジェクト構造

### 主要ディレクトリ
- `src/`: ソースコード
- `tests/` または `__tests__/`: テストファイル
- `docs/`: ドキュメント
- `scripts/`: ビルドとユーティリティスクリプト
- `public/`: 静的アセット

### 重要なファイル
- `pyproject.toml`: 依存関係とプロジェクト設定
- `uv.lock`: ロックファイル
- `requirements.txt`: 依存関係（存在する場合）
- `.env`: 環境変数設定
- `README.md`: プロジェクト概要とセットアップ

## ワークフロー設定

### 開発ワークフロー
1. 探索フェーズから開始 - まず関連ファイルを読む
2. コーディング前に実装計画を作成
3. 大きなタスクを小さくテスト可能なチャンクに分割
4. 開発中は頻繁にテストを実行
5. 明確なメッセージで早期かつ頻繁にコミット

### コードレビュープロセス
- 大きな変更前に人間のレビューを要求
- 複雑なロジックはコメントで説明
- 自明でない決定には文脈を提供
- マージ前にすべてのレビューフィードバックに対応

### パフォーマンス考慮事項
- パフォーマンスボトルネックのプロファイリング
- データベースクエリの最適化
- Webアプリケーションのバンドルサイズ最小化
- 適切なキャッシュ戦略の使用

## セキュリティガイドライン

### ベストプラクティス
- 機密データ（APIキー、パスワード）は絶対にコミットしない
- すべてのユーザー入力を検証
- 外部通信にはHTTPSを使用
- セキュリティパッチのために依存関係を最新に保つ

### コードセキュリティ
- データベース操作にはパラメータ化クエリを使用
- 適切な認証と認可を実装
- ユーザーに表示する前にデータをサニタイズ
- 最小権限の原則に従う

---

これらのガイドラインにより、Claude Code での協調的で効率的、高品質なソフトウェア開発を確保します。