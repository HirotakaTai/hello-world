# VSCode DevContainer Python Project

このプロジェクトは、VSCode DevContainerを使用した開発環境と、本番環境へのDockerベースのデプロイメントをサポートしています。

## 開発環境

### 前提条件
- Docker
- Visual Studio Code
- Dev Containers拡張機能

### 開発環境の起動
1. このリポジトリをクローン
2. VSCodeで開く
3. "Reopen in Container"を選択

## 本番環境

### Docker イメージのビルド
```bash
make build
```

### ローカルでの実行
```bash
make run
```

### 本番環境へのデプロイ
```bash
make deploy
```

## テスト実行
```bash
make test
```

## 利用可能なコマンド
```bash
make help
```

## CI/CD
GitHub Actionsを使用してCI/CDパイプラインが設定されています：
- テストの自動実行
- Dockerイメージの自動ビルド・プッシュ
- セキュリティスキャン

## セキュリティ
- 非rootユーザーでの実行
- マルチステージビルドによる軽量イメージ
- セキュリティスキャンの統合
- 環境変数による設定管理
