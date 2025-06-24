#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AIエージェントパターン実践プロジェクト - メインランチャー
Anthropic社の「Building effective agents」で紹介されている
AIエージェント作成パターンの学習・実践用プロジェクトです。
"""

import importlib.util
import os


def load_pattern_module(pattern_name: str, file_name: str):
    """指定されたパターンのモジュールを動的に読み込みます。"""
    pattern_path = f"patterns/{pattern_name}/{file_name}"
    if not os.path.exists(pattern_path):
        print(f"❌ エラー: {pattern_path} が見つかりません。")
        return None

    spec = importlib.util.spec_from_file_location(
        f"{pattern_name}_module", pattern_path
    )
    if spec is None or spec.loader is None:
        print(f"❌ エラー: {pattern_path} の読み込みに失敗しました。")
        return None

    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
        return module
    except Exception as e:
        print(f"❌ エラー: {pattern_path} の実行に失敗しました: {e}")
        return None


def show_menu():
    """メニューを表示します。"""
    print("\n" + "=" * 60)
    print("🤖 AIエージェントパターン実践プロジェクト")
    print("=" * 60)
    print("Anthropic社の「Building effective agents」パターンを学習できます。")
    print("\n📚 利用可能なパターン:")
    print("  1. Augmented LLM Pattern        - LLMの機能拡張パターン")
    print("  2. Prompt Chaining Pattern      - プロンプト連鎖パターン")
    print("  3. Routing Pattern              - 条件分岐・ルーティングパターン")
    print("  4. Parallelization Pattern      - 並列処理パターン")
    print("  5. Orchestrator-Workers Pattern - オーケストレーター・ワーカーパターン")
    print("  6. Evaluator-Optimizer Pattern  - 評価・最適化パターン")
    print("  7. Agents Pattern               - エージェントパターン")
    print("\n  0. 終了")
    print("-" * 60)


def run_pattern(choice: str):
    """選択されたパターンを実行します。"""
    patterns = {
        "1": ("01_augmented_llm", "augmented_llm.py", "Augmented LLM Pattern"),
        "2": ("02_prompt_chaining", "prompt_chaining.py", "Prompt Chaining Pattern"),
        "3": ("03_routing", "routing.py", "Routing Pattern"),
        "4": ("04_parallelization", "parallelization.py", "Parallelization Pattern"),
        "5": (
            "05_orchestrator_workers",
            "orchestrator_workers.py",
            "Orchestrator-Workers Pattern",
        ),
        "6": (
            "06_evaluator_optimizer",
            "evaluator_optimizer.py",
            "Evaluator-Optimizer Pattern",
        ),
        "7": ("07_agents", "agents.py", "Agents Pattern"),
    }

    if choice not in patterns:
        print("❌ 無効な選択です。")
        return

    pattern_dir, file_name, pattern_name = patterns[choice]
    print(f"\n🚀 {pattern_name} を実行中...")
    print("-" * 40)

    # パターンモジュールを読み込んで実行
    module = load_pattern_module(pattern_dir, file_name)
    if module and hasattr(module, "main"):
        try:
            module.main()
        except KeyboardInterrupt:
            print("\n⚠️  実行が中断されました。")
        except Exception as e:
            print(f"❌ エラーが発生しました: {e}")
            print("💡 APIキーが設定されているか確認してください。")
    else:
        print("❌ パターンの実行に失敗しました。")


def check_environment():
    """環境設定をチェックします。"""
    print("\n🔍 環境設定をチェック中...")

    # .envファイルの存在確認
    if not os.path.exists(".env"):
        print("⚠️  .envファイルが見つかりません。")
        print("💡 .envファイルを作成してAPIキーを設定してください:")
        print("   OPENAI_API_KEY=your_openai_api_key_here")
        print("   ANTHROPIC_API_KEY=your_anthropic_api_key_here")
        return False

    # パターンディレクトリの存在確認
    patterns_dir = "patterns"
    if not os.path.exists(patterns_dir):
        print(f"❌ {patterns_dir} ディレクトリが見つかりません。")
        return False

    print("✅ 環境設定OK")
    return True


def main():
    """メインアプリケーション"""
    print("🎯 AIエージェントパターン実践プロジェクトへようこそ！")

    # 環境設定チェック
    if not check_environment():
        print("\n❌ 環境設定に問題があります。READMEを参照して設定してください。")
        return

    while True:
        try:
            show_menu()
            choice = input("\n選択してください (0-7): ").strip()

            if choice == "0":
                print("\n👋 プロジェクトを終了します。学習お疲れさまでした！")
                break
            elif choice in ["1", "2", "3", "4", "5", "6", "7"]:
                run_pattern(choice)
                input("\n📚 Enterキーを押してメニューに戻る...")
            else:
                print("❌ 無効な選択です。0-7の数字を入力してください。")

        except KeyboardInterrupt:
            print("\n\n👋 プロジェクトを終了します。")
            break
        except Exception as e:
            print(f"\n❌ 予期しないエラーが発生しました: {e}")


if __name__ == "__main__":
    main()
