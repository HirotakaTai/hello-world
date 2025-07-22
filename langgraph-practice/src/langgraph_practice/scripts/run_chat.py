#!/usr/bin/env python3
"""
LangGraphチャットボット実行スクリプト

コマンドライン引数でさまざまなモードを実行可能
"""
import asyncio
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

app = typer.Typer(help="LangGraphチャットボット実行ツール")
console = Console()


@app.command()
def basic(
    input_text: Optional[str] = typer.Option(
        None, 
        "--input", 
        "-i", 
        help="入力テキスト（指定しない場合は対話モード）"
    )
):
    """基本状態グラフの例を実行"""
    console.print(Panel.fit(
        "🎯 基本状態グラフの例を実行します",
        style="bold blue"
    ))
    
    if input_text:
        asyncio.run(run_basic_example_with_input(input_text))
    else:
        # デフォルトの例を実行
        asyncio.run(run_basic_default_examples())


@app.command()
def streaming(
    input_text: Optional[str] = typer.Option(
        None, 
        "--input", 
        "-i", 
        help="入力テキスト（指定しない場合は対話モード）"
    )
):
    """ストリーミングレスポンスの例を実行"""
    console.print(Panel.fit(
        "🔄 ストリーミングレスポンスの例を実行します",
        style="bold green"
    ))
    
    if input_text:
        asyncio.run(run_streaming_example_with_input(input_text))
    else:
        asyncio.run(run_streaming_default_examples())


@app.command()
def interactive():
    """対話型チャットモードを開始"""
    console.print(Panel.fit(
        "💬 対話型チャットモードを開始します\n終了するには 'quit' または 'exit' を入力してください",
        style="bold yellow"
    ))
    
    asyncio.run(interactive_chat())


@app.command()
def list_examples():
    """利用可能な例の一覧を表示"""
    examples = [
        ("basic", "基本状態グラフ - TypedDict状態とシンプルなノード"),
        ("streaming", "ストリーミングレスポンス - リアルタイム表示"),
        ("interactive", "対話型チャット - 継続的な会話"),
    ]
    
    console.print("\n[bold]利用可能な例:[/bold]")
    for command, description in examples:
        console.print(f"  [cyan]{command}[/cyan]: {description}")
    
    console.print("\n[dim]使用例: uv run chat basic --input 'こんにちは'[/dim]")


async def run_basic_example_with_input(input_text: str):
    """単一入力での基本例実行"""
    console.print(f"👤 入力: {input_text}")
    
    # 基本的なルールベース応答
    user_message = input_text.lower()
    
    if "こんにちは" in user_message or "hello" in user_message:
        response = "こんにちは！どのようにお手伝いできますか？"
    elif "ありがとう" in user_message or "thank" in user_message:
        response = "どういたしまして！他にご質問はありますか？"
    elif "さようなら" in user_message or "bye" in user_message:
        response = "さようなら！また何かありましたらお声をかけてください。"
    else:
        response = f"「{input_text}」について理解しました。詳しく教えていただけますか？"
    
    console.print(f"🤖 AI: {response}")
    console.print(f"✅ 基本状態グラフのシミュレーション完了")


async def run_basic_default_examples():
    """基本例のデフォルト実行"""
    test_inputs = [
        "こんにちは",
        "LangGraphについて教えて",
        "ありがとうございました"
    ]
    
    for i, user_input in enumerate(test_inputs, 1):
        console.print(f"\n--- 実行例 {i} ---")
        await run_basic_example_with_input(user_input)


async def run_streaming_example_with_input(input_text: str):
    """単一入力でのストリーミング例実行"""
    console.print(f"👤 入力: {input_text}")
    
    try:
        from ..streaming.stream_processor import StreamingChatDisplay
        
        chat_display = StreamingChatDisplay()
        
        async def simple_response_stream():
            import random
            responses = ["これは", "ストリーミング", "レスポンス", "の", "テスト", "です。"]
            for word in responses:
                await asyncio.sleep(random.uniform(0.1, 0.3))
                yield word + " "
        
        response = await chat_display.display_streaming_response(
            simple_response_stream(),
            title="🤖 AI応答（ストリーミング）"
        )
        console.print(f"✅ ストリーミング完了")
    except ImportError:
        console.print("[yellow]ストリーミング機能が利用できません（依存関係を確認してください）[/yellow]")
        # フォールバック: 基本レスポンス
        await run_basic_example_with_input(input_text)


async def run_streaming_default_examples():
    """ストリーミング例のデフォルト実行"""
    test_inputs = [
        "こんにちは",
        "LangGraphについて教えて",
        "ストリーミング処理の仕組みは？"
    ]
    
    for i, user_input in enumerate(test_inputs, 1):
        console.print(f"\n--- ストリーミング例 {i} ---")
        await run_streaming_example_with_input(user_input)


async def interactive_chat():
    """対話型チャットの実装"""
    session_count = 0
    
    console.print("[green]チャットボットの準備ができました！[/green]\n")
    
    while True:
        try:
            user_input = Prompt.ask("\n[bold blue]あなた[/bold blue]")
            
            if user_input.lower() in ['quit', 'exit', '終了', 'q']:
                console.print("\n[yellow]チャットを終了します。[/yellow]")
                break
            
            if not user_input.strip():
                continue
            
            session_count += 1
            
            try:
                from ..streaming.stream_processor import StreamingChatDisplay
                
                chat_display = StreamingChatDisplay()
                
                async def simple_response_stream():
                    import random
                    responses = [
                        "それは", "興味深い", "質問", "ですね。", "もう少し", "詳しく", 
                        "教えて", "いただけ", "ますか？"
                    ]
                    
                    for word in responses:
                        await asyncio.sleep(random.uniform(0.1, 0.3))
                        yield word + " "
                
                response = await chat_display.display_streaming_response(
                    simple_response_stream(),
                    title=f"🤖 AI応答 (セッション {session_count})"
                )
            except ImportError:
                # フォールバック: 基本レスポンス
                console.print(f"🤖 AI応答 (セッション {session_count}): それは興味深い質問ですね。")
            
        except KeyboardInterrupt:
            console.print("\n[yellow]チャットを終了します。[/yellow]")
            break
        except Exception as e:
            console.print(f"[red]エラーが発生しました: {str(e)}[/red]")


def main():
    """メイン関数"""
    app()


if __name__ == "__main__":
    main()