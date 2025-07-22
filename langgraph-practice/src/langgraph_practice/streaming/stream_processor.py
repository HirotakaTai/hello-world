"""ストリーミング処理クラス

LLMレスポンスのリアルタイムストリーミング表示
"""
import asyncio
import time
from typing import AsyncGenerator, Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field

from rich.console import Console
from rich.text import Text
from rich.live import Live
from rich.spinner import Spinner


@dataclass
class StreamChunk:
    """ストリームチャンクのデータクラス"""
    content: str
    timestamp: float = field(default_factory=time.time)
    chunk_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class StreamProcessor:
    """ストリーミング処理クラス"""
    
    def __init__(
        self,
        chunk_size: int = 1024,
        delay_ms: int = 50,
        buffer_size: int = 100,
        console: Optional[Console] = None,
    ):
        self.chunk_size = chunk_size
        self.delay_ms = delay_ms
        self.buffer_size = buffer_size
        self.console = console or Console()
        self.buffer: List[StreamChunk] = []
        self._processing = False
        self._total_content = ""
    
    async def process_stream(
        self,
        stream_generator: AsyncGenerator[str, None],
        on_chunk: Optional[Callable[[StreamChunk], None]] = None,
        display_progress: bool = True,
    ) -> str:
        """ストリームを処理して結果を返す"""
        self._processing = True
        self._total_content = ""
        chunk_count = 0
        
        try:
            if display_progress:
                with Live(self._create_progress_display(), refresh_per_second=10):
                    async for content in stream_generator:
                        chunk = StreamChunk(
                            content=content,
                            chunk_id=f"chunk_{chunk_count}",
                        )
                        
                        await self._add_to_buffer(chunk)
                        
                        if on_chunk:
                            on_chunk(chunk)
                        
                        self._total_content += content
                        chunk_count += 1
                        
                        # 遅延を追加（よりリアルなストリーミング体験のため）
                        if self.delay_ms > 0:
                            await asyncio.sleep(self.delay_ms / 1000.0)
            else:
                async for content in stream_generator:
                    chunk = StreamChunk(
                        content=content,
                        chunk_id=f"chunk_{chunk_count}",
                    )
                    
                    await self._add_to_buffer(chunk)
                    
                    if on_chunk:
                        on_chunk(chunk)
                    
                    self._total_content += content
                    chunk_count += 1
                    
                    if self.delay_ms > 0:
                        await asyncio.sleep(self.delay_ms / 1000.0)
        
        finally:
            self._processing = False
        
        return self._total_content
    
    async def _add_to_buffer(self, chunk: StreamChunk) -> None:
        """バッファにチャンクを追加"""
        self.buffer.append(chunk)
        
        # バッファサイズ制限
        if len(self.buffer) > self.buffer_size:
            self.buffer = self.buffer[-self.buffer_size:]
    
    def _create_progress_display(self) -> Text:
        """プログレス表示の作成"""
        if self._processing:
            spinner = Spinner("dots", text="処理中...")
            return Text.from_markup(f"[blue]{spinner}[/blue] ストリーミング中...")
        else:
            return Text.from_markup("[green]✓[/green] ストリーミング完了")
    
    def get_buffer_content(self) -> str:
        """バッファの内容を結合して返す"""
        return "".join(chunk.content for chunk in self.buffer)
    
    def clear_buffer(self) -> None:
        """バッファをクリア"""
        self.buffer.clear()
        self._total_content = ""
    
    def get_chunk_statistics(self) -> Dict[str, Any]:
        """チャンク統計情報を取得"""
        if not self.buffer:
            return {"total_chunks": 0, "total_content_length": 0, "average_chunk_size": 0}
        
        total_length = sum(len(chunk.content) for chunk in self.buffer)
        return {
            "total_chunks": len(self.buffer),
            "total_content_length": total_length,
            "average_chunk_size": total_length / len(self.buffer) if self.buffer else 0,
            "processing_status": "active" if self._processing else "idle",
        }


class RealTimeDisplay:
    """リアルタイム表示クラス"""
    
    def __init__(self, console: Optional[Console] = None):
        self.console = console or Console()
        self.current_text = ""
        self._display_active = False
    
    async def start_display(self, title: str = "AI応答") -> None:
        """表示開始"""
        self._display_active = True
        self.console.print(f"\n[bold blue]{title}[/bold blue]")
        self.console.print("[dim]" + "="*50 + "[/dim]")
    
    async def update_display(self, new_content: str) -> None:
        """表示更新"""
        if not self._display_active:
            return
        
        self.current_text += new_content
        # リアルタイムで内容を表示
        self.console.print(new_content, end="")
    
    async def finish_display(self) -> None:
        """表示終了"""
        self._display_active = False
        self.console.print("\n[dim]" + "="*50 + "[/dim]")
        self.console.print("[green]完了[/green]")
    
    def clear_display(self) -> None:
        """表示クリア"""
        self.current_text = ""
        self.console.clear()


class StreamingChatDisplay:
    """ストリーミングチャット表示クラス"""
    
    def __init__(self, console: Optional[Console] = None):
        self.console = console or Console()
        self.processor = StreamProcessor(console=console)
        self.display = RealTimeDisplay(console=console)
    
    async def display_streaming_response(
        self,
        response_stream: AsyncGenerator[str, None],
        title: str = "🤖 AI応答",
    ) -> str:
        """ストリーミングレスポンスを表示"""
        await self.display.start_display(title)
        
        def on_chunk_received(chunk: StreamChunk):
            # リアルタイムで表示更新
            asyncio.create_task(self.display.update_display(chunk.content))
        
        try:
            result = await self.processor.process_stream(
                response_stream,
                on_chunk=on_chunk_received,
                display_progress=False  # カスタム表示を使用
            )
        finally:
            await self.display.finish_display()
        
        return result
    
    def display_statistics(self) -> None:
        """統計情報を表示"""
        stats = self.processor.get_chunk_statistics()
        self.console.print(f"\n[dim]統計: {stats['total_chunks']}チャンク, "
                          f"{stats['total_content_length']}文字[/dim]")