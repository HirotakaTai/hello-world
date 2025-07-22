"""エラーハンドリングとリトライ戦略

LangGraphでの堅牢なエラー処理実装
"""
import asyncio
import logging
import time
from typing import Any, Callable, Dict, Optional, Type
from functools import wraps
from enum import Enum

from .state import AgentState


class ErrorType(str, Enum):
    """エラーの種類"""
    API_ERROR = "api_error"
    TIMEOUT_ERROR = "timeout_error"
    VALIDATION_ERROR = "validation_error"
    NETWORK_ERROR = "network_error"
    PROCESSING_ERROR = "processing_error"
    UNKNOWN_ERROR = "unknown_error"


class RetryStrategy(str, Enum):
    """リトライ戦略"""
    FIXED_DELAY = "fixed_delay"
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    NO_RETRY = "no_retry"


class ErrorHandler:
    """エラーハンドリングクラス"""
    
    def __init__(
        self,
        max_retries: int = 3,
        retry_strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
    ):
        self.max_retries = max_retries
        self.retry_strategy = retry_strategy
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.logger = logging.getLogger(__name__)
    
    def calculate_delay(self, attempt: int) -> float:
        """リトライ遅延時間の計算"""
        if self.retry_strategy == RetryStrategy.FIXED_DELAY:
            return self.base_delay
        elif self.retry_strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            delay = self.base_delay * (2 ** attempt)
            return min(delay, self.max_delay)
        elif self.retry_strategy == RetryStrategy.LINEAR_BACKOFF:
            delay = self.base_delay * (attempt + 1)
            return min(delay, self.max_delay)
        else:
            return 0.0
    
    def classify_error(self, error: Exception) -> ErrorType:
        """エラーの分類"""
        error_name = type(error).__name__.lower()
        
        if "timeout" in error_name:
            return ErrorType.TIMEOUT_ERROR
        elif "connection" in error_name or "network" in error_name:
            return ErrorType.NETWORK_ERROR
        elif "validation" in error_name or "value" in error_name:
            return ErrorType.VALIDATION_ERROR
        elif "api" in error_name or "client" in error_name:
            return ErrorType.API_ERROR
        else:
            return ErrorType.UNKNOWN_ERROR
    
    def should_retry(self, error: Exception, attempt: int) -> bool:
        """リトライすべきかどうかの判定"""
        if attempt >= self.max_retries:
            return False
        
        error_type = self.classify_error(error)
        
        # リトライしないエラータイプ
        no_retry_errors = {ErrorType.VALIDATION_ERROR}
        if error_type in no_retry_errors:
            return False
        
        return True
    
    async def execute_with_retry(
        self,
        func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """リトライ付きでの関数実行"""
        last_error = None
        
        for attempt in range(self.max_retries + 1):
            try:
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                return result
            
            except Exception as e:
                last_error = e
                error_type = self.classify_error(e)
                
                self.logger.warning(
                    f"実行エラー (試行 {attempt + 1}/{self.max_retries + 1}): "
                    f"{error_type.value} - {str(e)}"
                )
                
                if not self.should_retry(e, attempt):
                    break
                
                if attempt < self.max_retries:
                    delay = self.calculate_delay(attempt)
                    if delay > 0:
                        await asyncio.sleep(delay)
        
        # 最終的に失敗した場合
        self.logger.error(f"全てのリトライが失敗: {str(last_error)}")
        raise last_error


def with_error_handling(
    max_retries: int = 3,
    retry_strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF,
    base_delay: float = 1.0,
):
    """エラーハンドリングデコレータ"""
    def decorator(func):
        error_handler = ErrorHandler(
            max_retries=max_retries,
            retry_strategy=retry_strategy,
            base_delay=base_delay
        )
        
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await error_handler.execute_with_retry(func, *args, **kwargs)
        
        return wrapper
    return decorator


class RecoveryManager:
    """チェックポイントからの回復管理"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def can_recover_from_checkpoint(self, state: AgentState) -> bool:
        """チェックポイントから回復可能かチェック"""
        return (
            state.get("current_step") is not None and
            state.get("error_count", 0) < 5  # 最大エラー数制限
        )
    
    def create_recovery_state(
        self, 
        original_state: AgentState, 
        error: Exception
    ) -> AgentState:
        """回復用状態の作成"""
        recovery_state = original_state.copy()
        recovery_state.update({
            "error_count": original_state.get("error_count", 0) + 1,
            "last_error": str(error),
            "metadata": {
                **original_state.get("metadata", {}),
                "recovery_timestamp": time.time(),
                "recovery_reason": type(error).__name__,
            }
        })
        return recovery_state
    
    def should_fallback(self, state: AgentState) -> bool:
        """フォールバック処理が必要かチェック"""
        return state.get("error_count", 0) >= 3