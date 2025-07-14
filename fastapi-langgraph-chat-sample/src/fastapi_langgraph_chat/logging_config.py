"""シンプルなロギング設定."""

import logging
import logging.handlers
import os
from pathlib import Path


def setup_logging():
    """ロギングを設定する."""
    # ログレベルを環境変数から取得（デフォルト: INFO）
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    
    # ログディレクトリを作成
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # ルートロガーの設定
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level, logging.INFO))
    
    # 既存のハンドラーをクリア
    root_logger.handlers.clear()
    
    # ログフォーマット
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # コンソールハンドラー
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level, logging.INFO))
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # ファイルハンドラー（日次ローテーション）
    file_handler = logging.handlers.TimedRotatingFileHandler(
        filename=log_dir / "app.log",
        when="midnight",
        interval=1,
        backupCount=30,
        encoding="utf-8"
    )
    file_handler.setLevel(getattr(logging, log_level, logging.INFO))
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    
    # uvicornのロガー設定
    uvicorn_logger = logging.getLogger("uvicorn")
    uvicorn_logger.setLevel(getattr(logging, log_level, logging.INFO))
    
    # FastAPIのロガー設定
    fastapi_logger = logging.getLogger("fastapi")
    fastapi_logger.setLevel(getattr(logging, log_level, logging.INFO))
    
    logging.info("ロギング設定が完了しました")