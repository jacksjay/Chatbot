"""
Records every LLM call to SQLite (structured, queryable) and to a text log
file"""

import logging
from pathlib import Path

from core.llm_client import LLMResponse
from database.db_manager import DatabaseManager

# Ensure the logs directory exists on startup
Path("logs").mkdir(exist_ok=True)

# Setup the standard Python file logger
_file_logger = logging.getLogger("llm_calls")
_file_logger.setLevel(logging.INFO)
if not _file_logger.handlers:
    _handler = logging.FileHandler("logs/llm_calls.log")
    _handler.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
    _file_logger.addHandler(_handler)


class LLMLogger:
    def __init__(self, db: DatabaseManager):
        self._db = db

    def log(self, username: str, response: LLMResponse, input_chars: int) -> None:
        status = "success" if response.success else "error"
        total_tokens = response.prompt_tokens + response.completion_tokens

        # LOG TO DATABASE:(for UI)
        self._db.log_llm_call(
            username=username,
            model=response.model,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            total_tokens=total_tokens,
            latency_ms=round(response.latency_ms, 2),
            input_chars=input_chars,
            output_chars=len(response.content),
            status=status,
            error=response.error or "",
        )
        # LOG TO FILE:(for dev)
        _file_logger.info(
            "user=%s model=%s status=%s prompt_tokens=%d completion_tokens=%d "
            "total_tokens=%d latency_ms=%.2f input_chars=%d output_chars=%d error=%s",
            username,
            response.model,
            status,
            response.prompt_tokens,
            response.completion_tokens,
            total_tokens,
            response.latency_ms,
            input_chars,
            len(response.content),
            response.error or "-",
        )
