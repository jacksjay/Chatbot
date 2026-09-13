"""
Thin wrapper around the local Ollama chat API."""

import time
from dataclasses import dataclass
from typing import Optional

import ollama

from config import CFG

# Standardized response format
@dataclass
class LLMResponse:
    content: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    success: bool
    error: Optional[str] = None


class OllamaClient:
    def __init__(self, model: str = CFG.OLLAMA_CHAT_MODEL, base_url: str = CFG.OLLAMA_BASE_URL, timeout:float = CFG.OLLAMA_TIMEOUT_SECONDS):
        self._model = model
        self._client = ollama.Client(host=base_url, timeout=timeout)

    def chat(self, messages: list[dict], max_tokens: int = CFG.MAX_OUTPUT_TOKENS) -> LLMResponse:
        start = time.perf_counter() #much more accurate for benchmarking
        try:
            #Actual call to the LLM
            response = self._client.chat(
                model=self._model,
                messages=messages,
                options={"num_predict": max_tokens},
            )
            # Calculate exactly how long the generation took
            latency_ms = (time.perf_counter() - start) * 1000
            return LLMResponse(
                content=response["message"]["content"],
                model=self._model,
                # Extract exact token usage natively provided by Ollama
                completion_tokens=response.get("eval_count", 0),
                prompt_tokens=response.get("prompt_eval_count", 0), 
                latency_ms=latency_ms,
                success=True,
            )
        except Exception as exc:  # surfaced to caller/logger
            latency_ms = (time.perf_counter() - start) * 1000
            return LLMResponse(
                content="",
                model=self._model,
                prompt_tokens=0,
                completion_tokens=0,
                latency_ms=latency_ms,
                success=False,
                error=str(exc),
            )

    """Utility to check if models are correctly installed."""
    def list_models(self) -> list[str]:
        try:
            data = self._client.list()
            return [m["model"] for m in data.get("models", [])]
        except Exception:
            return []
