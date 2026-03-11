from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Any, Optional

from app.config.settings import settings
from app.utils.logger import logger

_client: Optional[Any] = None


def get_groq_client() -> Any:
    """Create and cache Groq client."""

    global _client
    if _client is None:
        from groq import Groq

        _client = Groq(api_key=settings.GROQ_API_KEY)
    return _client


def _create_completion(
    messages: list[dict[str, Any]],
    model: str,
    temperature: float,
):
    """Execute one Groq chat completion request."""

    client = get_groq_client()
    return client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )


def generate_answer(
    messages: list[dict[str, Any]],
    model: str | None = None,
    temperature: float | None = None,
    timeout_seconds: int | None = None,
    max_retries: int | None = None,
):
    """Generate LLM answer with timeout and retry controls."""

    effective_model = model or settings.GROQ_MODEL
    effective_temp = settings.LLM_TEMPERATURE if temperature is None else temperature
    effective_timeout = timeout_seconds or settings.LLM_TIMEOUT_SECONDS
    effective_retries = settings.LLM_MAX_RETRIES if max_retries is None else max_retries

    logger.info("Calling LLM model=%s messages=%s", effective_model, len(messages))

    for attempt in range(effective_retries + 1):
        try:
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(
                    _create_completion,
                    messages,
                    effective_model,
                    effective_temp,
                )
                return future.result(timeout=effective_timeout)
        except FuturesTimeoutError as exc:
            logger.warning("LLM timeout on attempt %s", attempt + 1)
            if attempt >= effective_retries:
                raise TimeoutError("LLM request timed out") from exc
        except Exception:
            logger.exception("LLM request failed on attempt %s", attempt + 1)
            if attempt >= effective_retries:
                raise
        time.sleep(0.5 * (attempt + 1))

    raise RuntimeError("LLM request failed after retries")
