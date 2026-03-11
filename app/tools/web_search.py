from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

from app.config.settings import settings
from app.utils.logger import logger


def _run_ddgs_search(query: str, max_results: int) -> list[dict]:
    """Execute DDGS text search and return raw result rows."""

    from ddgs import DDGS

    with DDGS() as ddgs:
        return list(ddgs.text(query, max_results=max_results))


def web_search(
    query: str,
    max_results: int | None = None,
    timeout_seconds: int | None = None,
    max_retries: int | None = None,
) -> list[dict]:
    """Perform web search with timeout and retry behavior."""

    effective_results = max_results or settings.WEB_SEARCH_MAX_RESULTS
    effective_timeout = timeout_seconds or settings.WEB_SEARCH_TIMEOUT_SECONDS
    effective_retries = (
        settings.WEB_SEARCH_MAX_RETRIES if max_retries is None else max_retries
    )

    for attempt in range(effective_retries + 1):
        try:
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(_run_ddgs_search, query, effective_results)
                results = future.result(timeout=effective_timeout)

            logger.info("Web search returned %s results", len(results))
            formatted_results: list[dict] = []
            for row in results:
                formatted_results.append(
                    {
                        "title": row.get("title", ""),
                        "snippet": row.get("body", ""),
                        "url": row.get("href", ""),
                    }
                )
            return formatted_results
        except FuturesTimeoutError as exc:
            logger.warning("Web search timeout on attempt %s", attempt + 1)
            if attempt >= effective_retries:
                raise TimeoutError("Web search timed out") from exc
        except Exception:
            logger.exception("Web search failed on attempt %s", attempt + 1)
            if attempt >= effective_retries:
                raise
        time.sleep(0.4 * (attempt + 1))

    return []
