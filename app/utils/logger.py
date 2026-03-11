import functools
import inspect
import logging
import sys
import time
import traceback
from collections.abc import Callable
from typing import Any


def setup_logger() -> logging.Logger:
    """Configure and return application logger."""

    app_logger = logging.getLogger("agentic_rag")
    app_logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] - %(message)s"
    )
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    if not app_logger.handlers:
        app_logger.addHandler(handler)

    return app_logger


logger = setup_logger()


def log_execution(func: Callable) -> Callable:
    """Log function start/end/error for sync and async callables."""

    @functools.wraps(func)
    async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = time.time()
        logger.info("Started: %s", func.__name__)
        try:
            result = await func(*args, **kwargs)
            execution_time = round(time.time() - start_time, 4)
            logger.info(
                "Completed: %s | Execution Time: %ss", func.__name__, execution_time
            )
            return result
        except Exception as exc:
            logger.error(
                "Error in %s: %s - %s", func.__name__, type(exc).__name__, str(exc)
            )
            logger.error(traceback.format_exc())
            raise

    @functools.wraps(func)
    def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = time.time()
        logger.info("Started: %s", func.__name__)
        try:
            result = func(*args, **kwargs)
            execution_time = round(time.time() - start_time, 4)
            logger.info(
                "Completed: %s | Execution Time: %ss", func.__name__, execution_time
            )
            return result
        except Exception as exc:
            logger.error(
                "Error in %s: %s - %s", func.__name__, type(exc).__name__, str(exc)
            )
            logger.error(traceback.format_exc())
            raise

    if inspect.iscoroutinefunction(func):
        return async_wrapper
    return sync_wrapper
