import logging
import sys
import time
import functools
import traceback
import inspect

def setup_logger():
    logger = logging.getLogger("agentic_rag")
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] - %(message)s"
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    if not logger.handlers:
        logger.addHandler(handler)

    return logger


logger = setup_logger()

def log_execution(func):
    """
    Decorator for logging:
    - Function start
    - Execution time
    - Success
    - Errors with traceback
    Supports both sync and async functions
    """

    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs):
        start_time = time.time()
        logger.info(f"Started: {func.__name__}")

        try:
            result = await func(*args, **kwargs)
            execution_time = round(time.time() - start_time, 4)
            logger.info(
                f"Completed: {func.__name__} | Execution Time: {execution_time}s"
            )
            return result

        except Exception as e:
            logger.error(
                f"Error in {func.__name__}: {type(e).__name__} - {str(e)}"
            )
            logger.error(traceback.format_exc())
            raise

    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs):
        start_time = time.time()
        logger.info(f"Started: {func.__name__}")

        try:
            result = func(*args, **kwargs)
            execution_time = round(time.time() - start_time, 4)
            logger.info(
                f"Completed: {func.__name__} | Execution Time: {execution_time}s"
            )
            return result

        except Exception as e:
            logger.error(
                f"Error in {func.__name__}: {type(e).__name__} - {str(e)}"
            )
            logger.error(traceback.format_exc())
            raise

    if inspect.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper