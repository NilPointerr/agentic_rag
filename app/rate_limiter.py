from app.config.settings import settings

try:
    from slowapi import Limiter
    from slowapi.errors import RateLimitExceeded
    from slowapi.middleware import SlowAPIMiddleware
    from slowapi.util import get_remote_address
except ModuleNotFoundError:
    Limiter = None
    RateLimitExceeded = None
    SlowAPIMiddleware = None


class NoOpLimiter:
    def limit(self, _value):
        def decorator(func):
            return func

        return decorator


limiter = Limiter(key_func=get_remote_address) if Limiter else NoOpLimiter()


def configure_rate_limiter(app):
    if not settings.RATE_LIMIT_ENABLED:
        return

    if not Limiter or not SlowAPIMiddleware or not RateLimitExceeded:
        raise RuntimeError(
            "slowapi must be installed when RATE_LIMIT_ENABLED is true"
        )

    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    app.add_middleware(SlowAPIMiddleware)


async def _rate_limit_exceeded_handler(request, exc):
    from fastapi.responses import JSONResponse

    return JSONResponse(status_code=429, content={"detail": "Rate limit exceeded"})
