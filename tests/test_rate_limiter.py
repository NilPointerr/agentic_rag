import pytest
from fastapi import FastAPI

from app import rate_limiter


def test_configure_rate_limiter_is_noop_when_disabled(monkeypatch):
    """Rate limiter setup should do nothing when the feature is disabled."""
    app = FastAPI()
    monkeypatch.setattr(rate_limiter.settings, "RATE_LIMIT_ENABLED", False)

    rate_limiter.configure_rate_limiter(app)

    assert not hasattr(app.state, "limiter")


def test_configure_rate_limiter_raises_when_dependencies_missing(monkeypatch):
    """Enabled rate limiting should fail clearly without slowapi installed."""
    app = FastAPI()
    monkeypatch.setattr(rate_limiter.settings, "RATE_LIMIT_ENABLED", True)
    monkeypatch.setattr(rate_limiter, "Limiter", None)
    monkeypatch.setattr(rate_limiter, "SlowAPIMiddleware", None)
    monkeypatch.setattr(rate_limiter, "RateLimitExceeded", None)

    with pytest.raises(RuntimeError):
        rate_limiter.configure_rate_limiter(app)


def test_configure_rate_limiter_attaches_handler_and_middleware(monkeypatch):
    """Enabled rate limiting should attach app state and middleware hooks."""
    app = FastAPI()
    handler_calls = []
    middleware_calls = []

    monkeypatch.setattr(rate_limiter.settings, "RATE_LIMIT_ENABLED", True)
    monkeypatch.setattr(rate_limiter, "Limiter", object)
    monkeypatch.setattr(rate_limiter, "SlowAPIMiddleware", type("SlowAPIMiddleware", (), {}))
    monkeypatch.setattr(rate_limiter, "RateLimitExceeded", type("RateLimitExceeded", (), {}))
    monkeypatch.setattr(rate_limiter, "limiter", object())
    monkeypatch.setattr(app, "add_exception_handler", lambda exc, handler: handler_calls.append((exc, handler)))
    monkeypatch.setattr(app, "add_middleware", lambda middleware: middleware_calls.append(middleware))

    rate_limiter.configure_rate_limiter(app)

    assert app.state.limiter is rate_limiter.limiter
    assert handler_calls
    assert middleware_calls


@pytest.mark.anyio
async def test_rate_limit_exceeded_handler_returns_standard_payload():
    """The rate-limit handler should return a standard 429 response."""
    response = await rate_limiter._rate_limit_exceeded_handler(None, None)

    assert response.status_code == 429
    assert response.body == b'{"detail":"Rate limit exceeded"}'
