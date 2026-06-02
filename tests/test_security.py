import sys
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from app.config.settings import settings
from app.security import verify_bearer_token


def test_verify_bearer_token_returns_none_when_auth_disabled():
    """Auth should be bypassed entirely when the feature flag is disabled."""
    original_auth_enabled = settings.AUTH_ENABLED
    settings.AUTH_ENABLED = False

    try:
        assert verify_bearer_token(None) is None
    finally:
        settings.AUTH_ENABLED = original_auth_enabled


def test_verify_bearer_token_requires_secret_when_enabled():
    """Enabled auth should fail fast when the signing secret is missing."""
    original_auth_enabled = settings.AUTH_ENABLED
    original_secret = settings.JWT_SECRET_KEY
    settings.AUTH_ENABLED = True
    settings.JWT_SECRET_KEY = ""

    try:
        with pytest.raises(HTTPException) as exc_info:
            verify_bearer_token("Bearer token")
    finally:
        settings.AUTH_ENABLED = original_auth_enabled
        settings.JWT_SECRET_KEY = original_secret

    assert exc_info.value.status_code == 500


def test_verify_bearer_token_rejects_missing_bearer_prefix():
    """Enabled auth should reject requests without a bearer header."""
    original_auth_enabled = settings.AUTH_ENABLED
    original_secret = settings.JWT_SECRET_KEY
    settings.AUTH_ENABLED = True
    settings.JWT_SECRET_KEY = "secret"

    try:
        with pytest.raises(HTTPException) as exc_info:
            verify_bearer_token(None)
    finally:
        settings.AUTH_ENABLED = original_auth_enabled
        settings.JWT_SECRET_KEY = original_secret

    assert exc_info.value.status_code == 401


def test_verify_bearer_token_returns_decoded_payload(monkeypatch):
    """Valid JWTs should decode into the returned auth payload."""
    original_auth_enabled = settings.AUTH_ENABLED
    original_secret = settings.JWT_SECRET_KEY
    original_algorithm = settings.JWT_ALGORITHM
    settings.AUTH_ENABLED = True
    settings.JWT_SECRET_KEY = "secret"
    settings.JWT_ALGORITHM = "HS256"

    fake_jwt = SimpleNamespace(
        decode=lambda token, secret, algorithms: {"sub": "user-1"},
        PyJWTError=RuntimeError,
    )
    monkeypatch.setitem(sys.modules, "jwt", fake_jwt)

    try:
        payload = verify_bearer_token("Bearer valid-token")
    finally:
        settings.AUTH_ENABLED = original_auth_enabled
        settings.JWT_SECRET_KEY = original_secret
        settings.JWT_ALGORITHM = original_algorithm

    assert payload == {"sub": "user-1"}


def test_verify_bearer_token_rejects_invalid_token(monkeypatch):
    """JWT decode failures should be surfaced as unauthorized requests."""
    original_auth_enabled = settings.AUTH_ENABLED
    original_secret = settings.JWT_SECRET_KEY
    settings.AUTH_ENABLED = True
    settings.JWT_SECRET_KEY = "secret"

    class FakeJWTError(Exception):
        pass

    fake_jwt = SimpleNamespace(
        decode=lambda token, secret, algorithms: (_ for _ in ()).throw(FakeJWTError("bad token")),
        PyJWTError=FakeJWTError,
    )
    monkeypatch.setitem(sys.modules, "jwt", fake_jwt)

    try:
        with pytest.raises(HTTPException) as exc_info:
            verify_bearer_token("Bearer invalid-token")
    finally:
        settings.AUTH_ENABLED = original_auth_enabled
        settings.JWT_SECRET_KEY = original_secret

    assert exc_info.value.status_code == 401
