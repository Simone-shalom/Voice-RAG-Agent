import pytest
from fastapi import HTTPException

from app.core.auth import require_api_key


def test_allows_request_when_api_key_env_unset(monkeypatch):
    monkeypatch.delenv("API_KEY", raising=False)
    require_api_key(x_api_key=None)  # does not raise


def test_rejects_missing_header_when_api_key_configured(monkeypatch):
    monkeypatch.setenv("API_KEY", "secret123")
    with pytest.raises(HTTPException) as exc_info:
        require_api_key(x_api_key=None)
    assert exc_info.value.status_code == 401


def test_rejects_wrong_header_when_api_key_configured(monkeypatch):
    monkeypatch.setenv("API_KEY", "secret123")
    with pytest.raises(HTTPException):
        require_api_key(x_api_key="wrong")


def test_allows_correct_header_when_api_key_configured(monkeypatch):
    monkeypatch.setenv("API_KEY", "secret123")
    require_api_key(x_api_key="secret123")  # does not raise
