from unittest.mock import MagicMock

from app.core.rate_limit import _client_key


def _fake_request(headers: dict, client_host: str = "1.1.1.1") -> MagicMock:
    request = MagicMock()
    request.headers = headers
    request.client.host = client_host
    return request


def test_uses_forwarded_for_header_when_present():
    request = _fake_request({"x-forwarded-for": "203.0.113.5, 10.0.0.1"})
    assert _client_key(request) == "203.0.113.5"


def test_falls_back_to_remote_address_without_header():
    request = _fake_request({})
    assert _client_key(request) == "1.1.1.1"
