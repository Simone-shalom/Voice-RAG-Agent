import socket
from unittest.mock import patch

import pytest

from app.core.url_safety import assert_safe_url


def _addrinfo(ip: str):
    return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, 0))]


def test_rejects_non_http_scheme():
    with pytest.raises(ValueError, match="scheme"):
        assert_safe_url("file:///etc/passwd")


def test_rejects_url_with_no_hostname():
    with pytest.raises(ValueError, match="hostname"):
        assert_safe_url("http://")


def test_rejects_unresolvable_host():
    with patch("app.core.url_safety.socket.getaddrinfo", side_effect=socket.gaierror("nope")):
        with pytest.raises(ValueError, match="could not resolve"):
            assert_safe_url("http://nonexistent.example")


def test_rejects_loopback_address():
    with patch("app.core.url_safety.socket.getaddrinfo", return_value=_addrinfo("127.0.0.1")):
        with pytest.raises(ValueError, match="disallowed address"):
            assert_safe_url("http://localhost/feed.xml")


def test_rejects_private_address():
    with patch("app.core.url_safety.socket.getaddrinfo", return_value=_addrinfo("10.0.0.5")):
        with pytest.raises(ValueError, match="disallowed address"):
            assert_safe_url("http://internal.example/feed.xml")


def test_rejects_link_local_metadata_address():
    with patch("app.core.url_safety.socket.getaddrinfo", return_value=_addrinfo("169.254.169.254")):
        with pytest.raises(ValueError, match="disallowed address"):
            assert_safe_url("http://metadata.example/latest")


def test_allows_public_address():
    with patch("app.core.url_safety.socket.getaddrinfo", return_value=_addrinfo("93.184.216.34")):
        assert_safe_url("https://example.com/feed.xml")


def test_rejects_unspecified_address():
    with patch("app.core.url_safety.socket.getaddrinfo", return_value=_addrinfo("0.0.0.0")):
        with pytest.raises(ValueError, match="disallowed address"):
            assert_safe_url("http://zero.example/")


def test_rejects_ipv4_mapped_loopback_address():
    with patch("app.core.url_safety.socket.getaddrinfo", return_value=_addrinfo("::ffff:127.0.0.1")):
        with pytest.raises(ValueError, match="disallowed address"):
            assert_safe_url("http://mapped.example/")
