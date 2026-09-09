from unittest.mock import patch

from app.core.allowed_origins import get_allowed_origins


def test_parses_comma_separated_origins():
    with patch.dict("os.environ", {"ALLOWED_ORIGINS": "https://a.com, https://b.com"}):
        assert get_allowed_origins() == ["https://a.com", "https://b.com"]


def test_defaults_to_localhost_3000():
    with patch.dict("os.environ", {}, clear=True):
        assert get_allowed_origins() == ["http://localhost:3000"]


def test_strips_blank_entries():
    with patch.dict("os.environ", {"ALLOWED_ORIGINS": "https://a.com,,  "}):
        assert get_allowed_origins() == ["https://a.com"]
