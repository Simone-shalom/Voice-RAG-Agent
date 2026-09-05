import os

from fastapi import Header, HTTPException


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """
    Gates a router behind a shared API key passed as the X-API-Key header.
    No-op when the API_KEY env var is unset (local dev / test default) —
    only enforced once an operator opts in for a public deployment.
    """
    expected = os.environ.get("API_KEY")
    if not expected:
        return
    if x_api_key != expected:
        raise HTTPException(status_code=401, detail="invalid or missing API key")
