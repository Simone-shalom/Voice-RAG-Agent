import os

from fastapi import Request
from slowapi import Limiter
from slowapi.util import get_remote_address


def _client_key(request: Request) -> str:
    """
    Prefers the original client IP from X-Forwarded-For (set by a reverse
    proxy such as the Next.js rewrite in dev, or a production edge/load
    balancer) over the immediate TCP peer, which would otherwise be the
    proxy itself for every request and collapse all users into one shared
    rate-limit bucket.
    """
    forwarded_for = request.headers.get("x-forwarded-for")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    return get_remote_address(request)


limiter = Limiter(
    key_func=_client_key,
    enabled=os.environ.get("RATE_LIMIT_ENABLED", "true").lower() == "true",
)
