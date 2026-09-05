import ipaddress
import socket
from urllib.parse import urlparse

ALLOWED_SCHEMES = {"http", "https"}


def assert_safe_url(url: str) -> None:
    """
    Raises ValueError if `url` is unsafe for the server to fetch: a
    non-http(s) scheme, or a hostname that resolves to a private,
    loopback, link-local, reserved, or multicast address. Blocks SSRF
    against internal services and cloud metadata endpoints reachable
    from server-side URL/RSS ingest.
    """
    parsed = urlparse(url)
    if parsed.scheme not in ALLOWED_SCHEMES:
        raise ValueError(f"unsupported URL scheme: {parsed.scheme!r}")
    if not parsed.hostname:
        raise ValueError("URL has no hostname")

    try:
        addrinfo = socket.getaddrinfo(parsed.hostname, None)
    except socket.gaierror as e:
        raise ValueError(f"could not resolve host: {parsed.hostname}") from e

    for _family, _type, _proto, _canon, sockaddr in addrinfo:
        ip = ipaddress.ip_address(sockaddr[0])
        if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped:
            ip = ip.ipv4_mapped
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
            or ip.is_unspecified
        ):
            raise ValueError(f"URL resolves to a disallowed address: {ip}")
