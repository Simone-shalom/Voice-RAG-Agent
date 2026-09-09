import logging

logger = logging.getLogger(__name__)


def log_latency(stage: str, ms: float, thread_id: str) -> None:
    """
    Structured latency log line — stage is a short label (e.g.
    "first_audio_byte"), ms is the measured duration. Read these back
    with `docker compose logs backend | grep latency_ms` to compute
    p50/p95 for DECISIONS.md.
    """
    logger.info(
        "voice latency: %s = %.1fms (thread %s)",
        stage,
        ms,
        thread_id,
        extra={"stage": stage, "latency_ms": ms, "thread_id": thread_id},
    )
