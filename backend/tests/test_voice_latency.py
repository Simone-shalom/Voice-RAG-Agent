import logging

from app.voice.latency import log_latency


def test_log_latency_emits_structured_log_line(caplog):
    with caplog.at_level(logging.INFO, logger="app.voice.latency"):
        log_latency("first_audio_byte", 642.5, "thread-abc")

    assert len(caplog.records) == 1
    record = caplog.records[0]
    assert record.stage == "first_audio_byte"
    assert record.latency_ms == 642.5
    assert record.thread_id == "thread-abc"
    assert "642" in record.message
