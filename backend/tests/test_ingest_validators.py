import io

import pytest

from app.ingest.validators import MAX_UPLOAD_BYTES, is_probably_audio, read_capped


def test_recognizes_mp3_with_id3_tag():
    assert is_probably_audio(b"ID3\x03\x00\x00\x00")


def test_recognizes_mp3_frame_sync_without_id3():
    assert is_probably_audio(b"\xff\xfb\x90\x00")


def test_recognizes_wav():
    assert is_probably_audio(b"RIFF\x00\x00\x00\x00WAVEfmt ")


def test_recognizes_ogg():
    assert is_probably_audio(b"OggS\x00\x02\x00\x00")


def test_recognizes_m4a():
    assert is_probably_audio(b"\x00\x00\x00\x18ftypM4A ")


def test_recognizes_webm():
    assert is_probably_audio(b"\x1a\x45\xdf\xa3\x01\x00\x00\x00")


def test_rejects_unknown_binary():
    assert not is_probably_audio(b"PK\x03\x04random zip bytes")


def test_rejects_riff_without_wave_subtype():
    assert not is_probably_audio(b"RIFF\x00\x00\x00\x00WEBPVP8 ")


def test_rejects_empty_header():
    assert not is_probably_audio(b"")


def test_read_capped_returns_full_content_under_limit():
    data = b"a" * 100
    assert read_capped(io.BytesIO(data), max_bytes=1000) == data


def test_read_capped_raises_when_over_limit():
    data = b"a" * 2000
    with pytest.raises(ValueError, match="exceeds"):
        read_capped(io.BytesIO(data), max_bytes=1000)


def test_max_upload_bytes_is_reasonable():
    assert 1_000_000 < MAX_UPLOAD_BYTES <= 500 * 1024 * 1024
