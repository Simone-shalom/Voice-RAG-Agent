MAX_UPLOAD_BYTES = 200 * 1024 * 1024  # 200 MB

_MAGIC_SIGNATURES = (
    (0, b"ID3"),               # MP3 with ID3 tag
    (0, b"OggS"),              # Ogg
    (4, b"ftyp"),              # MP4 / M4A
    (0, b"\x1a\x45\xdf\xa3"),  # WebM / Matroska (EBML header)
)
_MPEG_FRAME_SYNCS = (b"\xff\xfb", b"\xff\xf3", b"\xff\xfa", b"\xff\xf2")


def is_probably_audio(header: bytes) -> bool:
    """
    Sniffs the first bytes of a file for known audio-container magic
    numbers. Conservative allow-list (not exhaustive — e.g. a bare MPEG
    frame with an unusual sync byte still passes since Whisper handles it)
    — rejects obviously non-audio uploads without a libmagic dependency.
    """
    if header[:2] in _MPEG_FRAME_SYNCS:
        return True
    if header[:4] == b"RIFF" and header[8:12] == b"WAVE":
        return True  # RIFF alone also matches WebP/AVI — require the WAVE subtype too
    return any(header[offset:offset + len(sig)] == sig for offset, sig in _MAGIC_SIGNATURES)


def read_capped(fileobj, max_bytes: int = MAX_UPLOAD_BYTES) -> bytes:
    """Reads fileobj in chunks, raising ValueError if it exceeds max_bytes."""
    chunks = []
    total = 0
    while True:
        chunk = fileobj.read(1024 * 1024)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise ValueError(f"upload exceeds {max_bytes} byte limit")
        chunks.append(chunk)
    return b"".join(chunks)
