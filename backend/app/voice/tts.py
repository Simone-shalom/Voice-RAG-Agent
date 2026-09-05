import os

import httpx

ELEVENLABS_TTS_URL = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
DEFAULT_VOICE_ID = "JBFqnCBsd6RMkjVDRZzb"  # ElevenLabs "George"
MAX_TTS_CHARS = 4000


def synthesise(text: str, voice_id: str = DEFAULT_VOICE_ID) -> bytes:
    """
    Convert text to MP3 audio using ElevenLabs TTS REST API (via httpx).
    Returns raw MP3 bytes.
    Raises ValueError for empty/oversized text, RuntimeError if API key unset.
    """
    if not text:
        raise ValueError("text is empty")
    if len(text) > MAX_TTS_CHARS:
        raise ValueError(f"text exceeds {MAX_TTS_CHARS} character limit")
    api_key = os.environ.get("ELEVENLABS_API_KEY")
    if not api_key:
        raise RuntimeError("ELEVENLABS_API_KEY not set — cannot use TTS")

    response = httpx.post(
        ELEVENLABS_TTS_URL.format(voice_id=voice_id),
        headers={"xi-api-key": api_key, "Content-Type": "application/json"},
        json={"text": text, "model_id": "eleven_multilingual_v2"},
        timeout=30.0,
    )
    response.raise_for_status()
    return response.content
