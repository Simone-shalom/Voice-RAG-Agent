import io
import os
from functools import lru_cache

from openai import OpenAI


@lru_cache(maxsize=1)
def _client() -> OpenAI:
    return OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def transcribe_audio(file_bytes: bytes, filename: str) -> str:
    """
    Transcribe audio bytes to text using Whisper API.
    Returns the full transcript as a plain string.
    Raises ValueError for empty input.
    """
    if not file_bytes:
        raise ValueError("audio bytes are empty")
    audio_file = io.BytesIO(file_bytes)
    audio_file.name = filename
    response = _client().audio.transcriptions.create(
        model="whisper-1",
        file=audio_file,
    )
    return response.text
