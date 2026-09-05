import os
from functools import lru_cache
from pathlib import Path

from openai import OpenAI


@lru_cache(maxsize=1)
def _client() -> OpenAI:
    return OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def transcribe(audio_path: Path) -> list[dict]:
    """
    Transcribe audio using the OpenAI Whisper API with word-level timestamps.
    Returns list of segment dicts: {text, start, end}.
    """
    with open(audio_path, "rb") as f:
        response = _client().audio.transcriptions.create(
            model="whisper-1",
            file=f,
            response_format="verbose_json",
            timestamp_granularities=["word", "segment"],
        )
    return [{"text": seg.text, "start": seg.start, "end": seg.end} for seg in response.segments]
