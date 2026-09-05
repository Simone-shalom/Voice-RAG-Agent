import re
from typing import Generator

_BOUNDARY = re.compile(r"(?<=[.!?])\s+")
MIN_WORDS = 8  # minimum words before a chunk is sent to TTS — keeps audio segments natural-sounding


class SentenceChunker:
    """
    Accumulates streamed tokens and yields sentence-boundary chunks
    of at least MIN_WORDS words. Short leading sentences (e.g. "Hi.")
    are merged with subsequent sentences until the threshold is met.

    Usage:
        chunker = SentenceChunker()
        for token in token_stream:
            for chunk in chunker.push(token):
                synthesise(chunk)
        for chunk in chunker.flush():
            synthesise(chunk)
    """

    def __init__(self) -> None:
        self._buf = ""

    def push(self, token: str) -> Generator[str, None, None]:
        """Feed one streamed token; yields 0+ complete chunks ready for TTS."""
        self._buf += token
        while True:
            match = None
            for match in _BOUNDARY.finditer(self._buf):
                candidate = self._buf[: match.start()]
                if len(candidate.split()) >= MIN_WORDS:
                    yield candidate.strip()
                    self._buf = self._buf[match.end():]
                    break
            else:
                break

    def flush(self) -> Generator[str, None, None]:
        """Call once the token stream ends — yields any remaining buffered text as a final chunk."""
        remainder = self._buf.strip()
        self._buf = ""
        if remainder:
            yield remainder
