PAUSE_THRESHOLD = 0.5   # seconds — pauses longer than this trigger a chunk boundary
MAX_CHUNK_WORDS = 300    # hard cap to prevent runaway chunks


def chunk_segments(segments: list[dict]) -> list[dict]:
    """
    Split Whisper segments into chunks at natural speech boundaries.

    A boundary occurs when:
    - the pause between two consecutive segments exceeds PAUSE_THRESHOLD, OR
    - the accumulated word count hits MAX_CHUNK_WORDS.

    Each returned chunk has: {start_ts, end_ts, text}
    """
    # Whisper sometimes emits whitespace-only segments (silence markers).
    # Discard them early: an empty string passed to the embedding API raises HTTP 400.
    segments = [s for s in segments if s["text"].strip()]
    if not segments:
        return []

    chunks: list[dict] = []
    current_texts: list[str] = []
    current_start: float = segments[0]["start"]
    current_words = 0

    for i, seg in enumerate(segments):
        seg_text = seg["text"].strip()
        current_texts.append(seg_text)
        current_words += len(seg_text.split())

        is_last = i == len(segments) - 1
        next_seg = segments[i + 1] if not is_last else None
        pause = (next_seg["start"] - seg["end"]) if next_seg else 0.0

        should_split = is_last or pause >= PAUSE_THRESHOLD or current_words >= MAX_CHUNK_WORDS

        if should_split:
            chunks.append(
                {
                    "start_ts": current_start,
                    "end_ts": seg["end"],
                    "text": " ".join(current_texts),
                }
            )
            if not is_last:
                current_texts = []
                current_start = next_seg["start"]
                current_words = 0

    return chunks
