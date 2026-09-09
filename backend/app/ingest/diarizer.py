import os
from pathlib import Path

import httpx

DEEPGRAM_PRERECORDED_URL = (
    "https://api.deepgram.com/v1/listen?model=nova-2&diarize=true&punctuate=true"
)


def diarize_audio(audio_path: Path) -> list[dict]:
    """
    Calls Deepgram's prerecorded API with diarize=true on the given audio
    file (the same file already transcribed by Whisper for chunking).
    Returns word-level entries: [{"start": float, "end": float, "speaker": int}, ...].
    Raises RuntimeError if DEEPGRAM_API_KEY is unset.

    NOTE: the response shape parsed here (results.channels[0].alternatives[0].words[],
    each word carrying a "speaker" int) is Deepgram's documented prerecorded-API
    diarization shape but is UNVERIFIED against a live call — this project has
    shipped bugs before from trusting a documented wire shape without checking it
    against a real response (see CLAUDE.md "Known gotchas": Anthropic streaming
    content blocks, SQLAlchemy `:param::type`). Etap 15 flagged the same risk for
    Deepgram's streaming API. Before relying on this in production, run a real
    multi-speaker clip through this function with a genuine DEEPGRAM_API_KEY,
    print the raw JSON, and add a regression test if the shape differs. That
    live verification is Task 10's job, not this module's.
    """
    api_key = os.environ.get("DEEPGRAM_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPGRAM_API_KEY not set — cannot diarize episode")

    with open(audio_path, "rb") as f:
        response = httpx.post(
            DEEPGRAM_PRERECORDED_URL,
            headers={"Authorization": f"Token {api_key}", "Content-Type": "audio/*"},
            content=f,
            timeout=120.0,
        )
    response.raise_for_status()
    data = response.json()
    words = data["results"]["channels"][0]["alternatives"][0]["words"]
    return [{"start": w["start"], "end": w["end"], "speaker": w["speaker"]} for w in words]


def merge_into_speaker_segments(words: list[dict]) -> list[dict]:
    """Collapse consecutive same-speaker words into segments: [{start_ts, end_ts, speaker}, ...]."""
    if not words:
        return []
    segments: list[dict] = []
    current = {"start_ts": words[0]["start"], "end_ts": words[0]["end"], "speaker": words[0]["speaker"]}
    for w in words[1:]:
        if w["speaker"] == current["speaker"]:
            current["end_ts"] = w["end"]
        else:
            segments.append(current)
            current = {"start_ts": w["start"], "end_ts": w["end"], "speaker": w["speaker"]}
    segments.append(current)
    return segments


def assign_speaker_labels(
    chunks: list[dict], words: list[dict], threshold: float = 0.5
) -> list[str | None]:
    """
    For each chunk (dict with start_ts/end_ts), returns the speaker label
    ("Speaker N") whose total overlap covers more than `threshold` of the
    chunk's duration, else None — an ambiguous or no-overlap chunk is never
    guessed at, per this etap's explicit design (overlapping/interrupting
    speech near a chunk boundary can otherwise misassign a speaker).
    """
    segments = merge_into_speaker_segments(words)
    labels: list[str | None] = []
    for chunk in chunks:
        chunk_start, chunk_end = chunk["start_ts"], chunk["end_ts"]
        duration = chunk_end - chunk_start
        if duration <= 0:
            labels.append(None)
            continue
        overlap_by_speaker: dict[int, float] = {}
        for seg in segments:
            overlap = min(chunk_end, seg["end_ts"]) - max(chunk_start, seg["start_ts"])
            if overlap > 0:
                overlap_by_speaker[seg["speaker"]] = overlap_by_speaker.get(seg["speaker"], 0.0) + overlap
        if not overlap_by_speaker:
            labels.append(None)
            continue
        best_speaker, best_overlap = max(overlap_by_speaker.items(), key=lambda kv: kv[1])
        labels.append(f"Speaker {best_speaker}" if best_overlap / duration > threshold else None)
    return labels
