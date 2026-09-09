import os
from functools import lru_cache

import anthropic

SUMMARIZER_MODEL = "claude-haiku-4-5-20251001"
MAX_TRANSCRIPT_CHARS = 70000  # ~ a 60-70 min podcast's worth of chunked transcript

_SUMMARY_TOOL = {
    "name": "emit_episode_summary",
    "description": "Emit a structured summary of a podcast episode transcript.",
    "input_schema": {
        "type": "object",
        "properties": {
            "summary": {
                "type": "string",
                "description": "A 2-4 sentence summary of the episode's content.",
            },
            "chapters": {
                "type": "array",
                "description": "Chapter markers at natural topic-change points in the episode.",
                "items": {
                    "type": "object",
                    "properties": {
                        "start_ts": {"type": "number", "description": "Start time in seconds"},
                        "title": {"type": "string", "description": "Short chapter title"},
                    },
                    "required": ["start_ts", "title"],
                },
            },
            "speaker_names": {
                "type": "object",
                "description": (
                    "Best-effort mapping from 'Speaker N' labels to real names actually "
                    "stated in the transcript (e.g. someone saying 'I'm Gary Jordan, your "
                    "host'). Use the exact 'Speaker N' labels that prefix lines in the "
                    "transcript above — do not invent or renumber speaker indices. Omit "
                    "any speaker whose real name is never stated, and omit this mapping "
                    "entirely if the transcript has no speaker labels — never guess."
                ),
                "additionalProperties": {"type": "string"},
            },
        },
        "required": ["summary", "chapters", "speaker_names"],
    },
}


@lru_cache(maxsize=1)
def _client() -> anthropic.Anthropic:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set — cannot summarize episode")
    return anthropic.Anthropic(api_key=api_key)


def _format_transcript(chunks: list[dict]) -> str:
    lines = []
    for c in chunks:
        speaker = c.get("speaker_label")
        prefix = f"{speaker}: " if speaker else ""
        lines.append(f"[{c['start_ts']:.1f}s] {prefix}{c['text']}")
    text = "\n".join(lines)
    if len(text) > MAX_TRANSCRIPT_CHARS:
        text = text[:MAX_TRANSCRIPT_CHARS] + "\n[...transcript truncated...]"
    return text


def summarize_episode(chunks: list[dict]) -> dict:
    """
    One Claude Haiku call over the chunked transcript, forced into structured
    output via tool-use (never parses free text with regex — free-text
    structured output is known to be unstable in this exact way, see
    CLAUDE.md's gotchas for two prior incidents of trusting an assumed shape).

    Returns {"summary": str, "chapters": [{"start_ts": float, "title": str}, ...],
    "speaker_names": {"Speaker N": str, ...}} — chapters/speaker_names are
    always present, possibly empty. Raises RuntimeError if ANTHROPIC_API_KEY
    is unset, or if the model doesn't return the expected tool-use block.
    """
    if not chunks:
        return {"summary": "", "chapters": [], "speaker_names": {}}

    transcript = _format_transcript(chunks)
    message = _client().messages.create(
        model=SUMMARIZER_MODEL,
        max_tokens=1024,
        tools=[_SUMMARY_TOOL],
        tool_choice={"type": "tool", "name": "emit_episode_summary"},
        messages=[
            {
                "role": "user",
                "content": (
                    "Summarize this podcast episode transcript. Each line is "
                    "timestamped in seconds from the start of the episode.\n\n"
                    f"{transcript}"
                ),
            }
        ],
    )
    for block in message.content:
        if block.type == "tool_use":
            return {
                "summary": block.input.get("summary", ""),
                "chapters": block.input.get("chapters", []),
                "speaker_names": block.input.get("speaker_names", {}),
            }
    raise RuntimeError("Claude did not return a structured summary")
