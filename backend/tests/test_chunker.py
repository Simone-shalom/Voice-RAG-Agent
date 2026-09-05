from app.ingest.chunker import chunk_segments


def make_seg(text, start, end):
    return {"text": text, "start": start, "end": end}


def test_empty_input_returns_empty():
    assert chunk_segments([]) == []


def test_single_segment_returns_one_chunk():
    segs = [make_seg("Hello world.", 0.0, 1.5)]
    chunks = chunk_segments(segs)
    assert len(chunks) == 1
    assert chunks[0]["start_ts"] == 0.0
    assert chunks[0]["end_ts"] == 1.5
    assert "Hello world" in chunks[0]["text"]


def test_pause_splits_into_two_chunks():
    segs = [
        make_seg("First sentence.", 0.0, 2.0),
        make_seg("Second sentence.", 3.0, 5.0),  # pause = 1.0s, above threshold
    ]
    chunks = chunk_segments(segs)
    assert len(chunks) == 2
    assert chunks[0]["start_ts"] == 0.0
    assert chunks[0]["end_ts"] == 2.0
    assert chunks[1]["start_ts"] == 3.0
    assert chunks[1]["end_ts"] == 5.0


def test_no_pause_keeps_in_one_chunk():
    segs = [
        make_seg("First.", 0.0, 1.0),
        make_seg("Second.", 1.2, 2.0),  # pause = 0.2s, below threshold
    ]
    chunks = chunk_segments(segs)
    assert len(chunks) == 1
    assert chunks[0]["start_ts"] == 0.0
    assert chunks[0]["end_ts"] == 2.0


def test_chunk_text_has_no_leading_trailing_whitespace():
    segs = [
        make_seg("Alpha beta.", 0.0, 2.0),
        make_seg("Gamma delta.", 3.0, 5.0),
    ]
    chunks = chunk_segments(segs)
    for chunk in chunks:
        assert chunk["text"] == chunk["text"].strip()


def test_whitespace_only_segment_is_dropped():
    segs = [
        make_seg("  ", 0.0, 0.5),  # noise / silence marker — dropped
        make_seg("Hello world.", 1.0, 2.5),
    ]
    chunks = chunk_segments(segs)
    assert len(chunks) == 1
    assert chunks[0]["text"] == "Hello world."
    assert chunks[0]["start_ts"] == 1.0


def test_all_whitespace_segments_returns_empty():
    segs = [make_seg("  ", 0.0, 0.5), make_seg("\n", 1.0, 1.5)]
    assert chunk_segments(segs) == []


def test_word_count_limit_splits_long_chunk():
    # 30 segments of 15 words each = 450 words, above MAX_CHUNK_WORDS=300, pauses all small
    segs = [make_seg(" ".join(["word"] * 15), i * 1.1, i * 1.1 + 1.0) for i in range(30)]
    chunks = chunk_segments(segs)
    assert len(chunks) >= 2
    for chunk in chunks:
        assert len(chunk["text"].split()) <= 350  # cap plus one segment of slack

    # No word from the original segments is lost or duplicated across the split.
    original_words = [w for seg in segs for w in seg["text"].split()]
    reconstructed_words = [w for chunk in chunks for w in chunk["text"].split()]
    assert reconstructed_words == original_words

    # Boundaries are derived from real segment timestamps, contiguous and non-overlapping.
    seg_starts = {seg["start"] for seg in segs}
    seg_ends = {seg["end"] for seg in segs}
    for chunk in chunks:
        assert chunk["start_ts"] in seg_starts
        assert chunk["end_ts"] in seg_ends
    for i in range(len(chunks) - 1):
        assert chunks[i]["end_ts"] <= chunks[i + 1]["start_ts"]
