from app.search.rrf import reciprocal_rank_fusion


def make_chunk(chunk_id, similarity=0.5):
    return {"chunk_id": chunk_id, "episode_id": "ep1", "start_ts": 0.0, "end_ts": 5.0,
            "text": f"text {chunk_id}", "similarity": similarity}


def test_single_list_preserves_order():
    results = [make_chunk("a", 0.9), make_chunk("b", 0.5), make_chunk("c", 0.1)]
    fused = reciprocal_rank_fusion([results])
    assert [r["chunk_id"] for r in fused] == ["a", "b", "c"]


def test_chunk_in_both_lists_ranked_first():
    list1 = [make_chunk("a"), make_chunk("b")]
    list2 = [make_chunk("b"), make_chunk("c")]
    fused = reciprocal_rank_fusion([list1, list2])
    assert fused[0]["chunk_id"] == "b"


def test_empty_lists_return_empty():
    assert reciprocal_rank_fusion([[], []]) == []


def test_one_empty_one_nonempty():
    results = [make_chunk("x"), make_chunk("y")]
    fused = reciprocal_rank_fusion([results, []])
    assert [r["chunk_id"] for r in fused] == ["x", "y"]


def test_result_contains_rrf_score():
    results = [make_chunk("a")]
    fused = reciprocal_rank_fusion([results])
    assert "rrf_score" in fused[0]
    assert isinstance(fused[0]["rrf_score"], float)


def test_rrf_uses_ranks_not_raw_scores():
    # "a" is rank 1 in list1 and rank 2 in list2; "b" is rank 1 in list2 only
    # a: 1/(60+1) + 1/(60+2) = 0.01639 + 0.01613 = 0.03252
    # b: 1/(60+1) = 0.01639
    list1 = [make_chunk("a", similarity=0.99)]
    list2 = [make_chunk("b", similarity=0.01), make_chunk("a", similarity=0.98)]
    fused = reciprocal_rank_fusion([list1, list2])
    assert fused[0]["chunk_id"] == "a"
