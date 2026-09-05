def reciprocal_rank_fusion(result_lists: list[list[dict]], k: int = 60) -> list[dict]:
    """
    Fuse multiple ranked result lists using Reciprocal Rank Fusion.

    Each list must be sorted best-first. Chunks are identified by 'chunk_id'.
    The same chunk appearing in multiple lists receives contributions from each.

    Returns a new list sorted by total RRF score descending, with 'rrf_score' added.
    """
    scores: dict[str, float] = {}
    chunk_data: dict[str, dict] = {}

    for results in result_lists:
        for rank, chunk in enumerate(results, start=1):
            cid = chunk["chunk_id"]
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
            chunk_data.setdefault(cid, chunk)

    return [
        {**chunk_data[cid], "rrf_score": score}
        for cid, score in sorted(scores.items(), key=lambda x: -x[1])
    ]
