from unittest.mock import patch

MOCK_RESULTS = [
    {
        "chunk_id": "aaa",
        "episode_id": "bbb",
        "start_ts": 10.0,
        "end_ts": 25.0,
        "text": "Relevant content here.",
        "similarity": 0.87,
    }
]

MOCK_HYBRID_RESULTS = [
    {
        "chunk_id": "hhh",
        "episode_id": "eee",
        "start_ts": 5.0,
        "end_ts": 20.0,
        "text": "Hybrid result.",
        "similarity": 0.75,
    }
]


def test_search_returns_results(client):
    with patch("app.search.router.semantic_search", return_value=MOCK_RESULTS):
        response = client.get("/search?q=test+query")

    assert response.status_code == 200
    body = response.json()
    assert len(body) == 1
    assert body[0]["chunk_id"] == "aaa"
    assert body[0]["start_ts"] == 10.0


def test_search_missing_query_returns_422(client):
    response = client.get("/search")
    assert response.status_code == 422


def test_search_passes_limit_to_searcher(client):
    with patch("app.search.router.semantic_search", return_value=[]) as mock_search:
        client.get("/search?q=hello&limit=5")

    assert mock_search.call_args.kwargs["limit"] == 5


def test_search_returns_400_on_searcher_failure(client):
    with patch("app.search.router.semantic_search", side_effect=RuntimeError("embedding failed")):
        response = client.get("/search?q=test+query")

    assert response.status_code == 400
    assert "embedding failed" in response.json()["detail"]


def test_search_defaults_to_semantic_mode(client):
    with patch("app.search.router.semantic_search", return_value=MOCK_RESULTS) as m:
        response = client.get("/search?q=hello")
    assert response.status_code == 200
    m.assert_called_once()


def test_search_bm25_mode_calls_bm25(client):
    with patch("app.search.router.bm25_search", return_value=MOCK_RESULTS) as m:
        response = client.get("/search?q=hello&mode=bm25")
    assert response.status_code == 200
    m.assert_called_once()


def test_search_hybrid_mode_calls_hybrid(client):
    with patch("app.search.router.hybrid_search", return_value=MOCK_HYBRID_RESULTS) as m:
        response = client.get("/search?q=hello&mode=hybrid")
    assert response.status_code == 200
    m.assert_called_once()


def test_search_hybrid_rerank_mode_calls_hybrid_rerank(client):
    with patch("app.search.router.hybrid_rerank_search", return_value=MOCK_HYBRID_RESULTS) as m:
        response = client.get("/search?q=hello&mode=hybrid%2Brerank")
    assert response.status_code == 200
    m.assert_called_once()


def test_search_invalid_mode_returns_422(client):
    response = client.get("/search?q=hello&mode=unknown")
    assert response.status_code == 422


def test_search_hybrid_rerank_no_key_returns_400(client):
    with patch("app.search.router.hybrid_rerank_search",
               side_effect=RuntimeError("COHERE_API_KEY not set")):
        response = client.get("/search?q=hello&mode=hybrid%2Brerank")
    assert response.status_code == 400
    assert "COHERE_API_KEY" in response.json()["detail"]
