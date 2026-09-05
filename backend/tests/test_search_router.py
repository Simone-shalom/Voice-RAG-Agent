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
