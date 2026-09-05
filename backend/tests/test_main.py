def test_health_still_works(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ingest_router_mounted(client):
    response = client.post("/ingest/upload")
    assert response.status_code == 422  # missing file, not 404


def test_search_router_mounted(client):
    response = client.get("/search")
    assert response.status_code == 422  # missing q, not 404
