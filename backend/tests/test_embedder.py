import os
os.environ.setdefault("OPENAI_API_KEY", "test-key")

from unittest.mock import MagicMock, patch
from app.ingest.embedder import embed_texts


def make_mock_embedding_response(n: int):
    response = MagicMock()
    response.data = [MagicMock(embedding=[0.1] * 1536) for _ in range(n)]
    return response


def make_mock_client(n: int = 1):
    mock = MagicMock()
    mock.embeddings.create.return_value = make_mock_embedding_response(n)
    return mock


def test_embed_texts_returns_list_of_vectors():
    with patch("app.ingest.embedder._client", return_value=make_mock_client(2)):
        result = embed_texts(["Hello world.", "Another sentence."])

    assert len(result) == 2
    assert len(result[0]) == 1536


def test_embed_texts_uses_correct_model():
    mock_client = make_mock_client(1)
    with patch("app.ingest.embedder._client", return_value=mock_client):
        embed_texts(["test"])

    assert mock_client.embeddings.create.call_args.kwargs["model"] == "text-embedding-3-small"


def test_embed_empty_list_returns_empty():
    assert embed_texts([]) == []
