import os
from functools import lru_cache

from openai import OpenAI

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536


@lru_cache(maxsize=1)
def _client() -> OpenAI:
    return OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed texts with OpenAI text-embedding-3-small. Returns one 1536-dim vector per text."""
    if not texts:
        return []
    response = _client().embeddings.create(input=texts, model=EMBEDDING_MODEL)
    return [item.embedding for item in response.data]
