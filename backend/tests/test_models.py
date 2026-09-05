from app.db.models import Episode, Chunk


def test_episode_has_required_columns():
    cols = {c.name for c in Episode.__table__.columns}
    assert {"id", "filename", "source_url", "created_at"} <= cols


def test_chunk_has_required_columns():
    cols = {c.name for c in Chunk.__table__.columns}
    assert {"id", "episode_id", "start_ts", "end_ts", "text", "embedding"} <= cols


def test_chunk_embedding_dimension():
    embedding_col = Chunk.__table__.columns["embedding"]
    assert embedding_col.type.dim == 1536
