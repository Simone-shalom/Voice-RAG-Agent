import os
os.environ.setdefault("DATABASE_URL", "postgresql://voicerag:voicerag@localhost:5433/voicerag")

import inspect
from unittest.mock import MagicMock, patch
from app.db.session import get_db, SessionLocal, init_db


def test_get_db_is_generator():
    assert inspect.isgeneratorfunction(get_db)


def test_session_local_exists():
    assert SessionLocal is not None


def test_init_db_is_callable():
    assert callable(init_db)


def test_init_db_sql_contains_tsvector():
    """init_db must run the tsvector migration — verify it calls execute at least 3 times."""
    mock_engine = MagicMock()
    mock_conn = MagicMock()
    mock_engine.connect.return_value.__enter__ = lambda s: mock_conn
    mock_engine.connect.return_value.__exit__ = MagicMock(return_value=False)

    with patch("app.db.models.Base"):
        init_db(mock_engine)

    # CREATE EXTENSION, tsvector DO block, CREATE INDEX
    assert mock_conn.execute.call_count >= 3


def test_init_db_creates_tables_before_altering_chunks():
    """
    Regression test: on a brand-new database, `chunks` doesn't exist until
    Base.metadata.create_all() runs — the ALTER TABLE/CREATE INDEX
    statements that target `chunks` must run after that, not before, or
    they fail with psycopg.errors.UndefinedTable. This shipped broken
    once — every local dev DB already had `chunks` from earlier runs,
    masking the ordering bug; it only surfaced against a genuinely fresh
    database during a first production deploy.
    """
    mock_engine = MagicMock()
    mock_conn = MagicMock()
    mock_engine.connect.return_value.__enter__ = lambda s: mock_conn
    mock_engine.connect.return_value.__exit__ = MagicMock(return_value=False)

    call_order = []
    mock_conn.execute.side_effect = lambda stmt, *a, **k: call_order.append(str(stmt))

    with patch("app.db.models.Base") as mock_base:
        mock_base.metadata.create_all.side_effect = lambda **k: call_order.append("create_all")
        init_db(mock_engine)

    create_all_index = call_order.index("create_all")
    chunks_stmt_index = next(i for i, c in enumerate(call_order) if "chunks" in c)
    assert create_all_index < chunks_stmt_index


def test_init_db_adds_speaker_label_and_summary_columns():
    """
    Regression test for the same class of bug test_init_db_creates_tables_before_altering_chunks
    guards: Base.metadata.create_all() only creates missing tables, it never alters
    an existing chunks/episodes table already deployed without these columns — an
    idempotent ALTER TABLE is required, same as the text_search column (Etap 13).
    """
    mock_engine = MagicMock()
    mock_conn = MagicMock()
    mock_engine.connect.return_value.__enter__ = lambda s: mock_conn
    mock_engine.connect.return_value.__exit__ = MagicMock(return_value=False)

    executed_sql = []
    mock_conn.execute.side_effect = lambda stmt, *a, **k: executed_sql.append(str(stmt))

    with patch("app.db.models.Base"):
        init_db(mock_engine)

    combined = "\n".join(executed_sql)
    assert "speaker_label" in combined
    assert "ADD COLUMN" in combined and "speaker_label" in combined
    assert "summary" in combined
    assert "chapters" in combined
