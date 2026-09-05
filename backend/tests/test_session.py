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
