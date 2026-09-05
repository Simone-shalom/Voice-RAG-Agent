import os
os.environ.setdefault("DATABASE_URL", "postgresql://voicerag:voicerag@localhost:5433/voicerag")

import inspect
from app.db.session import get_db, SessionLocal, init_db


def test_get_db_is_generator():
    assert inspect.isgeneratorfunction(get_db)


def test_session_local_exists():
    assert SessionLocal is not None


def test_init_db_is_callable():
    assert callable(init_db)
