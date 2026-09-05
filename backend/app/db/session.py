import os

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

DATABASE_URL = os.environ["DATABASE_URL"]

# This project depends on psycopg3 (`psycopg[binary]` in requirements.txt),
# not the legacy psycopg2. SQLAlchemy's bare "postgresql://" scheme defaults
# to the psycopg2 dialect, which isn't installed here, so normalize to the
# psycopg3 dialect explicitly. Values that already specify a driver
# (e.g. "postgresql+psycopg://") are left untouched.
_ENGINE_URL = DATABASE_URL
if _ENGINE_URL.startswith("postgresql://"):
    _ENGINE_URL = "postgresql+psycopg://" + _ENGINE_URL[len("postgresql://"):]

engine = create_engine(_ENGINE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def init_db(engine_):
    """Enable pgvector extension and create all tables."""
    from .models import Base

    with engine_.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()
    Base.metadata.create_all(bind=engine_)
