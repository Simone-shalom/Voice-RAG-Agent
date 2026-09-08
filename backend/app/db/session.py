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
    """Enable pgvector extension, create all tables, add tsvector search column."""
    from .models import Base

    with engine_.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()

    # Must run after the extension exists (Vector columns need it) and before
    # the ALTER/INDEX below (both target the `chunks` table this creates) —
    # on a brand-new database `chunks` doesn't exist yet, so those would fail
    # with psycopg.errors.UndefinedTable if run first. Only ever caught this
    # against a fresh DB (e.g. a first production deploy); local dev DBs keep
    # `chunks` around from earlier runs, which masked the ordering bug.
    Base.metadata.create_all(bind=engine_)

    with engine_.connect() as conn:
        conn.execute(
            text(
                """
                DO $$
                BEGIN
                  IF NOT EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_name = 'chunks' AND column_name = 'text_search'
                  ) THEN
                    ALTER TABLE chunks
                    ADD COLUMN text_search tsvector
                    GENERATED ALWAYS AS (to_tsvector('english', text)) STORED;
                  END IF;
                END
                $$
                """
            )
        )
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS chunks_text_search_gin "
                "ON chunks USING GIN(text_search)"
            )
        )
        conn.commit()
