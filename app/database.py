"""SQLAlchemy engine + session factory.

Supports both SQLite (default, no server needed) and PostgreSQL.
SQLite is the default for laptops; set DATABASE_URL=postgresql://... for Postgres.
"""
from __future__ import annotations

from sqlalchemy import create_engine, event
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from app.config import get_settings


class Base(DeclarativeBase):
    pass


_engine = None
_SessionLocal = None


def _is_sqlite(url: str) -> bool:
    return url.startswith("sqlite")


def get_engine():
    global _engine
    if _engine is None:
        settings = get_settings()
        url = settings.database_url
        if _is_sqlite(url):
            # SQLite: multi-thread safe with 30s busy timeout + WAL mode
            _engine = create_engine(
                url,
                connect_args={"check_same_thread": False, "timeout": 30},
                echo=False,
            )
            # Enable WAL mode, busy timeout, and foreign keys for SQLite
            @event.listens_for(_engine, "connect")
            def _sqlite_pragmas(dbapi_conn, _):
                cursor = dbapi_conn.cursor()
                cursor.execute("PRAGMA journal_mode=WAL")
                cursor.execute("PRAGMA busy_timeout=30000")
                cursor.execute("PRAGMA synchronous=NORMAL")
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.close()
        else:
            _engine = create_engine(
                url,
                pool_pre_ping=True,
                pool_size=5,
                max_overflow=10,
                echo=False,
            )
    return _engine


def get_session_local():
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(
            autocommit=False, autoflush=False, bind=get_engine()
        )
    return _SessionLocal


def get_db():
    """FastAPI dependency: yields a DB session, rolls back on error, closes on exit."""
    SL = get_session_local()
    db = SL()
    try:
        yield db
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
