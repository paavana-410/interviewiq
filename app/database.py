"""SQLAlchemy engine + session factory."""
from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from app.config import get_settings


class Base(DeclarativeBase):
    pass


_engine = None
_SessionLocal = None


def get_engine():
    global _engine
    if _engine is None:
        settings = get_settings()
        _engine = create_engine(
            settings.database_url,
            pool_pre_ping=True,
            pool_size=10,
            max_overflow=20,
            echo=False,
        )
    return _engine


def get_session_local():
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=get_engine())
    return _SessionLocal


# Keep these module-level names for Alembic compatibility
@property
def engine():
    return get_engine()


SessionLocal = None  # lazy; use get_session_local() directly


def get_db():
    """FastAPI dependency: yields a DB session, closes on exit."""
    SL = get_session_local()
    db = SL()
    try:
        yield db
    finally:
        db.close()
