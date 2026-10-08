"""API endpoint smoke tests using TestClient (no real DB or audio)."""
from __future__ import annotations

import io
import os
import pytest

# Set mock environment before any app imports
os.environ["LLM_PROVIDER"] = "mock"
os.environ["DATABASE_URL"] = "sqlite:///./test_api.db"

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base, get_db
from app.main import app


# ── In-memory SQLite DB for API tests ────────────────────────────────────────

TEST_DB_URL = "sqlite:///./test_api.db"

test_engine = create_engine(TEST_DB_URL, connect_args={"check_same_thread": False})
TestSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)


def override_get_db():
    db = TestSessionLocal()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture(scope="module", autouse=True)
def setup_db():
    Base.metadata.create_all(bind=test_engine)
    yield
    Base.metadata.drop_all(bind=test_engine)
    test_engine.dispose()
    import os as _os, time
    for _ in range(5):
        try:
            _os.remove("test_api.db")
            break
        except (PermissionError, FileNotFoundError):
            time.sleep(0.2)


@pytest.fixture(scope="module")
def client():
    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_create_session(client):
    r = client.post("/sessions", json={"mode": "practice"})
    assert r.status_code == 201
    data = r.json()
    assert "session_id" in data
    assert data["mode"] == "practice"
    assert data["status"] == "active"


def test_create_session_invalid_mode(client):
    r = client.post("/sessions", json={"mode": "invalid_mode"})
    assert r.status_code == 422


def test_submit_response_no_session(client):
    """Submitting audio to a non-existent session returns 404."""
    audio_bytes = b"RIFF" + b"\x00" * 44  # fake WAV header
    r = client.post(
        "/sessions/99999/responses",
        files={"audio": ("test.wav", io.BytesIO(audio_bytes), "audio/wav")},
    )
    assert r.status_code == 404


def test_submit_response_to_valid_session(client):
    """Submit audio to a real session; pipeline runs in mock mode."""
    # Create session
    s = client.post("/sessions", json={"mode": "practice"})
    session_id = s.json()["session_id"]

    # Submit fake audio
    audio_bytes = b"RIFF" + b"\x00" * 100
    r = client.post(
        f"/sessions/{session_id}/responses",
        files={"audio": ("test.wav", io.BytesIO(audio_bytes), "audio/wav")},
    )
    assert r.status_code == 201
    data = r.json()
    assert "response_id" in data
    assert data["session_id"] == session_id


def test_get_evaluation_after_submit(client):
    """After submitting, evaluation should be retrievable."""
    # Create session
    s = client.post("/sessions", json={"mode": "practice"})
    session_id = s.json()["session_id"]

    # Submit audio
    audio_bytes = b"RIFF" + b"\x00" * 100
    resp = client.post(
        f"/sessions/{session_id}/responses",
        files={"audio": ("test.wav", io.BytesIO(audio_bytes), "audio/wav")},
    )
    response_id = resp.json()["response_id"]

    # Fetch evaluation
    r = client.get(f"/responses/{response_id}/evaluation")
    assert r.status_code == 200
    data = r.json()
    assert data["response_id"] == response_id
    assert "transcript" in data


def test_get_evaluation_not_found(client):
    r = client.get("/responses/99999/evaluation")
    assert r.status_code == 404
