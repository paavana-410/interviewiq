"""Tests for the UI-facing interview endpoints.

All using the mock LLM + SQLite — no real API calls.
"""
from __future__ import annotations

import io
import os

os.environ["LLM_PROVIDER"] = "mock"
os.environ["DATABASE_URL"] = "sqlite:///./test_interview_ui.db"

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base, get_db
from app.main import app

# ── In-process SQLite DB ─────────────────────────────────────────────────────

_DB_URL = "sqlite:///./test_interview_ui.db"
_engine = create_engine(_DB_URL, connect_args={"check_same_thread": False})
_SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=_engine)


def _override_db():
    db = _SessionLocal()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture(scope="module", autouse=True)
def setup_db():
    Base.metadata.create_all(bind=_engine)
    # Seed questions so session creation works
    from app.seed.questions import seed
    seed(db=_SessionLocal())
    yield
    Base.metadata.drop_all(bind=_engine)
    _engine.dispose()
    import os as _os, time
    for _ in range(5):
        try:
            _os.remove("test_interview_ui.db")
            break
        except (PermissionError, FileNotFoundError):
            time.sleep(0.2)


@pytest.fixture(scope="module")
def client():
    app.dependency_overrides[get_db] = _override_db
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


# ─────────────────────────────────────────────────────────────────────────────
# POST /interview/sessions
# ─────────────────────────────────────────────────────────────────────────────

def test_create_interview_session_mixed(client):
    r = client.post("/interview/sessions", json={"mode": "mixed", "n_questions": 3})
    assert r.status_code == 201
    d = r.json()
    assert "session_id" in d
    assert d["mode"] == "mixed"
    assert d["n_questions"] >= 1
    assert len(d["question_ids"]) >= 1


def test_create_interview_session_behavioral(client):
    r = client.post("/interview/sessions", json={"mode": "behavioral", "n_questions": 4})
    assert r.status_code == 201
    assert r.json()["mode"] == "behavioral"


def test_create_interview_session_technical(client):
    r = client.post("/interview/sessions", json={"mode": "technical", "n_questions": 3})
    assert r.status_code == 201


def test_create_interview_session_invalid_mode(client):
    r = client.post("/interview/sessions", json={"mode": "invalid", "n_questions": 3})
    assert r.status_code == 422


# ─────────────────────────────────────────────────────────────────────────────
# GET /interview/sessions/{id}/next-question
# ─────────────────────────────────────────────────────────────────────────────

def test_next_question_first(client):
    s = client.post("/interview/sessions", json={"mode": "mixed", "n_questions": 3}).json()
    sid = s["session_id"]
    r = client.get(f"/interview/sessions/{sid}/next-question")
    assert r.status_code == 200
    d = r.json()
    assert d["done"] is False
    assert d["question_text"]
    assert d["question_number"] == 1
    assert d["total_questions"] == s["n_questions"]


def test_next_question_session_not_found(client):
    r = client.get("/interview/sessions/99999/next-question")
    assert r.status_code == 404


# ─────────────────────────────────────────────────────────────────────────────
# POST /interview/sessions/{id}/responses (audio upload)
# ─────────────────────────────────────────────────────────────────────────────

def test_submit_response_mock(client):
    """Upload fake audio to the slim pipeline (mock STT + mock LLM)."""
    s = client.post("/interview/sessions", json={"mode": "mixed", "n_questions": 3}).json()
    sid = s["session_id"]

    # Get the first question
    nq = client.get(f"/interview/sessions/{sid}/next-question").json()
    qid = nq["question_id"]

    # Submit fake audio bytes
    fake_audio = b"RIFF" + b"\x00" * 200
    r = client.post(
        f"/interview/sessions/{sid}/responses?question_id={qid}",
        files={"audio": ("test.wav", io.BytesIO(fake_audio), "audio/wav")},
    )
    assert r.status_code == 201
    d = r.json()
    assert "overall_score" in d
    assert "wpm" in d
    assert "transcript" in d
    assert "judge_model" in d
    assert d["judge_provider"] == "mock"
    assert 0.0 <= d["overall_score"] <= 1.0


def test_submit_response_session_not_found(client):
    r = client.post(
        "/interview/sessions/99999/responses?question_id=1",
        files={"audio": ("t.wav", io.BytesIO(b"\x00" * 10), "audio/wav")},
    )
    assert r.status_code == 404


# ─────────────────────────────────────────────────────────────────────────────
# GET /interview/sessions/{id}/summary
# ─────────────────────────────────────────────────────────────────────────────

def test_session_summary_empty(client):
    """A fresh session with no answers returns an empty answers list."""
    s = client.post("/interview/sessions", json={"mode": "mixed", "n_questions": 2}).json()
    r = client.get(f"/interview/sessions/{s['session_id']}/summary")
    assert r.status_code == 200
    d = r.json()
    assert "answers" in d
    assert isinstance(d["answers"], list)
    assert "judge_provider" in d
    assert "is_mock" in d


def test_session_summary_after_answer(client):
    """After submitting one answer, summary contains that answer."""
    s = client.post("/interview/sessions", json={"mode": "mixed", "n_questions": 2}).json()
    sid = s["session_id"]
    nq = client.get(f"/interview/sessions/{sid}/next-question").json()
    qid = nq["question_id"]
    client.post(
        f"/interview/sessions/{sid}/responses?question_id={qid}",
        files={"audio": ("t.wav", io.BytesIO(b"RIFF" + b"\x00" * 100), "audio/wav")},
    )
    r = client.get(f"/interview/sessions/{sid}/summary")
    assert r.status_code == 200
    d = r.json()
    assert len(d["answers"]) == 1
    ans = d["answers"][0]
    assert "overall_score" in ans
    assert "transcript" in ans
    assert "wpm" in ans


def test_session_summary_not_found(client):
    r = client.get("/interview/sessions/99999/summary")
    assert r.status_code == 404


# ─────────────────────────────────────────────────────────────────────────────
# GET /interview/judge-info
# ─────────────────────────────────────────────────────────────────────────────

def test_judge_info_mock(client):
    r = client.get("/interview/judge-info")
    assert r.status_code == 200
    d = r.json()
    assert "provider" in d
    assert "model" in d
    assert "is_mock" in d
    assert d["is_mock"] is True   # we're in mock mode
