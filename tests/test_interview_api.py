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
    assert d["question_number"] == 0
    assert d["is_intro"] is True
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
    """After submitting answers, summary contains introduction and answers."""
    s = client.post("/interview/sessions", json={"mode": "mixed", "n_questions": 2}).json()
    sid = s["session_id"]
    # 1. Intro question
    nq1 = client.get(f"/interview/sessions/{sid}/next-question").json()
    qid1 = nq1["question_id"]
    client.post(
        f"/interview/sessions/{sid}/responses?question_id={qid1}",
        files={"audio": ("t1.wav", io.BytesIO(b"RIFF" + b"\x00" * 100), "audio/wav")},
    )
    # 2. First core question
    nq2 = client.get(f"/interview/sessions/{sid}/next-question").json()
    qid2 = nq2["question_id"]
    client.post(
        f"/interview/sessions/{sid}/responses?question_id={qid2}",
        files={"audio": ("t2.wav", io.BytesIO(b"RIFF" + b"\x00" * 100), "audio/wav")},
    )
    r = client.get(f"/interview/sessions/{sid}/summary")
    assert r.status_code == 200
    d = r.json()
    assert d["introduction"] is not None
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


# ─────────────────────────────────────────────────────────────────────────────
# New tests: UI_JUDGE, Whisper guard, synthetic transcript per-question
# ─────────────────────────────────────────────────────────────────────────────

def test_submit_response_returns_transcript_source_synthetic_in_mock(client):
    """In mock mode the response must include transcript_source='synthetic'."""
    s = client.post("/interview/sessions", json={"mode": "mixed", "n_questions": 2}).json()
    sid = s["session_id"]
    nq = client.get(f"/interview/sessions/{sid}/next-question").json()
    qid = nq["question_id"]
    r = client.post(
        f"/interview/sessions/{sid}/responses?question_id={qid}",
        files={"audio": ("t.wav", io.BytesIO(b"RIFF" + b"\x00" * 100), "audio/wav")},
    )
    assert r.status_code == 201
    d = r.json()
    assert d["transcript_source"] == "synthetic"
    assert d["is_mock"] is True


def test_synthetic_transcripts_differ_per_question(client):
    """Two different questions must produce different synthetic transcripts."""
    s = client.post("/interview/sessions", json={"mode": "mixed", "n_questions": 4}).json()
    sid = s["session_id"]
    transcripts = []
    for _ in range(2):
        nq = client.get(f"/interview/sessions/{sid}/next-question").json()
        if nq["done"]:
            break
        qid = nq["question_id"]
        r = client.post(
            f"/interview/sessions/{sid}/responses?question_id={qid}",
            files={"audio": ("t.wav", io.BytesIO(b"RIFF" + b"\x00" * 100), "audio/wav")},
        )
        assert r.status_code == 201
        transcripts.append(r.json()["transcript"])
    # Two distinct questions → two distinct synthetic transcripts
    assert len(transcripts) == 2
    assert transcripts[0] != transcripts[1], (
        "synthetic transcripts must differ per question"
    )


def test_synthetic_transcript_labelled(client):
    """Synthetic transcript text must start with [SYNTHETIC TRANSCRIPT (demo)]."""
    s = client.post("/interview/sessions", json={"mode": "mixed", "n_questions": 2}).json()
    sid = s["session_id"]
    nq = client.get(f"/interview/sessions/{sid}/next-question").json()
    qid = nq["question_id"]
    r = client.post(
        f"/interview/sessions/{sid}/responses?question_id={qid}",
        files={"audio": ("t.wav", io.BytesIO(b"RIFF" + b"\x00" * 100), "audio/wav")},
    )
    d = r.json()
    assert d["transcript"].startswith("[SYNTHETIC TRANSCRIPT (demo)]"), (
        f"expected synthetic prefix, got: {d['transcript'][:60]}"
    )


def test_ui_judge_not_configured_returns_503():
    """In real (non-mock) mode with UI_JUDGE unset, submit must return 503."""
    import os as _os
    from fastapi.testclient import TestClient
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    # Stand up a separate real-mode client
    _os.environ["LLM_PROVIDER"] = "live"
    _os.environ["UI_JUDGE"] = ""   # explicitly unset
    _os.environ["UI_JUDGE_ROUTER"] = "static"

    # Reset the settings singleton so it picks up the change
    import app.config as _cfg
    _cfg._settings = None

    from app.main import create_app
    from app.database import Base, get_db

    _engine2 = create_engine(
        "sqlite:///./test_ui_judge.db",
        connect_args={"check_same_thread": False},
    )
    _SL2 = sessionmaker(autocommit=False, autoflush=False, bind=_engine2)
    Base.metadata.create_all(bind=_engine2)
    from app.seed.questions import seed
    seed(db=_SL2())

    def _db2():
        db = _SL2()
        try:
            yield db
        finally:
            db.close()

    _app2 = create_app()
    _app2.dependency_overrides[get_db] = _db2

    try:
        with TestClient(_app2, raise_server_exceptions=False) as c2:
            # Create session
            s = c2.post("/interview/sessions", json={"mode": "mixed", "n_questions": 2})
            sid = s.json()["session_id"]
            nq = c2.get(f"/interview/sessions/{sid}/next-question").json()
            qid = nq["question_id"]
            r = c2.post(
                f"/interview/sessions/{sid}/responses?question_id={qid}",
                files={"audio": ("t.wav", io.BytesIO(b"RIFF" + b"\x00" * 100), "audio/wav")},
            )
            assert r.status_code == 503, f"expected 503, got {r.status_code}: {r.text[:200]}"
            assert "UI_JUDGE" in r.json().get("detail", ""), (
                "error detail must mention UI_JUDGE"
            )
    finally:
        # Restore mock mode so other tests are unaffected
        _os.environ["LLM_PROVIDER"] = "mock"
        _os.environ.pop("UI_JUDGE", None)
        _os.environ.pop("UI_JUDGE_ROUTER", None)
        _cfg._settings = None
        Base.metadata.drop_all(bind=_engine2)
        _engine2.dispose()
        import time
        for _ in range(5):
            try:
                _os.remove("test_ui_judge.db")
                break
            except (PermissionError, FileNotFoundError):
                time.sleep(0.2)


def test_whisper_unavailable_returns_503(monkeypatch):
    """In real mode with Whisper not installed, submit must return 503
    with 'speech recognition unavailable' in the detail."""
    import app.orchestration.slim_pipeline as _sp
    # Patch _whisper_installed to return False for this test
    monkeypatch.setattr(_sp, "_whisper_installed", lambda: False)

    # Also patch _get_ui_llm_service to not raise UIJudgeNotConfigured
    from app.services.llm import MockLLMService
    monkeypatch.setattr(_sp, "_get_ui_llm_service", lambda: MockLLMService())

    # Temporarily switch to real mode
    import app.config as _cfg
    _orig = _cfg._settings
    import os as _os
    _os.environ["LLM_PROVIDER"] = "live"
    _cfg._settings = None

    from app.database import get_db
    from fastapi.testclient import TestClient
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from app.database import Base
    from app.main import create_app

    _engine3 = create_engine(
        "sqlite:///./test_whisper_guard.db",
        connect_args={"check_same_thread": False},
    )
    _SL3 = sessionmaker(autocommit=False, autoflush=False, bind=_engine3)
    Base.metadata.create_all(bind=_engine3)
    from app.seed.questions import seed
    seed(db=_SL3())

    def _db3():
        db = _SL3()
        try:
            yield db
        finally:
            db.close()

    _app3 = create_app()
    _app3.dependency_overrides[get_db] = _db3

    try:
        with TestClient(_app3, raise_server_exceptions=False) as c3:
            s = c3.post("/interview/sessions", json={"mode": "mixed", "n_questions": 2})
            sid = s.json()["session_id"]
            nq = c3.get(f"/interview/sessions/{sid}/next-question").json()
            qid = nq["question_id"]
            r = c3.post(
                f"/interview/sessions/{sid}/responses?question_id={qid}",
                files={"audio": ("t.wav", io.BytesIO(b"RIFF" + b"\x00" * 100), "audio/wav")},
            )
            assert r.status_code == 503, f"expected 503, got {r.status_code}: {r.text[:200]}"
            detail = r.json().get("detail", "")
            assert "speech recognition unavailable" in detail.lower(), (
                f"expected 'speech recognition unavailable' in detail, got: {detail}"
            )
    finally:
        _os.environ["LLM_PROVIDER"] = "mock"
        _cfg._settings = None
        Base.metadata.drop_all(bind=_engine3)
        _engine3.dispose()
        import time
        for _ in range(5):
            try:
                _os.remove("test_whisper_guard.db")
                break
            except (PermissionError, FileNotFoundError):
                time.sleep(0.2)
