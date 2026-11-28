"""FastAPI application entry point."""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.routes import admin, annotations, interview, responses, sessions
from app.config import get_settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    # Pre-warm Whisper in background thread so model stays in memory
    if settings.llm_provider != "mock":
        import threading
        def _warm_whisper():
            try:
                from app.services.stt import WhisperSTTService
                WhisperSTTService(
                    model_size=settings.whisper_model_size,
                    device=settings.whisper_device,
                    compute_type=settings.whisper_compute_type,
                )
            except Exception:
                pass
        threading.Thread(target=_warm_whisper, daemon=True).start()
    yield


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="InterviewIQ API",
        description=(
            "GenAI-Based Virtual Interviewer for Campus Placement Coaching.\n\n"
            "Formative coaching only — no hireability prediction, "
            "no facial/emotion/personality/accent assessment."
        ),
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(sessions.router)
    app.include_router(responses.router)
    app.include_router(annotations.router)
    app.include_router(interview.router)
    app.include_router(admin.router)

    # Serve plain-HTML UI from app/static
    static_dir = Path(__file__).parent / "static"
    static_dir.mkdir(exist_ok=True)
    app.mount("/ui", StaticFiles(directory=str(static_dir), html=True), name="ui")

    @app.get("/", include_in_schema=False)
    def root():
        from fastapi.responses import RedirectResponse
        return RedirectResponse(url="/ui/")

    @app.get("/health", tags=["meta"])
    def health():
        from app.utils.audio import audio_dep_info
        return {
            "status": "ok",
            "provider": settings.llm_provider,
            "audit_inline": settings.audit_inline,
            "audio": audio_dep_info(),
        }

    return app


app = create_app()
