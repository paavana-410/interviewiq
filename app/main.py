"""FastAPI application entry point."""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import annotations, responses, sessions
from app.config import get_settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: ensure upload / experiment dirs exist
    get_settings()
    yield
    # Shutdown: nothing to clean up


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

    @app.get("/health", tags=["meta"])
    def health():
        return {"status": "ok", "provider": settings.llm_provider}

    return app


app = create_app()
