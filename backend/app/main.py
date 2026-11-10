
"""
FastAPI entry point for the Multi-Agent Clinical Trial
Feasibility Assessment project.

Responsibilities:
- Application startup and shutdown.
- API router registration.
- Health and readiness endpoints.
- Structured application logging.
- Optional CORS configuration.
- Safe, generic handling of unexpected API errors.

Run from the project root:
    uvicorn app.main:app --reload
"""

from __future__ import annotations


import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.routes.feasibility_router import router as feasibility_router


# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
from app.config.settings import settings
logger = settings.get_logger()






# ---------------------------------------------------------------------
# Application lifecycle
# ---------------------------------------------------------------------

@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    """
    Initialize application-level resources and release them on shutdown.

    Database pools, embedding clients, and other shared resources should
    be initialized here when their concrete implementations are available.
    """
    logger.info("Clinical Trial Feasibility API starting")

    try:
        # Place concrete startup initialization here when configured:
        # - Verify PostgreSQL connectivity.
        # - Initialize the configured repository/session factory.
        # - Initialize the Chroma client and embedding service.
        # - Initialize telemetry exporters.
        #
        # Do not mark the service ready until required dependencies
        # have actually passed their readiness checks.

        application.state.startup_complete = True

        logger.info("Clinical Trial Feasibility API startup complete")
        yield

    except Exception:
        logger.exception("Clinical Trial Feasibility API lifecycle failed")
        raise

    finally:
        application.state.startup_complete = False

        # Close only resources that were actually initialized:
        # - Database connection pools.
        # - Vector-store clients, if they require explicit cleanup.
        # - Telemetry providers and exporters.

        logger.info("Clinical Trial Feasibility API shutting down")


# ---------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------

app = FastAPI(
    title="Multi-Agent Clinical Trial Feasibility Assessment API",
    description=(
        "API for protocol feasibility, patient population analysis, "
        "site scoring, enrollment forecasting, cost estimation, "
        "evidence traceability, and human review."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------
# Optional CORS configuration
# ---------------------------------------------------------------------

# Configure explicit frontend origins, for example:
# CORS_ORIGINS=http://localhost:3000,http://localhost:8501
#
# An empty list means CORS middleware is not installed. Configure
# production origins explicitly rather than allowing every origin.

cors_origins = [
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", "").split(",")
    if origin.strip()
]

if cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
        allow_headers=["Authorization", "Content-Type", "X-Correlation-ID"],
    )


# ---------------------------------------------------------------------
# API routers
# ---------------------------------------------------------------------

app.include_router(feasibility_router)


# ---------------------------------------------------------------------
# Health and readiness endpoints
# ---------------------------------------------------------------------

@app.get("/", tags=["System"])
async def root() -> dict[str, str]:
    """Basic service identity endpoint."""
    return {
        "service": "clinical-trial-feasibility-assessment",
        "status": "running",
        "version": app.version,
    }


@app.get("/health", tags=["System"])
async def health() -> dict[str, str]:
    """
    Liveness endpoint.

    This confirms that the application process can answer requests.
    It does not prove that PostgreSQL, Chroma, or an LLM provider is
    available.
    """
    return {
        "status": "healthy",
        "service": "clinical-trial-feasibility-assessment",
    }


@app.get("/ready", tags=["System"])
async def readiness(request: Request) -> JSONResponse:
    """
    Readiness endpoint.

    This minimal version checks application startup only. Add actual
    dependency checks before using it as a production readiness probe.
    """
    startup_complete = getattr(
        request.app.state,
        "startup_complete",
        False,
    )

    if not startup_complete:
        return JSONResponse(
            status_code=503,
            content={
                "status": "not_ready",
                "reason": "Application startup is incomplete.",
            },
        )

    return JSONResponse(
        status_code=200,
        content={"status": "ready"},
    )


# ---------------------------------------------------------------------
# Generic exception handler
# ---------------------------------------------------------------------

@app.exception_handler(Exception)
async def unexpected_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    """
    Return a generic response for unexpected errors.

    Detailed exceptions remain in server logs. Avoid including request
    bodies, protocol contents, patient data, credentials, or model
    prompts in error logs.
    """
    logger.error(
        "Unhandled API exception | method=%s | path=%s | exception_type=%s",
        request.method,
        request.url.path,
        type(exc).__name__,
        exc_info=(type(exc), exc, exc.__traceback__),
    )

    return JSONResponse(
        status_code=500,
        content={
            "detail": "An unexpected server error occurred.",
        },
    )
