"""FastAPI application entry point."""

from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.analysis.registry import all_specs, load_all_methods
from app.api.v1.router import api_router
from app.core.config import settings
from app.core.errors import PlatformError
from app.core.logging import configure_logging, get_logger
from app.forecasting.base import all_models

log = get_logger(__name__)

DESCRIPTION = """\
An evidence-driven stock intelligence API.

**Operating principle:** data before intelligence, validation before
prediction, evidence before recommendations.

Three conventions apply to every response:

* **Every conclusion carries an epistemic label.** A computed return is a
  `deterministic_calculation`; a regime label is a `model_estimate`; an
  unanswerable question is `unknown`. Nothing is presented as an observed fact
  unless it was observed.
* **Missing data is reported, never defaulted.** A method that cannot run
  returns the reason it could not. No value is interpolated, forward-filled or
  substituted.
* **Analysis carries its own limitations.** Responses include a `disclosures`
  or `limitations` field; a client that displays a figure should display these
  too.

This is research and educational tooling, not personalised investment advice.
It does not execute trades. See `GET /api/v1/limitations`.
"""


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    load_all_methods()
    log.info(
        "application_starting",
        environment=settings.environment,
        methods=len(all_specs()),
        models=len(all_models()),
        providers=settings.enabled_provider_ids,
    )
    yield
    log.info("application_stopping")


app = FastAPI(
    title=settings.app_name,
    description=DESCRIPTION,
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_context(request: Request, call_next):
    """Bind a request id and log the outcome of every request."""
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(
        request_id=request_id,
        path=request.url.path,
        method=request.method,
    )
    started = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        log.exception("request_failed", duration_ms=(time.perf_counter() - started) * 1000)
        raise
    duration = (time.perf_counter() - started) * 1000
    response.headers["X-Request-ID"] = request_id
    log.info("request_completed", status=response.status_code, duration_ms=round(duration, 2))
    return response


@app.exception_handler(PlatformError)
async def platform_error_handler(request: Request, exc: PlatformError) -> JSONResponse:
    """Map a domain error to its HTTP status, keeping the explanation intact.

    ``InsufficientDataError`` is a 409 rather than a 500: it is the designed
    outcome when the data does not support an analysis, and the message says
    exactly what was missing.
    """
    log.info(
        "platform_error",
        code=exc.code,
        status=exc.status_code,
        message=exc.message,
        details=exc.details,
    )
    return JSONResponse(
        status_code=exc.status_code,
        content={"code": exc.code, "message": exc.message, "details": exc.details},
    )


@app.exception_handler(RequestValidationError)
async def validation_error_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content={
            "code": "request_validation_error",
            "message": "the request parameters are invalid",
            "details": {"errors": exc.errors()},
        },
    )


@app.get("/", include_in_schema=False)
async def root() -> dict:
    return {
        "name": settings.app_name,
        "version": "0.1.0",
        "docs": "/docs",
        "api": settings.api_v1_prefix,
        "principle": (
            "data before intelligence, validation before prediction, evidence "
            "before recommendations"
        ),
        "disclaimer": (
            "Research and educational tooling. Not personalised investment "
            "advice. No trade execution."
        ),
    }


app.include_router(api_router, prefix=settings.api_v1_prefix)
