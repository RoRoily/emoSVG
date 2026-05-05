"""
FastAPI middleware: request logging and structured error handling.
"""
from __future__ import annotations

import logging
import time
import uuid

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from src.core.exceptions import EmoSVGError

logger = logging.getLogger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = str(uuid.uuid4())[:8]
        t0 = time.monotonic()
        logger.info("[%s] %s %s", request_id, request.method, request.url.path)
        try:
            response = await call_next(request)
        except Exception as exc:
            logger.exception("[%s] Unhandled error: %s", request_id, exc)
            return JSONResponse(
                status_code=500,
                content={"error": "internal_server_error", "detail": str(exc)},
            )
        elapsed_ms = (time.monotonic() - t0) * 1000
        logger.info(
            "[%s] %s %s -> %d (%.0f ms)",
            request_id, request.method, request.url.path,
            response.status_code, elapsed_ms,
        )
        return response
