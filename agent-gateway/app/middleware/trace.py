import uuid
import time

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


from app.core.logger import app_logger


class TraceMiddleware(BaseHTTPMiddleware):

    async def dispatch(self, request: Request, call_next):
        request_id=str(uuid.uuid4())
        trace_id=str(uuid.uuid4())

        request.state.request_id = request_id
        request.state.trace_id = trace_id

        start_time = time.perf_counter()

        try:
            app_logger.bind(
                request_id=request_id,
                trace_id=trace_id
            ).info(
                f"HTTP request started: {request.method} {request.url.path}"
            )

            response = await call_next(request)

            duration_ms = round(
                (time.perf_counter() - start_time) * 1000,
                2
            )

            response.headers["X-Request-Id"] = request_id
            response.headers["X-Trace-Id"] = trace_id

            app_logger.bind(
                request_id=request_id,
                trace_id=trace_id
            ).info(
                 f"HTTP request completed: "
                f"status={response.status_code}, "
                f"duration_ms={duration_ms}"
            )

            return response
        except Exception as e:
            duration_ms = round(
                (time.perf_counter() - start_time) * 1000,
                2
            )

            app_logger.bind(
                request_id=request_id,
                trace_id=trace_id
            ).exception(
                f"HTTP request failed, duration_ms={duration_ms}"
            )

            raise
