
import uuid
import time

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import update

from app.db.session import AsyncSessionLocal
from app.models.observability import (
    AgentSpan,
    AgentEvent
)

from app.observability.context import (
    trace_id_var,
    run_id_var,
    span_id_var
)


def utc_now():
    return datetime.now(timezone.utc)


class SpanRecorder:

    def __init__(
        self,
        span_name: str,
        span_type: str,
        input_data: dict | None = None
    ):

        self.span_id = str(uuid.uuid4())

        self.span_name = span_name
        self.span_type = span_type

        self.input_data = input_data or {}

        self.output_data = None
        self.error_message = None

        self.status = "running"

        self.trace_id = trace_id_var.get()
        self.run_id = run_id_var.get()

        self.parent_span_id = span_id_var.get()

        self.started_at = utc_now()
        self.start_clock = None

        self._span_token = None

    async def __aenter__(self):

        if not self.trace_id or not self.run_id:
            raise RuntimeError(
                "Trace context is missing. "
                "Please initialize execution context first."
            )

        self.start_clock = time.perf_counter()

        async with AsyncSessionLocal() as session:

            span = AgentSpan(
                span_id=self.span_id,
                trace_id=self.trace_id,
                run_id=self.run_id,
                parent_span_id=self.parent_span_id,
                span_name=self.span_name,
                span_type=self.span_type,
                status="running",
                input_data=self.input_data,
                started_at=self.started_at
            )

            session.add(span)
            await session.commit()

        self._span_token = span_id_var.set(self.span_id)

        return self

    def set_output(self, output_data: dict[str, Any]):

        self.output_data = output_data

    async def __aexit__(
        self,
        exc_type,
        exc_value,
        traceback
    ):

        try:

            ended_at = utc_now()

            duration_ms = int(
                (time.perf_counter() - self.start_clock) * 1000
            )

            if exc_type is not None:

                self.status = "error"

                self.error_message = str(exc_value)

            else:

                self.status = "success"

            async with AsyncSessionLocal() as session:

                await session.execute(
                    update(AgentSpan)
                    .where(
                        AgentSpan.span_id == self.span_id
                    )
                    .values(
                        status=self.status,
                        output_data=self.output_data,
                        error_message=self.error_message,
                        ended_at=ended_at,
                        duration_ms=duration_ms
                    )
                )

                await session.commit()

        finally:

            if self._span_token is not None:
                span_id_var.reset(self._span_token)

        # 不吞掉异常
        return False


async def record_event(
    event_type: str,
    event_data: dict | None = None
):
    from sqlalchemy import select




    trace_id = trace_id_var.get()
    run_id = run_id_var.get()
    span_id = span_id_var.get()


    from app.models.observability import AgentRun
    async with AsyncSessionLocal() as session:
        saved_run = await session.scalar(
            select(AgentRun.run_id).where(
                AgentRun.run_id == run_id
            )
        )

        print("\n========== TEST EVENT DEBUG ==========")
        print("event_type:", event_type)
        print("trace_id:", trace_id)
        print("context_run_id:", run_id)
        print("context_span_id:", span_id)
        print("database_run_id:", saved_run)
        print("======================================\n")

    if not trace_id or not run_id:
        raise RuntimeError(
            "Cannot record event without trace context."
        )

    async with AsyncSessionLocal() as session:
        event = AgentEvent(
            event_id=str(uuid.uuid4()),
            trace_id=trace_id,
            run_id=run_id,
            span_id=span_id,
            event_type=event_type,
            event_data=event_data or {}
        )

        session.add(event)

        await session.commit()
