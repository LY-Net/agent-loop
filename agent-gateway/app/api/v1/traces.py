
from fastapi import APIRouter, HTTPException

from sqlalchemy import select

from app.db.session import AsyncSessionLocal

from app.models.observability import (
    AgentTrace,
    AgentRun,
    AgentSpan,
    AgentEvent
)


router = APIRouter(
    prefix="/traces",
    tags=["Observability"]
)


@router.get("/{trace_id}")
async def get_trace(trace_id: str):

    async with AsyncSessionLocal() as session:

        # 1. 查询 Trace
        trace = await session.get(
            AgentTrace,
            trace_id
        )

        if trace is None:
            raise HTTPException(
                status_code=404,
                detail="Trace not found"
            )

        # 2. 查询 Run
        runs_result = await session.execute(
            select(AgentRun)
            .where(AgentRun.trace_id == trace_id)
        )

        runs = runs_result.scalars().all()

        # 3. 查询 Span
        spans_result = await session.execute(
            select(AgentSpan)
            .where(AgentSpan.trace_id == trace_id)
            .order_by(AgentSpan.started_at)
        )

        spans = spans_result.scalars().all()

        # 4. 查询 Event
        events_result = await session.execute(
            select(AgentEvent)
            .where(AgentEvent.trace_id == trace_id)
            .order_by(AgentEvent.created_at)
        )

        events = events_result.scalars().all()

        return {
            "trace": {
                "trace_id": trace.trace_id,
                "request_id": trace.request_id,
                "user_id": trace.user_id,
                "session_id": trace.session_id,
                "status": trace.status,
                "input": trace.input_data,
                "output": trace.output_data,
                "duration_ms": trace.duration_ms,
                "started_at": trace.started_at,
                "ended_at": trace.ended_at
            },

            "runs": [
                {
                    "run_id": item.run_id,
                    "agent_name": item.agent_name,
                    "status": item.status,
                    "duration_ms": item.duration_ms
                }
                for item in runs
            ],

            "spans": [
                {
                    "span_id": item.span_id,
                    "parent_span_id": item.parent_span_id,
                    "span_name": item.span_name,
                    "span_type": item.span_type,
                    "status": item.status,
                    "input": item.input_data,
                    "output": item.output_data,
                    "error": item.error_message,
                    "duration_ms": item.duration_ms,
                    "started_at": item.started_at,
                    "ended_at": item.ended_at
                }
                for item in spans
            ],

            "events": [
                {
                    "event_id": item.event_id,
                    "span_id": item.span_id,
                    "event_type": item.event_type,
                    "event_data": item.event_data,
                    "created_at": item.created_at
                }
                for item in events
            ]
        }
