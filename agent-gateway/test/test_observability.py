
import pytest

from sqlalchemy import select

from app.db.session import AsyncSessionLocal

from app.models.observability import (
    AgentTrace,
    AgentRun,
    AgentSpan,
    AgentEvent
)


@pytest.mark.asyncio
async def test_observability_persistence():

    from fastapi.testclient import TestClient
    from app.main import app

    client = TestClient(app)

    response = client.post(
        "/api/v1/agent/chat",
        json={
            "user_id": "test_user",
            "session_id": "test_session",
            "message": "分析销售数据"
        }
    )

    assert response.status_code == 200

    trace_id = response.json()["trace_id"]

    async with AsyncSessionLocal() as session:

        trace = await session.get(
            AgentTrace,
            trace_id
        )

        assert trace is not None
        assert trace.status == "success"

        runs = (
            await session.execute(
                select(AgentRun)
                .where(AgentRun.trace_id == trace_id)
            )
        ).scalars().all()

        assert len(runs) == 1

        spans = (
            await session.execute(
                select(AgentSpan)
                .where(AgentSpan.trace_id == trace_id)
            )
        ).scalars().all()

        assert len(spans) == 4

        root = next(
            item for item in spans
            if item.span_name == "agent_loop"
        )

        children = [
            item for item in spans
            if item.parent_span_id == root.span_id
        ]

        assert len(children) == 3

        events = (
            await session.execute(
                select(AgentEvent)
                .where(AgentEvent.trace_id == trace_id)
            )
        ).scalars().all()

        assert len(events) == 4
