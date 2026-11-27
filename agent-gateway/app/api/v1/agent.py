
from fastapi import APIRouter, Request

from app.schemas.agent import (
    AgentChatRequest,
    AgentChatResponse
)

from app.services.agent_service import agent_service


router = APIRouter(
    prefix="/agent",
    tags=["Agent"]
)


@router.post(
    "/chat",
    response_model=AgentChatResponse
)
async def chat(
    body: AgentChatRequest,
    request: Request
):

    request_id = request.state.request_id
    trace_id = request.state.trace_id

    return await agent_service.chat(
        request=body,
        request_id=request_id,
        trace_id=trace_id
    )
