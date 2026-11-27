
from typing import Any, Optional

from pydantic import BaseModel, Field


class AgentChatRequest(BaseModel):

    user_id: str = Field(
        ...,
        description="用户唯一标识"
    )

    session_id: Optional[str] = Field(
        default=None,
        description="会话 ID"
    )

    message: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="用户输入内容"
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="扩展业务参数"
    )


class AgentChatResponse(BaseModel):

    success: bool

    request_id: str

    trace_id: str

    session_id: str

    message: str

    data: Optional[dict[str, Any]] = None

    error: Optional[str] = None
