
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field


class LLMError(Exception):
    """大模型调用基础异常。"""

    def __init__(
        self,
        message: str,
        error_type: str = "llm_error",
    ):
        super().__init__(message)
        self.error_type = error_type


class LLMResponse(BaseModel):
    """统一的大模型响应结构。"""

    content: str = ""
    model: str = ""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    finish_reason: str | None = None


class LLMClient(ABC):
    """大模型客户端统一接口。"""

    @abstractmethod
    async def chat(
        self,
        messages: list[dict[str, Any]],
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """发送对话请求。"""
        raise NotImplementedError

    @abstractmethod
    async def close(self) -> None:
        """释放客户端资源。"""
        raise NotImplementedError
