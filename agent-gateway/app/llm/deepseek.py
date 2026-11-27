
import logging
from typing import Any

from openai import (
    AsyncOpenAI,
    APIError,
    APITimeoutError,
    AuthenticationError,
    RateLimitError,
)

from app.core.config import settings
from app.llm.client import LLMClient, LLMError, LLMResponse


logger = logging.getLogger(__name__)


class DeepSeekClient(LLMClient):
    """DeepSeek 异步客户端。"""

    def __init__(self):
        if not settings.DEEPSEEK_API_KEY:
            raise ValueError("DEEPSEEK_API_KEY 未配置")

        self.model = settings.DEEPSEEK_MODEL

        self.client = AsyncOpenAI(
            api_key=settings.DEEPSEEK_API_KEY,
            base_url=settings.DEEPSEEK_BASE_URL,
            timeout=settings.DEEPSEEK_TIMEOUT,
            max_retries=settings.DEEPSEEK_MAX_RETRIES,
        )

    async def chat(
        self,
        messages: list[dict[str, Any]],
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """调用 DeepSeek Chat Completion。"""

        try:
            request_params: dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
            }

            if max_tokens is not None:
                request_params["max_tokens"] = max_tokens

            response = await self.client.chat.completions.create(
                **request_params
            )

            if not response.choices:
                raise LLMError(
                    "模型没有返回有效结果",
                    error_type="empty_response",
                )

            choice = response.choices[0]
            message = choice.message

            usage = response.usage

            return LLMResponse(
                content=message.content or "",
                model=response.model or self.model,
                prompt_tokens=usage.prompt_tokens if usage else 0,
                completion_tokens=usage.completion_tokens if usage else 0,
                total_tokens=usage.total_tokens if usage else 0,
                finish_reason=choice.finish_reason,
            )

        except AuthenticationError as exc:
            logger.error("DeepSeek API 认证失败")
            raise LLMError(
                "DeepSeek API 认证失败，请检查 API Key",
                error_type="authentication_error",
            ) from exc

        except RateLimitError as exc:
            logger.warning("DeepSeek API 请求频率超限")
            raise LLMError(
                "模型请求频率超限，请稍后重试",
                error_type="rate_limit",
            ) from exc

        except APITimeoutError as exc:
            logger.warning("DeepSeek API 请求超时")
            raise LLMError(
                "模型请求超时",
                error_type="timeout",
            ) from exc

        except APIError as exc:
            logger.exception("DeepSeek API 调用失败")
            raise LLMError(
                "模型服务调用失败",
                error_type="api_error",
            ) from exc

        except LLMError:
            raise

        except Exception as exc:
            logger.exception("LLM 调用发生未知异常")
            raise LLMError(
                "模型调用发生未知异常",
                error_type="unknown_error",
            ) from exc

    async def close(self) -> None:
        """关闭 HTTP 客户端连接。"""
        await self.client.close()


# 暂时使用模块级单例，后续接入 FastAPI 生命周期统一管理
deepseek_client = DeepSeekClient()
