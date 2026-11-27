

import json
import logging
import re
from typing import Any

from pydantic import ValidationError

from app.intent.schemas import CoarseIntentResult
from app.llm.client import LLMClient, LLMError
from app.llm.deepseek import deepseek_client


logger = logging.getLogger(__name__)


COARSE_INTENT_SYSTEM_PROMPT = """
你是一个专业的 AI Agent 意图识别引擎。

你的任务是对用户输入进行第一阶段的粗粒度意图识别。

你只负责判断用户想做什么，不负责真正执行任务。

一、意图分类

必须从以下类别中选择一个：

1. general
普通聊天、问候、闲聊、简单解释等。

2. knowledge_qa
知识问答、概念解释、技术原理、事实性问题等。

3. data_analysis
数据分析、统计分析、趋势分析、数据洞察、原因分析等。

4. data_query
查询数据库、查询业务数据、获取指定记录、检索结构化数据等。

5. content_creation
文章创作、文案撰写、报告生成、代码编写、内容编辑等。

6. task_execution
需要执行一系列步骤才能完成的任务，例如制定计划、处理文件、完成复杂工作流等。

7. system_operation
系统配置、系统管理、服务操作、应用管理等。

8. other
无法归入以上类别的请求。

二、识别规则

1. 必须理解用户真实目的，不要只根据关键词分类。
2. 用户要求分析数据时，优先考虑 data_analysis。
3. 用户要求获取已有数据时，优先考虑 data_query。
4. 用户要求生成文章、文案、代码等内容时，优先考虑 content_creation。
5. 用户要求执行多步骤任务时，考虑 task_execution。
6. 简单知识解释归入 knowledge_qa。
7. 普通问候和闲聊归入 general。
8. 不确定时降低 confidence，不允许伪造确定性。
9. 可以返回 1 到 3 个候选意图，按可能性从高到低排列。
10. 如果用户输入信息不足，但仍能判断大致类别，可以先分类，并通过 reason 说明不确定性。

三、输出要求

只输出合法 JSON，不要输出 Markdown，不要输出代码块。

JSON 结构：

{
    "category": "data_analysis",
    "candidates": [
        {
            "intent_name": "具体意图名称",
            "description": "意图描述",
            "confidence": 0.9
        }
    ],
    "confidence": 0.9,
    "need_fine": true,
    "reason": "分类理由"
}

category 必须使用上述八个类别之一。

confidence 和 candidates 中的 confidence 必须是 0 到 1 之间的数字。

need_fine 表示是否需要进一步进行精细意图识别。
对于存在明确执行目标、复杂参数或多个可能意图的请求，通常设置为 true。
对于简单问候和简单知识问答，可以设置为 false。

不要在 JSON 外输出任何内容。
"""


class CoarseIntentRecognizer:
    """粗粒度意图识别器。"""

    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    @staticmethod
    def _extract_json(content: str) -> dict[str, Any]:
        """从模型输出中提取 JSON。"""

        content = content.strip()

        # 兼容模型偶尔返回 Markdown 代码块
        content = re.sub(
            r"^```(?:json)?\s*",
            "",
            content,
            flags=re.IGNORECASE,
        )
        content = re.sub(r"\s*```$", "", content)

        try:
            result = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ValueError("模型返回的内容不是合法 JSON") from exc

        if not isinstance(result, dict):
            raise ValueError("模型返回的 JSON 顶层必须是对象")

        return result

    async def recognize(self, user_input: str) -> CoarseIntentResult:
        """
        执行粗粒度意图识别。

        Args:
            user_input: 用户输入文本

        Returns:
            CoarseIntentResult: 粗粒度识别结果
        """

        user_input = user_input.strip()

        if not user_input:
            raise ValueError("用户输入不能为空")

        try:
            response = await self.llm_client.chat(
                messages=[
                    {
                        "role": "system",
                        "content": COARSE_INTENT_SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": user_input,
                    },
                ],
                temperature=0.1,
                max_tokens=800,
            )

            raw_result = self._extract_json(response.content)

            result = CoarseIntentResult.model_validate(raw_result)

            logger.info(
                "粗粒度意图识别完成: category=%s, confidence=%.2f",
                result.category.value,
                result.confidence,
            )

            return result

        except ValidationError as exc:
            logger.exception("粗粒度意图结果结构校验失败")
            raise LLMError(
                "粗粒度意图识别结果不符合数据结构要求",
                error_type="intent_validation_error",
            ) from exc

        except ValueError as exc:
            logger.exception("粗粒度意图结果解析失败")
            raise LLMError(
                str(exc),
                error_type="intent_parse_error",
            ) from exc

        except LLMError:
            logger.exception("粗粒度意图识别调用失败")
            raise

        except Exception as exc:
            logger.exception("粗粒度意图识别发生未知异常")
            raise LLMError(
                "粗粒度意图识别失败",
                error_type="coarse_intent_error",
            ) from exc


# 暂时使用全局识别器
coarse_intent_recognizer = CoarseIntentRecognizer(
    llm_client=deepseek_client
)
