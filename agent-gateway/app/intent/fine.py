
import json
import logging
import re
from typing import Any

from pydantic import ValidationError

from app.intent.schemas import (
    CoarseIntentResult,
    IntentFineResult,
)
from app.llm.client import LLMClient, LLMError
from app.llm.deepseek import deepseek_client


logger = logging.getLogger(__name__)


FINE_INTENT_SYSTEM_PROMPT = """
你是一个专业的 AI Agent 精细化意图识别引擎。

你的任务是理解用户的真实需求，并将自然语言请求转换成结构化任务描述。
你只负责分析和识别，不负责执行任务，也不能假装已经查询数据或调用工具。

一、需要识别的内容

1. primary_intent
具体、明确的意图名称，使用英文 snake_case。
例如：
- query_user_list
- regional_user_registration_statistics
- analyze_sales_trend
- generate_technical_document
- explain_python_async

2. intent_description
用中文简要描述用户真正想完成的任务。

3. confidence
对本次精细识别结果的整体置信度，范围为 0 到 1。

4. entities
提取用户输入中明确出现或可以可靠推断的实体。
每个实体包含：
- name：实体名称
- value：实体值
- entity_type：实体类型

常见实体类型：
person、organization、time_range、date、location、
number、dimension、metric、file、product、technology、
system、unknown。

不要凭空创造用户没有提供的信息。
对于无法确定的实体，可以不提取。

5. parameters
将用户需求转换为结构化参数。
例如：
{
    "time_range": "最近一个月",
    "group_by": "region",
    "metric": "registration_count"
}

参数名称应尽量清晰、稳定，便于后续 Agent 使用。

6. required_capabilities
列出完成任务所需的能力。
例如：
- knowledge_retrieval
- database_query
- data_aggregation
- data_analysis
- web_search
- file_processing
- code_execution
- content_generation
- tool_calling

只列出确实需要的能力，不要为了丰富结果而随意添加。

7. execution_mode
必须从以下值中选择一个：

direct：可以直接回答，不需要外部数据或工具。
rag：需要检索知识库。
tool：需要调用外部工具或业务系统。
hybrid：同时需要知识检索和工具调用。
planning：需要先制定多步骤执行计划。

8. need_rag
是否需要知识库检索。

9. need_tool
是否需要调用外部工具或业务系统。

10. need_clarification
是否需要向用户追问必要信息。

11. clarification_question
如果确实需要追问，给出一个简洁、明确的问题。
如果不需要追问，返回 null。

12. reason
解释判断依据，简明扼要。

二、执行模式判断规则

1. 普通知识解释、简单问答，通常使用 direct。
2. 需要查询内部知识库、文档、知识条目，使用 rag。
3. 需要访问数据库、业务 API、外部系统，使用 tool。
4. 同时需要知识库和外部工具，使用 hybrid。
5. 需要多个步骤、多个工具协同完成的复杂任务，使用 planning。

注意：
- 不能因为用户提到了数据库，就认为已经具备数据库访问权限。
- 不能因为用户要求查询，就假设工具已经存在。
- 不能将缺少的信息自行编造。
- 如果用户的任务目标明确，只是执行环境或工具尚未接入，不应因此追问用户。
- 如果缺少的信息会直接影响任务执行，才需要追问。
- 不要把所有复杂请求都归类为 planning。

三、参数提取规则

1. 优先提取用户明确提供的参数。
2. 对于明确的相对时间表达，可以保留原始表达，例如“最近一个月”。
3. 不要擅自将相对时间转换为具体日期，除非输入中提供了当前时间或明确的时间上下文。
4. 对于未提供的参数，不要编造默认值。
5. 可以使用空对象表示没有可提取的参数。

四、输出格式

只输出合法 JSON，不要输出 Markdown，不要输出代码块。

必须符合以下结构：

{
    "primary_intent": "具体意图名称",
    "intent_description": "任务描述",
    "confidence": 0.9,
    "entities": [],
    "parameters": {},
    "required_capabilities": [],
    "execution_mode": "direct",
    "need_rag": false,
    "need_tool": false,
    "need_clarification": false,
    "clarification_question": null,
    "reason": "识别依据"
}

execution_mode 只能是：
direct、rag、tool、hybrid、planning。

confidence 必须在 0 到 1 之间。

不要在 JSON 外输出任何内容。
"""


class FineIntentRecognizer:
    """精细化意图识别器。"""

    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    @staticmethod
    def _extract_json(content: str) -> dict[str, Any]:
        """解析模型返回的 JSON。"""

        content = content.strip()

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

    async def recognize(
        self,
        user_input: str,
        coarse_result: CoarseIntentResult | None = None,
    ) -> IntentFineResult:
        """
        执行精细化意图识别。

        Args:
            user_input: 用户原始输入
            coarse_result: 可选的粗粒度识别结果

        Returns:
            IntentFineResult: 精细化意图结果
        """

        user_input = user_input.strip()

        if not user_input:
            raise ValueError("用户输入不能为空")

        messages = [
            {
                "role": "system",
                "content": FINE_INTENT_SYSTEM_PROMPT,
            }
        ]

        if coarse_result is not None:
            coarse_context = {
                "category": coarse_result.category.value,
                "candidates": [
                    candidate.model_dump(mode="json")
                    for candidate in coarse_result.candidates
                ],
                "confidence": coarse_result.confidence,
                "reason": coarse_result.reason,
            }

            messages.append(
                {
                    "role": "system",
                    "content": (
                        "以下是前一阶段的粗粒度识别结果，"
                        "请将其作为参考，但仍需以用户原始输入为准：\n"
                        + json.dumps(
                            coarse_context,
                            ensure_ascii=False,
                        )
                    ),
                }
            )

        messages.append(
            {
                "role": "user",
                "content": user_input,
            }
        )

        try:
            response = await self.llm_client.chat(
                messages=messages,
                temperature=0.1,
                max_tokens=1200,
            )

            raw_result = self._extract_json(response.content)

            result = IntentFineResult.model_validate(raw_result)

            # 对追问字段做基本一致性校验
            if (
                result.need_clarification
                and not result.clarification_question
            ):
                raise ValueError(
                    "需要澄清时必须提供 clarification_question"
                )

            logger.info(
                "精细意图识别完成: intent=%s, mode=%s, confidence=%.2f",
                result.primary_intent,
                result.execution_mode.value,
                result.confidence,
            )

            return result

        except ValidationError as exc:
            logger.exception("精细意图结果结构校验失败")
            raise LLMError(
                "精细意图识别结果不符合数据结构要求",
                error_type="intent_validation_error",
            ) from exc

        except ValueError as exc:
            logger.exception("精细意图结果解析或校验失败")
            raise LLMError(
                str(exc),
                error_type="intent_parse_error",
            ) from exc

        except LLMError:
            logger.exception("精细意图识别调用失败")
            raise

        except Exception as exc:
            logger.exception("精细意图识别发生未知异常")
            raise LLMError(
                "精细意图识别失败",
                error_type="fine_intent_error",
            ) from exc


# 暂时使用模块级单例
fine_intent_recognizer = FineIntentRecognizer(
    llm_client=deepseek_client
)
