
import json
from typing import Any

from app.llm.deepseek import deepseek_client


class LLMGenerationHandler:
    """
    LLM 生成处理器。

    负责：
    1. 获取前置工具执行结果。
    2. 组织大模型 Prompt。
    3. 调用 DeepSeek。
    4. 返回最终自然语言回答。

    当前版本主要支持：
    TOOL_CALL -> LLM_GENERATION

    后续可以扩展为：
    RAG_RETRIEVAL -> LLM_GENERATION
    多个 TOOL_CALL -> LLM_GENERATION
    """

    def __init__(self, llm_client=None):
        """
        支持依赖注入。

        默认使用项目已有的 DeepSeek 客户端。
        测试时也可以传入 Mock LLM Client。
        """
        self.llm_client = llm_client or deepseek_client

    async def _get_dependency_outputs(
        self,
        step,
        context,
    ) -> list[dict[str, Any]]:
        """
        获取当前 Step 的前置步骤执行结果。

        例如：

        TOOL_CALL
            |
            v
        LLM_GENERATION

        LLM_GENERATION 需要读取 TOOL_CALL 的输出。
        """

        outputs = []

        for dependency_id in step.depends_on or []:

            # ExecutionContext 使用 Step ID 保存执行结果。
            output = context.get_step_output(
                str(dependency_id)
            )

            if output is None:
                raise ValueError(
                    f"无法获取前置 Step 的执行结果: {dependency_id}"
                )

            outputs.append({
                "step_id": str(dependency_id),
                "output": output,
            })

        return outputs

    def _build_messages(
        self,
        user_input: str,
        dependency_outputs: list[dict[str, Any]],
    ) -> list[dict[str, str]]:
        """
        构建发送给 DeepSeek 的消息。

        将用户问题与工具结果一起提供给模型，
        避免模型脱离实际查询数据进行回答。
        """

        # 将工具结果转换为 JSON 字符串。
        tool_result_json = json.dumps(
            dependency_outputs,
            ensure_ascii=False,
            indent=2,
            default=str,
        )

        system_prompt = """
你是一个企业级数据分析 Agent。

你的职责是根据用户问题和工具返回的数据，
生成准确、清晰、自然的分析答复。

请遵循以下规则：

1. 必须以工具返回的数据为依据。
2. 不允许编造查询结果。
3. 不允许擅自修改数据中的数值。
4. 如果数据不足以回答用户问题，明确说明缺少的信息。
5. 使用中文回答。
6. 先回答核心结论，再补充必要说明。
7. 不要向用户暴露内部工具调用、JSON 或程序执行细节。
"""

        user_prompt = f"""
用户问题：

{user_input}

工具执行结果：

{tool_result_json}

请根据以上信息回答用户问题。
"""

        return [
            {
                "role": "system",
                "content": system_prompt.strip(),
            },
            {
                "role": "user",
                "content": user_prompt.strip(),
            },
        ]

    async def handle(
        self,
        step,
        context,
    ) -> dict[str, Any]:
        """
        ExecutionEngine 调用的统一入口。

        参数：
            step:
                当前 ExecutionStep。

            context:
                当前执行上下文 ExecutionContext。

        返回：
            dict:
                LLM 生成结果。
        """

        print("\n[LLM HANDLER] 开始执行")

        # 1. 获取前置步骤的执行结果。
        dependency_outputs = await self._get_dependency_outputs(
            step,
            context,
        )

        # 2. 获取用户原始问题。
        user_input = context.user_input or ""

        if not user_input:
            raise ValueError("ExecutionContext 中缺少用户原始问题")

        # 3. 构建 Prompt。
        messages = self._build_messages(
            user_input=user_input,
            dependency_outputs=dependency_outputs,
        )

        print("[LLM HANDLER] Prompt 构建完成")
        print("[LLM HANDLER] 正在请求 DeepSeek")

        # 4. 调用 DeepSeek。
        response = await self.llm_client.chat(
            messages=messages,
            temperature=0.2,
            max_tokens=2000,
        )

        # 5. 校验模型响应。
        answer = (response.content or "").strip()

        if not answer:
            raise RuntimeError("DeepSeek 返回了空内容")

        print("[LLM HANDLER] DeepSeek 响应成功")
        print("[LLM HANDLER] 回答生成完成")

        # 6. 返回标准化执行结果。
        return {
            "answer": answer,
            "model": response.model,
            "usage": {
                "prompt_tokens": response.prompt_tokens,
                "completion_tokens": response.completion_tokens,
                "total_tokens": response.total_tokens,
            },
            "finish_reason": response.finish_reason,
            "source_data": dependency_outputs,
        }


# 创建默认实例。
llm_generation_handler = LLMGenerationHandler()
