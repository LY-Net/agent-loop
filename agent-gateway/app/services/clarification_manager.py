
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any


class ClarificationManager:
    """
    澄清管理器。

    负责：
    1. 构建初始会话上下文
    2. 构建待澄清信息
    3. 接收用户补充内容
    4. 合并多轮澄清记录
    5. 生成重新识别所需的完整输入

    本类不直接操作数据库。
    """

    @staticmethod
    def build_initial_context(
        original_message: str,
        intent_name: str | None = None,
        known_parameters: dict[str, Any] | None = None,
        missing_parameters: list[str] | None = None,
        required_capabilities: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        构建首次请求的会话上下文。
        """
        if not original_message or not original_message.strip():
            raise ValueError("original_message 不能为空")

        return {
            "original_message": original_message.strip(),
            "intent": intent_name,
            "known_parameters": known_parameters or {},
            "missing_parameters": missing_parameters or [],
            "required_capabilities": required_capabilities or [],
            "clarification_history": [],
        }

    @staticmethod
    def build_pending_clarification(
        question: str,
        missing_parameters: list[str] | None = None,
        known_parameters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        构建当前待澄清信息。
        """
        if not question or not question.strip():
            raise ValueError("clarification question 不能为空")

        return {
            "question": question.strip(),
            "missing_parameters": missing_parameters or [],
            "known_parameters": known_parameters or {},
        }


    @staticmethod
    def apply_user_response(
        context_data: dict[str, Any],
        pending_clarification: dict[str, Any] | None,
        user_response: str,
    ) -> dict[str, Any]:

        if not user_response or not user_response.strip():
            raise ValueError("用户补充内容不能为空")

        if not pending_clarification:
            raise ValueError("当前会话没有待处理的澄清问题")

        question = pending_clarification.get("question")

        if not question:
            raise ValueError("待澄清信息缺少 question")

        updated_context = deepcopy(context_data or {})

        original_message = updated_context.get("original_message")

        if not original_message:
            raise ValueError("会话上下文缺少 original_message")

        history = updated_context.get("clarification_history", [])

        if not isinstance(history, list):
            raise ValueError("clarification_history 必须是列表")

        response = user_response.strip()

        history.append({
            "question": question,
            "answer": response,
            "created_at": datetime.now(timezone.utc).isoformat(),
        })

        updated_context["clarification_history"] = history
        updated_context["latest_clarification"] = response

        # 保留之前已经识别出的参数
        known_parameters = dict(
            updated_context.get("known_parameters") or {}
        )

        # 合并待澄清状态中已有的参数
        pending_parameters = (
            pending_clarification.get("known_parameters") or {}
        )
        known_parameters.update(pending_parameters)

        updated_context["known_parameters"] = known_parameters

        # 生成完整输入
        message_parts = [
            f"原始问题：{original_message}"
        ]

        for index, item in enumerate(history, start=1):
            message_parts.append(
                f"第{index}轮澄清问题：{item['question']}"
            )
            message_parts.append(
                f"第{index}轮用户补充：{item['answer']}"
            )

        combined_input = "\n".join(message_parts)

        return {
            "context_data": updated_context,
            "pending_clarification": None,
            "combined_input": combined_input,
        }
