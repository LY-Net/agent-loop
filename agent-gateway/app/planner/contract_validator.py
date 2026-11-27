
from typing import Any, Set

from app.planner.schemas import ExecutionPlan, StepType
from app.planner.validator import PlanValidationError


class IntentPlanContractError(ValueError):
    """Intent Engine 与 Execution Planner 之间的契约异常。"""


class IntentPlanContractValidator:
    """
    校验 Intent Engine 的识别结果与 ExecutionPlan 是否一致。
    """

    @staticmethod
    def _step_type_value(step_type: Any) -> str:
        """兼容 Enum 和字符串形式的步骤类型。"""
        return str(getattr(step_type, "value", step_type))

    def validate(
        self,
        intent_result: Any,
        plan: ExecutionPlan,
    ) -> None:
        errors = []

        # 1. 获取精细意图识别结果
        fine = getattr(intent_result, "fine", None)

        # 获取执行计划中的步骤类型
        step_types = {
            getattr(step.step_type, "value", step.step_type)
            for step in plan.steps
        }

        # 判断计划中是否包含对应步骤
        has_clarify = StepType.CLARIFY.value in step_types
        has_rag = StepType.RAG_RETRIEVAL.value in step_types
        has_tool = StepType.TOOL_CALL.value in step_types


        # 兼容可能使用 fine_result 命名的情况
        if fine is None:
            fine = getattr(intent_result, "fine_result", None)

        # 2. 提取 Intent Engine 的执行要求
        need_rag = fine.need_rag if fine is not None else False
        need_tool = fine.need_tool if fine is not None else False

        # 是否需要澄清
        need_clarification = (
            intent_result.need_clarification
            or (
                fine is not None
                and fine.need_clarification
            )
        )

        # 3. 获取 Planner 实际生成的步骤类型
        actual_types: Set[str] = {
            self._step_type_value(step.step_type)
            for step in plan.steps
        }

        rag_type = self._step_type_value(StepType.RAG_RETRIEVAL)
        tool_type = self._step_type_value(StepType.TOOL_CALL)
        clarify_type = self._step_type_value(StepType.CLARIFY)
        generation_type = self._step_type_value(StepType.LLM_GENERATION)

        # # 4. 检查 RAG 要求
        # if need_rag and rag_type not in actual_types:
        #     errors.append(
        #         "Intent Engine 要求执行 RAG，"
        #         "但 ExecutionPlan 中缺少 RAG_RETRIEVAL 步骤"
        #     )
        #
        # # 5. 检查 Tool 要求
        # if need_tool and tool_type not in actual_types:
        #     errors.append(
        #         "Intent Engine 要求执行 Tool，"
        #         "但 ExecutionPlan 中缺少 TOOL_CALL 步骤"
        #     )

        # 6. 检查澄清要求
        # 需要澄清时，只要求存在 CLARIFY 步骤
        if need_clarification:
            if not has_clarify:
                errors.append(
                    "Intent Engine 要求澄清，但 ExecutionPlan 中缺少 CLARIFY 步骤"
                )

            # 尚未补充信息，不应要求执行 RAG 或 Tool
        else:
            if need_rag and not has_rag:
                errors.append(
                    "Intent Engine 要求执行 RAG，但 ExecutionPlan 中缺少 RAG_RETRIEVAL 步骤"
                )

            if need_tool and not has_tool:
                errors.append(
                    "Intent Engine 要求执行 Tool，但 ExecutionPlan 中缺少 TOOL_CALL 步骤"
                )

        # 7. 非澄清计划必须包含生成步骤
        if not need_clarification:
            if generation_type not in actual_types:
                errors.append(
                    "非澄清计划必须包含 LLM_GENERATION 步骤"
                )

        if errors:
            raise IntentPlanContractError(
                "\n".join(errors)
            )


# 全局实例
intent_plan_contract_validator = IntentPlanContractValidator()
