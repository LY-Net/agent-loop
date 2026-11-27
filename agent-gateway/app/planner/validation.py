
from app.planner.schemas import ExecutionPlan
from app.planner.validator import (
    PlanValidator,
    PlanValidationError,
)
from app.planner.contract_validator import (
    IntentPlanContractValidator,
)


class ExecutionPlanValidator:
    """
    执行计划统一校验入口。

    负责：
    1. 校验计划结构和依赖关系
    2. 校验 Intent 与 Plan 的业务契约
    """

    def __init__(self):
        self.plan_validator = PlanValidator()
        self.contract_validator = IntentPlanContractValidator()

    def validate(
        self,
        intent_result,
        plan: ExecutionPlan,
    ) -> None:
        # 第一层：结构校验
        self.plan_validator.validate(plan)

        # 第二层：业务契约校验
        self.contract_validator.validate(
            intent_result,
            plan,
        )


# 全局实例
execution_plan_validator = ExecutionPlanValidator()
