
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.planner.planner import execution_planner
from app.planner.validation import execution_plan_validator
from app.intent.schemas import ExecutionMode

from scripts.test_planner import build_intent


def test_unified_validation():
    # 构造一个 DIRECT 意图
    intent = build_intent(
        mode=ExecutionMode.DIRECT,
    )

    # 生成执行计划
    plan = execution_planner.build_plan(
        intent,
        "你好，请介绍一下你自己",
    )

    # 执行统一校验
    execution_plan_validator.validate(
        intent,
        plan,
    )

    print("[PASS] DIRECT 计划统一校验")


def test_contract_violation():
    # Intent 要求 RAG
    intent = build_intent(
        mode=ExecutionMode.DIRECT,
        need_rag=True,
    )

    # 故意构造一个缺少 RAG 的计划
    from app.planner.schemas import (
        ExecutionPlan,
        ExecutionStep,
        PlanStatus,
        StepType,
        StepStatus,
    )
    from uuid import uuid4

    step = ExecutionStep(
        step_id=str(uuid4()),
        step_type=StepType.LLM_GENERATION,
        name="generation",
        description="测试生成",
        depends_on=[],
        input_data={},
        output_data={},
        status=StepStatus.PENDING,
        required_capabilities=[],
    )

    invalid_plan = ExecutionPlan(
        plan_id=str(uuid4()),
        intent_name="GENERAL",
        execution_mode=ExecutionMode.DIRECT,
        description="故意构造的错误计划",
        status=PlanStatus.CREATED,
        steps=[step],
        need_clarification=False,
        clarification_question=None,
        metadata={},
    )

    try:
        execution_plan_validator.validate(
            intent,
            invalid_plan,
        )
    except ValueError:
        print("[PASS] 成功拦截违反 Intent 契约的计划")
        return

    raise AssertionError("统一校验器未能拦截错误计划")


if __name__ == "__main__":
    print("\n========== Unified Plan Validation ==========\n")

    test_unified_validation()
    test_contract_violation()

    print("\n========== All Tests Passed ==========\n")
