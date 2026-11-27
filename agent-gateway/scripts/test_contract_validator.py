
import sys
from pathlib import Path
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.planner.contract_validator import (
    intent_plan_contract_validator,
    IntentPlanContractError,
)
from app.planner.schemas import (
    ExecutionPlan,
    ExecutionStep,
    PlanStatus,
    StepType,
    StepStatus,
)
from app.intent.schemas import ExecutionMode

from scripts.test_planner import build_intent


def make_step(step_type, name):
    return ExecutionStep(
        step_id=str(uuid4()),
        step_type=step_type,
        name=name,
        description=f"{name} test",
        depends_on=[],
        input_data={},
        output_data={},
        status=StepStatus.PENDING,
        required_capabilities=[],
    )


def make_plan(steps):
    return ExecutionPlan(
        plan_id=str(uuid4()),
        intent_name="GENERAL",
        execution_mode=ExecutionMode.DIRECT,
        description="Contract test",
        status=PlanStatus.CREATED,
        steps=steps,
        need_clarification=False,
        clarification_question=None,
        metadata={},
    )


def expect_failure(intent, plan, message):
    try:
        intent_plan_contract_validator.validate(intent, plan)
    except IntentPlanContractError:
        print(f"[PASS] {message}")
        return

    raise AssertionError(f"未能拦截：{message}")


def test_missing_rag():
    intent = build_intent(
        mode=ExecutionMode.DIRECT,
        need_rag=True,
    )
    plan = make_plan([
        make_step(StepType.LLM_GENERATION, "generation")
    ])

    expect_failure(intent, plan, "缺少 RAG 步骤")


def test_missing_tool():
    intent = build_intent(
        mode=ExecutionMode.DIRECT,
        need_tool=True,
    )
    plan = make_plan([
        make_step(StepType.LLM_GENERATION, "generation")
    ])

    expect_failure(intent, plan, "缺少 Tool 步骤")


def test_missing_generation():
    intent = build_intent(
        mode=ExecutionMode.DIRECT,
    )
    plan = make_plan([
        make_step(StepType.RAG_RETRIEVAL, "rag")
    ])

    expect_failure(intent, plan, "缺少 LLM 生成步骤")


def test_valid_direct_plan():
    intent = build_intent(
        mode=ExecutionMode.DIRECT,
    )
    plan = make_plan([
        make_step(StepType.LLM_GENERATION, "generation")
    ])

    intent_plan_contract_validator.validate(intent, plan)
    print("[PASS] 合法 DIRECT 计划")


if __name__ == "__main__":
    print("\n========== Contract Validator Test ==========\n")

    test_missing_rag()
    test_missing_tool()
    test_missing_generation()
    test_valid_direct_plan()

    print("\n========== All Tests Passed ==========\n")
