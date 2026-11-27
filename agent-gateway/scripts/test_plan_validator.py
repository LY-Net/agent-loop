
import sys
from pathlib import Path
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.planner.schemas import (
    ExecutionPlan,
    ExecutionStep,
    PlanStatus,
    StepType,
    StepStatus,
)
from app.intent.schemas import ExecutionMode
from app.planner.validator import (
    PlanValidator,
    PlanValidationError,
)


validator = PlanValidator()


def make_step(step_id=None, depends_on=None):
    """构造一个测试步骤。"""
    return ExecutionStep(
        step_id=step_id or str(uuid4()),
        step_type=StepType.LLM_GENERATION,
        name="test_generation",
        description="测试生成步骤",
        depends_on=depends_on or [],
        input_data={},
        output_data={},
        status=StepStatus.PENDING,
        required_capabilities=[],
    )


def make_plan(steps):
    """构造一个测试执行计划。"""
    return ExecutionPlan(
        plan_id=str(uuid4()),
        intent_name="GENERAL",
        execution_mode=ExecutionMode.DIRECT,
        description="Validator 测试计划",
        status=PlanStatus.CREATED,
        steps=steps,
        need_clarification=False,
        clarification_question=None,
        metadata={},
    )


def test_valid_plan():
    step_a = make_step()
    step_b = make_step(depends_on=[str(step_a.step_id)])

    plan = make_plan([step_a, step_b])
    validator.validate(plan)

    print("[PASS] 正常计划")


def test_missing_dependency():
    step = make_step(depends_on=["missing-step"])
    plan = make_plan([step])

    try:
        validator.validate(plan)
        raise AssertionError("未能识别不存在的依赖")
    except PlanValidationError:
        print("[PASS] 不存在的依赖")


def test_self_dependency():
    step_id = str(uuid4())
    step = make_step(
        step_id=step_id,
        depends_on=[step_id],
    )
    plan = make_plan([step])

    try:
        validator.validate(plan)
        raise AssertionError("未能识别自我依赖")
    except PlanValidationError:
        print("[PASS] 自我依赖")


def test_circular_dependency():
    step_a_id = str(uuid4())
    step_b_id = str(uuid4())

    step_a = make_step(
        step_id=step_a_id,
        depends_on=[step_b_id],
    )

    step_b = make_step(
        step_id=step_b_id,
        depends_on=[step_a_id],
    )

    plan = make_plan([step_a, step_b])

    try:
        validator.validate(plan)
        raise AssertionError("未能识别循环依赖")
    except PlanValidationError:
        print("[PASS] 循环依赖")


def test_duplicate_step_id():
    duplicated_id = str(uuid4())

    step_a = make_step(step_id=duplicated_id)
    step_b = make_step(step_id=duplicated_id)

    plan = make_plan([step_a, step_b])

    try:
        validator.validate(plan)
        raise AssertionError("未能识别重复 step_id")
    except PlanValidationError:
        print("[PASS] 重复 step_id")


def test_empty_plan():
    plan = make_plan([])

    try:
        validator.validate(plan)
        raise AssertionError("未能识别空计划")
    except PlanValidationError:
        print("[PASS] 空执行计划")


if __name__ == "__main__":
    tests = [
        test_valid_plan,
        test_missing_dependency,
        test_self_dependency,
        test_circular_dependency,
        test_duplicate_step_id,
        test_empty_plan,
    ]

    print("\n========== Plan Validator Test ==========\n")

    for test in tests:
        test()

    print("\n========== All Tests Passed ==========\n")
