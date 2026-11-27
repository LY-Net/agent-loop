from app.intent.schemas import (
    IntentCategory,
    IntentStatus,
    IntentResult,
    CoarseIntentResult,
    IntentFineResult,
    ExecutionMode,
)

from app.planner import (
    execution_planner,
    StepType,
    PlanStatus,
)
from app.planner.validator import plan_validator

def build_intent(
    mode: ExecutionMode,
    need_rag: bool = False,
    need_tool: bool = False,
):
    """构造模拟的 Intent Engine 输出"""

    coarse = CoarseIntentResult(
        category=IntentCategory.DATA_ANALYSIS,
        confidence=0.95,
        need_fine=True,
        reason="测试意图",
    )

    fine = IntentFineResult(
        primary_intent="sales_analysis",
        intent_description="分析销售数据",
        confidence=0.96,
        execution_mode=mode,
        need_rag=need_rag,
        need_tool=need_tool,
        required_capabilities=[
            "sales_data_query",
            "data_analysis",
        ],
    )

    return IntentResult(
        status=IntentStatus.SUCCESS,
        coarse=coarse,
        fine=fine,
        final_intent="sales_analysis",
        total_confidence=0.95,
    )


def print_plan(name, plan):
    print(f"\n{'=' * 60}")
    print(f"测试场景：{name}")
    print(f"执行模式：{plan.execution_mode}")
    print(f"计划状态：{plan.status.value}")
    print(f"意图名称：{plan.intent_name}")
    print("-" * 60)

    for index, step in enumerate(plan.steps, 1):
        print(f"步骤 {index}: {step.name}")
        print(f"  类型: {step.step_type.value}")
        print(f"  依赖: {step.depends_on}")
        print(f"  能力: {step.required_capabilities}")

    print(f"{'=' * 60}")


def test_direct():
    intent = build_intent(ExecutionMode.DIRECT)

    plan = execution_planner.build_plan(
        intent,
        "你好，请介绍一下你自己",
    )
    plan_validator.validate(plan)
    print("Plan Validation: PASSED")
    assert len(plan.steps) == 1
    assert plan.steps[0].step_type == StepType.LLM_GENERATION

    print_plan("DIRECT", plan)


def test_rag():
    intent = build_intent(ExecutionMode.RAG)

    plan = execution_planner.build_plan(
        intent,
        "公司的销售指标有哪些？",
    )
    plan_validator.validate(plan)
    print("Plan Validation: PASSED")
    assert len(plan.steps) == 2
    assert plan.steps[0].step_type == StepType.RAG_RETRIEVAL
    assert plan.steps[1].step_type == StepType.LLM_GENERATION

    assert plan.steps[0].step_id in plan.steps[1].depends_on

    print_plan("RAG", plan)


def test_tool():
    intent = build_intent(ExecutionMode.TOOL, need_tool=True)

    plan = execution_planner.build_plan(
        intent,
        "查询最近三个月的销售数据",
    )
    plan_validator.validate(plan)
    print("Plan Validation: PASSED")
    assert any(
        step.step_type == StepType.TOOL_CALL
        for step in plan.steps
    )

    print_plan("TOOL", plan)


def test_hybrid():
    intent = build_intent(
        ExecutionMode.HYBRID,
        need_rag=True,
        need_tool=True,
    )

    plan = execution_planner.build_plan(
        intent,
        "结合公司销售指标分析最近三个月的销售数据",
    )

    rag_steps = [
        step for step in plan.steps
        if step.step_type == StepType.RAG_RETRIEVAL
    ]

    tool_steps = [
        step for step in plan.steps
        if step.step_type == StepType.TOOL_CALL
    ]

    generation_steps = [
        step for step in plan.steps
        if step.step_type == StepType.LLM_GENERATION
    ]

    assert len(rag_steps) == 1
    assert len(tool_steps) == 1
    assert len(generation_steps) == 1

    assert rag_steps[0].step_id in tool_steps[0].depends_on
    assert rag_steps[0].step_id in generation_steps[0].depends_on
    assert tool_steps[0].step_id in generation_steps[0].depends_on
    plan_validator.validate(plan)
    print("Plan Validation: PASSED")
    print_plan("HYBRID", plan)


def test_planning():
    intent = build_intent(ExecutionMode.PLANNING)

    plan = execution_planner.build_plan(
        intent,
        "帮我制定一个完整的销售数据分析方案",
    )
    plan_validator.validate(plan)
    print("Plan Validation: PASSED")
    assert plan.steps[0].step_type == StepType.TASK_PLANNING
    assert plan.steps[1].step_type == StepType.LLM_GENERATION

    print_plan("PLANNING", plan)


def test_need_rag():
    """验证 need_rag 能否覆盖 DIRECT 模式"""

    intent = build_intent(
        ExecutionMode.DIRECT,
        need_rag=True,
    )

    plan = execution_planner.build_plan(
        intent,
        "查询公司的销售制度",
    )
    plan_validator.validate(plan)
    print("Plan Validation: PASSED")
    assert any(
        step.step_type == StepType.RAG_RETRIEVAL
        for step in plan.steps
    )

    print_plan("DIRECT + NEED_RAG", plan)


if __name__ == "__main__":
    tests = [
        test_direct,
        test_rag,
        test_tool,
        test_hybrid,
        test_planning,
        test_need_rag,
    ]

    for test in tests:
        test()

    print("\n所有 Planner 测试通过！")
