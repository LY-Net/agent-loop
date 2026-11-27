"""
Execution Engine 异常处理测试

测试目标：
1. TOOL_CALL 执行失败时，Step 是否标记为 FAILED。
2. 依赖失败 Step 的下游任务是否标记为 SKIPPED。
3. 不依赖失败 Step 的独立任务是否仍然执行。
4. 整个 Plan 是否最终标记为 FAILED。
"""

import asyncio
import uuid

from app.planner.schemas import (
    ExecutionPlan,
    ExecutionStep,
    PlanStatus,
    StepStatus,
    StepType,
)

from app.intent.schemas import ExecutionMode

from app.executor import (
    execution_engine,
    executor_registry,
)


# ============================================================
# 1. Mock Handler：模拟工具执行失败
# ============================================================

async def mock_failed_tool_handler(step, context):
    """
    模拟一个执行失败的数据库工具。

    这里主动抛出异常，用于验证 ExecutionEngine 的异常捕获机制。
    """

    print("\n[MOCK TOOL] 开始执行数据库查询")
    print("[MOCK TOOL] 模拟数据库连接失败")

    raise RuntimeError("Database connection timeout")


# ============================================================
# 2. Mock Handler：模拟正常 LLM 执行
# ============================================================

async def mock_llm_handler(step, context):
    """
    模拟 LLM 生成回答。

    如果该 Step 能被执行，说明它没有被失败依赖阻断。
    """

    print(f"\n[MOCK LLM] 执行独立任务：{step.name}")

    return {
        "answer": f"{step.name} 执行成功",
        "status": "success",
    }


# ============================================================
# 3. 注册 Handler
# ============================================================

def register_handlers():
    """
    注册本次测试所需的 Handler。

    注意：
    ExecutorRegistry 管理的是 StepType 与 Handler 的映射，
    不是具体业务工具的注册中心。
    """

    executor_registry.register(
        StepType.TOOL_CALL,
        mock_failed_tool_handler,
    )

    executor_registry.register(
        StepType.LLM_GENERATION,
        mock_llm_handler,
    )


# ============================================================
# 4. 构建测试 Plan
# ============================================================

def build_failure_test_plan():
    """
    构造一个包含三个 Step 的执行计划：

        Step 1：执行数据库查询（必定失败）
                    |
                    v
        Step 2：生成数据库分析报告（依赖 Step 1，应被跳过）

        Step 3：生成独立说明（不依赖 Step 1，应正常执行）
    """

    tool_step_id = str(uuid.uuid4())
    dependent_step_id = str(uuid.uuid4())
    independent_step_id = str(uuid.uuid4())

    steps = [
        # ----------------------------------------------------
        # Step 1：模拟失败的工具调用
        # ----------------------------------------------------
        ExecutionStep(
            step_id=tool_step_id,
            step_type=StepType.TOOL_CALL,
            name="execute_tool",
            description="模拟执行数据库查询",
            depends_on=[],
            input_data={
                "region": "重庆地区",
                "time_range": "最近一个月",
                "metric": "sales_amount",
            },
            required_capabilities=[],
            status=StepStatus.PENDING,
        ),

        # ----------------------------------------------------
        # Step 2：依赖 Step 1
        # 如果 Step 1 失败，本步骤应该被跳过
        # ----------------------------------------------------
        ExecutionStep(
            step_id=dependent_step_id,
            step_type=StepType.LLM_GENERATION,
            name="generate_sales_report",
            description="根据数据库结果生成销售分析报告",
            depends_on=[tool_step_id],
            input_data={
                "task": "生成重庆地区销售分析报告",
            },
            required_capabilities=[],
            status=StepStatus.PENDING,
        ),

        # ----------------------------------------------------
        # Step 3：独立任务
        # 不依赖 Step 1，应该继续执行
        # ----------------------------------------------------
        ExecutionStep(
            step_id=independent_step_id,
            step_type=StepType.LLM_GENERATION,
            name="generate_independent_answer",
            description="生成与数据库查询无关的说明",
            depends_on=[],
            input_data={
                "task": "生成一段独立说明",
            },
            required_capabilities=[],
            status=StepStatus.PENDING,
        ),
    ]

    return ExecutionPlan(
        plan_id=str(uuid.uuid4()),
        intent_name="test_execution_failure",
        execution_mode=ExecutionMode.HYBRID,
        description="Execution Engine 异常处理测试",
        status=PlanStatus.CREATED,
        steps=steps,
        need_clarification=False,
        metadata={},
    )


# ============================================================
# 5. 执行测试
# ============================================================

async def main():
    print("\n========== EXECUTION FAILURE TEST START ==========")

    register_handlers()

    plan = build_failure_test_plan()

    result = await execution_engine.execute(
        plan,
        user_input="帮我分析重庆地区最近一个月的销售额",
    )

    print("\n========== EXECUTION FAILURE TEST RESULT ==========")

    print("Plan Status:", result.status)

    for index, step in enumerate(result.steps, start=1):
        print(f"\nStep {index}")
        print("Name:", step.name)
        print("Type:", step.step_type)
        print("Status:", step.status)
        print("Output:", step.output_data)

    print("\n========== ASSERTIONS ==========")

    tool_step = result.steps[0]
    dependent_step = result.steps[1]
    independent_step = result.steps[2]

    # 验证工具执行失败
    assert tool_step.status == StepStatus.FAILED, (
        "工具执行失败，但 Step 状态不是 FAILED"
    )

    # 验证依赖失败的 Step 被跳过
    assert dependent_step.status == StepStatus.SKIPPED, (
        "上游 Step 失败，但下游 Step 没有被跳过"
    )

    # 验证独立 Step 仍然执行成功
    assert independent_step.status == StepStatus.SUCCESS, (
        "独立 Step 没有继续执行"
    )

    # 验证整体 Plan 状态
    assert result.status == PlanStatus.FAILED, (
        "存在失败 Step，但 Plan 状态不是 FAILED"
    )

    print("[PASS] 工具异常捕获正常")
    print("[PASS] 下游依赖跳过正常")
    print("[PASS] 独立任务继续执行正常")
    print("[PASS] Plan 最终失败状态正常")

    print("\n========== ALL TESTS PASSED ==========")


if __name__ == "__main__":
    asyncio.run(main())