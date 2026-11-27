
import asyncio
import uuid

from app.executor import (
    execution_engine,
    executor_registry,
)

from app.planner.schemas import (
    ExecutionPlan,
    ExecutionStep,
    PlanStatus,
    StepStatus,
    StepType,
)

from app.intent.schemas import ExecutionMode


# ==================================================
# 1. 模拟 Tool Handler
# ==================================================

async def mock_tool_handler(step, context):
    """
    模拟数据库查询处理器。

    真实环境中，这里以后会调用：
        - SQL 查询工具
        - 数据库连接器
        - MCP Tool
        - 内部业务 API

    当前仅返回模拟数据。
    """

    print("\n[MOCK TOOL] 开始执行数据库查询")

    print(
        "[MOCK TOOL] 用户问题:",
        context.user_input,
    )

    print(
        "[MOCK TOOL] Step 输入:",
        step.input_data,
    )

    # 模拟数据库返回结果。
    return {
        "region": "重庆地区",
        "time_range": "最近一个月",
        "metric": "sales_amount",
        "total_sales": 1258000,
        "currency": "CNY",
    }


# ==================================================
# 2. 模拟 LLM Handler
# ==================================================

async def mock_llm_handler(step, context):
    """
    模拟 LLM 生成处理器。

    真实环境中，这里以后会调用 DeepSeek，
    根据 Tool 的查询结果生成自然语言回答。
    """

    print("\n[MOCK LLM] 开始生成分析报告")

    # 获取 Tool Step 的执行结果。
    dependency_id = str(step.depends_on[0])

    tool_result = context.get_step_output(
        dependency_id
    )

    if tool_result is None:
        raise ValueError(
            "没有获取到前置 Tool Step 的执行结果"
        )

    total_sales = tool_result["total_sales"]

    report = (
        f"重庆地区最近一个月的销售额为 "
        f"{total_sales:,} 元。"
    )

    print("[MOCK LLM] 报告生成完成")

    return {
        "answer": report,
        "source": tool_result,
    }


# ==================================================
# 3. 注册 Handler
# ==================================================

executor_registry.register(
    StepType.TOOL_CALL,
    mock_tool_handler,
)

executor_registry.register(
    StepType.LLM_GENERATION,
    mock_llm_handler,
)


# ==================================================
# 4. 构建模拟 ExecutionPlan
# ==================================================

def build_test_plan():

    tool_step_id = str(uuid.uuid4())
    llm_step_id = str(uuid.uuid4())

    tool_step = ExecutionStep(
        step_id=tool_step_id,
        step_type=StepType.TOOL_CALL,
        name="execute_tool",
        description="查询重庆地区最近一个月的销售额",
        depends_on=[],
        input_data={
            "region": "重庆地区",
            "time_range": "最近一个月",
            "metric": "sales_amount",
        },
        output_data={},
        status=StepStatus.PENDING,
        required_capabilities=[
            "database_query",
        ],
    )

    llm_step = ExecutionStep(
        step_id=llm_step_id,
        step_type=StepType.LLM_GENERATION,
        name="generate_response",
        description="根据销售额查询结果生成分析报告",
        depends_on=[
            tool_step_id,
        ],
        input_data={
            "task": "生成销售额分析报告",
        },
        output_data={},
        status=StepStatus.PENDING,
        required_capabilities=[],
    )

    return ExecutionPlan(
        plan_id=str(uuid.uuid4()),
        intent_name="regional_sales_amount_analysis",
        execution_mode=ExecutionMode.TOOL,
        description="重庆地区销售额分析",
        status=PlanStatus.CREATED,
        steps=[
            tool_step,
            llm_step,
        ],
        need_clarification=False,
        clarification_question=None,
        metadata={},
    )


# ==================================================
# 5. 执行测试
# ==================================================

async def main():

    plan = build_test_plan()

    print("\n========== EXECUTION START ==========")

    result = await execution_engine.execute(
        plan=plan,
        user_input="帮我分析重庆地区最近一个月的销售额",
        run_id="test_run_001",
        trace_id="test_trace_001",
    )

    print("\n========== EXECUTION RESULT ==========")

    print(
        "Plan Status:",
        result.status,
    )

    for index, step in enumerate(result.steps):

        print(f"\nStep {index + 1}")

        print("Name:", step.name)
        print("Type:", step.step_type)
        print("Status:", step.status)
        print("Output:", step.output_data)

    print("\n======================================")


if __name__ == "__main__":
    asyncio.run(main())
