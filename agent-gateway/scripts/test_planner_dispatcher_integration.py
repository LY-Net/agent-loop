
"""
Intent -> Planner -> ExecutionEngine -> Dispatcher 完整集成测试

测试目标：
1. 模拟 Intent Engine 输出。
2. ExecutionPlanner 根据意图生成 ExecutionPlan。
3. 将 ToolDispatcher 注册到 ExecutorRegistry。
4. ExecutionEngine 接收完整 Plan。
5. ExecutionEngine 根据 StepType 调用 ToolDispatcher。
6. ToolDispatcher 根据 required_capabilities 自动匹配工具。
7. 业务 Handler 执行并返回结果。
8. 检查 Plan 和 Step 最终状态。
"""

import asyncio
import uuid

from pydantic import BaseModel, Field

from app.intent.schemas import (
    IntentCategory,
    IntentCandidate,
    IntentFineResult,
    IntentResult,
    IntentStatus,
    CoarseIntentResult,
    ExecutionMode,
)

from app.planner.planner import execution_planner
from app.planner.schemas import (
    PlanStatus,
    StepStatus,
    StepType,
)

from app.executor.executor import execution_engine
from app.executor.registry import executor_registry

from app.tools.schemas import ToolDefinition
from app.tools.registry import tool_registry
from app.tools.dispatcher import tool_dispatcher

# 真实llm
from app.executor.llm_handler import (
    llm_generation_handler,
)


# ============================================================
# 1. 定义业务工具输入参数
# ============================================================

class QuerySalesInput(BaseModel):
    """销售数据查询参数"""

    region: str = Field(description="查询地区")
    time_range: str = Field(description="统计时间范围")
    metric: str = Field(description="统计指标")


# ============================================================
# 2. 定义模拟业务 Handler
# ============================================================

async def query_sales_handler(args, context) -> dict:
    """
    模拟销售数据查询。

    当前使用模拟数据。
    后续可以替换为真实 PostgreSQL 查询逻辑。
    """

    print("\n[BUSINESS TOOL] 开始执行销售数据查询")

    print(f"地区: {args.region}")
    print(f"时间范围: {args.time_range}")
    print(f"统计指标: {args.metric}")

    return {
        "region": args.region,
        "time_range": args.time_range,
        "metric": args.metric,
        "total_sales": 1258000,
        "currency": "CNY",
    }

# ============================================================
# 2.1 模拟 LLM 生成 Handler
# ============================================================

async def mock_llm_handler(step, context) -> dict:
    """
    模拟大模型生成最终答复。

    主要工作：
    1. 获取前置 TOOL_CALL Step 的执行结果。
    2. 提取业务数据。
    3. 模拟大模型组织自然语言。
    4. 返回最终回答。

    后续这里会替换成真实 DeepSeek 调用。
    """

    print("\n[MOCK LLM] 开始生成分析报告")

    # 获取当前 LLM_GENERATION Step 的前置依赖。
    if not step.depends_on:
        raise ValueError(
            "LLM_GENERATION Step 缺少 TOOL_CALL 前置依赖"
        )

    dependency_id = str(step.depends_on[0])

    # 从 ExecutionContext 中获取前置 Step 的输出。
    tool_output = context.get_step_output(dependency_id)

    if not tool_output:
        raise ValueError(
            f"无法获取前置工具步骤结果: {dependency_id}"
        )

    if not tool_output.get("success"):
        raise RuntimeError("前置工具调用没有成功")

    # 提取业务查询结果。
    sales_data = tool_output.get("result", {})

    region = sales_data.get("region")
    time_range = sales_data.get("time_range")
    total_sales = sales_data.get("total_sales")
    currency = sales_data.get("currency")

    # 当前使用模拟文本。
    # 后续将把 sales_data 和用户问题发送给 DeepSeek。
    answer = (
        f"{region}{time_range}的销售额为 "
        f"{total_sales:,} {currency}。"
    )

    print("[MOCK LLM] 报告生成完成")

    return {
        "answer": answer,
        "source_data": sales_data,
        "generation_model": "mock",
    }


# ============================================================
# 3. 模拟 Intent Engine 输出
# ============================================================

def build_intent_result() -> IntentResult:
    """
    构造一个模拟的意图识别结果。

    等价于用户提出：
    帮我分析一下最近一个月重庆地区的销售额
    """

    return IntentResult(
        status=IntentStatus.SUCCESS,

        coarse=CoarseIntentResult(
            category=IntentCategory.DATA_ANALYSIS,
            candidates=[
                IntentCandidate(
                    intent_name="regional_sales_amount_analysis",
                    description="地区销售额分析",
                    confidence=0.96,
                )
            ],
            confidence=0.96,
            need_fine=False,
            reason="明确的数据分析请求",
        ),

        fine=IntentFineResult(
            primary_intent="regional_sales_amount_analysis",
            intent_description="查询指定地区指定时间范围的销售额",
            confidence=0.95,

            parameters={
                "region": "重庆地区",
                "time_range": "最近一个月",
                "metric": "sales_amount",
            },

            required_capabilities=[
                "sales_query",
                "database_read",
            ],

            execution_mode=ExecutionMode.TOOL,
            need_tool=True,
            need_rag=False,
            need_clarification=False,

            reason="需要通过销售数据查询工具获取业务数据",
        ),

        final_intent="regional_sales_amount_analysis",
        total_confidence=0.95,
        need_clarification=False,
    )


# ============================================================
# 4. 主测试流程
# ============================================================

async def main():

    print("\n========== FULL AGENT EXECUTION INTEGRATION TEST ==========")

    user_input = "帮我分析一下最近一个月重庆地区的销售额"

    # --------------------------------------------------------
    # 第一步：注册业务工具
    # --------------------------------------------------------

    tool_definition = ToolDefinition(
        name="query_sales_integration",
        description="查询指定地区和时间范围的销售数据",
        input_model=QuerySalesInput,
        handler=query_sales_handler,
        capabilities=[
            "sales_query",
            "database_read",
        ],
    )

    tool_registry.register(tool_definition)

    print("[PASS] ToolRegistry 业务工具注册成功")

    # --------------------------------------------------------
    # 第二步：注册 Dispatcher 到 ExecutorRegistry
    # --------------------------------------------------------

    # ExecutionEngine 不直接认识 ToolDispatcher。
    # 它只根据 StepType 查找对应的 Handler。
    #
    # 因此需要把 TOOL_CALL 映射到 Dispatcher 的 handle_step 方法。

    executor_registry.register(
        StepType.TOOL_CALL,
        tool_dispatcher.handle_step,
    )

    # 注册模拟 LLM Handler。
    executor_registry.register(
        StepType.LLM_GENERATION,
        llm_generation_handler.handle,
    )

    print("[PASS] ExecutorRegistry 注册 Mock LLM Handler 成功")
    print("[PASS] ExecutorRegistry 注册 ToolDispatcher 成功")

    # --------------------------------------------------------
    # 第三步：模拟 Intent Engine 输出
    # --------------------------------------------------------

    intent_result = build_intent_result()

    print("\n[INTENT] 意图识别完成")

    print("Intent:", intent_result.fine.primary_intent)
    print("Parameters:", intent_result.fine.parameters)
    print(
        "Capabilities:",
        intent_result.fine.required_capabilities,
    )

    # --------------------------------------------------------
    # 第四步：Planner 自动生成执行计划
    # --------------------------------------------------------

    execution_plan = execution_planner.build_plan(
        intent_result,
        user_input,
    )

    print("\n[PLANNER] 执行计划生成成功")

    print("Plan ID:", execution_plan.plan_id)
    print("Plan Status:", execution_plan.status)

    # 确认 Planner 确实生成了 TOOL_CALL。
    tool_steps = [
        step
        for step in execution_plan.steps
        if step.step_type == StepType.TOOL_CALL
    ]

    assert len(tool_steps) == 1, (
        f"预期生成一个 TOOL_CALL，实际生成 {len(tool_steps)} 个"
    )

    tool_step = tool_steps[0]

    print("\n[PLANNER] TOOL_CALL INPUT")
    print("Step Name:", tool_step.name)
    print("Input Data:", tool_step.input_data)

    # --------------------------------------------------------
    # 第五步：检查 Planner 的参数传递
    # --------------------------------------------------------

    assert (
        tool_step.input_data.get("arguments")
        == intent_result.fine.parameters
    ), "Intent 参数传递失败"

    assert (
        tool_step.input_data.get("required_capabilities")
        == intent_result.fine.required_capabilities
    ), "工具能力传递失败"

    assert "tool_name" not in tool_step.input_data, (
        "测试要求通过能力自动匹配工具，不应依赖固定 tool_name"
    )

    print("\n[PASS] Intent 参数传递正确")
    print("[PASS] required_capabilities 传递正确")
    print("[PASS] Planner 没有写死 tool_name")

    # --------------------------------------------------------
    # 第六步：交给 ExecutionEngine 执行完整 Plan
    # --------------------------------------------------------

    print("\n[EXECUTION ENGINE] 开始执行完整 Plan")

    executed_plan = await execution_engine.execute(
        plan=execution_plan,
        user_input=user_input,
        run_id=str(uuid.uuid4()),
        trace_id=str(uuid.uuid4()),
    )

    print("\n========== EXECUTION RESULT ==========")

    print("Plan Status:", executed_plan.status)

    for step in executed_plan.steps:

        print("\nStep Name:", step.name)
        print("Step Type:", step.step_type)
        print("Step Status:", step.status)
        print("Step Output:", step.output_data)

    # --------------------------------------------------------
    # 第七步：检查执行结果
    # --------------------------------------------------------

    assert executed_plan.status == PlanStatus.COMPLETED, (
        f"Plan 执行失败，最终状态：{executed_plan.status}"
    )

    assert tool_step.status == StepStatus.SUCCESS, (
        f"TOOL_CALL 执行失败，状态：{tool_step.status}"
    )

    assert tool_step.output_data.get("success") is True

    business_result = tool_step.output_data["result"]

    assert business_result["region"] == "重庆地区"
    assert business_result["time_range"] == "最近一个月"
    assert business_result["total_sales"] == 1258000

    print("\n[PASS] ExecutionEngine 调度成功")
    print("[PASS] ToolDispatcher 被正确调用")
    print("[PASS] 工具自动匹配成功")
    print("[PASS] 业务 Handler 执行成功")
    print("[PASS] Step 状态更新成功")
    print("[PASS] Plan 最终状态正确")

    print("\n========== ALL INTEGRATION TESTS PASSED ==========")


if __name__ == "__main__":
    asyncio.run(main())
