
"""
Tool Dispatcher 集成测试

测试内容：
1. 注册销售查询工具。
2. 将 ToolDispatcher 注册到 ExecutorRegistry。
3. ExecutionEngine 执行 TOOL_CALL。
4. Dispatcher 根据 tool_name 找到工具。
5. Pydantic 校验工具参数。
6. 调用实际 Handler。
7. 返回统一结果。
"""
import uuid
from typing import Any, Dict

from pydantic import BaseModel, Field

from app.intent.schemas import ExecutionMode

from app.planner.schemas import (
    ExecutionPlan,
    ExecutionStep,
    PlanStatus,
    StepStatus,
    StepType,
)

from app.executor import (
    execution_engine,
    executor_registry,
)

from app.tools import (
    ToolDefinition,
    tool_registry,
    tool_dispatcher,
)


# ============================================================
# 1. 定义销售查询参数
# ============================================================

class QuerySalesInput(BaseModel):

    region: str = Field(
        description="查询地区"
    )

    time_range: str = Field(
        description="查询时间范围"
    )

    metric: str = Field(
        default="sales_amount",
        description="统计指标",
    )


# ============================================================
# 2. 定义实际工具 Handler
# ============================================================

async def query_sales_handler(
    args: QuerySalesInput,
    context: Any,
) -> Dict[str, Any]:
    """
    模拟真正的销售数据查询。

    未来这里可以替换成：
        SQLAlchemy 数据库查询
        内部数据 API
        数据仓库查询
    """

    print("\n[BUSINESS TOOL] 正在执行销售数据查询")

    print("地区:", args.region)
    print("时间范围:", args.time_range)
    print("统计指标:", args.metric)

    return {
        "region": args.region,
        "time_range": args.time_range,
        "metric": args.metric,
        "total_sales": 1258000,
        "currency": "CNY",
    }


# ============================================================
# 3. 注册业务工具
# ============================================================

def register_business_tools():

    tool_registry.clear()

    sales_tool = ToolDefinition(
        name="query_sales",
        description="查询指定地区和时间范围内的销售数据",
        input_model=QuerySalesInput,
        handler=query_sales_handler,
        capabilities=[
            "sales_query",
            "database_read",
        ],
        requires_confirmation=False,
        metadata={
            "version": "1.0.0",
        },
    )

    tool_registry.register(sales_tool)

    print("[INIT] query_sales 工具注册成功")


# ============================================================
# 4. 注册 TOOL_CALL 对应的 Step Handler
# ============================================================

def register_step_handlers():

    executor_registry.register(
        StepType.TOOL_CALL,
        tool_dispatcher.handle_step,
        # 如果你当前 ExecutorRegistry.register
        # 不支持 replace 参数，保持这个调用即可。
    )

    print("[INIT] ToolDispatcher 注册成功")


# ============================================================
# 5. 构造 ExecutionPlan
# ============================================================

def build_test_plan():

    tool_step = ExecutionStep(
        step_id=str(uuid.uuid4()),
        step_type=StepType.TOOL_CALL,
        name="query_sales",
        description="查询重庆地区最近一个月的销售额",
        depends_on=[],
        input_data={
            "query": "帮我查询重庆地区最近一个月的销售额",
            "arguments": {
                "region": "重庆地区",
                "time_range": "最近一个月",
                "metric": "sales_amount",
            },
            "required_capabilities": [
                "sales_query",
                "database_read",
            ],
        },
        required_capabilities=[
            "sales_query",
            "database_read",
        ],
        status=StepStatus.PENDING,
    )

    return ExecutionPlan(
        plan_id=str(uuid.uuid4()),
        intent_name="query_sales",
        execution_mode=ExecutionMode.TOOL,
        description="Tool Dispatcher 集成测试",
        status=PlanStatus.CREATED,
        steps=[tool_step],
        need_clarification=False,
        metadata={},
    )


# ============================================================
# 6. 执行测试
# ============================================================

async def main():

    print("\n========== TOOL DISPATCHER TEST START ==========")

    # 注册具体业务工具
    register_business_tools()

    # 将 Dispatcher 接入执行器
    register_step_handlers()

    # 构造执行计划
    plan = build_test_plan()

    # 通过 ExecutionEngine 执行
    result = await execution_engine.execute(
        plan,
        user_input="帮我查询重庆地区最近一个月的销售额",
    )

    print("\n========== EXECUTION RESULT ==========")

    print("Plan Status:", result.status)

    for step in result.steps:
        print("Step Name:", step.name)
        print("Step Status:", step.status)
        print("Step Output:", step.output_data)

    # 验证 Plan 状态
    assert result.status == PlanStatus.COMPLETED

    # 验证 Step 状态
    assert result.steps[0].status == StepStatus.SUCCESS

    # 验证工具返回结果
    output = result.steps[0].output_data

    assert output["tool_name"] == "query_sales"
    assert output["success"] is True
    assert output["result"]["total_sales"] == 1258000

    print("\n[PASS] ToolRegistry 工具注册正常")
    print("[PASS] ExecutorRegistry 调度正常")
    print("[PASS] ToolDispatcher 工具查找正常")
    print("[PASS] Pydantic 参数校验正常")
    print("[PASS] 业务 Handler 调用正常")
    print("[PASS] ExecutionEngine 集成正常")

    print("\n========== ALL TESTS PASSED ==========")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
