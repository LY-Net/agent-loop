
"""
Tool Registry 功能测试

验证：
1. 工具注册
2. 工具查询
3. 重复注册保护
4. 工具参数校验
5. JSON Schema 生成
6. 能力标签筛选
7. 工具注销
"""

from typing import Any, Dict

from pydantic import BaseModel, Field

from app.tools import (
    ToolDefinition,
    ToolRegistry,
)


# ============================================================
# 1. 定义销售查询工具的输入参数
# ============================================================

class QuerySalesInput(BaseModel):
    """
    销售查询工具的参数模型。

    后续大模型生成工具参数时，
    就需要符合这个模型定义。
    """

    region: str = Field(
        description="查询地区，例如重庆地区"
    )

    time_range: str = Field(
        description="时间范围，例如最近一个月"
    )

    metric: str = Field(
        default="sales_amount",
        description="统计指标",
    )


# ============================================================
# 2. 定义 Mock 工具 Handler
# ============================================================

async def mock_query_sales(
    args: QuerySalesInput,
    context: Any,
) -> Dict[str, Any]:
    """
    模拟销售查询工具。

    注意：
    目前只测试注册中心，不会真正执行这个 Handler。
    """

    return {
        "region": args.region,
        "time_range": args.time_range,
        "metric": args.metric,
        "total_sales": 1258000,
        "currency": "CNY",
    }


async def mock_query_orders(
    args: BaseModel,
    context: Any,
) -> Dict[str, Any]:

    return {
        "total_orders": 3200,
    }


# ============================================================
# 3. 执行测试
# ============================================================

def main():

    print("\n========== TOOL REGISTRY TEST START ==========")

    # 使用独立实例，避免影响应用全局注册中心
    registry = ToolRegistry()

    # --------------------------------------------------------
    # 测试一：注册销售查询工具
    # --------------------------------------------------------

    sales_tool = ToolDefinition(
        name="query_sales",
        description="查询指定地区和时间范围内的销售数据",
        input_model=QuerySalesInput,
        handler=mock_query_sales,
        capabilities=[
            "sales_query",
            "database_read",
            "regional_analysis",
        ],
        requires_confirmation=False,
        metadata={
            "version": "1.0.0",
            "category": "data_analysis",
        },
    )

    registry.register(sales_tool)

    print("[PASS] 销售查询工具注册成功")

    # --------------------------------------------------------
    # 测试二：查询工具
    # --------------------------------------------------------

    tool = registry.get("query_sales")

    assert tool is not None
    assert tool.name == "query_sales"

    print("[PASS] 工具查询成功")

    # --------------------------------------------------------
    # 测试三：重复注册保护
    # --------------------------------------------------------

    duplicate_blocked = False

    try:
        registry.register(sales_tool)

    except ValueError:
        duplicate_blocked = True

    assert duplicate_blocked

    print("[PASS] 重复注册保护正常")

    # --------------------------------------------------------
    # 测试四：参数校验
    # --------------------------------------------------------

    validated_args = tool.validate_input({
        "region": "重庆地区",
        "time_range": "最近一个月",
        "metric": "sales_amount",
    })

    assert validated_args.region == "重庆地区"
    assert validated_args.metric == "sales_amount"

    print("[PASS] 工具参数校验成功")

    # --------------------------------------------------------
    # 测试五：参数错误校验
    # --------------------------------------------------------

    invalid_blocked = False

    try:
        tool.validate_input({
            "region": "重庆地区",
            # 故意缺少 time_range
        })

    except Exception:
        invalid_blocked = True

    assert invalid_blocked

    print("[PASS] 非法参数拦截正常")

    # --------------------------------------------------------
    # 测试六：JSON Schema 生成
    # --------------------------------------------------------

    input_schema = tool.get_input_schema()

    assert "properties" in input_schema
    assert "region" in input_schema["properties"]
    assert "time_range" in input_schema["properties"]

    print("[PASS] JSON Schema 生成成功")

    # --------------------------------------------------------
    # 测试七：注册第二个工具
    # --------------------------------------------------------

    order_tool = ToolDefinition(
        name="query_orders",
        description="查询订单数据",
        input_model=BaseModel,
        handler=mock_query_orders,
        capabilities=[
            "order_query",
            "database_read",
        ],
    )

    registry.register(order_tool)

    assert registry.count() == 2

    print("[PASS] 多工具注册成功")

    # --------------------------------------------------------
    # 测试八：根据能力筛选工具
    # --------------------------------------------------------

    matched_tools = registry.find_by_capabilities([
        "database_read",
        "sales_query",
    ])

    assert len(matched_tools) == 1
    assert matched_tools[0].name == "query_sales"

    print("[PASS] 工具能力筛选成功")

    # --------------------------------------------------------
    # 测试九：工具列表
    # --------------------------------------------------------

    tools = registry.list_tools()

    assert len(tools) == 2

    print("[PASS] 工具列表查询成功")

    # --------------------------------------------------------
    # 测试十：工具注销
    # --------------------------------------------------------

    removed = registry.unregister("query_orders")

    assert removed is True
    assert registry.has("query_orders") is False
    assert registry.count() == 1

    print("[PASS] 工具注销成功")

    print("\n========== ALL TESTS PASSED ==========")


if __name__ == "__main__":
    main()
