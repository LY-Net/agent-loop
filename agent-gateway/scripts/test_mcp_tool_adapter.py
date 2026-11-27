import asyncio

from app.tools.catalog.runtime import mcp_tool_adapter


async def main():

    tool_id = "15c65ef9-3192-450b-87f4-dbc4e3fc2dc6"

    print("\n========== MCP ADAPTER TEST ==========")

    # 1. 动态加载 MCP 工具
    definition = await mcp_tool_adapter.load_tool(
        tool_id
    )

    print("\n[1] 工具动态加载成功")
    print("Gateway Name:", definition.name)
    print("Description:", definition.description)

    # 2. 检查动态生成的参数模型
    print("\n[2] 动态参数模型")
    print(
        definition.input_model.model_json_schema()
    )

    # 3. 检查运行时注册结果
    print("\n[3] 工具注册成功")
    print("Tool Name:", definition.name)

    # 4. 验证参数
    validated = definition.input_model(
        message="人工智能 Agent 技术研究",
        page=1,
        size=10,
    )

    print("\n[4] 参数验证成功")
    print(validated.model_dump())

    print("\n========== ALL TESTS PASSED ==========")


if __name__ == "__main__":
    asyncio.run(main())