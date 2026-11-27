import asyncio
import json

from app.tools.catalog.runtime import mcp_tool_adapter


async def main():
    tool_id = "15c65ef9-3192-450b-87f4-dbc4e3fc2dc6"

    print("\n========== MCP TOOL EXECUTION TEST ==========")

    # 1. 从 Catalog 动态加载工具
    definition = await mcp_tool_adapter.load_tool(tool_id)

    print(f"\n[1] 工具加载成功: {definition.name}")

    # 2. 构建并验证参数
    arguments = {
        "message": "人工智能 Agent 技术在企业业务流程自动化中的应用研究",
        "page": 1,
        "size": 5,
    }

    validated_input = definition.input_model(**arguments)

    print("\n[2] 参数验证成功")
    print(validated_input.model_dump())

    # 3. 调用通用 MCP Handler
    print("\n[3] 开始调用远端 MCP Server...")

    result = await definition.handler(
        input_data=validated_input,
        context=None,
    )

    # 4. 输出执行结果
    print("\n[4] MCP 执行结果")

    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    )

    # 5. 判断调用结果
    if not result.get("success"):
        raise RuntimeError(
            f"MCP 工具执行失败: {result}"
        )

    print("\n========== MCP EXECUTION PASSED ==========")


if __name__ == "__main__":
    asyncio.run(main())