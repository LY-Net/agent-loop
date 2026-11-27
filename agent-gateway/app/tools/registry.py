
"""
Tool Registry

负责管理系统中所有具体业务工具。

注意：
这里的 Registry 与 app.executor.registry 中的
ExecutorRegistry 是两个不同层级的注册中心。

ExecutorRegistry：
    StepType -> Step Handler

ToolRegistry：
    Tool Name -> Tool Definition
"""

from typing import Dict, List, Optional

from app.tools.schemas import ToolDefinition


class ToolRegistry:
    """
    业务工具注册中心。

    内部通过字典保存所有工具定义：

        {
            "query_sales": ToolDefinition(...),
            "query_orders": ToolDefinition(...),
        }
    """

    def __init__(self):
        self._tools: Dict[str, ToolDefinition] = {}

    # ========================================================
    # 1. 注册工具
    # ========================================================

    def register(
        self,
        tool: ToolDefinition,
        replace: bool = False,
    ) -> None:
        """
        注册一个业务工具。

        参数：
            tool：
                工具定义对象。

            replace：
                是否允许覆盖已有同名工具。

        默认不允许重复注册，避免工具被意外覆盖。
        """

        if not isinstance(tool, ToolDefinition):
            raise TypeError("注册对象必须是 ToolDefinition")

        tool_name = tool.name.strip()

        if tool_name in self._tools and not replace:
            raise ValueError(
                f"工具已经注册：{tool_name}"
            )

        self._tools[tool_name] = tool

    # ========================================================
    # 2. 注销工具
    # ========================================================

    def unregister(self, tool_name: str) -> bool:
        """
        注销指定工具。

        返回：
            True：注销成功
            False：工具不存在
        """

        return self._tools.pop(tool_name, None) is not None

    # ========================================================
    # 3. 获取工具
    # ========================================================

    def get(self, tool_name: str) -> Optional[ToolDefinition]:
        """
        根据工具名称获取工具定义。

        如果不存在，返回 None。
        """

        return self._tools.get(tool_name)

    # ========================================================
    # 4. 检查工具是否存在
    # ========================================================

    def has(self, tool_name: str) -> bool:
        return tool_name in self._tools

    # ========================================================
    # 5. 获取所有工具
    # ========================================================

    def list_tools(self) -> List[ToolDefinition]:
        """
        返回所有已注册工具。
        """

        return list(self._tools.values())

    # ========================================================
    # 6. 根据能力筛选工具
    # ========================================================

    def find_by_capabilities(
        self,
        required_capabilities: List[str],
    ) -> List[ToolDefinition]:
        """
        根据能力标签筛选工具。

        采用 AND 匹配：
        一个工具必须同时具备所有要求的能力，才会被返回。

        例如：
            required_capabilities = [
                "database_read",
                "sales_query"
            ]

        只有同时具备这两个能力的工具才会被选中。
        """

        if not required_capabilities:
            return self.list_tools()

        required = set(required_capabilities)

        matched_tools = []

        for tool in self._tools.values():
            tool_capabilities = set(tool.capabilities)

            if required.issubset(tool_capabilities):
                matched_tools.append(tool)

        return matched_tools

    # ========================================================
    # 7. 清空注册中心
    # ========================================================

    def clear(self) -> None:
        """
        清空所有已注册工具。

        主要用于测试或系统重新初始化。
        """

        self._tools.clear()

    # ========================================================
    # 8. 获取工具数量
    # ========================================================

    def count(self) -> int:
        return len(self._tools)


# ============================================================
# 全局 ToolRegistry 单例
# ============================================================

# 整个应用统一使用这个注册中心。
tool_registry = ToolRegistry()
