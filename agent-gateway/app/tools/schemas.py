
"""
Tool Registry 数据结构定义

主要职责：
1. 定义工具处理函数的统一接口。
2. 定义工具元数据结构。
3. 提供工具参数校验能力。
4. 为后续大模型 Tool Calling 提供 JSON Schema。
"""

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Type

from pydantic import BaseModel


# ============================================================
# 1. 定义工具处理函数类型
# ============================================================

# args：通过 Pydantic 校验后的工具参数对象
# context：当前 Agent 执行上下文
# 返回值：必须是字典，方便 ExecutionEngine 保存 Step 输出
ToolHandler = Callable[
    [BaseModel, Any],
    Awaitable[Dict[str, Any]]
]


# ============================================================
# 2. 工具定义
# ============================================================

@dataclass
class ToolDefinition:
    """
    一个 ToolDefinition 代表一个具体的业务工具。

    例如：
        query_sales：查询销售数据
        query_orders：查询订单数据
        query_customers：查询客户数据
    """

    # 工具唯一名称
    name: str

    # 工具功能描述
    description: str

    # 工具参数对应的 Pydantic 模型
    input_model: Type[BaseModel]

    # 工具实际执行函数
    handler: ToolHandler

    # 工具支持的能力标签
    capabilities: List[str] = field(default_factory=list)

    # 是否需要用户确认后才能执行
    requires_confirmation: bool = False

    # 扩展元数据
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """
        对工具定义进行基础合法性检查。

        避免错误的工具定义进入注册中心。
        """

        if not self.name or not self.name.strip():
            raise ValueError("工具名称不能为空")

        if not self.description or not self.description.strip():
            raise ValueError("工具描述不能为空")

        if not isinstance(self.input_model, type):
            raise TypeError("input_model 必须是 Pydantic 模型类")

        if not issubclass(self.input_model, BaseModel):
            raise TypeError("input_model 必须继承 BaseModel")

        if not callable(self.handler):
            raise TypeError("handler 必须是可调用对象")

        # 统一清理能力标签
        self.capabilities = list(
            dict.fromkeys(
                capability.strip()
                for capability in self.capabilities
                if capability and capability.strip()
            )
        )

    # ========================================================
    # 3. 参数校验
    # ========================================================

    def validate_input(
        self,
        input_data: Dict[str, Any],
    ) -> BaseModel:
        """
        使用工具对应的 Pydantic 模型校验输入参数。

        例如：
            {
                "region": "重庆地区",
                "time_range": "最近一个月"
            }

        如果参数缺失或类型错误，Pydantic 会抛出 ValidationError。
        """

        if not isinstance(input_data, dict):
            raise TypeError("工具输入参数必须是字典")

        return self.input_model.model_validate(input_data)

    # ========================================================
    # 4. 获取 JSON Schema
    # ========================================================

    def get_input_schema(self) -> Dict[str, Any]:
        """
        获取工具参数的 JSON Schema。

        后续可以将该 Schema 提供给大模型，
        让大模型按照工具定义生成参数。
        """

        return self.input_model.model_json_schema()
