
"""
Tool Catalog 数据结构定义

主要职责：
1. 定义工具提供方类型。
2. 定义 Server 连接状态。
3. 定义工具注册状态。
4. 定义 MCP Server 配置的数据结构。
5. 定义工具元数据的数据结构。
6. 区分轻量工具信息和完整工具 Schema。

注意：
本文件只定义数据结构，不负责数据库操作和 MCP 连接。
"""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, ConfigDict, HttpUrl


# ============================================================
# 1. 工具来源类型
# ============================================================

class ToolProviderType(str, Enum):
    """
    工具提供方类型。

    BUILTIN:
        Gateway 内置工具。

    MCP:
        通过 MCP 协议连接的外部工具。

    REMOTE:
        通过 HTTP / RPC 等自定义协议连接的工具服务。
    """

    BUILTIN = "builtin"
    MCP = "mcp"
    REMOTE = "remote"


# ============================================================
# 2. Server 状态
# ============================================================

class ToolServerStatus(str, Enum):
    """
    工具 Server 的连接状态。
    """

    PENDING = "pending"
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    ERROR = "error"
    DISABLED = "disabled"


# ============================================================
# 3. Tool 状态
# ============================================================

class CatalogToolStatus(str, Enum):
    """
    工具目录中的工具状态。

    ACTIVE:
        工具正常，可以参与发现和调用。

    DISABLED:
        管理员主动禁用。

    UNAVAILABLE:
        工具所属 Server 不可用。

    DELETED:
        工具已经从 Server 移除，保留记录用于审计。
    """

    ACTIVE = "active"
    DISABLED = "disabled"
    UNAVAILABLE = "unavailable"
    DELETED = "deleted"


# ============================================================
# 4. Server 配置
# ============================================================

class ToolServerCreate(BaseModel):
    """
    创建工具 Server 的请求结构。

    后续新增 MCP Server 时，最基本只需要：
        name
        provider_type
        server_url

    authentication_ref:
        认证信息引用，不直接保存明文 Token。

    connection_config:
        连接扩展配置，例如超时时间等。
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(
        min_length=1,
        max_length=100,
        description="Server 唯一标识名称",
    )

    provider_type: ToolProviderType = Field(
        description="工具提供方类型",
    )

    server_url: HttpUrl | None = Field(
        default=None,
        description="远程 Server 地址，内置工具可以为空",
    )

    description: str = Field(
        default="",
        max_length=500,
        description="Server 功能描述",
    )

    authentication_ref: str | None = Field(
        default=None,
        description="认证信息引用，不保存明文凭证",
    )

    connection_config: dict[str, Any] = Field(
        default_factory=dict,
        description="连接扩展配置",
    )


# ============================================================
# 5. Server 信息
# ============================================================

class ToolServerInfo(BaseModel):
    """
    Server 基础信息。

    用于 Repository 查询结果和 API 返回。
    """

    model_config = ConfigDict(from_attributes=True)

    server_id: str
    name: str
    provider_type: ToolProviderType

    server_url: str | None = None

    description: str = ""

    status: ToolServerStatus = ToolServerStatus.PENDING

    authentication_ref: str | None = None

    connection_config: dict[str, Any] = Field(
        default_factory=dict
    )


# ============================================================
# 6. 工具元数据
# ============================================================

class CatalogToolCreate(BaseModel):
    """
    创建工具目录记录。

    这里保存的是工具的元数据和 JSON Schema，
    不包含 Python Handler 或 Pydantic Model 类。
    """

    model_config = ConfigDict(extra="forbid")

    server_id: str | None = Field(
        default=None,
        description="所属 Server ID，内置工具可以为空",
    )

    provider_type: ToolProviderType

    name: str = Field(
        min_length=1,
        max_length=200,
    )

    description: str = Field(
        default="",
        max_length=2000,
    )

    capabilities: list[str] = Field(
        default_factory=list,
    )

    keywords: list[str] = Field(
        default_factory=list,
    )

    input_schema: dict[str, Any] = Field(
        default_factory=dict,
        description="工具完整 JSON Schema",
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )

    status: CatalogToolStatus = CatalogToolStatus.ACTIVE


# ============================================================
# 7. 工具轻量元数据
# ============================================================

class CatalogToolSummary(BaseModel):
    """
    工具发现阶段使用的轻量数据结构。

    特意不包含 input_schema。

    当系统拥有几千个 MCP 工具时，
    Discovery Engine 可以批量读取 Summary，
    而不必加载所有完整 Schema。
    """

    model_config = ConfigDict(from_attributes=True)

    tool_id: str

    server_id: str | None = None

    provider_type: ToolProviderType

    name: str

    description: str

    capabilities: list[str] = Field(
        default_factory=list,
    )

    keywords: list[str] = Field(
        default_factory=list,
    )

    status: CatalogToolStatus


# ============================================================
# 8. 完整工具定义
# ============================================================

class CatalogToolDetail(CatalogToolSummary):
    """
    工具按需加载时使用的完整数据结构。

    只有确定需要使用某个工具时，
    才加载 input_schema。
    """

    input_schema: dict[str, Any] = Field(
        default_factory=dict,
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )


# ============================================================
# 9. 工具发现查询条件
# ============================================================

class ToolDiscoveryQuery(BaseModel):
    """
    工具发现查询条件。

    后续 Tool Discovery Engine 使用。

    capabilities:
        按能力标签过滤。

    keywords:
        按关键词匹配。

    query:
        用户问题或语义检索文本。

    top_k:
        限制返回的候选工具数量。
    """

    query: str = ""

    capabilities: list[str] = Field(
        default_factory=list,
    )

    keywords: list[str] = Field(
        default_factory=list,
    )

    provider_types: list[ToolProviderType] = Field(
        default_factory=list,
    )

    top_k: int = Field(
        default=10,
        ge=1,
        le=100,
    )
