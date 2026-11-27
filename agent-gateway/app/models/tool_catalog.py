
"""
Tool Catalog 数据模型

负责持久化：
1. ToolServer：外部工具服务配置，例如 MCP Server。
2. CatalogTool：从内置工具或外部 Server 同步得到的工具目录信息。

设计原则：
- Server 配置与工具元数据分开存储。
- 工具参数 Schema 使用 PostgreSQL JSONB 存储。
- Catalog 不直接存储 Python handler。
- 同名工具允许来自不同 Server。
- 内置工具允许 server_id 为空。
"""

import uuid

from sqlalchemy import (
    String,
    Text,
    DateTime,
    ForeignKey,
    Index,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


def generate_uuid() -> str:
    """生成字符串格式的 UUID。"""
    return str(uuid.uuid4())


class ToolServer(Base):
    """
    工具服务配置表。

    一条记录代表一个工具服务来源：
    - MCP Server
    - Remote Tool Server
    - 未来也可以扩展其他 Provider

    注意：
    authentication_ref 只保存凭证引用，不保存真实 Token。
    """

    __tablename__ = "tool_servers"

    # Server 唯一 ID
    server_id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=generate_uuid,
    )

    # Server 名称，用于管理后台展示
    name: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
        unique=True,
        index=True,
    )

    # Provider 类型：builtin / mcp / remote
    provider_type: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        index=True,
    )

    # Server 连接地址
    # MCP Server 通常只需要提供 URL 即可接入
    server_url: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    # Server 描述
    description: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    # 认证配置引用，例如凭证管理系统中的 credential_id
    # 不在数据库中保存明文 Token
    authentication_ref: Mapped[str | None] = mapped_column(
        String(256),
        nullable=True,
    )

    # 连接状态：
    # pending / connected / disconnected / error / disabled
    status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="pending",
        server_default="pending",
        index=True,
    )

    # 连接配置，例如超时时间、重试策略等
    # 这里不放敏感凭证
    connection_config: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        server_default="{}",
    )

    created_at: Mapped[DateTime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    updated_at: Mapped[DateTime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )


class CatalogTool(Base):
    """
    工具目录表。

    记录所有可发现的工具，包括：
    - 系统内置工具
    - MCP Server 提供的工具
    - Remote Tool Server 提供的工具

    这里只保存工具的描述信息和 JSON Schema，
    不保存 Python handler，也不负责实际执行。
    """

    __tablename__ = "tool_catalog"

    # 工具目录记录 ID
    tool_id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=generate_uuid,
    )

    # 工具来源 Server。
    # 内置工具可以没有 Server，因此允许为空。
    server_id: Mapped[str | None] = mapped_column(
        String(36),
        ForeignKey(
            "tool_servers.server_id",
            ondelete="CASCADE",
        ),
        nullable=True,
        index=True,
    )

    # Provider 类型：builtin / mcp / remote
    provider_type: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        index=True,
    )

    # 工具名称
    # 注意：不同 Server 可以存在相同名称的工具
    name: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
    )

    # 工具描述
    description: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        default="",
        server_default="",
    )

    # 工具能力标签
    # 例如：["sales_query", "database_read"]
    capabilities: Mapped[list] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
        server_default="[]",
    )

    # 工具关键词
    # 用于后续关键词检索和工具发现
    keywords: Mapped[list] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
        server_default="[]",
    )

    # 工具输入参数 JSON Schema
    # Discovery 阶段只查轻量字段，选中工具后再加载此字段
    input_schema: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        server_default="{}",
    )

    # 工具扩展元数据
    # SQLAlchemy 的 metadata 是保留属性，因此 Python 属性命名为 tool_metadata。
    # 数据库列名仍然使用 metadata。
    tool_metadata: Mapped[dict] = mapped_column(
        "metadata",
        JSONB,
        nullable=False,
        default=dict,
        server_default="{}",
    )

    # 工具状态：
    # active / disabled / unavailable / deleted
    status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="active",
        server_default="active",
        index=True,
    )

    # Schema 版本
    # MCP Server 工具定义变化时，可以用于版本追踪
    schema_version: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
    )

    # Schema 内容哈希
    # 后续同步工具时，可以通过哈希判断 Schema 是否发生变化
    schema_hash: Mapped[str | None] = mapped_column(
        String(128),
        nullable=True,
    )

    created_at: Mapped[DateTime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    updated_at: Mapped[DateTime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    # 来源唯一标识。
    #
    # 例如：
    # mcp:server-001:query_sales
    # mcp:server-002:query_sales
    # builtin:system:generate_report
    #
    # 这样可以避免不同 Server 的同名工具发生冲突。
    source_key: Mapped[str] = mapped_column(
        String(512),
        nullable=False,
        unique=True,
    )

    __table_args__ = (
        # 常见的工具发现过滤条件
        Index(
            "ix_tool_catalog_provider_status",
            "provider_type",
            "status",
        ),

        # 后续按能力标签筛选工具
        # JSONB GIN 索引，支持 JSONB 包含查询
        Index(
            "ix_tool_catalog_capabilities",
            "capabilities",
            postgresql_using="gin",
        ),

        # 按 Server 查询工具
        Index(
            "ix_tool_catalog_server_name",
            "server_id",
            "name",
        ),
    )
