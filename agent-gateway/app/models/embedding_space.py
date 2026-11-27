
"""
Embedding Space 持久化模型

负责持久化 Embedding Space 的元数据信息。

设计原则：
1. 统一管理不同业务的 Embedding Space。
2. Space 元数据与实际向量数据分离。
3. RAG 和 Tool Discovery 可以拥有独立的 Space。
4. 多个 Space 可以共享同一个 Embedding Provider。
5. 不在这里存储实际向量数据。
"""

import uuid
from datetime import datetime

from sqlalchemy import (
    String,
    Text,
    Integer,
    DateTime,
    Index,
    CheckConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


def generate_uuid() -> str:
    """生成字符串格式的 UUID。"""
    return str(uuid.uuid4())


class EmbeddingSpaceRecord(Base):
    """Embedding Space 元数据表。"""

    __tablename__ = "embedding_spaces"

    # 数据库记录 ID
    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=generate_uuid,
    )

    # Space 逻辑唯一标识
    # 例如：rag_documents、tool_discovery
    space_id: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
        unique=True,
    )

    # 业务名称
    business_name: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
    )

    # Space 描述
    description: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        default="",
        server_default="",
    )

    # Embedding Provider 名称
    # 例如：local_bge
    provider_name: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
    )

    # 模型名称
    # 例如：BAAI/bge-m3
    model_name: Mapped[str] = mapped_column(
        String(256),
        nullable=False,
    )

    # 模型逻辑版本
    model_version: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
    )

    # 向量维度
    dimension: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    # 模型配置指纹
    fingerprint: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        index=True,
    )

    # 预留的物理向量索引名称
    # 这里只保存元数据，不代表数据库索引已经创建
    index_name: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
    )

    # 创建时间
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    # 更新时间
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    __table_args__ = (
        CheckConstraint(
            "dimension > 0",
            name="ck_embedding_spaces_dimension_positive",
        ),
        Index(
            "ix_embedding_spaces_provider_model",
            "provider_name",
            "model_name",
        ),
    )
