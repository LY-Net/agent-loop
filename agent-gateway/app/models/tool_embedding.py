
"""
Tool Discovery Embedding 模型

只保存工具检索需要的文本快照和向量。
工具的权威定义仍然保存在 tool_catalog。
"""

import uuid
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    String,
    Text,
    DateTime,
    ForeignKey,
    Index,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


def generate_uuid() -> str:
    return str(uuid.uuid4())


class ToolEmbedding(Base):
    __tablename__ = "tool_embeddings"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=generate_uuid,
    )

    tool_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("tool_catalog.tool_id", ondelete="CASCADE"),
        nullable=False,
    )

    space_id: Mapped[str] = mapped_column(
        String(128),
        ForeignKey("embedding_spaces.space_id", ondelete="RESTRICT"),
        nullable=False,
    )

    embedding_text: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    embedding: Mapped[list[float]] = mapped_column(
        Vector(1024),
        nullable=False,
    )

    model_fingerprint: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    __table_args__ = (
        UniqueConstraint(
            "tool_id",
            "space_id",
            name="uq_tool_embeddings_tool_space",
        ),
        Index(
            "ix_tool_embeddings_space_id",
            "space_id",
        ),
    )
