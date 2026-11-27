
"""
RAG 数据模型

rag_documents：原始文档元数据
rag_chunks：文档分块及 Dense Embedding
"""

import uuid
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    String,
    Text,
    Integer,
    DateTime,
    ForeignKey,
    Index,
    CheckConstraint,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


def generate_uuid() -> str:
    return str(uuid.uuid4())


class RagDocument(Base):
    __tablename__ = "rag_documents"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=generate_uuid,
    )

    # 外部文档标识
    document_id: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
        unique=True,
    )

    title: Mapped[str] = mapped_column(
        String(512),
        nullable=False,
    )

    source: Mapped[str] = mapped_column(
        String(1024),
        nullable=False,
        default="",
        server_default="",
    )

    content_hash: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        index=True,
    )

    status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="active",
        server_default="active",
    )

    metadata_json: Mapped[dict] = mapped_column(
        "metadata",
        JSONB,
        nullable=False,
        default=dict,
        server_default=text("'{}'::jsonb"),
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

    chunks: Mapped[list["RagChunk"]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class RagChunk(Base):
    __tablename__ = "rag_chunks"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=generate_uuid,
    )

    document_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("rag_documents.id", ondelete="CASCADE"),
        nullable=False,
    )

    space_id: Mapped[str] = mapped_column(
        String(128),
        ForeignKey("embedding_spaces.space_id", ondelete="RESTRICT"),
        nullable=False,
    )

    chunk_index: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    content: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    content_hash: Mapped[str] = mapped_column(
        String(64),
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

    metadata_json: Mapped[dict] = mapped_column(
        "metadata",
        JSONB,
        nullable=False,
        default=dict,
        server_default=text("'{}'::jsonb"),
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    document: Mapped["RagDocument"] = relationship(
        back_populates="chunks",
    )

    __table_args__ = (
        CheckConstraint(
            "chunk_index >= 0",
            name="ck_rag_chunks_index_nonnegative",
        ),
        UniqueConstraint(
            "document_id",
            "chunk_index",
            name="uq_rag_chunks_document_index",
        ),
        Index(
            "ix_rag_chunks_space_document",
            "space_id",
            "document_id",
        ),
    )
