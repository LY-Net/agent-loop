
"""
验证 RAG 和 Tool Discovery ORM 模型注册。
不连接数据库，只检查 SQLAlchemy 元数据。
"""

import app.models

from sqlalchemy.dialects.postgresql import JSONB
from app.db.base import Base


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)
    print(f"[PASS] {message}")


def main() -> None:
    print("========== VECTOR ORM MODEL TEST ==========")

    tables = Base.metadata.tables

    expected_tables = {
        "rag_documents",
        "rag_chunks",
        "tool_embeddings",
    }

    for table_name in expected_tables:
        check(table_name in tables, f"{table_name} 注册成功")

    rag_documents = tables["rag_documents"]
    rag_chunks = tables["rag_chunks"]
    tool_embeddings = tables["tool_embeddings"]

    # JSONB 检查
    check(
        isinstance(rag_documents.c.metadata.type, JSONB),
        "rag_documents metadata 使用 JSONB",
    )
    check(
        isinstance(rag_chunks.c.metadata.type, JSONB),
        "rag_chunks metadata 使用 JSONB",
    )

    # 向量维度检查
    for table in (rag_chunks, tool_embeddings):
        vector_type = table.c.embedding.type
        check(
            vector_type.dim == 1024,
            f"{table.name} embedding 维度为 1024",
        )

    # 外键检查
    rag_chunk_fks = {
        fk.target_fullname for fk in rag_chunks.foreign_keys
    }
    check(
        "rag_documents.id" in rag_chunk_fks,
        "RAG Chunk 文档外键正确",
    )
    check(
        "embedding_spaces.space_id" in rag_chunk_fks,
        "RAG Chunk Space 外键正确",
    )

    tool_embedding_fks = {
        fk.target_fullname for fk in tool_embeddings.foreign_keys
    }
    check(
        "tool_catalog.tool_id" in tool_embedding_fks,
        "Tool Embedding 工具外键正确",
    )
    check(
        "embedding_spaces.space_id" in tool_embedding_fks,
        "Tool Embedding Space 外键正确",
    )

    # 复合唯一约束检查
    tool_unique_constraints = [
        {column.name for column in constraint.columns}
        for constraint in tool_embeddings.constraints
        if constraint.__class__.__name__ == "UniqueConstraint"
    ]

    check(
        {"tool_id", "space_id"} in tool_unique_constraints,
        "Tool Embedding 复合唯一约束正确",
    )

    print("============================================")
    print("Result: ALL PASSED")


if __name__ == "__main__":
    main()
