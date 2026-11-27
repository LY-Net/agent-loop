"""add vector hnsw indexes

Revision ID: 8b266bd56850
Revises: 47616ad12ae8
Create Date: 2026-11-26 15:27:59.123269

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '8b266bd56850'
down_revision: Union[str, Sequence[str], None] = '47616ad12ae8'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # RAG Chunk HNSW 索引
    op.execute("""
        CREATE INDEX IF NOT EXISTS ix_rag_chunks_embedding_hnsw
        ON rag_chunks
        USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64)
    """)

    # Tool Discovery HNSW 索引
    op.execute("""
        CREATE INDEX IF NOT EXISTS ix_tool_embeddings_embedding_hnsw
        ON tool_embeddings
        USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64)
    """)


def downgrade() -> None:
    op.execute("""
        DROP INDEX IF EXISTS ix_tool_embeddings_embedding_hnsw
    """)

    op.execute("""
        DROP INDEX IF EXISTS ix_rag_chunks_embedding_hnsw
    """)