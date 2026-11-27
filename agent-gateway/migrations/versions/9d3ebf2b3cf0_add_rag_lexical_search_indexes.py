"""add rag lexical search indexes

Revision ID: 9d3ebf2b3cf0
Revises: 8b266bd56850
Create Date: 2026-11-26 16:50:30.665138

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '9d3ebf2b3cf0'
down_revision: Union[str, Sequence[str], None] = '8b266bd56850'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 中文字符片段和模糊关键词检索
    op.execute("""
        CREATE EXTENSION IF NOT EXISTS pg_trgm
    """)

    op.execute("""
        CREATE INDEX IF NOT EXISTS ix_rag_chunks_content_trgm
        ON rag_chunks
        USING gin (content gin_trgm_ops)
    """)

    # 英文、数字和空格分隔术语的全文检索
    op.execute("""
        CREATE INDEX IF NOT EXISTS ix_rag_chunks_content_fts
        ON rag_chunks
        USING gin (to_tsvector('simple', content))
    """)


def downgrade() -> None:
    op.execute("""
        DROP INDEX IF EXISTS ix_rag_chunks_content_fts
    """)

    op.execute("""
        DROP INDEX IF EXISTS ix_rag_chunks_content_trgm
    """)

    # pg_trgm 是数据库级共享扩展，这里不主动删除。