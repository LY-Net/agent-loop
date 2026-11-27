
"""
Tool Embedding Repository

职责：
1. 保存工具的语义检索文本和向量。
2. 按 tool_id + space_id 更新向量。
3. 清理已禁用、失效或删除工具的向量。
4. 不创建 Session，不主动提交事务。
"""

from typing import Any

from sqlalchemy import delete, func
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.tool_embedding import ToolEmbedding


class ToolEmbeddingRepository:

    def __init__(self, session: AsyncSession):
        self.session = session

    async def upsert_many(
        self,
        items: list[dict[str, Any]],
    ) -> int:
        """
        批量新增或更新工具向量。

        每个 item 包含：
        tool_id
        space_id
        embedding_text
        embedding
        model_fingerprint
        """

        if not items:
            return 0

        statement = insert(ToolEmbedding).values(items)

        statement = statement.on_conflict_do_update(
            index_elements=[
                ToolEmbedding.tool_id,
                ToolEmbedding.space_id,
            ],
            set_={
                "embedding_text": statement.excluded.embedding_text,
                "embedding": statement.excluded.embedding,
                "model_fingerprint": (
                    statement.excluded.model_fingerprint
                ),
                "updated_at": func.now(),
            },
        )

        result = await self.session.execute(statement)

        return result.rowcount or 0

    async def delete_by_tool_ids(
        self,
        tool_ids: list[str],
        space_id: str,
    ) -> int:
        """批量删除指定工具在某个 Space 中的向量。"""

        if not tool_ids:
            return 0

        statement = delete(ToolEmbedding).where(
            ToolEmbedding.tool_id.in_(tool_ids),
            ToolEmbedding.space_id == space_id,
        )

        result = await self.session.execute(statement)

        return result.rowcount or 0

    async def get_by_tool_id(
        self,
        tool_id: str,
        space_id: str,
    ) -> ToolEmbedding | None:
        """查询单个工具的向量记录。"""

        from sqlalchemy import select

        result = await self.session.execute(
            select(ToolEmbedding).where(
                ToolEmbedding.tool_id == tool_id,
                ToolEmbedding.space_id == space_id,
            )
        )

        return result.scalar_one_or_none()

    async def count_by_space(
        self,
        space_id: str,
    ) -> int:
        """统计某个 Space 的工具向量数量。"""

        from sqlalchemy import select, func

        result = await self.session.execute(
            select(func.count())
            .select_from(ToolEmbedding)
            .where(ToolEmbedding.space_id == space_id)
        )

        return result.scalar_one()
