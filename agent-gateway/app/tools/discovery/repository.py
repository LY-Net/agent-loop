import re

from sqlalchemy import select, and_, or_, case
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.tool_catalog import CatalogTool
from app.models.tool_embedding import ToolEmbedding


class ToolDiscoveryRepository:
    """
    Tool Discovery 数据访问层。

    负责：
    1. 向量候选检索。
    2. 关键词候选检索。
    3. 统一应用工具状态和元数据过滤条件。

    不负责：
    - Embedding 生成。
    - RRF 排序。
    - 数据库事务提交。
    """

    def __init__(self, session: AsyncSession):
        self.session = session

    @staticmethod
    def _provider_values(provider_types: list) -> list[str]:
        return [
            item.value if hasattr(item, "value") else item
            for item in provider_types
        ]

    @staticmethod
    def _base_filters(
        provider_types: list,
        capabilities: list[str],
    ) -> list:
        filters = [
            CatalogTool.status == "active",
        ]

        if provider_types:
            filters.append(
                CatalogTool.provider_type.in_(
                    ToolDiscoveryRepository._provider_values(
                        provider_types
                    )
                )
            )

        if capabilities:
            filters.append(
                CatalogTool.capabilities.contains(capabilities)
            )

        return filters

    @staticmethod
    def _summary_columns():
        return (
            CatalogTool.tool_id,
            CatalogTool.server_id,
            CatalogTool.provider_type,
            CatalogTool.name,
            CatalogTool.description,
            CatalogTool.capabilities,
            CatalogTool.keywords,
            CatalogTool.status,
        )

    async def search_vector(
        self,
        query_vector: list[float],
        space_id: str,
        model_fingerprint: str,
        provider_types: list,
        capabilities: list[str],
        limit: int,
    ) -> list[dict]:
        """通过 pgvector cosine distance 检索候选工具。"""

        distance = ToolEmbedding.embedding.cosine_distance(
            query_vector
        ).label("distance")

        statement = (
            select(
                *self._summary_columns(),
                distance,
            )
            .join(
                ToolEmbedding,
                ToolEmbedding.tool_id == CatalogTool.tool_id,
            )
            .where(
                ToolEmbedding.space_id == space_id,
                ToolEmbedding.model_fingerprint == model_fingerprint,
                *self._base_filters(
                    provider_types,
                    capabilities,
                ),
            )
            .order_by(distance.asc())
            .limit(limit)
        )

        result = await self.session.execute(statement)

        return [
            dict(row)
            for row in result.mappings().all()
        ]

    async def search_keyword(
        self,
        terms: list[str],
        provider_types: list,
        capabilities: list[str],
        limit: int,
    ) -> list[dict]:
        """通过名称、描述、关键词和能力标签进行关键词检索。"""

        if not terms:
            return []

        conditions = []
        score_expressions = []

        for term in terms:
            pattern = f"%{term}%"

            name_match = CatalogTool.name.ilike(pattern)
            description_match = CatalogTool.description.ilike(
                pattern
            )
            keyword_match = CatalogTool.keywords.contains([term])
            capability_match = CatalogTool.capabilities.contains(
                [term]
            )

            conditions.append(
                or_(
                    name_match,
                    description_match,
                    keyword_match,
                    capability_match,
                )
            )

            # 名称匹配优先于描述和标签匹配
            score_expressions.extend([
                case((name_match, 4), else_=0),
                case((description_match, 2), else_=0),
                case((keyword_match, 3), else_=0),
                case((capability_match, 3), else_=0),
            ])

        relevance_score = sum(score_expressions).label(
            "relevance_score"
        )

        statement = (
            select(
                *self._summary_columns(),
                relevance_score,
            )
            .where(
                and_(
                    *self._base_filters(
                        provider_types,
                        capabilities,
                    ),
                    or_(*conditions),
                )
            )
            .order_by(
                relevance_score.desc(),
                CatalogTool.updated_at.desc(),
            )
            .limit(limit)
        )

        result = await self.session.execute(statement)

        return [
            dict(row)
            for row in result.mappings().all()
        ]