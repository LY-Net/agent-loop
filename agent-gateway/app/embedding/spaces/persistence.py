
"""
Embedding Space Persistence Service

数据库是持久化事实来源，内存 Registry 是运行时缓存。
"""

from collections.abc import Sequence

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.embedding.spaces.registry import EmbeddingSpaceRegistry
from app.embedding.spaces.repository import EmbeddingSpaceRepository
from app.embedding.spaces.schemas import EmbeddingSpace
from app.models.embedding_space import EmbeddingSpaceRecord


class EmbeddingSpacePersistenceService:

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        registry: EmbeddingSpaceRegistry,
        default_spaces: Sequence[EmbeddingSpace],
    ) -> None:
        self.session_factory = session_factory
        self.registry = registry
        self.default_spaces = tuple(default_spaces)

    @staticmethod
    def _to_space(record: EmbeddingSpaceRecord) -> EmbeddingSpace:
        return EmbeddingSpace(
            space_id=record.space_id,
            business_name=record.business_name,
            description=record.description,
            provider_name=record.provider_name,
            model_name=record.model_name,
            model_version=record.model_version,
            dimension=record.dimension,
            fingerprint=record.fingerprint,
            index_name=record.index_name,
        )

    async def initialize(self) -> tuple[str, ...]:
        # 第一步：确保默认 Space 存在，并读取数据库完整数据。
        async with self.session_factory() as session:
            async with session.begin():
                repository = EmbeddingSpaceRepository(session)

                for space in self.default_spaces:
                    existing = await repository.get_by_space_id(
                        space.space_id
                    )

                    if existing is None:
                        await repository.create(
                            space_id=space.space_id,
                            business_name=space.business_name,
                            description=space.description,
                            provider_name=space.provider_name,
                            model_name=space.model_name,
                            model_version=space.model_version,
                            dimension=space.dimension,
                            fingerprint=space.fingerprint,
                            index_name=space.index_name,
                        )

                records = await repository.list_all()

        # 第二步：数据库事务成功后，刷新内存 Registry。
        for space_id in self.registry.space_ids:
            self.registry.unregister(space_id)

        for record in records:
            self.registry.register(self._to_space(record))

        return tuple(record.space_id for record in records)
