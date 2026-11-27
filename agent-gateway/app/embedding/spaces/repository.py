
"""
Embedding Space Repository

负责 Embedding Space 元数据的数据库访问。

设计原则：
1. Repository 只负责数据库访问，不承担业务规则。
2. 使用外部注入的 AsyncSession。
3. 不主动 commit，由上层 Service 管理事务。
4. flush 后可在当前事务内继续读取数据。
"""

from typing import Any

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.embedding_space import EmbeddingSpaceRecord


class EmbeddingSpaceRepository:
    """Embedding Space 数据访问层。"""

    _UPDATABLE_FIELDS = {
        "business_name",
        "description",
        "provider_name",
        "model_name",
        "model_version",
        "dimension",
        "fingerprint",
        "index_name",
    }

    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_by_space_id(
        self,
        space_id: str,
    ) -> EmbeddingSpaceRecord | None:
        statement = select(EmbeddingSpaceRecord).where(
            EmbeddingSpaceRecord.space_id == space_id
        )
        result = await self.session.execute(statement)
        return result.scalar_one_or_none()

    async def get_by_id(
        self,
        record_id: str,
    ) -> EmbeddingSpaceRecord | None:
        statement = select(EmbeddingSpaceRecord).where(
            EmbeddingSpaceRecord.id == record_id
        )
        result = await self.session.execute(statement)
        return result.scalar_one_or_none()

    async def list_all(self) -> list[EmbeddingSpaceRecord]:
        statement = select(EmbeddingSpaceRecord).order_by(
            EmbeddingSpaceRecord.space_id
        )
        result = await self.session.execute(statement)
        return list(result.scalars().all())

    async def create(
        self,
        *,
        space_id: str,
        business_name: str,
        description: str,
        provider_name: str,
        model_name: str,
        model_version: str,
        dimension: int,
        fingerprint: str,
        index_name: str,
    ) -> EmbeddingSpaceRecord:
        record = EmbeddingSpaceRecord(
            space_id=space_id,
            business_name=business_name,
            description=description,
            provider_name=provider_name,
            model_name=model_name,
            model_version=model_version,
            dimension=dimension,
            fingerprint=fingerprint,
            index_name=index_name,
        )

        self.session.add(record)
        await self.session.flush()
        await self.session.refresh(record)

        return record

    async def update(
        self,
        space_id: str,
        values: dict[str, Any],
    ) -> EmbeddingSpaceRecord | None:
        record = await self.get_by_space_id(space_id)

        if record is None:
            return None

        invalid_fields = set(values) - self._UPDATABLE_FIELDS
        if invalid_fields:
            raise ValueError(
                f"Unsupported Embedding Space fields: "
                f"{sorted(invalid_fields)}"
            )

        for field, value in values.items():
            setattr(record, field, value)

        await self.session.flush()
        await self.session.refresh(record)

        return record

    async def delete(self, space_id: str) -> bool:
        statement = (
            delete(EmbeddingSpaceRecord)
            .where(EmbeddingSpaceRecord.space_id == space_id)
            .returning(EmbeddingSpaceRecord.id)
        )

        result = await self.session.execute(statement)
        deleted_id = result.scalar_one_or_none()

        return deleted_id is not None

    async def exists(self, space_id: str) -> bool:
        statement = select(EmbeddingSpaceRecord.id).where(
            EmbeddingSpaceRecord.space_id == space_id
        )
        result = await self.session.execute(statement)
        return result.scalar_one_or_none() is not None
