from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.rag import RagDocument, RagChunk


class RagRepository:
    """RAG 数据访问层，不负责事务提交。"""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def get_document(
        self,
        document_id: str,
    ) -> RagDocument | None:
        stmt = select(RagDocument).where(
            RagDocument.document_id == document_id
        )
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def create_document(
        self,
        *,
        document_id: str,
        title: str,
        source: str,
        content_hash: str,
        metadata: dict,
    ) -> RagDocument:
        document = RagDocument(
            document_id=document_id,
            title=title,
            source=source,
            content_hash=content_hash,
            status="active",
            metadata_json=metadata,
        )

        self.session.add(document)
        await self.session.flush()

        return document

    async def delete_chunks(
        self,
        document_pk: str,
    ) -> int:
        stmt = delete(RagChunk).where(
            RagChunk.document_id == document_pk
        )
        result = await self.session.execute(stmt)
        return result.rowcount or 0

    async def delete_document(
        self,
        document: RagDocument,
    ) -> None:
        await self.session.delete(document)
        await self.session.flush()

    async def create_chunks(
        self,
        *,
        document_pk: str,
        space_id: str,
        chunks: list[str],
        embeddings: list[list[float]],
        content_hashes: list[str],
        model_fingerprint: str,
    ) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("Chunk and embedding count mismatch")

        if len(chunks) != len(content_hashes):
            raise ValueError("Chunk and hash count mismatch")

        records = [
            RagChunk(
                document_id=document_pk,
                space_id=space_id,
                chunk_index=index,
                content=content,
                content_hash=content_hashes[index],
                embedding=embeddings[index],
                model_fingerprint=model_fingerprint,
            )
            for index, content in enumerate(chunks)
        ]

        self.session.add_all(records)
        await self.session.flush()

    async def list_chunks(
        self,
        document_pk: str,
    ) -> list[RagChunk]:
        stmt = (
            select(RagChunk)
            .where(RagChunk.document_id == document_pk)
            .order_by(RagChunk.chunk_index)
        )

        result = await self.session.execute(stmt)
        return list(result.scalars().all())