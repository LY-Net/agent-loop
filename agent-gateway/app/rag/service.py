import hashlib
import math

from sqlalchemy.ext.asyncio import AsyncSession

from app.embedding.schemas import EmbeddingInput
from app.embedding.spaces.service import EmbeddingSpaceService
from app.embedding.spaces.registry import EmbeddingSpaceRegistry

from app.rag.chunker import TextChunker
from app.rag.repository import RagRepository
from app.rag.schemas import (
    RagDocumentIngestRequest,
    RagDocumentIngestResponse,
    RagDocumentDeleteResponse,
)


RAG_SPACE_ID = "rag_documents"


class RagService:
    """RAG 文档生命周期管理服务。"""

    def __init__(
        self,
        session: AsyncSession,
        embedding_space_service: EmbeddingSpaceService,
        space_registry: EmbeddingSpaceRegistry,
    ) -> None:
        self.session = session
        self.repository = RagRepository(session)
        self.embedding_space_service = embedding_space_service
        self.space_registry = space_registry

    async def ingest(
        self,
        request: RagDocumentIngestRequest,
    ) -> RagDocumentIngestResponse:
        # 1. 文本切片
        chunker = TextChunker(
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap,
        )

        chunks = chunker.split(request.content)

        if not chunks:
            raise ValueError("Document content is empty")

        # 2. 获取当前 Space 元数据
        space = self.space_registry.get(RAG_SPACE_ID)

        if space is None:
            raise RuntimeError(
                f"Embedding Space not found: {RAG_SPACE_ID}"
            )

        # 3. 生成向量
        # 向量生成放在数据库事务外，避免长时间占用事务。
        embeddings = await self.embedding_space_service.embed(
            space_id=RAG_SPACE_ID,
            request=EmbeddingInput(texts=chunks),
        )

        if len(embeddings) != len(chunks):
            raise RuntimeError("Embedding result count mismatch")

        for vector in embeddings:
            if len(vector) != space.dimension:
                raise RuntimeError("Embedding dimension mismatch")

            if not all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(value)
                for value in vector
            ):
                raise RuntimeError("Invalid embedding vector")

        # 4. 计算内容指纹
        document_hash = self._hash(request.content)

        chunk_hashes = [
            self._hash(chunk)
            for chunk in chunks
        ]

        # 5. 文档和 Chunk 在同一事务中替换
        async with self.session.begin():
            existing = await self.repository.get_document(
                request.document_id
            )

            if existing is not None:
                await self.repository.delete_chunks(existing.id)
                await self.repository.delete_document(existing)

            document = await self.repository.create_document(
                document_id=request.document_id,
                title=request.title,
                source=request.source,
                content_hash=document_hash,
                metadata=request.metadata,
            )

            await self.repository.create_chunks(
                document_pk=document.id,
                space_id=RAG_SPACE_ID,
                chunks=chunks,
                embeddings=embeddings,
                content_hashes=chunk_hashes,
                model_fingerprint=space.fingerprint,
            )

        # 事务成功提交后才返回
        return RagDocumentIngestResponse(
            document_id=request.document_id,
            chunk_count=len(chunks),
            embedding_dimension=space.dimension,
            model_fingerprint=space.fingerprint,
        )

    async def delete(
        self,
        document_id: str,
    ) -> RagDocumentDeleteResponse:
        async with self.session.begin():
            document = await self.repository.get_document(
                document_id
            )

            if document is None:
                return RagDocumentDeleteResponse(
                    document_id=document_id,
                    deleted=False,
                )

            await self.repository.delete_chunks(document.id)
            await self.repository.delete_document(document)

        return RagDocumentDeleteResponse(
            document_id=document_id,
            deleted=True,
        )

    @staticmethod
    def _hash(content: str) -> str:
        return hashlib.sha256(
            content.encode("utf-8")
        ).hexdigest()