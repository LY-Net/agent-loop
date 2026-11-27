import asyncio
from uuid import uuid4

from app.main import app
from app.db.session import AsyncSessionLocal

from app.rag.service import RagService
from app.rag.hybrid import RagHybridSearchService
from app.rag.schemas import (
    RagDocumentIngestRequest,
    RagSearchRequest,
)


DOCUMENTS = [
    {
        "title": "PostgreSQL 数据库说明",
        "content": """
        PostgreSQL 是一个开源关系型数据库。
        PostgreSQL 支持事务、索引、JSONB。
        pgvector 是 PostgreSQL 的向量扩展。
        pgvector 支持 HNSW 向量索引和 Cosine Distance。
        """
        * 8,
    },
    {
        "title": "BGE-M3 模型说明",
        "content": """
        BGE-M3 是一个多语言 Embedding 模型。
        本项目使用 BGE-M3 Dense Embedding。
        向量维度是 1024。
        模型通过 CPU 本地服务生成文本向量。
        """
        * 8,
    },
    {
        "title": "Agent Gateway 架构说明",
        "content": """
        Agent Gateway 是统一的 AI Agent 网关。
        它负责模型调用、工具发现、工具调度。
        RAG 为 Agent 提供知识检索能力。
        """
        * 8,
    },
]


async def main() -> None:
    print("========== RAG HYBRID SEARCH TEST ==========")

    document_ids = [
        f"rag-hybrid-{uuid4().hex[:12]}"
        for _ in DOCUMENTS
    ]

    async with app.router.lifespan_context(app):
        embedding_service = app.state.embedding_space_service
        space_registry = app.state.embedding_space_registry

        try:
            # 1. 写入三个不同主题的文档
            for document_id, item in zip(
                document_ids,
                DOCUMENTS,
            ):
                async with AsyncSessionLocal() as session:
                    service = RagService(
                        session,
                        embedding_service,
                        space_registry,
                    )

                    response = await service.ingest(
                        RagDocumentIngestRequest(
                            document_id=document_id,
                            title=item["title"],
                            content=item["content"],
                            source="hybrid-test",
                            chunk_size=250,
                            chunk_overlap=40,
                        )
                    )

                    assert response.chunk_count > 0

            print("[PASS] 三个不同主题的文档入库成功")

            # 2. 执行中文关键词查询
            query = "PostgreSQL 的 HNSW 向量索引"

            async with AsyncSessionLocal() as session:
                service = RagHybridSearchService(
                    session,
                    embedding_service,
                    space_registry,
                )

                response = await service.search(
                    RagSearchRequest(
                        query=query,
                        top_k=5,
                    )
                )

            assert response.results

            print("[PASS] 混合检索成功")

            # 3. 输出排名及召回来源
            for rank, item in enumerate(
                response.results,
                start=1,
            ):
                print(f"\n[RESULT {rank}]")
                print(f"标题: {item.title}")
                print(f"Chunk: {item.chunk_index}")
                print(f"RRF: {item.rrf_score:.6f}")
                print(f"Vector Rank: {item.vector_rank}")
                print(f"Keyword Rank: {item.keyword_rank}")
                print(f"Matched By: {item.matched_by}")
                print(f"内容: {item.content[:120]}")

            # 4. 验证 PostgreSQL 文档确实被召回
            assert any(
                item.document_id == document_ids[0]
                for item in response.results
            ), "PostgreSQL 文档没有进入最终结果"

            print("\n[PASS] PostgreSQL 主题文档成功召回")

            # 5. 验证 RRF 分数
            assert all(
                item.rrf_score is not None
                and item.rrf_score > 0
                for item in response.results
            )

            print("[PASS] RRF 融合分数正常")

        finally:
            for document_id in document_ids:
                async with AsyncSessionLocal() as session:
                    service = RagService(
                        session,
                        embedding_service,
                        space_registry,
                    )

                    await service.delete(document_id)

            print("\n[PASS] 测试文档清理成功")

    print("============================================")
    print("Result: ALL PASSED")


if __name__ == "__main__":
    asyncio.run(main())