import asyncio
from uuid import uuid4

from app.main import app
from app.db.session import AsyncSessionLocal

from app.rag.service import RagService
from app.rag.retrieval import RagRetrievalService
from app.rag.schemas import (
    RagDocumentIngestRequest,
    RagSearchRequest,
)


async def main() -> None:
    print("========== RAG RETRIEVAL INTEGRATION TEST ==========")

    document_id = f"rag-search-{uuid4().hex[:12]}"

    content = """
    PostgreSQL 是一个开源的关系型数据库管理系统。
    它支持事务、索引、JSONB 和复杂 SQL 查询。

    pgvector 是 PostgreSQL 的向量扩展。
    它支持向量存储、余弦距离、欧氏距离和内积检索。
    pgvector 可以使用 HNSW 和 IVFFlat 索引加速向量搜索。

    BGE-M3 是一个多语言 Embedding 模型。
    它可以将文本转换成高维向量，用于语义检索和 RAG。
    本项目使用 BGE-M3 的 Dense Embedding，向量维度为 1024。

    Agent Gateway 负责统一管理 AI Agent 的模型调用、
    工具发现、工具调度和 RAG 知识检索。
    """

    content = content * 8

    async with app.router.lifespan_context(app):
        embedding_service = app.state.embedding_space_service
        space_registry = app.state.embedding_space_registry

        try:
            # ======================================
            # 1. 写入测试文档
            # ======================================
            async with AsyncSessionLocal() as session:
                service = RagService(
                    session=session,
                    embedding_space_service=embedding_service,
                    space_registry=space_registry,
                )

                response = await service.ingest(
                    RagDocumentIngestRequest(
                        document_id=document_id,
                        title="PostgreSQL 和 pgvector 技术文档",
                        content=content,
                        source="rag-retrieval-test",
                        chunk_size=300,
                        chunk_overlap=50,
                    )
                )

            assert response.chunk_count > 0

            print("[PASS] 测试文档入库成功")
            print(f"[INFO] Chunk 数量: {response.chunk_count}")

            # ======================================
            # 2. 执行向量检索
            # ======================================
            query = "PostgreSQL 如何进行向量相似度检索？"

            async with AsyncSessionLocal() as session:
                retrieval = RagRetrievalService(
                    session=session,
                    embedding_space_service=embedding_service,
                    space_registry=space_registry,
                )

                result = await retrieval.search(
                    RagSearchRequest(
                        query=query,
                        top_k=5,
                    )
                )

            assert result.query == query
            assert len(result.results) > 0

            print("[PASS] 向量检索成功")
            print(f"[INFO] 返回结果数: {len(result.results)}")

            # ======================================
            # 3. 验证检索结果
            # ======================================
            for index, item in enumerate(result.results, start=1):
                assert item.document_id == document_id
                assert item.content
                assert 0 <= item.distance <= 2

                print(f"\n[RESULT {index}]")
                print(f"标题: {item.title}")
                print(f"Chunk: {item.chunk_index}")
                print(f"Distance: {item.distance:.6f}")
                print(f"Similarity: {item.similarity:.6f}")
                print(f"内容: {item.content[:100]}")

            print("\n[PASS] 检索结果字段校验成功")
            print("[PASS] 检索结果属于目标文档")

        finally:
            # 清理测试数据
            async with AsyncSessionLocal() as session:
                service = RagService(
                    session=session,
                    embedding_space_service=embedding_service,
                    space_registry=space_registry,
                )

                await service.delete(document_id)

            print("\n[PASS] 测试文档清理成功")

    print("====================================================")
    print("Result: ALL PASSED")


if __name__ == "__main__":
    asyncio.run(main())