import asyncio
from uuid import uuid4

from app.main import app
from app.db.session import AsyncSessionLocal
from app.rag.service import RagService
from app.rag.repository import RagRepository
from app.rag.schemas import RagDocumentIngestRequest


async def main() -> None:
    print("========== RAG INGESTION INTEGRATION TEST ==========")

    document_id = f"rag-test-{uuid4().hex[:12]}"

    async with app.router.lifespan_context(app):
        embedding_space_service = app.state.embedding_space_service
        embedding_space_registry = app.state.embedding_space_registry

        assert embedding_space_service is not None
        assert embedding_space_registry is not None

        print("[PASS] Gateway 生命周期初始化成功")
        print("[PASS] Embedding Space Service 获取成功")

        try:
            # ==========================================
            # 1. 第一次入库
            # ==========================================
            content_v1 = (
                "Agent Gateway 是一个统一的 AI Agent 网关。"
                "它负责管理模型调用、工具发现、工具调度和上下文处理。"
                "RAG 系统通过文档切片和向量检索，为 Agent 提供相关知识。"
            ) * 30

            request_v1 = RagDocumentIngestRequest(
                document_id=document_id,
                title="RAG 入库集成测试 V1",
                content=content_v1,
                source="integration-test",
                metadata={"test": True, "version": 1},
                chunk_size=300,
                chunk_overlap=50,
            )

            async with AsyncSessionLocal() as session:
                service = RagService(
                    session=session,
                    embedding_space_service=embedding_space_service,
                    space_registry=embedding_space_registry,
                )

                response_v1 = await service.ingest(request_v1)

            assert response_v1.document_id == document_id
            assert response_v1.chunk_count > 0
            assert response_v1.embedding_dimension == 1024
            assert response_v1.model_fingerprint

            print("[PASS] 第一次文档入库成功")
            print(f"[INFO] Chunk 数量: {response_v1.chunk_count}")
            print(f"[INFO] 向量维度: {response_v1.embedding_dimension}")

            # ==========================================
            # 2. 验证数据库真实数据
            # ==========================================
            async with AsyncSessionLocal() as session:
                repository = RagRepository(session)

                document = await repository.get_document(document_id)

                assert document is not None
                assert document.title == "RAG 入库集成测试 V1"
                assert document.content_hash

                chunks = await repository.list_chunks(document.id)

                assert len(chunks) == response_v1.chunk_count

                for chunk in chunks:
                    assert len(chunk.embedding) == 1024
                    assert chunk.space_id == "rag_documents"
                    assert chunk.model_fingerprint == (
                        response_v1.model_fingerprint
                    )

            print("[PASS] 文档数据读取成功")
            print("[PASS] Chunk 向量真实持久化成功")
            print("[PASS] Space ID 和模型指纹正确")

            # ==========================================
            # 3. 重复入库，验证替换逻辑
            # ==========================================
            content_v2 = (
                "这是更新后的 Agent Gateway 文档。"
                "新版本介绍了 RAG 文档更新和向量重新生成机制。"
            ) * 15

            request_v2 = RagDocumentIngestRequest(
                document_id=document_id,
                title="RAG 入库集成测试 V2",
                content=content_v2,
                source="integration-test",
                metadata={"test": True, "version": 2},
                chunk_size=300,
                chunk_overlap=50,
            )

            async with AsyncSessionLocal() as session:
                service = RagService(
                    session=session,
                    embedding_space_service=embedding_space_service,
                    space_registry=embedding_space_registry,
                )

                response_v2 = await service.ingest(request_v2)

            assert response_v2.chunk_count > 0

            async with AsyncSessionLocal() as session:
                repository = RagRepository(session)

                document = await repository.get_document(document_id)

                assert document is not None
                assert document.title == "RAG 入库集成测试 V2"
                assert document.content_hash

                chunks_v2 = await repository.list_chunks(document.id)

                assert len(chunks_v2) == response_v2.chunk_count
                assert all(
                    chunk.model_fingerprint == response_v2.model_fingerprint
                    for chunk in chunks_v2
                )

            print("[PASS] 重复入库替换成功")
            print(f"[INFO] V1 Chunk 数量: {response_v1.chunk_count}")
            print(f"[INFO] V2 Chunk 数量: {response_v2.chunk_count}")

            # ==========================================
            # 4. 删除文档，验证清理
            # ==========================================
            async with AsyncSessionLocal() as session:
                service = RagService(
                    session=session,
                    embedding_space_service=embedding_space_service,
                    space_registry=embedding_space_registry,
                )

                delete_response = await service.delete(document_id)

            assert delete_response.deleted is True

            async with AsyncSessionLocal() as session:
                repository = RagRepository(session)

                document = await repository.get_document(document_id)

                assert document is None

            print("[PASS] 文档删除成功")
            print("[PASS] 文档及关联 Chunk 已清理")

            print("====================================================")
            print("Result: ALL PASSED")

        finally:
            # 测试异常时也尽可能清理测试数据
            async with AsyncSessionLocal() as session:
                service = RagService(
                    session=session,
                    embedding_space_service=embedding_space_service,
                    space_registry=embedding_space_registry,
                )

                await service.delete(document_id)


if __name__ == "__main__":
    asyncio.run(main())