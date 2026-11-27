
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from app.embedding.schemas import EmbeddingInput
from app.embedding.spaces.service import EmbeddingSpaceService


def build_space(
    space_id: str,
    fingerprint: str = "a" * 64,
):
    return SimpleNamespace(
        space_id=space_id,
        provider_name="local_bge",
        model_name="BAAI/bge-m3",
        model_version="v1",
        dimension=1024,
        fingerprint=fingerprint,
    )


def build_model_info():
    return SimpleNamespace(
        provider_name="local_bge",
        model_name="BAAI/bge-m3",
        model_version="v1",
        dimension=1024,
        fingerprint="a" * 64,
    )


async def main():
    print("\n========== EMBEDDING SPACE SERVICE TEST ==========")

    # 1. 构造 Mock 依赖
    embedding_service = Mock()
    embedding_service.model_info = build_model_info()
    embedding_service.embed = AsyncMock(
        return_value=[
            [0.1] * 1024,
            [0.2] * 1024,
        ]
    )

    space_registry = Mock()

    spaces = {
        "rag_documents": build_space("rag_documents"),
        "tool_discovery": build_space("tool_discovery"),
    }

    space_registry.has.side_effect = lambda space_id: (
        space_id in spaces
    )
    space_registry.get.side_effect = lambda space_id: spaces[space_id]

    space_service = EmbeddingSpaceService(
        embedding_service=embedding_service,
        space_registry=space_registry,
    )

    request = EmbeddingInput(
        texts=[
            "Agent Gateway 架构设计",
            "查询重庆地区销售额",
        ]
    )

    # 2. 测试 RAG Space 调用
    rag_vectors = await space_service.embed(
        space_id="rag_documents",
        request=request,
    )

    assert len(rag_vectors) == 2
    assert len(rag_vectors[0]) == 1024

    print("[PASS] RAG Space 调用成功")

    # 3. 测试 Tool Discovery Space 调用
    tool_vectors = await space_service.embed(
        space_id="tool_discovery",
        request=request,
    )

    assert len(tool_vectors) == 2
    assert len(tool_vectors[0]) == 1024

    print("[PASS] Tool Discovery Space 调用成功")

    # 4. 验证两个 Space 都调用了底层 EmbeddingService
    assert embedding_service.embed.await_count == 2

    print("[PASS] 两个 Space 共用底层 EmbeddingService")

    # 5. 测试不存在的 Space
    try:
        await space_service.embed(
            space_id="unknown_space",
            request=request,
        )
        raise AssertionError("Expected ValueError was not raised")

    except ValueError as exc:
        assert "does not exist" in str(exc)

    assert embedding_service.embed.await_count == 2

    print("[PASS] 不存在的 Space 被正确拦截")

    # 6. 测试模型指纹不匹配
    spaces["rag_documents"] = build_space(
        "rag_documents",
        fingerprint="b" * 64,
    )

    try:
        await space_service.embed(
            space_id="rag_documents",
            request=request,
        )
        raise AssertionError("Expected RuntimeError was not raised")

    except RuntimeError as exc:
        assert "Fingerprint mismatch" in str(exc)

    assert embedding_service.embed.await_count == 2

    print("[PASS] 模型指纹不匹配时拒绝调用")

    print("=================================================")
    print("Result: ALL PASSED")


if __name__ == "__main__":
    asyncio.run(main())
