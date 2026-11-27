
import asyncio

from app.embedding.providers.base import EmbeddingProvider
from app.embedding.providers.local_bge import LocalBGEProvider
from app.embedding.registry import EmbeddingProviderRegistry
from app.embedding.schemas import EmbeddingInput
from app.embedding.service import EmbeddingService
from app.embedding.policy import DataClassification


class MockEmbeddingProvider(EmbeddingProvider):
    """模拟 Provider，用于验证批量处理和安全策略。"""

    def __init__(self, name: str, is_remote: bool):
        self._name = name
        self._is_remote = is_remote
        self.calls = []
        self.closed = False

    @property
    def provider_name(self) -> str:
        return self._name

    @property
    def model_name(self) -> str:
        return "mock-model"

    @property
    def dimension(self) -> int:
        return 1024

    @property
    def is_remote(self) -> bool:
        return self._is_remote

    async def embed(self, texts):
        self.calls.append(list(texts))

        return [
            [float(index)] * self.dimension
            for index, _ in enumerate(texts)
        ]

    async def close(self):
        self.closed = True


async def test_real_local_embedding():
    """验证 Service 到真实 BGE-M3 的完整调用链路。"""

    registry = EmbeddingProviderRegistry()
    provider = LocalBGEProvider()

    registry.register(provider)

    service = EmbeddingService(registry)

    try:
        request = EmbeddingInput(
            texts=[
                "重庆是一座美丽的山城",
                "Agent Gateway 是一个智能体网关",
            ]
        )

        embeddings = await service.embed(request)

        assert len(embeddings) == 2
        assert all(len(vector) == 1024 for vector in embeddings)

        print("[PASS] Service 调用真实 Local BGE")
        print("[PASS] 返回 2 条向量")
        print("[PASS] 每条向量维度为 1024")

    finally:
        await registry.close_all()


async def test_batch_processing():
    """验证批量拆分及结果数量。"""

    registry = EmbeddingProviderRegistry()
    provider = MockEmbeddingProvider(
        name="mock_local",
        is_remote=False,
    )

    registry.register(provider)

    service = EmbeddingService(
        registry=registry,
        provider_name="mock_local",
    )

    try:
        request = EmbeddingInput(
            texts=[f"测试文本 {index}" for index in range(10)]
        )

        embeddings = await service.embed(request)

        assert len(embeddings) == 10

        batch_sizes = [
            len(batch)
            for batch in provider.calls
        ]

        assert batch_sizes == [8, 2], (
            f"Unexpected batch sizes: {batch_sizes}"
        )

        print("[PASS] 批量处理")
        print("[PASS] 批次大小:", batch_sizes)
        print("[PASS] 输出数量与输入数量一致")

    finally:
        await registry.close_all()


async def test_remote_policy():
    """验证远程 Provider 的外发权限控制。"""

    registry = EmbeddingProviderRegistry()
    provider = MockEmbeddingProvider(
        name="mock_remote",
        is_remote=True,
    )

    registry.register(provider)

    service = EmbeddingService(
        registry=registry,
        provider_name="mock_remote",
    )

    try:
        # 1. 默认内部数据，不允许远程调用
        request = EmbeddingInput(
            texts=["内部业务数据"]
        )

        try:
            await service.embed(request)
            raise AssertionError("未授权的远程调用应该被拦截")

        except PermissionError:
            assert len(provider.calls) == 0
            print("[PASS] 未授权远程调用被拦截")
            print("[PASS] 拦截前没有调用 Provider")

        # 2. 公开数据，明确授权后允许调用
        request = EmbeddingInput(
            texts=["公开产品介绍"],
            classification=DataClassification.PUBLIC,
            allow_external=True,
        )

        embeddings = await service.embed(request)

        assert len(embeddings) == 1
        assert len(provider.calls) == 1

        print("[PASS] 明确授权后允许远程调用")

        # 3. 机密数据即使授权也不能外发
        request = EmbeddingInput(
            texts=["企业机密数据"],
            classification=DataClassification.CONFIDENTIAL,
            allow_external=True,
        )

        try:
            await service.embed(request)
            raise AssertionError("机密数据不应该允许远程调用")

        except PermissionError:
            assert len(provider.calls) == 1
            print("[PASS] 机密数据始终禁止外发")

    finally:
        await registry.close_all()


async def main():
    print("\n========== EMBEDDING SERVICE TEST ==========")

    await test_real_local_embedding()
    await test_batch_processing()
    await test_remote_policy()

    print("============================================")
    print("Result: ALL PASSED")


if __name__ == "__main__":
    asyncio.run(main())
