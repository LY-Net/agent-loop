
import asyncio

from app.embedding.providers.base import EmbeddingProvider
from app.embedding.registry import EmbeddingProviderRegistry
from app.embedding.schemas import EmbeddingInput
from app.embedding.service import EmbeddingService


class InvalidEmbeddingProvider(EmbeddingProvider):

    def __init__(self, embeddings):
        self._embeddings = embeddings

    @property
    def provider_name(self):
        return "invalid_mock"

    @property
    def model_name(self):
        return "invalid-model"

    @property
    def dimension(self):
        return 1024

    @property
    def is_remote(self):
        return False

    async def embed(self, texts):
        return self._embeddings

    async def close(self):
        pass


async def assert_rejected(name, embeddings):
    registry = EmbeddingProviderRegistry()

    provider = InvalidEmbeddingProvider(embeddings)
    registry.register(provider)

    service = EmbeddingService(
        registry=registry,
        provider_name="invalid_mock",
    )

    try:
        request = EmbeddingInput(texts=["测试文本"])

        try:
            await service.embed(request)

        except RuntimeError:
            print(f"[PASS] {name} 被成功拦截")
            return

        raise AssertionError(f"{name} 未被拦截")

    finally:
        await registry.close_all()


async def main():
    print("\n========== EMBEDDING VALIDATION TEST ==========")

    # 1. 错误的向量数量
    await assert_rejected(
        "错误向量数量",
        [],
    )

    # 2. 错误的向量维度
    await assert_rejected(
        "错误向量维度",
        [[0.1, 0.2, 0.3]],
    )

    # 3. NaN
    await assert_rejected(
        "NaN 向量值",
        [[float("nan")] * 1024],
    )

    # 4. Infinity
    await assert_rejected(
        "Infinity 向量值",
        [[float("inf")] * 1024],
    )

    # 5. 布尔值不能作为向量数值
    await assert_rejected(
        "布尔值向量元素",
        [[True] * 1024],
    )

    # 6. 非数值元素
    await assert_rejected(
        "字符串向量元素",
        [["invalid"] * 1024],
    )

    print("==============================================")
    print("Result: ALL PASSED")


if __name__ == "__main__":
    asyncio.run(main())
