
import asyncio

from app.embedding.providers.base import EmbeddingProvider
from app.embedding.registry import EmbeddingProviderRegistry


class MockEmbeddingProvider(EmbeddingProvider):
    """用于测试的模拟 Provider，不发起真实网络请求。"""

    def __init__(self, name: str):
        self._name = name
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
        return self._name != "local_bge"

    async def embed(self, texts):
        return [[0.1] * self.dimension for _ in texts]

    async def close(self) -> None:
        self.closed = True


async def main():
    registry = EmbeddingProviderRegistry()

    local = MockEmbeddingProvider("local_bge")
    remote = MockEmbeddingProvider("remote_test")

    print("\n========== EMBEDDING REGISTRY TEST ==========")

    # 1. 注册 Provider
    registry.register(local)
    registry.register(remote)

    assert registry.has("local_bge")
    assert registry.has("remote_test")

    print("[PASS] Provider 注册")

    # 2. 获取 Provider
    provider = registry.get("local_bge")

    assert provider is local
    assert provider.model_name == "mock-model"

    print("[PASS] Provider 获取")

    # 3. 查询 Provider 列表
    names = registry.provider_names

    assert set(names) == {"local_bge", "remote_test"}

    print("[PASS] Provider 列表查询")

    # 4. 重复注册保护
    try:
        registry.register(local)
        raise AssertionError("重复注册应该失败")
    except ValueError:
        print("[PASS] 重复注册保护")

    # 5. 不存在的 Provider
    try:
        registry.get("not_exists")
        raise AssertionError("不存在的 Provider 应该失败")
    except KeyError:
        print("[PASS] 不存在的 Provider 异常处理")

    # 6. 关闭所有 Provider
    await registry.close_all()

    assert local.closed is True
    assert remote.closed is True
    assert registry.provider_names == ()

    print("[PASS] Provider 资源释放")
    print("[PASS] 注册表清空")

    print("=============================================")
    print("Result: ALL PASSED")


if __name__ == "__main__":
    asyncio.run(main())
