
from app.embedding.providers.base import EmbeddingProvider


class EmbeddingProviderRegistry:
    """
    Embedding Provider 注册中心。

    负责：
    1. 注册 Provider
    2. 根据名称获取 Provider
    3. 查询已注册的 Provider
    4. 统一释放 Provider 资源

    不负责：
    - 数据分类
    - 安全策略判断
    - Provider 路由决策
    - Embedding 请求处理
    """

    def __init__(self):
        self._providers: dict[str, EmbeddingProvider] = {}

    def register(self, provider: EmbeddingProvider) -> None:
        """注册一个 Provider。"""

        name = provider.provider_name

        if not name:
            raise ValueError("Provider name cannot be empty")

        if name in self._providers:
            raise ValueError(
                f"Provider already registered: {name}"
            )

        self._providers[name] = provider

    def get(self, provider_name: str) -> EmbeddingProvider:
        """根据名称获取 Provider。"""

        provider = self._providers.get(provider_name)

        if provider is None:
            raise KeyError(
                f"Embedding Provider not found: {provider_name}"
            )

        return provider

    def has(self, provider_name: str) -> bool:
        """判断 Provider 是否已注册。"""

        return provider_name in self._providers

    @property
    def provider_names(self) -> tuple[str, ...]:
        """返回已注册 Provider 的名称。"""

        return tuple(self._providers.keys())

    async def close_all(self) -> None:
        """释放所有已注册 Provider 的资源。"""

        providers = list(self._providers.values())
        self._providers.clear()

        errors = []

        for provider in providers:
            try:
                await provider.close()
            except Exception as exc:
                errors.append(
                    f"{provider.provider_name}: {exc}"
                )

        if errors:
            raise RuntimeError(
                "Failed to close Embedding Providers: "
                + "; ".join(errors)
            )
