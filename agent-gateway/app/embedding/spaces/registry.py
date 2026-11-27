
from app.embedding.registry import EmbeddingProviderRegistry
from app.embedding.spaces.schemas import EmbeddingSpace


class EmbeddingSpaceRegistry:
    """
    Embedding Space 注册中心。

    负责：
    1. 注册业务向量空间
    2. 校验 Space 与 Provider 的一致性
    3. 根据 Space ID 获取空间
    4. 查询和注销空间

    不负责：
    - 创建 Embedding Provider
    - 执行 Embedding 推理
    - 管理 PostgreSQL 向量数据
    - 创建或删除实际数据库索引
    """

    def __init__(
        self,
        provider_registry: EmbeddingProviderRegistry,
    ):
        self._provider_registry = provider_registry
        self._spaces: dict[str, EmbeddingSpace] = {}

    def register(self, space: EmbeddingSpace) -> None:
        """校验并注册 Embedding Space。"""

        space_id = space.space_id

        if space_id in self._spaces:
            raise ValueError(
                f"Embedding Space already registered: {space_id}"
            )

        # 1. 检查 Provider 是否存在
        if not self._provider_registry.has(space.provider_name):
            raise ValueError(
                f"Embedding Provider not found: {space.provider_name}"
            )

        provider = self._provider_registry.get(
            space.provider_name
        )

        # 2. 检查模型名称
        if provider.model_name != space.model_name:
            raise ValueError(
                f"Embedding model mismatch: "
                f"space={space.model_name}, "
                f"provider={provider.model_name}"
            )

        # 3. 检查模型版本
        if provider.model_version != space.model_version:
            raise ValueError(
                f"Embedding model version mismatch: "
                f"space={space.model_version}, "
                f"provider={provider.model_version}"
            )

        # 4. 检查向量维度
        if provider.dimension != space.dimension:
            raise ValueError(
                f"Embedding dimension mismatch: "
                f"space={space.dimension}, "
                f"provider={provider.dimension}"
            )

        # 5. 检查模型配置指纹
        if provider.fingerprint != space.fingerprint:
            raise ValueError(
                f"Embedding model fingerprint mismatch: "
                f"space={space.fingerprint}, "
                f"provider={provider.fingerprint}"
            )

        # 所有检查通过后才注册
        self._spaces[space_id] = space

    def get(self, space_id: str) -> EmbeddingSpace:
        space = self._spaces.get(space_id)

        if space is None:
            raise KeyError(
                f"Embedding Space not found: {space_id}"
            )

        return space

    def has(self, space_id: str) -> bool:
        return space_id in self._spaces

    def unregister(self, space_id: str) -> None:
        if space_id not in self._spaces:
            raise KeyError(
                f"Embedding Space not found: {space_id}"
            )

        del self._spaces[space_id]

    @property
    def space_ids(self) -> tuple[str, ...]:
        return tuple(self._spaces.keys())

    @property
    def count(self) -> int:
        return len(self._spaces)
