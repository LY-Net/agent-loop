
from app.embedding.schemas import EmbeddingInput
from app.embedding.service import EmbeddingService
from app.embedding.spaces.registry import EmbeddingSpaceRegistry


class EmbeddingSpaceService:

    def __init__(
        self,
        embedding_service: EmbeddingService,
        space_registry: EmbeddingSpaceRegistry,
    ):
        self._embedding_service = embedding_service
        self._space_registry = space_registry

    async def embed(
        self,
        space_id: str,
        request: EmbeddingInput,
    ) -> list[list[float]]:

        # 1. 检查 Space 是否存在
        if not self._space_registry.has(space_id):
            raise ValueError(
                f"Embedding Space does not exist: {space_id}"
            )

        space = self._space_registry.get(space_id)

        # 2. 获取当前 Embedding 模型元数据
        model_info = self._embedding_service.model_info

        # 3. 校验 Space 与当前 EmbeddingService 配置是否一致
        if space.provider_name != model_info.provider_name:
            raise RuntimeError(
                f"Embedding Provider mismatch for Space: {space_id}"
            )

        if space.model_name != model_info.model_name:
            raise RuntimeError(
                f"Embedding Model mismatch for Space: {space_id}"
            )

        if space.model_version != model_info.model_version:
            raise RuntimeError(
                f"Embedding Model Version mismatch for Space: {space_id}"
            )

        if space.dimension != model_info.dimension:
            raise RuntimeError(
                f"Embedding Dimension mismatch for Space: {space_id}"
            )

        if space.fingerprint != model_info.fingerprint:
            raise RuntimeError(
                f"Embedding Fingerprint mismatch for Space: {space_id}"
            )

        # 4. 调用现有 EmbeddingService
        return await self._embedding_service.embed(request)

    @property
    def model_info(self):
        """当前 Embedding 模型元数据。"""
        return self._embedding_service.model_info

    def get_space(self, space_id: str):
        """获取指定 Embedding Space 配置。"""
        if not self._space_registry.has(space_id):
            raise ValueError(
                f"Embedding Space does not exist: {space_id}"
            )

        return self._space_registry.get(space_id)