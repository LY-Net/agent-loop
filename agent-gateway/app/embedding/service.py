
import math

from app.core.config import settings
from app.embedding.policy import (
    EmbeddingPolicy,
    embedding_policy,
)
from app.embedding.registry import EmbeddingProviderRegistry
from app.embedding.schemas import (
    EmbeddingInput,
    EmbeddingModelInfo,
)


class EmbeddingService:
    """
    Embedding 统一服务入口。
    """

    def __init__(
        self,
        registry: EmbeddingProviderRegistry,
        policy: EmbeddingPolicy | None = None,
        provider_name: str | None = None,
    ):
        self._registry = registry
        self._policy = policy or embedding_policy

        self._provider_name = (
            provider_name or settings.EMBEDDING_PROVIDER
        )


    @property
    def model_info(self) -> EmbeddingModelInfo:
        provider = self._registry.get(self._provider_name)

        return EmbeddingModelInfo(
            provider_name=provider.provider_name,
            model_name=provider.model_name,
            model_version=provider.model_version,
            dimension=provider.dimension,
            is_remote=provider.is_remote,
            normalize_embeddings=provider.normalize_embeddings,
            fingerprint=provider.fingerprint,
        )

    async def embed(
        self,
        request: EmbeddingInput,
    ) -> list[list[float]]:

        # 1. 输入数量校验
        if len(request.texts) > settings.EMBEDDING_MAX_TEXTS:
            raise ValueError(
                f"Too many texts: maximum "
                f"{settings.EMBEDDING_MAX_TEXTS}"
            )

        # 2. 单条文本长度校验
        for index, text in enumerate(request.texts):
            if len(text) > settings.EMBEDDING_MAX_TEXT_LENGTH:
                raise ValueError(
                    f"texts[{index}] exceeds maximum length "
                    f"{settings.EMBEDDING_MAX_TEXT_LENGTH}"
                )

        # 3. 数据安全策略
        decision = self._policy.evaluate(
            classification=request.classification,
            allow_external=request.allow_external,
        )

        # 4. 获取 Provider
        provider = self._registry.get(self._provider_name)

        # 5. 远程调用权限检查
        if provider.is_remote and not decision.allow_remote:
            raise PermissionError(
                "Embedding data is not allowed to be sent "
                "to a remote Provider: "
                f"{decision.reason}"
            )

        # 6. 向量维度检查
        if provider.dimension != settings.EMBEDDING_DIMENSION:
            raise RuntimeError(
                f"Provider dimension mismatch: "
                f"configured={settings.EMBEDDING_DIMENSION}, "
                f"provider={provider.dimension}"
            )

        # 7. 批次配置检查
        batch_size = settings.EMBEDDING_BATCH_SIZE

        if batch_size <= 0:
            raise RuntimeError(
                "EMBEDDING_BATCH_SIZE must be greater than zero"
            )

        # 8. 批量生成向量
        all_embeddings: list[list[float]] = []

        for start in range(0, len(request.texts), batch_size):
            batch = request.texts[start:start + batch_size]

            embeddings = await provider.embed(batch)

            validated_embeddings = self._validate_embeddings(
                embeddings=embeddings,
                expected_count=len(batch),
                expected_dimension=provider.dimension,
            )

            all_embeddings.extend(validated_embeddings)

        if len(all_embeddings) != len(request.texts):
            raise RuntimeError(
                "Final embedding count does not match input count"
            )

        return all_embeddings

    @staticmethod
    def _validate_embeddings(
        embeddings: list[list[float]],
        expected_count: int,
        expected_dimension: int,
    ) -> list[list[float]]:

        if not isinstance(embeddings, list):
            raise RuntimeError(
                "Provider embeddings must be a list"
            )

        if len(embeddings) != expected_count:
            raise RuntimeError(
                f"Embedding count mismatch: "
                f"expected {expected_count}, "
                f"got {len(embeddings)}"
            )

        validated = []

        for index, vector in enumerate(embeddings):

            if not isinstance(vector, (list, tuple)):
                raise RuntimeError(
                    f"Invalid vector structure at index {index}"
                )

            if len(vector) != expected_dimension:
                raise RuntimeError(
                    f"Embedding dimension mismatch at index {index}: "
                    f"expected {expected_dimension}, "
                    f"got {len(vector)}"
                )

            validated_vector = []

            for value in vector:

                if isinstance(value, bool) or not isinstance(
                    value, (int, float)
                ):
                    raise RuntimeError(
                        f"Invalid vector value at index {index}"
                    )

                if not math.isfinite(value):
                    raise RuntimeError(
                        f"Non-finite vector value at index {index}"
                    )

                validated_vector.append(float(value))

            validated.append(validated_vector)

        return validated
