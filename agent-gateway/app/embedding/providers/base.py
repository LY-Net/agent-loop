
from abc import ABC, abstractmethod
from collections.abc import Sequence



from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any


class EmbeddingProvider(ABC):

    @property
    @abstractmethod
    def provider_name(self) -> str:
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        ...

    @property
    @abstractmethod
    def model_version(self) -> str:
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        ...

    @property
    @abstractmethod
    def is_remote(self) -> bool:
        ...

    @property
    @abstractmethod
    def normalize_embeddings(self) -> bool:
        ...

    @property
    @abstractmethod
    def runtime_config(self) -> Mapping[str, Any]:
        ...

    @property
    def fingerprint(self) -> str:
        from app.embedding.fingerprint import build_model_fingerprint

        return build_model_fingerprint(
            model_name=self.model_name,
            model_version=self.model_version,
            dimension=self.dimension,
            normalize_embeddings=self.normalize_embeddings,
            runtime_config=self.runtime_config,
        )

    @abstractmethod
    async def embed(
        self,
        texts: Sequence[str],
    ) -> list[list[float]]:
        ...

    @abstractmethod
    async def close(self) -> None:
        ...
