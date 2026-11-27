
import math
from collections.abc import Sequence

import httpx

from app.core.config import settings
from app.embedding.providers.base import EmbeddingProvider



from collections.abc import Mapping, Sequence
from typing import Any

import httpx

from app.core.config import settings
from app.embedding.providers.base import EmbeddingProvider


class LocalBGEProvider(EmbeddingProvider):

    def __init__(self):
        self._client = httpx.AsyncClient(
            base_url=settings.EMBEDDING_BASE_URL,
            timeout=settings.EMBEDDING_TIMEOUT,
            trust_env=False,
        )

    @property
    def provider_name(self) -> str:
        return "local_bge"

    @property
    def model_name(self) -> str:
        return settings.EMBEDDING_MODEL_NAME

    @property
    def model_version(self) -> str:
        # 当前为逻辑版本标识，后续由 Embedding Server 提供真实权重版本
        return "v1"

    @property
    def dimension(self) -> int:
        return settings.EMBEDDING_DIMENSION

    @property
    def is_remote(self) -> bool:
        return False

    @property
    def normalize_embeddings(self) -> bool:
        return True

    @property
    def runtime_config(self) -> Mapping[str, Any]:
        return {
            "max_length": 512,
            "device": "cpu",
            "use_fp16": False,
            "pooling": "cls",
        }

    async def embed(
        self,
        texts: Sequence[str],
    ) -> list[list[float]]:

        if not texts:
            raise ValueError("texts cannot be empty")

        if any(
            not isinstance(text, str) or not text.strip()
            for text in texts
        ):
            raise ValueError("texts cannot contain empty values")

        try:
            response = await self._client.post(
                "/embed",
                json={"texts": list(texts)},
            )
            response.raise_for_status()

        except httpx.HTTPError as exc:
            raise RuntimeError(
                "Embedding Server request failed"
            ) from exc

        try:
            data = response.json()
        except ValueError as exc:
            raise RuntimeError(
                "Embedding Server returned invalid JSON"
            ) from exc

        embeddings = data.get("embeddings")

        if not isinstance(embeddings, list):
            raise RuntimeError(
                "Embedding Server returned invalid embeddings"
            )

        if len(embeddings) != len(texts):
            raise RuntimeError(
                "Embedding result count mismatch"
            )

        return embeddings

    async def close(self) -> None:
        await self._client.aclose()
