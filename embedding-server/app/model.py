
import threading
from typing import Any

import torch
from FlagEmbedding import BGEM3FlagModel

from app.config import settings


class BGEModel:

    def __init__(self):
        self._model = None
        self._lock = threading.Lock()
        self._inference_lock = threading.Lock()
        self._loaded = False

    def load(self) -> None:
        """加载本地 BGE-M3 模型，只加载一次。"""

        if self._loaded:
            return

        with self._lock:
            if self._loaded:
                return

            torch.set_num_threads(settings.EMBEDDING_CPU_THREADS)

            model = BGEM3FlagModel(
                settings.EMBEDDING_MODEL_PATH,
                use_fp16=settings.EMBEDDING_USE_FP16,
                devices=settings.EMBEDDING_DEVICE,
            )

            self._model = model
            self._loaded = True

    def unload(self) -> None:
        """释放模型资源。"""

        with self._lock:
            with self._inference_lock:
                self._model = None
                self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def encode(
        self,
        texts: list[str],
        batch_size: int | None = None,
        max_length: int | None = None,
    ) -> list[list[float]]:
        """批量生成 Dense Embedding。"""

        if not texts:
            return []

        self.load()

        with self._inference_lock:
            result: dict[str, Any] = self._model.encode(
                texts,
                batch_size=batch_size or settings.EMBEDDING_BATCH_SIZE,
                max_length=max_length or settings.EMBEDDING_MAX_LENGTH,
            )

            vectors = result["dense_vecs"]

            if vectors.shape[1] != settings.EMBEDDING_DIMENSION:
                raise RuntimeError(
                    f"Unexpected embedding dimension: {vectors.shape[1]}"
                )

            if not torch.isfinite(torch.as_tensor(vectors)).all().item():
                raise RuntimeError("Embedding contains non-finite values")

            return vectors.tolist()


bge_model = BGEModel()
