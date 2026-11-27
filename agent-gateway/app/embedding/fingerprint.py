
import hashlib
import json
from collections.abc import Mapping
from typing import Any


def build_model_fingerprint(
    *,
    model_name: str,
    model_version: str,
    dimension: int,
    normalize_embeddings: bool,
    runtime_config: Mapping[str, Any] | None = None,
) -> str:
    """
    根据模型及其关键配置生成稳定的 SHA256 指纹。

    相同配置生成相同指纹；
    任意关键配置发生变化，指纹也会发生变化。

    runtime_config 用于记录影响向量语义的运行配置，
    例如 max_length、pooling 等。
    """

    if not model_name.strip():
        raise ValueError("model_name cannot be empty")

    if not model_version.strip():
        raise ValueError("model_version cannot be empty")

    if dimension <= 0:
        raise ValueError("dimension must be greater than 0")

    if not isinstance(normalize_embeddings, bool):
        raise ValueError("normalize_embeddings must be bool")

    payload = {
        "model_name": model_name,
        "model_version": model_version,
        "dimension": dimension,
        "normalize_embeddings": normalize_embeddings,
        "runtime_config": dict(runtime_config or {}),
    }

    # 使用固定序列化规则，确保相同配置生成相同字符串
    serialized = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )

    return hashlib.sha256(
        serialized.encode("utf-8")
    ).hexdigest()
