
from dataclasses import dataclass
from enum import Enum


class DataClassification(str, Enum):
    """Embedding 输入数据的安全分类。"""

    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"


@dataclass(frozen=True, slots=True)
class EmbeddingPolicyDecision:
    """Embedding 数据安全策略判断结果。"""

    classification: DataClassification
    allow_remote: bool
    reason: str


class EmbeddingPolicy:
    """
    Embedding 数据外发策略。

    安全原则：
    1. 默认不允许外发。
    2. 机密和受限制数据禁止外发。
    3. 公开和内部数据必须明确授权才能外发。
    4. 本地 Provider 不受外发授权限制。
    """

    def evaluate(
        self,
        classification: DataClassification = DataClassification.INTERNAL,
        *,
        allow_external: bool = False,
    ) -> EmbeddingPolicyDecision:

        classification = DataClassification(classification)

        # 敏感数据始终禁止外发
        if classification in {
            DataClassification.CONFIDENTIAL,
            DataClassification.RESTRICTED,
        }:
            return EmbeddingPolicyDecision(
                classification=classification,
                allow_remote=False,
                reason="Sensitive data is not allowed to be sent externally",
            )

        # 没有明确授权，默认禁止外发
        if not allow_external:
            return EmbeddingPolicyDecision(
                classification=classification,
                allow_remote=False,
                reason="External transfer has not been explicitly authorized",
            )

        # 公开或内部数据经过明确授权后，可以使用远程 Provider
        return EmbeddingPolicyDecision(
            classification=classification,
            allow_remote=True,
            reason="External transfer is explicitly authorized",
        )


embedding_policy = EmbeddingPolicy()
