
from pydantic import ValidationError

from app.embedding.policy import DataClassification
from app.embedding.schemas import EmbeddingInput


def main():
    print("\n========== EMBEDDING SCHEMAS TEST ==========")

    # 1. 默认参数验证
    request = EmbeddingInput(
        texts=["重庆是一座美丽的山城"],
    )

    assert request.classification == DataClassification.INTERNAL
    assert request.allow_external is False

    print("[PASS] 默认分类为 INTERNAL")
    print("[PASS] 默认禁止外发")

    # 2. 多文本输入
    request = EmbeddingInput(
        texts=[
            "重庆是一座美丽的山城",
            "Agent Gateway 是一个智能体网关",
        ],
    )

    assert len(request.texts) == 2

    print("[PASS] 多文本输入")

    # 3. 公开数据，明确授权
    request = EmbeddingInput(
        texts=["公开产品介绍"],
        classification=DataClassification.PUBLIC,
        allow_external=True,
    )

    assert request.classification == DataClassification.PUBLIC
    assert request.allow_external is True

    print("[PASS] 公开数据及外发授权")

    # 4. 机密数据
    request = EmbeddingInput(
        texts=["企业机密经营数据"],
        classification=DataClassification.CONFIDENTIAL,
        allow_external=True,
    )

    assert request.classification == DataClassification.CONFIDENTIAL
    assert request.allow_external is True

    print("[PASS] 机密数据分类输入")

    # 5. 空文本列表
    try:
        EmbeddingInput(texts=[])
        raise AssertionError("空文本列表应该验证失败")
    except ValidationError:
        print("[PASS] 拒绝空文本列表")

    # 6. 空字符串
    try:
        EmbeddingInput(texts=[""])
        raise AssertionError("空字符串应该验证失败")
    except ValidationError:
        print("[PASS] 拒绝空字符串")

    # 7. 纯空白文本
    try:
        EmbeddingInput(texts=["   "])
        raise AssertionError("纯空白文本应该验证失败")
    except ValidationError:
        print("[PASS] 拒绝纯空白文本")

    # 8. 未知字段
    try:
        EmbeddingInput(
            texts=["测试"],
            unknown_field="invalid",
        )
        raise AssertionError("未知字段应该验证失败")
    except ValidationError:
        print("[PASS] 拒绝未知字段")

    print("============================================")
    print("Result: ALL PASSED")


if __name__ == "__main__":
    main()
