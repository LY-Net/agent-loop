
from app.embedding.policy import (
    DataClassification,
    EmbeddingPolicy,
)


def main():
    policy = EmbeddingPolicy()

    test_cases = [
        {
            "name": "公开数据，未授权",
            "classification": DataClassification.PUBLIC,
            "allow_external": False,
            "expected": False,
        },
        {
            "name": "公开数据，明确授权",
            "classification": DataClassification.PUBLIC,
            "allow_external": True,
            "expected": True,
        },
        {
            "name": "内部数据，未授权",
            "classification": DataClassification.INTERNAL,
            "allow_external": False,
            "expected": False,
        },
        {
            "name": "内部数据，明确授权",
            "classification": DataClassification.INTERNAL,
            "allow_external": True,
            "expected": True,
        },
        {
            "name": "机密数据，明确授权",
            "classification": DataClassification.CONFIDENTIAL,
            "allow_external": True,
            "expected": False,
        },
        {
            "name": "受限制数据，明确授权",
            "classification": DataClassification.RESTRICTED,
            "allow_external": True,
            "expected": False,
        },
    ]

    print("\n========== EMBEDDING POLICY TEST ==========")

    for case in test_cases:
        decision = policy.evaluate(
            classification=case["classification"],
            allow_external=case["allow_external"],
        )

        assert decision.allow_remote == case["expected"], (
            f"测试失败：{case['name']}，"
            f"预期 {case['expected']}，"
            f"实际 {decision.allow_remote}"
        )

        print(
            f"[PASS] {case['name']} -> "
            f"allow_remote={decision.allow_remote}"
        )

    # 验证默认分类和默认授权
    default_decision = policy.evaluate()

    assert default_decision.classification == DataClassification.INTERNAL
    assert default_decision.allow_remote is False

    print("[PASS] 默认策略 -> INTERNAL / 禁止外发")
    print("==========================================")
    print("Result: ALL PASSED")


if __name__ == "__main__":
    main()
