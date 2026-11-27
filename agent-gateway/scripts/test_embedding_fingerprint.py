
from app.embedding.fingerprint import build_model_fingerprint


def build_fingerprint(**kwargs):
    return build_model_fingerprint(
        model_name="BAAI/bge-m3",
        model_version="v1",
        dimension=1024,
        normalize_embeddings=True,
        runtime_config={"max_length": 512},
        **kwargs,
    )


def main():
    print("\n========== EMBEDDING FINGERPRINT TEST ==========")

    # 1. 相同配置生成相同指纹
    fingerprint_a = build_model_fingerprint(
        model_name="BAAI/bge-m3",
        model_version="v1",
        dimension=1024,
        normalize_embeddings=True,
        runtime_config={"max_length": 512},
    )

    fingerprint_b = build_model_fingerprint(
        model_name="BAAI/bge-m3",
        model_version="v1",
        dimension=1024,
        normalize_embeddings=True,
        runtime_config={"max_length": 512},
    )

    assert fingerprint_a == fingerprint_b
    print("[PASS] 相同配置生成相同指纹")

    # 2. 指纹格式
    assert len(fingerprint_a) == 64
    assert all(c in "0123456789abcdef" for c in fingerprint_a)
    print("[PASS] SHA256 指纹格式正确")

    # 3. 不同模型名称
    assert build_model_fingerprint(
        model_name="another-model",
        model_version="v1",
        dimension=1024,
        normalize_embeddings=True,
        runtime_config={"max_length": 512},
    ) != fingerprint_a
    print("[PASS] 模型名称变化会改变指纹")

    # 4. 不同模型版本
    assert build_model_fingerprint(
        model_name="BAAI/bge-m3",
        model_version="v2",
        dimension=1024,
        normalize_embeddings=True,
        runtime_config={"max_length": 512},
    ) != fingerprint_a
    print("[PASS] 模型版本变化会改变指纹")

    # 5. 不同向量维度
    assert build_model_fingerprint(
        model_name="BAAI/bge-m3",
        model_version="v1",
        dimension=768,
        normalize_embeddings=True,
        runtime_config={"max_length": 512},
    ) != fingerprint_a
    print("[PASS] 向量维度变化会改变指纹")

    # 6. 不同归一化配置
    assert build_model_fingerprint(
        model_name="BAAI/bge-m3",
        model_version="v1",
        dimension=1024,
        normalize_embeddings=False,
        runtime_config={"max_length": 512},
    ) != fingerprint_a
    print("[PASS] 归一化配置变化会改变指纹")

    # 7. 不同运行配置
    assert build_model_fingerprint(
        model_name="BAAI/bge-m3",
        model_version="v1",
        dimension=1024,
        normalize_embeddings=True,
        runtime_config={"max_length": 256},
    ) != fingerprint_a
    print("[PASS] 运行配置变化会改变指纹")

    # 8. JSON 字典键顺序不影响指纹
    fingerprint_c = build_model_fingerprint(
        model_name="BAAI/bge-m3",
        model_version="v1",
        dimension=1024,
        normalize_embeddings=True,
        runtime_config={
            "pooling": "cls",
            "max_length": 512,
        },
    )

    fingerprint_d = build_model_fingerprint(
        model_name="BAAI/bge-m3",
        model_version="v1",
        dimension=1024,
        normalize_embeddings=True,
        runtime_config={
            "max_length": 512,
            "pooling": "cls",
        },
    )

    assert fingerprint_c == fingerprint_d
    print("[PASS] 配置字典键顺序不影响指纹")

    # 9. 非法维度
    try:
        build_model_fingerprint(
            model_name="BAAI/bge-m3",
            model_version="v1",
            dimension=0,
            normalize_embeddings=True,
        )
        raise AssertionError("非法维度应该失败")
    except ValueError:
        print("[PASS] 非法向量维度被拒绝")

    # 10. 空模型名称
    try:
        build_model_fingerprint(
            model_name=" ",
            model_version="v1",
            dimension=1024,
            normalize_embeddings=True,
        )
        raise AssertionError("空模型名称应该失败")
    except ValueError:
        print("[PASS] 空模型名称被拒绝")

    print("================================================")
    print("Result: ALL PASSED")


if __name__ == "__main__":
    main()
