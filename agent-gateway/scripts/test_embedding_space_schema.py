
from pydantic import ValidationError

from app.embedding.spaces.schemas import EmbeddingSpace


def main():
    print("\n========== EMBEDDING SPACE SCHEMA TEST ==========")

    space = EmbeddingSpace(
        space_id="rag_documents",
        business_name="RAG",
        description="RAG 文档语义检索",
        provider_name="local_bge",
        model_name="BAAI/bge-m3",
        model_version="v1",
        dimension=1024,
        index_name="rag_documents_v1",
    )

    assert space.space_id == "rag_documents"
    assert space.dimension == 1024

    print("[PASS] Embedding Space 创建成功")

    data = space.model_dump()

    assert data["model_name"] == "BAAI/bge-m3"
    assert data["index_name"] == "rag_documents_v1"

    print("[PASS] Space 元数据序列化成功")

    try:
        space.dimension = 768
        raise AssertionError("Space 元数据不应该允许修改")

    except ValidationError:
        print("[PASS] Space 元数据不可修改")

    try:
        EmbeddingSpace(
            space_id="invalid",
            business_name="Invalid",
            provider_name="local_bge",
            model_name="BAAI/bge-m3",
            model_version="v1",
            dimension=0,
            index_name="invalid_index",
        )

        raise AssertionError("非法维度应该被拒绝")

    except ValidationError:
        print("[PASS] 非法维度被拒绝")

    print("=================================================")
    print("Result: ALL PASSED")


if __name__ == "__main__":
    main()
