
from app.db.base import Base
import app.models

from app.models.embedding_space import EmbeddingSpaceRecord


def main():
    print("\n========== EMBEDDING SPACE MODEL TEST ==========")

    # 1. 检查 ORM 模型导入
    print("[PASS] EmbeddingSpaceRecord 导入成功")

    # 2. 检查表是否注册到 SQLAlchemy Metadata
    table_name = "embedding_spaces"

    if table_name not in Base.metadata.tables:
        raise AssertionError(
            f"{table_name} 未注册到 Base.metadata"
        )

    table = Base.metadata.tables[table_name]

    print("[PASS] embedding_spaces 已注册")

    # 3. 检查字段
    expected_columns = {
        "id",
        "space_id",
        "business_name",
        "description",
        "provider_name",
        "model_name",
        "model_version",
        "dimension",
        "fingerprint",
        "index_name",
        "created_at",
        "updated_at",
    }

    actual_columns = set(table.columns.keys())

    missing_columns = expected_columns - actual_columns

    if missing_columns:
        raise AssertionError(
            f"缺少字段: {missing_columns}"
        )

    print("[PASS] 所有预期字段均已注册")

    # 4. 检查主键
    primary_keys = {
        column.name
        for column in table.primary_key.columns
    }

    if primary_keys != {"id"}:
        raise AssertionError(
            f"主键不符合预期: {primary_keys}"
        )

    print("[PASS] 主键配置正确")

    # 5. 检查约束和索引
    constraint_names = {
        constraint.name
        for constraint in table.constraints
    }

    if "ck_embedding_spaces_dimension_positive" not in constraint_names:
        raise AssertionError("向量维度约束缺失")

    index_names = {
        index.name
        for index in table.indexes
    }

    if "ix_embedding_spaces_provider_model" not in index_names:
        raise AssertionError("Provider/Model 联合索引缺失")

    print("[PASS] 约束和索引注册成功")

    print("================================================")
    print("Result: ALL PASSED")


if __name__ == "__main__":
    main()
