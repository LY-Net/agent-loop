
"""
Embedding Space Repository 真实数据库测试

测试内容：
1. 创建 Space
2. 查询 Space
3. 查询全部 Space
4. 更新 Space
5. 检查非法字段拦截
6. 删除 Space

使用真实 PostgreSQL，测试数据使用唯一 ID。
"""

import asyncio
import uuid

from app.db.session import AsyncSessionLocal
from app.embedding.spaces.repository import EmbeddingSpaceRepository


async def main() -> None:
    test_space_id = f"repository_test_{uuid.uuid4().hex[:12]}"

    print("========== EMBEDDING SPACE REPOSITORY TEST ==========")

    async with AsyncSessionLocal() as session:
        async with session.begin():
            repository = EmbeddingSpaceRepository(session)

            # 1. Create
            record = await repository.create(
                space_id=test_space_id,
                business_name="Repository Test",
                description="Repository integration test",
                provider_name="local_bge",
                model_name="BAAI/bge-m3",
                model_version="v1",
                dimension=1024,
                fingerprint="a" * 64,
                index_name="test_embedding_index",
            )

            assert record.space_id == test_space_id
            assert record.id is not None
            print("[PASS] Space 创建成功")

            # 2. Get by space_id
            found = await repository.get_by_space_id(test_space_id)

            assert found is not None
            assert found.model_name == "BAAI/bge-m3"
            print("[PASS] 按 space_id 查询成功")

            # 3. Get by record ID
            found_by_id = await repository.get_by_id(record.id)

            assert found_by_id is not None
            assert found_by_id.space_id == test_space_id
            print("[PASS] 按数据库主键查询成功")

            # 4. Exists
            assert await repository.exists(test_space_id)
            assert not await repository.exists("not_existing_space")
            print("[PASS] Space 存在性检查成功")

            # 5. List
            records = await repository.list_all()

            assert any(
                item.space_id == test_space_id
                for item in records
            )
            print("[PASS] Space 列表查询成功")

            # 6. Update
            updated = await repository.update(
                test_space_id,
                {
                    "business_name": "Repository Test Updated",
                    "description": "Updated description",
                },
            )

            assert updated is not None
            assert updated.business_name == "Repository Test Updated"
            assert updated.description == "Updated description"
            print("[PASS] Space 更新成功")

            # 7. Invalid update field
            try:
                await repository.update(
                    test_space_id,
                    {"space_id": "should_not_be_allowed"},
                )
            except ValueError:
                print("[PASS] 非法更新字段拦截成功")
            else:
                raise AssertionError("非法更新字段未被拦截")

            # 8. Delete
            deleted = await repository.delete(test_space_id)

            assert deleted
            assert not await repository.exists(test_space_id)
            print("[PASS] Space 删除成功")

    print("=====================================================")
    print("Result: ALL PASSED")


if __name__ == "__main__":
    asyncio.run(main())
