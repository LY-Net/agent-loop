
import asyncio
import uuid

from app.db.session import AsyncSessionLocal
from app.repositories.agent_session_repository import AgentSessionRepository


async def main():
    session_id = f"test_{uuid.uuid4().hex[:12]}"
    user_id = "repository_test_user"

    async with AsyncSessionLocal() as db:
        repository = AgentSessionRepository(db)

        # 1. 创建会话
        agent_session = await repository.create_session(
            user_id=user_id,
            session_id=session_id,
            context_data={
                "original_message": "查询上个月的销售数据",
                "intent": "analyze_last_month_sales_data",
            },
        )

        print("创建会话：", agent_session.session_id)
        print("初始状态：", agent_session.status)

        # 2. 查询会话
        found = await repository.get_by_session_id(
            session_id=session_id,
            user_id=user_id,
        )

        print("查询成功：", found is not None)

        # 3. 保存澄清信息
        updated = await repository.save_clarification(
            session_id=session_id,
            user_id=user_id,
            context_data={
                "original_message": "查询上个月的销售数据",
                "intent": "analyze_last_month_sales_data",
                "known_parameters": {},
            },
            pending_clarification={
                "question": "请问你要查询哪个区域的销售数据？",
                "missing_parameters": ["region"],
            },
        )

        print("澄清状态：", updated.status)
        print("澄清次数：", updated.clarification_count)

        # 4. 完成会话
        completed = await repository.complete_session(
            user_id=user_id,
            session_id=session_id
        )

        print("完成状态：", completed.status)
        print("待澄清内容：", completed.pending_clarification)


if __name__ == "__main__":
    asyncio.run(main())
