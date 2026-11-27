
from datetime import datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.observability import AgentSession


class AgentSessionRepository:
    """
    AgentSession 数据库仓储层。

    负责会话的创建、查询、上下文更新和状态管理。
    """

    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_by_session_id(
        self,
        session_id: str,
        user_id: str,
    ) -> AgentSession | None:
        """
        根据 session_id 和 user_id 查询会话。
        """
        stmt = select(AgentSession).where(
            AgentSession.session_id == session_id,
            AgentSession.user_id == user_id,
        )

        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def create_session(
        self,
        user_id: str,
        session_id: str,
        context_data: dict[str, Any] | None = None,
        expires_at: datetime | None = None,
    ) -> AgentSession:
        """
        创建新的 AgentSession。
        """
        agent_session = AgentSession(
            user_id=user_id,
            session_id=session_id,
            status="active",
            context_data=context_data or {},
            pending_clarification=None,
            clarification_count=0,
            expires_at=expires_at,
        )

        self.session.add(agent_session)

        await self.session.flush()
        await self.session.commit()
        await self.session.refresh(agent_session)

        return agent_session

    async def update_context(
        self,
        session_id: str,
        user_id: str,
        context_data: dict[str, Any],
    ) -> AgentSession | None:
        """
        更新会话上下文。
        """
        agent_session = await self.get_by_session_id(
            session_id=session_id,
            user_id=user_id,
        )

        if agent_session is None:
            return None

        agent_session.context_data = context_data

        await self.session.commit()
        await self.session.refresh(agent_session)

        return agent_session

    async def save_clarification(
        self,
        session_id: str,
        user_id: str,
        context_data: dict[str, Any],
        pending_clarification: dict[str, Any],
    ) -> AgentSession | None:
        """
        保存待澄清任务。

        用户需要补充信息时，将会话状态设置为
        waiting_clarification。
        """
        agent_session = await self.get_by_session_id(
            session_id=session_id,
            user_id=user_id,
        )

        if agent_session is None:
            return None

        agent_session.context_data = context_data
        agent_session.pending_clarification = pending_clarification
        agent_session.status = "waiting_clarification"
        agent_session.clarification_count += 1

        await self.session.commit()
        await self.session.refresh(agent_session)

        return agent_session

    async def complete_session(
        self,
        session_id: str,
        user_id: str,
        context_data: dict[str, Any] | None = None,
    ) -> AgentSession | None:
        """
        标记会话完成，并清理待澄清任务。
        """
        agent_session = await self.get_by_session_id(
            session_id=session_id,
            user_id=user_id,
        )

        if agent_session is None:
            return None

        if context_data is not None:
            agent_session.context_data = context_data

        agent_session.status = "completed"
        agent_session.pending_clarification = None

        await self.session.commit()
        await self.session.refresh(agent_session)

        return agent_session
