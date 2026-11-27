
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine
)

from app.core.config import settings


engine = create_async_engine(
    settings.database_url,
    echo=False,
    pool_size=10,  # 常驻数据库连接数量
    max_overflow=20, # 	高并发时允许额外创建的连接
    pool_pre_ping=True # 使用连接前检查连接是否有效
)


AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False
)


async def get_db():
    async with AsyncSessionLocal() as session:
        yield session


async def close_db():
    await engine.dispose()
