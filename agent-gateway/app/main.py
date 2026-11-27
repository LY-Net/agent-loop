
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.core.config import settings
from app.core.logger import app_logger

from app.middleware.trace import TraceMiddleware

from app.api.v1.agent import router as agent_router
from app.api.v1.traces import router as trace_router
from app.api.tool_servers import router as tool_server_router
from app.api.v1.rag import router as rag_router

from app.db.session import AsyncSessionLocal, close_db

from app.embedding.providers.local_bge import LocalBGEProvider
from app.embedding.registry import EmbeddingProviderRegistry
from app.embedding.service import EmbeddingService

from app.embedding.spaces.registry import EmbeddingSpaceRegistry
from app.embedding.spaces.service import EmbeddingSpaceService
from app.embedding.spaces.defaults import register_default_spaces
from app.embedding.spaces.persistence import (
    EmbeddingSpacePersistenceService,
)
from app.api.tool_discovery import router as tool_discovery_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    app_logger.info("Agent Gateway starting...")

    # 初始化 Provider Registry
    embedding_registry = EmbeddingProviderRegistry()

    try:
        # ==========================================
        # 1. 初始化 Embedding Provider
        # ==========================================
        if settings.EMBEDDING_PROVIDER == "local_bge":
            local_bge_provider = LocalBGEProvider()
            embedding_registry.register(local_bge_provider)
        else:
            raise RuntimeError(
                f"Unsupported Embedding Provider: "
                f"{settings.EMBEDDING_PROVIDER}"
            )

        app_logger.info(
            "Embedding Provider initialized: {}",
            settings.EMBEDDING_PROVIDER,
        )

        # ==========================================
        # 2. 初始化 EmbeddingService
        # ==========================================
        embedding_service = EmbeddingService(
            registry=embedding_registry,
        )

        app_logger.info("Embedding Service initialized")

        # ==========================================
        # 3. 初始化 Space Registry
        # ==========================================
        embedding_space_registry = EmbeddingSpaceRegistry(
            provider_registry=embedding_registry,
        )

        # ==========================================
        # 4. 构建默认业务 Space
        # ==========================================
        register_default_spaces(
            space_registry=embedding_space_registry,
            provider_registry=embedding_registry,
            provider_name=settings.EMBEDDING_PROVIDER,
        )

        # 获取默认 Space 配置，作为首次初始化的数据来源
        default_spaces = tuple(
            embedding_space_registry.get(space_id)
            for space_id in embedding_space_registry.space_ids
        )

        # ==========================================
        # 5. 初始化 Space 持久化服务
        # ==========================================
        embedding_space_persistence = (
            EmbeddingSpacePersistenceService(
                session_factory=AsyncSessionLocal,
                registry=embedding_space_registry,
                default_spaces=default_spaces,
            )
        )

        # 从数据库初始化默认 Space，并恢复全部 Space
        loaded_space_ids = (
            await embedding_space_persistence.initialize()
        )

        app_logger.info(
            "Embedding Spaces restored from database: {}",
            loaded_space_ids,
        )

        # ==========================================
        # 6. 初始化 EmbeddingSpaceService
        # ==========================================
        embedding_space_service = EmbeddingSpaceService(
            embedding_service=embedding_service,
            space_registry=embedding_space_registry,
        )

        app_logger.info("Embedding Space Service initialized")

        # ==========================================
        # 7. 保存到 FastAPI app.state
        # ==========================================
        app.state.embedding_registry = embedding_registry
        app.state.embedding_service = embedding_service
        app.state.embedding_space_registry = embedding_space_registry
        app.state.embedding_space_service = embedding_space_service
        app.state.embedding_space_persistence = (
            embedding_space_persistence
        )

        app_logger.info(
            "Agent Gateway startup completed"
        )

        yield

    finally:
        # ==========================================
        # 8. 关闭资源
        # ==========================================
        try:
            await embedding_registry.close_all()
        finally:
            try:
                await close_db()
            finally:
                app.state.embedding_registry = None
                app.state.embedding_service = None
                app.state.embedding_space_registry = None
                app.state.embedding_space_service = None
                app.state.embedding_space_persistence = None

                app_logger.info(
                    "Agent Gateway shutting down..."
                )


app = FastAPI(
    title=settings.APP_NAME,
    version="0.1.0",
    description="AI Agent Gateway",
    lifespan=lifespan,
)

app.add_middleware(TraceMiddleware)

app.include_router(agent_router)
app.include_router(trace_router)
app.include_router(tool_server_router)
app.include_router(rag_router)
app.include_router(tool_discovery_router)

@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "service": settings.APP_NAME,
    }
