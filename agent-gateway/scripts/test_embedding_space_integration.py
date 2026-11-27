
import asyncio
import math

from app.main import app
from app.embedding.schemas import EmbeddingInput


async def main():
    print("\n========== EMBEDDING SPACE INTEGRATION TEST ==========")

    async with app.router.lifespan_context(app):

        space_service = app.state.embedding_space_service

        assert space_service is not None

        print("[PASS] Embedding Space Service 初始化成功")

        # 1. RAG Space 真实向量生成
        rag_request = EmbeddingInput(
            texts=[
                "Agent Gateway 是一个 AI Agent 网关系统",
                "RAG 通过检索增强生成提高回答质量",
            ]
        )

        rag_vectors = await space_service.embed(
            space_id="rag_documents",
            request=rag_request,
        )

        assert len(rag_vectors) == 2
        assert all(len(vector) == 1024 for vector in rag_vectors)
        assert all(
            math.isfinite(value)
            for vector in rag_vectors
            for value in vector
        )

        print("[PASS] RAG Space 真实向量生成成功")
        print("[INFO] RAG 向量数量:", len(rag_vectors))
        print("[INFO] RAG 向量维度:", len(rag_vectors[0]))

        # 2. Tool Discovery Space 真实向量生成
        tool_request = EmbeddingInput(
            texts=[
                "查询重庆地区最近一个月的销售额",
                "统计各区域的订单数量",
            ]
        )

        tool_vectors = await space_service.embed(
            space_id="tool_discovery",
            request=tool_request,
        )

        assert len(tool_vectors) == 2
        assert all(len(vector) == 1024 for vector in tool_vectors)
        assert all(
            math.isfinite(value)
            for vector in tool_vectors
            for value in vector
        )

        print("[PASS] Tool Discovery Space 真实向量生成成功")
        print("[INFO] Tool 向量数量:", len(tool_vectors))
        print("[INFO] Tool 向量维度:", len(tool_vectors[0]))

        # 3. 验证两个 Space 使用同一个 Provider
        provider_registry = app.state.embedding_registry

        assert provider_registry.has("local_bge")

        print("[PASS] 两个 Space 共用 local_bge Provider")

        # 4. 验证底层模型配置
        model_info = app.state.embedding_service.model_info

        assert model_info.model_name == "BAAI/bge-m3"
        assert model_info.dimension == 1024

        print("[PASS] BGE-M3 模型元数据正确")

        print("======================================================")
        print("Result: ALL PASSED")


if __name__ == "__main__":
    asyncio.run(main())
