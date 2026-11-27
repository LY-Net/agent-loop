
import asyncio

from app.embedding.providers.local_bge import LocalBGEProvider


async def main():
    provider = LocalBGEProvider(
        base_url="http://127.0.0.1:8090",
        model_name="BAAI/bge-m3",
        dimension=1024,
    )

    try:
        texts = [
            "重庆是一座美丽的山城",
            "Agent Gateway 是一个智能体网关",
        ]

        embeddings = await provider.embed(texts)

        print("\n========== LOCAL BGE TEST ==========")
        print("Provider:", provider.provider_name)
        print("Model:", provider.model_name)
        print("Dimension:", provider.dimension)
        print("Embedding count:", len(embeddings))
        print("First vector dimension:", len(embeddings[0]))
        print("First five values:", embeddings[0][:5])
        print("====================================")

        assert len(embeddings) == 2
        assert all(len(vector) == 1024 for vector in embeddings)

        print("Result: PASSED")

    finally:
        await provider.close()


if __name__ == "__main__":
    asyncio.run(main())
