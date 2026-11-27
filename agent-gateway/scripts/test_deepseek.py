
import asyncio

from app.llm.deepseek import deepseek_client


async def main():
    try:
        response = await deepseek_client.chat(
            messages=[
                {
                    "role": "system",
                    "content": "你是一个专业的AI助手。",
                },
                {
                    "role": "user",
                    "content": "请用一句话解释什么是AI Agent。",
                },
            ],
            temperature=0.2,
            max_tokens=200,
        )

        print("模型：", response.model)
        print("回答：", response.content)
        print("输入 Token：", response.prompt_tokens)
        print("输出 Token：", response.completion_tokens)
        print("总 Token：", response.total_tokens)
        print("结束原因：", response.finish_reason)

    finally:
        await deepseek_client.close()


if __name__ == "__main__":
    asyncio.run(main())
