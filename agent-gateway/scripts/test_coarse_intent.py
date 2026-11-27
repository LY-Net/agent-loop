
import asyncio
import json

from app.intent.coarse import coarse_intent_recognizer
from app.llm.deepseek import deepseek_client


TEST_CASES = [
    "你好，今天怎么样？",

    "什么是 Python 的异步编程？",

    "帮我分析一下最近三个月公司的销售数据，找出销售额下降的原因。",

    "帮我查询一下数据库中最近一个月注册的用户。",

    "帮我写一份 Agent Gateway 项目的技术方案。",

    "帮我设计一个包含意图识别、RAG 和工具调用的 Agent 执行流程。",

    "帮我检查一下服务器的运行状态。",
]


async def main():
    try:
        for index, user_input in enumerate(TEST_CASES, start=1):

            print("\n" + "=" * 70)
            print(f"测试用例 {index}")
            print("用户输入：", user_input)

            result = await coarse_intent_recognizer.recognize(
                user_input
            )

            print("\n识别结果：")

            print(
                json.dumps(
                    result.model_dump(mode="json"),
                    ensure_ascii=False,
                    indent=4,
                )
            )

    finally:
        await deepseek_client.close()


if __name__ == "__main__":
    asyncio.run(main())
