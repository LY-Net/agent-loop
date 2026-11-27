
import asyncio
import json

from app.intent.coarse import coarse_intent_recognizer
from app.intent.fine import fine_intent_recognizer
from app.llm.deepseek import deepseek_client


TEST_CASES = [
    "什么是 Python 的异步编程？",

    "帮我分析一下最近三个月公司的销售数据，找出销售额下降的原因。",

    "帮我查询一下最近一个月注册的用户，并统计各个地区的注册数量。",

    "帮我写一份 Agent Gateway 项目的技术方案。",

    "帮我设计一个包含意图识别、RAG 和工具调用的 Agent 执行流程。",

    "帮我查询一下数据。",
]


async def main():
    try:
        for index, user_input in enumerate(TEST_CASES, start=1):
            print("\n" + "=" * 80)
            print(f"测试用例 {index}")
            print("用户输入：", user_input)

            # 第一步：粗粒度识别
            coarse_result = await coarse_intent_recognizer.recognize(
                user_input
            )

            print("\n粗粒度识别：")
            print(
                json.dumps(
                    coarse_result.model_dump(mode="json"),
                    ensure_ascii=False,
                    indent=4,
                )
            )

            # 第二步：精细意图识别
            fine_result = await fine_intent_recognizer.recognize(
                user_input=user_input,
                coarse_result=coarse_result,
            )

            print("\n精细意图识别：")
            print(
                json.dumps(
                    fine_result.model_dump(mode="json"),
                    ensure_ascii=False,
                    indent=4,
                )
            )

    finally:
        await deepseek_client.close()


if __name__ == "__main__":
    asyncio.run(main())
