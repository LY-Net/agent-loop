import asyncio
from uuid import uuid4

from app.intent.engine import intent_engine
from app.llm.deepseek import deepseek_client
from app.observability.context import (
    set_execution_context,
    reset_execution_context,
)


async def main():
    tokens = set_execution_context(
        trace_id=str(uuid4()),
        run_id=str(uuid4()),
    )

    try:
        result = await intent_engine.recognize(
            "帮我分析一下最近三个月的销售数据"
        )

        print(result.model_dump_json(indent=2))

    finally:
        reset_execution_context(tokens)
        await deepseek_client.close()


if __name__ == "__main__":
    asyncio.run(main())