
from app.services.clarification_manager import ClarificationManager


def main():
    manager = ClarificationManager()

    # 1. 构建初始上下文
    context = manager.build_initial_context(
        original_message="查询上个月的销售数据",
        intent_name="analyze_last_month_sales_data",
        known_parameters={},
        missing_parameters=["region"],
        required_capabilities=["sales_query"],
    )

    print("初始上下文：")
    print(context)

    # 2. 构建待澄清问题
    pending = manager.build_pending_clarification(
        question="请问你要查询哪个区域的销售数据？",
        missing_parameters=["region"],
    )

    print("\n待澄清问题：")
    print(pending)

    # 3. 接收用户补充
    result = manager.apply_user_response(
        context_data=context,
        pending_clarification=pending,
        user_response="查询重庆地区的数据",
    )

    print("\n合并后的上下文：")
    print(result["context_data"])

    print("\n重新识别使用的完整输入：")
    print(result["combined_input"])

    # 4. 验证
    assert result["pending_clarification"] is None

    assert len(
        result["context_data"]["clarification_history"]
    ) == 1

    assert "查询上个月的销售数据" in result["combined_input"]
    assert "查询重庆地区的数据" in result["combined_input"]

    print("\nClarificationManager 测试通过！")


    # 第二轮澄清
    pending2 = manager.build_pending_clarification(
        question="请问需要统计销售额还是订单数量？",
        missing_parameters=["metric"],
    )

    result2 = manager.apply_user_response(
        context_data=result["context_data"],
        pending_clarification=pending2,
        user_response="统计销售额",
    )

    assert len(
        result2["context_data"]["clarification_history"]
    ) == 2

    assert "查询重庆地区的数据" in result2["combined_input"]
    assert "统计销售额" in result2["combined_input"]

    print("\n第二轮澄清测试通过！")

if __name__ == "__main__":
    main()
