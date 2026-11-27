
from collections import defaultdict
from typing import List, Set

from app.planner.schemas import ExecutionPlan


class PlanValidationError(ValueError):
    """执行计划校验异常。"""


class PlanValidator:
    """
    Execution Plan Validator

    校验执行计划的结构完整性和步骤依赖关系。
    """

    def validate(self, plan: ExecutionPlan) -> None:
        """
        校验执行计划。

        校验通过：正常返回 None。
        校验失败：抛出 PlanValidationError。
        """
        errors: List[str] = []

        steps = plan.steps

        # 1. 检查计划是否为空
        if not steps:
            raise PlanValidationError("ExecutionPlan 不包含任何执行步骤")

        # 2. 检查 step_id 是否重复
        step_map = {}

        for step in steps:
            step_id = str(step.step_id)

            if step_id in step_map:
                errors.append(f"重复的 step_id: {step_id}")
            else:
                step_map[step_id] = step

        if errors:
            raise PlanValidationError("\n".join(errors))

        step_ids = set(step_map.keys())

        # 3. 检查依赖是否存在、是否依赖自身
        for step in steps:
            current_id = str(step.step_id)

            for dependency_id in step.depends_on:
                dependency_id = str(dependency_id)

                if dependency_id == current_id:
                    errors.append(
                        f"步骤 {current_id} 不能依赖自身"
                    )

                elif dependency_id not in step_ids:
                    errors.append(
                        f"步骤 {current_id} 依赖了不存在的步骤 "
                        f"{dependency_id}"
                    )

        if errors:
            raise PlanValidationError("\n".join(errors))

        # 4. 检查是否存在循环依赖
        graph = defaultdict(list)

        for step in steps:
            current_id = str(step.step_id)

            for dependency_id in step.depends_on:
                graph[str(dependency_id)].append(current_id)

        visited: Set[str] = set()
        visiting: Set[str] = set()

        def dfs(node: str) -> bool:
            """
            使用 DFS 检测有向图中的环。
            True 表示发现循环依赖。
            """
            if node in visiting:
                return True

            if node in visited:
                return False

            visiting.add(node)

            for neighbor in graph[node]:
                if dfs(neighbor):
                    return True

            visiting.remove(node)
            visited.add(node)

            return False

        for step_id in step_ids:
            if dfs(step_id):
                raise PlanValidationError(
                    "执行计划存在循环依赖，无法确定执行顺序"
                )

        return None


# 全局实例
plan_validator = PlanValidator()
