"""
LangGraph Supervisor — 广告优化闭环的核心编排层。

架构：Supervisor Pattern（动态路由 + 条件分支 + 循环优化）

流程：
  1. Monitor Agent → 采集指标、检测异常
  2. Audience Agent → 受众分析（可与Monitor并行）
  3. Creative Agent → 生成新素材变体
  4. Bidding Agent  → 优化竞价策略
  5. Optimize Agent → 整合所有输出，生成优化方案
  6. 若仍有未解决告警且未达最大迭代数 → 回到步骤1
"""

from __future__ import annotations

import json
import operator
import os
from typing import Annotated, Any, TypedDict

import structlog

from ..agents.audience_agent import AudienceAgent
from ..agents.bidding_agent import BiddingAgent
from ..agents.creative_agent import CreativeAgent
from ..agents.monitor_agent import MonitorAgent
from ..agents.optimize_agent import OptimizeAgent
from ..data.mock_data import compute_mock_metrics, generate_full_mock_dataset
from ..models.schemas import CampaignMetrics

logger = structlog.get_logger()

# LangGraph 是可选依赖
try:
    from langgraph.graph import END, StateGraph
    HAS_LANGGRAPH = True
except ImportError:
    HAS_LANGGRAPH = False


# ────────── State Schema ──────────
# Annotated[list, operator.add] 告诉 LangGraph：当多个节点都写这个字段时，
# 用 operator.add（即列表拼接）合并，而不是后者覆盖前者。
# 这解决了多轮迭代和并行节点下 agent_messages / optimization_actions 被覆盖的问题。

def _take_last(a: Any, b: Any) -> Any:
    """并行节点同时写同一字段时，取最后写入的值（即 b）."""
    return b


class AdOptimizerStateDict(TypedDict, total=False):
    task: str
    campaign_ids: list
    metrics: list
    new_creatives: list
    audience_insights: dict
    bidding_decisions: list
    # 跨迭代累积：每轮新增的操作都保留，不覆盖
    optimization_actions: Annotated[list, operator.add]
    budget_allocations: list
    alerts: list
    # 跨迭代累积：每个 Agent 的每条消息都保留，形成完整执行日志
    agent_messages: Annotated[list, operator.add]
    # 并行节点（audience / creative）同时写 current_agent 时取最后一个
    current_agent: Annotated[str, _take_last]
    iteration: int
    max_iterations: int
    is_complete: bool


# ────────── Supervisor 构建 ──────────

class AdOptimizerSupervisor:
    """广告优化 Supervisor：编排5个Agent的闭环工作流."""

    def __init__(self, llm: Any = None, db_client: Any = None, ads_client: Any = None) -> None:
        self.creative = CreativeAgent(llm=llm)
        self.audience = AudienceAgent(llm=llm, db_client=db_client)
        self.bidding = BiddingAgent(llm=llm)
        self.monitor = MonitorAgent(llm=llm, db_client=db_client)
        self.optimize = OptimizeAgent(llm=llm, ads_client=ads_client)
        self.graph = self._build_graph() if HAS_LANGGRAPH else None

    def _build_graph(self) -> Any:
        """构建 LangGraph 状态图.

        拓扑结构（audience 和 creative 并行）：
          monitor → audience ──┐
                  └→ creative ─┴→ bidding → optimize → (条件边)
        """
        # 使用 TypedDict 而非 dict，LangGraph 才能识别 Annotated reducer
        graph = StateGraph(AdOptimizerStateDict)

        graph.add_node("monitor", self.monitor.run)
        graph.add_node("audience", self.audience.run)
        graph.add_node("creative", self.creative.run)
        graph.add_node("bidding", self.bidding.run)
        graph.add_node("optimize", self.optimize.run)

        graph.set_entry_point("monitor")

        # monitor 完成后，audience 和 creative 并行启动（fan-out）
        graph.add_edge("monitor", "audience")
        graph.add_edge("monitor", "creative")

        # audience 和 creative 都完成后 bidding 才启动（fan-in）
        graph.add_edge("audience", "bidding")
        graph.add_edge("creative", "bidding")

        graph.add_edge("bidding", "optimize")

        # 条件边：是否继续迭代
        graph.add_conditional_edges(
            "optimize",
            self._should_continue,
            {"continue": "monitor", "end": END},
        )

        return graph.compile()

    @staticmethod
    def _should_continue(state: dict) -> str:
        """条件分支：是否继续迭代优化."""
        if state.get("is_complete", False):
            return "end"
        iteration = state.get("iteration", 0) #从 state 这个字典里取 "iteration" 这个键的值如果没有这个键，就用默认值 0
        max_iter = state.get("max_iterations", 3)
        if iteration >= max_iter:
            return "end"
        if state.get("alerts"):
            return "continue"
        return "end"

    def run(self, campaign_ids: list[str] | None = None, max_iterations: int = 3) -> dict:
        """运行完整的优化闭环."""
        mock_data = generate_full_mock_dataset()
        metrics = mock_data["metrics"]

        if campaign_ids:
            metrics = [m for m in metrics if m.campaign_id in campaign_ids]
        else:
            campaign_ids = [m.campaign_id for m in metrics]

        initial_state = {
            "task": "optimize_campaigns",
            "campaign_ids": campaign_ids,
            "metrics": [m.model_dump() for m in metrics],
            "new_creatives": [],
            "audience_insights": {},
            "bidding_decisions": [],
            "optimization_actions": [],
            "budget_allocations": [],
            "alerts": [],
            "agent_messages": [],
            "current_agent": "",
            "iteration": 0,
            "max_iterations": max_iterations,
            "is_complete": False,
        }

        if self.graph is not None:
            logger.info("supervisor_start_langgraph", campaigns=len(campaign_ids))
            result = self.graph.invoke(initial_state)
        else:
            logger.info("supervisor_start_sequential", campaigns=len(campaign_ids))
            result = self._run_sequential(initial_state)

        return result

    @staticmethod
    def _merge_state(old: dict, updates: dict) -> dict:
        """智能合并 state：列表字段追加，字典字段合并，标量字段覆盖.

        与 LangGraph 的 Annotated reducer 保持相同语义，
        让串行回退路径和 LangGraph 路径的行为一致。
        """
        # 这两个字段要跨次调用累积，其余列表字段取最新值
        APPEND_KEYS = {"agent_messages", "optimization_actions"}
        result = dict(old)
        for key, value in updates.items():
            if key in APPEND_KEYS and isinstance(value, list):
                result[key] = result.get(key, []) + value
            elif key == "audience_insights" and isinstance(value, dict):
                result[key] = {**result.get(key, {}), **value}
            else:
                result[key] = value
        return result

    def _run_sequential(self, state: dict) -> dict:
        """无 LangGraph 时的顺序执行回退方案."""
        for iteration in range(state.get("max_iterations", 3)):
            logger.info("sequential_iteration", iteration=iteration + 1)

            state = self._merge_state(state, self.monitor.run(state))
            # audience 和 creative 串行模拟并行（结果都 merge 进 state）
            state = self._merge_state(state, self.audience.run(state))
            state = self._merge_state(state, self.creative.run(state))
            state = self._merge_state(state, self.bidding.run(state))
            state = self._merge_state(state, self.optimize.run(state))

            if state.get("is_complete", False):
                break
            if not state.get("alerts"):
                break

        return state

    def get_summary(self, result: dict) -> str:
        """生成人类可读的优化摘要."""
        messages = result.get("agent_messages", [])
        actions = result.get("optimization_actions", [])
        budgets = result.get("budget_allocations", [])
        iterations = result.get("iteration", 0)

        lines = [
            "=" * 60,
            "  多Agent广告优化系统 — 执行报告",
            "=" * 60,
            f"\n总迭代轮次: {iterations}",
            f"优化操作数: {len(actions)}",
            f"预算调整数: {len(budgets)}",
            "\n--- Agent 执行日志 ---",
        ]

        for msg in messages:
            lines.append(f"[{msg.get('agent', '?')}] {msg.get('content', '')}")

        if budgets:
            lines.append("\n--- 预算分配建议 ---")
            for b_data in budgets:
                b = b_data if isinstance(b_data, dict) else b_data
                lines.append(
                    f"  {b.get('campaign_id', '?')}: "
                    f"¥{b.get('current_budget', 0):.0f} → ¥{b.get('recommended_budget', 0):.0f} "
                    f"({b.get('change_pct', 0):+.1f}%) — {b.get('reason', '')}"
                )

        if actions:
            lines.append("\n--- 优化操作列表 ---")
            for a_data in actions:
                a = a_data if isinstance(a_data, dict) else a_data
                lines.append(
                    f"  [{a.get('action_type', '?')}] Campaign {a.get('campaign_id', '?')}: "
                    f"{a.get('reason', '')}"
                )

        lines.append("\n" + "=" * 60)
        return "\n".join(lines)


# ────────── CLI 入口 ──────────

def main() -> None:
    """命令行运行入口."""
    from dotenv import load_dotenv
    load_dotenv()

    llm = None
    api_key = os.getenv("OPENAI_API_KEY", "")
    if api_key and not api_key.startswith("sk-your"):
        try:
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0.7)
            logger.info("llm_initialized", model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
        except ImportError:
            logger.warning("langchain_openai not installed, using mock mode")

    supervisor = AdOptimizerSupervisor(llm=llm)
    result = supervisor.run(max_iterations=2)
    print(supervisor.get_summary(result))


if __name__ == "__main__":
    main()