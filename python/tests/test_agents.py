"""Agent 行为的集成测试：验证 state merge 是否正确、消息是否累积."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from src.models.schemas import CampaignMetrics
from src.agents.monitor_agent import MonitorAgent
from src.agents.bidding_agent import BiddingAgent
from src.agents.creative_agent import CreativeAgent
from src.agents.audience_agent import AudienceAgent
from src.agents.optimize_agent import OptimizeAgent
from src.orchestrator.supervisor import AdOptimizerSupervisor


# ─── 测试用的共享 fixtures ───

def _base_state():
    """最小可运行的初始 state."""
    metrics = [
        CampaignMetrics(
            campaign_id="camp_001",
            campaign_name="测试产品 - google投放",
            impressions=10000,
            clicks=300,
            conversions=30,
            total_cost=600.0,
            total_revenue=1500.0,
        ).model_dump()
    ]
    return {
        "task": "optimize_campaigns",
        "campaign_ids": ["camp_001"],
        "metrics": metrics,
        "new_creatives": [],
        "audience_insights": {},
        "bidding_decisions": [],
        "optimization_actions": [],
        "budget_allocations": [],
        "alerts": [],
        "agent_messages": [],
        "current_agent": "",
        "iteration": 0,
        "max_iterations": 3,
        "is_complete": False,
    }


# ─── 每个 Agent 返回 list 而非 full list ───

def test_monitor_returns_only_new_message():
    """MonitorAgent.run() 返回的 agent_messages 只包含自己新增的那条."""
    agent = MonitorAgent()
    state = _base_state()
    result = agent.run(state)
    msgs = result.get("agent_messages", [])
    assert len(msgs) == 1, f"应只返回 1 条新消息，实际: {len(msgs)}"
    assert msgs[0]["agent"] == "Monitor Agent"


def test_bidding_returns_only_new_message():
    agent = BiddingAgent()
    state = _base_state()
    result = agent.run(state)
    msgs = result.get("agent_messages", [])
    assert len(msgs) == 1
    assert msgs[0]["agent"] == "Bidding Agent"


def test_creative_returns_per_campaign_messages():
    """CreativeAgent 每个 campaign 生成一条消息."""
    agent = CreativeAgent()
    state = _base_state()
    result = agent.run(state)
    msgs = result.get("agent_messages", [])
    assert len(msgs) >= 1
    assert all(m["agent"] == "Creative Agent" for m in msgs)


def test_audience_returns_only_new_message():
    agent = AudienceAgent()
    state = _base_state()
    result = agent.run(state)
    msgs = result.get("agent_messages", [])
    assert len(msgs) == 1
    assert msgs[0]["agent"] == "Audience Agent"


def test_optimize_returns_only_new_message():
    agent = OptimizeAgent()
    state = _base_state()
    result = agent.run(state)
    msgs = result.get("agent_messages", [])
    assert len(msgs) == 1
    assert msgs[0]["agent"] == "Optimize Agent"


# ─── state merge 正确性 ───

def test_merge_state_appends_messages():
    """_merge_state 对 agent_messages 应追加，不覆盖."""
    supervisor = AdOptimizerSupervisor()
    old_state = {"agent_messages": [{"agent": "A", "content": "first"}], "iteration": 0}
    updates = {"agent_messages": [{"agent": "B", "content": "second"}], "iteration": 1}
    merged = supervisor._merge_state(old_state, updates)
    assert len(merged["agent_messages"]) == 2, "应有 2 条消息"
    assert merged["agent_messages"][0]["agent"] == "A"
    assert merged["agent_messages"][1]["agent"] == "B"
    assert merged["iteration"] == 1  # 标量字段正常覆盖


def test_merge_state_appends_optimization_actions():
    """optimization_actions 跨轮次应累积."""
    supervisor = AdOptimizerSupervisor()
    old_state = {"optimization_actions": [{"action_type": "pause_creative"}]}
    updates = {"optimization_actions": [{"action_type": "adjust_budget"}]}
    merged = supervisor._merge_state(old_state, updates)
    assert len(merged["optimization_actions"]) == 2


def test_merge_state_merges_audience_insights():
    """audience_insights（字典）应合并而非替换."""
    supervisor = AdOptimizerSupervisor()
    old_state = {"audience_insights": {"top_segments": [1, 2]}}
    updates = {"audience_insights": {"device_insights": {"mobile": 0.7}}}
    merged = supervisor._merge_state(old_state, updates)
    assert "top_segments" in merged["audience_insights"]
    assert "device_insights" in merged["audience_insights"]


# ─── 完整流程冒烟测试 ───

def test_supervisor_run_accumulates_messages():
    """跑一遍完整 Supervisor，agent_messages 应包含所有 5 个 Agent 的消息."""
    supervisor = AdOptimizerSupervisor()
    result = supervisor.run(max_iterations=1)

    messages = result.get("agent_messages", [])
    agent_names = {m["agent"] for m in messages}

    assert "Monitor Agent" in agent_names, "应有 Monitor Agent 消息"
    assert "Bidding Agent" in agent_names, "应有 Bidding Agent 消息"
    assert "Creative Agent" in agent_names, "应有 Creative Agent 消息"
    assert "Audience Agent" in agent_names, "应有 Audience Agent 消息"
    assert "Optimize Agent" in agent_names, "应有 Optimize Agent 消息"


def test_supervisor_run_produces_actions():
    """跑完之后应产生至少一条优化操作."""
    supervisor = AdOptimizerSupervisor()
    result = supervisor.run(max_iterations=1)
    assert len(result.get("optimization_actions", [])) > 0


def test_supervisor_run_produces_budget_allocations():
    """跑完之后应产生预算分配建议."""
    supervisor = AdOptimizerSupervisor()
    result = supervisor.run(max_iterations=1)
    assert len(result.get("budget_allocations", [])) > 0
