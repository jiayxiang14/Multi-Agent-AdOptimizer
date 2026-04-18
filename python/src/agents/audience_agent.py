"""Audience Agent — 人群画像分析、精准定向、Lookalike扩展."""

from __future__ import annotations

import random
import structlog
from typing import Any

from ..models.schemas import AdOptimizerState, AudienceProfile, CampaignMetrics

logger = structlog.get_logger()

SYSTEM_PROMPT = """你是一个专业的受众分析专家（Audience Agent）。
你的职责：
1. 分析现有转化用户画像，提取高价值人群特征
2. 推荐精准定向策略（年龄/性别/兴趣/地域/设备）
3. 基于种子人群生成 Lookalike 扩展建议
4. 评估各受众段的转化潜力
"""


class AudienceAgent:
    """受众 Agent：人群画像分析与定向优化."""

    def __init__(self, llm: Any = None, db_client: Any = None) -> None:
        self.llm = llm
        self.db = db_client
        self.name = "Audience Agent"

    def run(self, state: dict) -> dict:
        """LangGraph 节点入口."""
        logger.info("audience_agent_start")
        metrics = state.get("metrics", [])
        campaign_ids = state.get("campaign_ids", [])

        insights = self._analyze_audience(metrics, campaign_ids)
        #messages = list(state.get("agent_messages", []))

        top_segments = insights.get("top_segments", [])
        new_message = {
            "agent": self.name,
            "content": (
                f"受众分析完成。识别出 {len(top_segments)} 个高价值人群段。"
                f"推荐优先定向: {', '.join(s.get('name', '') for s in top_segments[:3])}"
            ),
        }

        logger.info("audience_agent_done", segments=len(top_segments))
        return {
            "audience_insights": insights,
            "agent_messages": [new_message],
            "current_agent": "audience",
        }

    def _analyze_audience(self, metrics: list, campaign_ids: list[str]) -> dict:
        """分析受众数据，返回洞察."""
        if self.db is not None and hasattr(self.db, "get_audience_breakdown"):
            return self._db_analyze(campaign_ids)
        return self._mock_analyze(metrics)

def _db_analyze(self, campaign_ids: list[str]) -> dict:
    breakdowns = {}
    top_segments = []

    for cid in campaign_ids:
        breakdown = self.db.get_audience_breakdown(cid)
        breakdowns[cid] = breakdown

        for row in breakdown.get("by_age", []):
            age = row.get("age_group", "unknown")
            clicks = row.get("clicks", 0)
            conversions = row.get("conversions", 0)
            cnt = row.get("cnt", 1) or 1

            ctr = clicks / cnt
            cvr = conversions / clicks if clicks > 0 else 0
            score = round(min(ctr / 0.05, 1.0) * 50 + min(cvr / 0.1, 1.0) * 50, 1)

            if ctr > 0.01 or cvr > 0.03:
                top_segments.append({
                    "name": f"{age}岁 ({cid})",
                    "age_range": age,
                    "gender": "all",
                    "interests": [],
                    "estimated_ctr": round(ctr, 4),
                    "estimated_cvr": round(cvr, 4),
                    "score": score,
                    "recommendation": (
                        "高价值核心人群，建议加大预算" if score > 70
                        else "潜力人群，可适量投放测试"
                    ),
                })

    top_segments.sort(key=lambda x: x["score"], reverse=True)

    lookalike = [
        {
            "seed_segment": seg["name"],
            "expansion": "相似人群扩展 1%",
            "estimated_reach": 500000,
            "expected_ctr_change": "+10%",
        }
        for seg in top_segments[:2]
    ]

    return {
        "breakdowns": breakdowns,
        "top_segments": top_segments[:6],
        "lookalike_suggestions": lookalike,
    }

    def _mock_analyze(self, metrics: list) -> dict:
        """Mock 模式：生成模拟受众洞察."""
        segments = [
            {
                "name": "高价值白领 25-34岁",
                "age_range": "25-34",
                "gender": "all",
                "interests": ["科技", "数码", "商务"],
                "estimated_ctr": 0.045,
                "estimated_cvr": 0.08,
                "score": 92,
                "recommendation": "核心人群，建议加大投放",
            },
            {
                "name": "年轻女性 18-24岁",
                "age_range": "18-24",
                "gender": "female",
                "interests": ["时尚", "美妆", "社交"],
                "estimated_ctr": 0.055,
                "estimated_cvr": 0.06,
                "score": 85,
                "recommendation": "CTR高但CVR偏低，优化落地页后可扩量",
            },
            {
                "name": "家庭决策者 35-44岁",
                "age_range": "35-44",
                "gender": "all",
                "interests": ["家居", "教育", "健康"],
                "estimated_ctr": 0.032,
                "estimated_cvr": 0.12,
                "score": 88,
                "recommendation": "CVR极高，适合高客单价产品定向",
            },
            {
                "name": "科技爱好者 25-44岁",
                "age_range": "25-44",
                "gender": "male",
                "interests": ["科技", "游戏", "数码"],
                "estimated_ctr": 0.038,
                "estimated_cvr": 0.09,
                "score": 80,
                "recommendation": "稳定人群，可作为Lookalike种子",
            },
        ]

        lookalike = [
            {
                "seed_segment": "高价值白领 25-34岁",
                "expansion": "相似人群扩展 1%",
                "estimated_reach": random.randint(500000, 2000000),
                "expected_ctr_change": "+10-15%",
            },
            {
                "seed_segment": "家庭决策者 35-44岁",
                "expansion": "相似人群扩展 2%",
                "estimated_reach": random.randint(1000000, 5000000),
                "expected_ctr_change": "+5-10%",
            },
        ]

        device_insights = {
            "mobile": {"share": 0.68, "ctr": 0.042, "cvr": 0.07},
            "desktop": {"share": 0.25, "ctr": 0.035, "cvr": 0.09},
            "tablet": {"share": 0.07, "ctr": 0.028, "cvr": 0.06},
        }

        return {
            "top_segments": segments,
            "lookalike_suggestions": lookalike,
            "device_insights": device_insights,
            "geo_recommendation": "一线城市转化率最高，建议二线城市扩量测试",
        }
