"""analytics.py 的单元测试."""

import sys
from pathlib import Path

# 把 python/ 加到搜索路径，保证 `from src...` 能找到
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from src.models.schemas import CampaignMetrics
from src.tools.analytics import (
    calculate_ecpm,
    detect_anomalies,
    score_creative_performance,
    optimize_budget_allocation,
)


# ─── calculate_ecpm ───

def test_calculate_ecpm_normal():
    """标准情况：CTR=5%, CVR=10%, CPA=100 → eCPM=500."""
    result = calculate_ecpm(ctr=0.05, cvr=0.10, target_cpa=100.0)
    assert abs(result - 500.0) < 0.01


def test_calculate_ecpm_zero_ctr():
    """CTR=0 时 eCPM 应为 0，不能报错."""
    result = calculate_ecpm(ctr=0.0, cvr=0.10, target_cpa=100.0)
    assert result == 0.0


# ─── detect_anomalies ───

def _make_metrics(campaign_id, impressions, clicks, conversions, cost, revenue):
    return CampaignMetrics(
        campaign_id=campaign_id,
        impressions=impressions,
        clicks=clicks,
        conversions=conversions,
        total_cost=cost,
        total_revenue=revenue,
    )


def test_detect_anomalies_low_ctr():
    """CTR 明显低于阈值时应触发 low_ctr 告警."""
    m = _make_metrics("c1", impressions=1000, clicks=2, conversions=0, cost=10.0, revenue=0.0)
    alerts = detect_anomalies([m], ctr_threshold=0.005)
    assert any(a["type"] == "low_ctr" for a in alerts), "应产生 low_ctr 告警"


def test_detect_anomalies_high_cpa():
    """CPA 超过上限时应触发 high_cpa 告警."""
    m = _make_metrics("c2", impressions=1000, clicks=100, conversions=1, cost=500.0, revenue=50.0)
    alerts = detect_anomalies([m], cpa_ceiling=200.0)
    assert any(a["type"] == "high_cpa" for a in alerts), "应产生 high_cpa 告警"


def test_detect_anomalies_healthy():
    """健康数据不应产生告警."""
    m = _make_metrics("c3", impressions=10000, clicks=500, conversions=50, cost=200.0, revenue=1000.0)
    alerts = detect_anomalies([m])
    assert len(alerts) == 0, f"不应有告警，实际: {alerts}"


def test_detect_anomalies_small_sample_no_alert():
    """曝光量太少时即使 CTR 低也不告警（避免小样本误报）."""
    m = _make_metrics("c4", impressions=10, clicks=0, conversions=0, cost=1.0, revenue=0.0)
    alerts = detect_anomalies([m])
    assert len(alerts) == 0


# ─── score_creative_performance ───

def test_score_zero_impressions():
    """零曝光时得分为 0，不抛异常."""
    score = score_creative_performance(0, 0, 0, 0.0)
    assert score == 0.0


def test_score_perfect():
    """完美表现：CTR=10%, CVR=20%, CPA=¥10 → 接近满分."""
    score = score_creative_performance(
        impressions=10000, clicks=1000, conversions=200, cost=2000.0
    )
    assert score > 80, f"完美表现应接近满分，实际得分: {score}"


def test_score_poor():
    """差劲表现：CTR=0.1%, CVR=0% → 低分."""
    score = score_creative_performance(
        impressions=10000, clicks=10, conversions=0, cost=100.0
    )
    assert score < 30, f"差劲表现应低分，实际: {score}"


# ─── optimize_budget_allocation ───

def test_budget_allocation_preserves_total():
    """重新分配后总预算不应变化（误差在 1% 以内）."""
    metrics = [
        _make_metrics("c1", 10000, 500, 50, 1000.0, 3000.0),
        _make_metrics("c2", 10000, 200, 10, 1000.0, 500.0),
        _make_metrics("c3", 10000, 400, 40, 1000.0, 2000.0),
    ]
    allocations = optimize_budget_allocation(metrics, total_budget=3000.0)
    total_recommended = sum(a.recommended_budget for a in allocations)
    assert abs(total_recommended - 3000.0) < 30.0, (
        f"总预算应约为 3000，实际: {total_recommended:.2f}"
    )


def test_budget_allocation_high_roas_gets_more():
    """ROAS 高的 Campaign 应该分配到更多预算."""
    metrics = [
        _make_metrics("high_roas", 10000, 500, 50, 1000.0, 5000.0),   # ROAS=5
        _make_metrics("low_roas",  10000, 500, 10, 1000.0,  500.0),   # ROAS=0.5
    ]
    allocations = optimize_budget_allocation(metrics, total_budget=2000.0)
    alloc_map = {a.campaign_id: a.recommended_budget for a in allocations}
    assert alloc_map["high_roas"] > alloc_map["low_roas"], (
        "高 ROAS Campaign 应分到更多预算"
    )
