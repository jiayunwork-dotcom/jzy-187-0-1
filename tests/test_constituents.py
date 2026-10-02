"""分潮可分辨判别（瑞利判据）的单元测试。"""
from __future__ import annotations

import math

from app import constituents as ct


def test_beat_periods_are_documented_values():
    # 度/时，会合周期（天）
    def beat_days(a, b):
        # 频率单位度/时，会合周期 360/Δf 小时，再 /24 得天
        return 360.0 / abs(ct.FREQ_DEG_PER_HOUR[a] - ct.FREQ_DEG_PER_HOUR[b]) / 24.0

    assert math.isclose(beat_days("S2", "K2"), 182.62, rel_tol=1e-3)
    assert math.isclose(beat_days("S2", "M2"), 14.77, rel_tol=1e-3)
    assert math.isclose(beat_days("K1", "O1"), 13.66, rel_tol=1e-3)
    assert math.isclose(beat_days("M2", "N2"), 27.56, rel_tol=1e-3)


def test_mean_flow_always_included():
    inc, _ = ct.select_constituents(3600.0, n_samples=100000)  # 1 小时
    assert inc[0].name == "Z0"


def test_very_short_record_keeps_only_m2_and_mean():
    # 20 小时：M2 是优先级最高的分潮，无人能遮住它；其余都被遮
    inc, exc = ct.select_constituents(20 * 3600.0, n_samples=100000)
    names = [c.name for c in inc]
    assert names == ["Z0", "M2"]
    excluded = {e.name: e for e in exc}
    # S2 被 M2 遮住，缺口约 14.7 天 - 20 小时
    assert excluded["S2"].masked_by == "M2"
    assert excluded["S2"].deficit_hours > 0
    assert excluded["K2"].masked_by in {"S2", "M2"}


def test_s2_k2_never_both_fit_for_short_record():
    # 30 天：S2 进得来，K2 必然被 S2 遮住（需要 ~182.6 天）
    inc, exc = ct.select_constituents(30 * 86400.0, n_samples=100000)
    names = {c.name for c in inc}
    excluded = {e.name: e for e in exc}
    assert "S2" in names
    assert "K2" not in names
    assert excluded["K2"].masked_by == "S2"
    # 注明还差约 152.6 天（数据类里以小时计）
    assert math.isclose(excluded["K2"].deficit_hours / 24.0, 152.6, rel_tol=2e-3)


def test_deficit_is_beat_minus_record():
    inc, exc = ct.select_constituents(100 * 86400.0, n_samples=100000)
    for e in exc:
        assert math.isclose(
            e.deficit_hours + e.record_hours, e.beat_period_hours, rel_tol=1e-9
        )


def test_long_record_admits_all_candidates():
    # 200 天：六对候选全部超过会合周期
    inc, exc = ct.select_constituents(200 * 86400.0, n_samples=100000)
    names = {c.name for c in inc}
    assert names == {"Z0", "M2", "S2", "N2", "K1", "O1", "K2"}
    assert exc == []
