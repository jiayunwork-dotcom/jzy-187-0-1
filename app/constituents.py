"""候选分潮频率表与“记录够不够长、分潮分不分得开”的判别。

频率为写死的天文角频率，单位 度/平太阳时（Schureman/Pawlowicz 常用值）：

    M2  28.9841042      S2  30.0
    N2  28.4397295      K2  30.0821373
    K1  15.0410686      O1  13.9430356

平均流 Z0 频率为 0，永远纳入，不参与遮蔽关系。

判据（保守版瑞利判据，Rayleigh criterion）
------------------------------------------
记录长度记为 T（首、末观测时刻之差，与中间缺测段无关——缺测只降低估计
稳定性，不改变频率分辨力）。频率为 f_a、f_b 的两个分潮要能在一次最小
二乘拟合中被分开，需要

    T >= R / |f_a - f_b|,     R 默认取 1

即记录至少覆盖一个会合周期（beat period）。常数 R=1 是调和分析中常用的
下限（T_TIDE 等默认 R=1）。

“遮不遮住”采用保守策略：按分潮的先验重要性排固定优先级
（M2 > S2 > N2 > K1 > O1 > K2），依次考虑每个候选分潮 c；它必须能与
**所有更高优先级的候选分潮**分开，才纳入拟合，否则被其中频率最近、且
无法分开的那个分潮遮住。之所以要求与所有更高优先级候选分开，而不只与
已纳入者分开，是为了避免链式遮蔽——例如短记录里 S2 未被纳入时，不能
因此就让与它更难分开的 K2 混进来。被遮住的分潮在结果中注明遮住它的分潮
以及还差多长记录。

除跨度判据外还有一道自由度门槛：N 个样本点最多纳入 floor((N-1)/2) 个
分潮（平均流 1 个系数 + 每分潮 cos/sin 2 个系数），不足时同样按优先级
从低到高剔除，并注明现有多少点、至少还需多少点。

典型会合周期：S2-K2 ≈ 182.6 天，S2-M2 ≈ 14.8 天，K1-O1 ≈ 13.7 天，
M2-N2 ≈ 27.6 天。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional

from . import config

# 度/太阳时 -> 弧度/秒
_DEG_PER_HOUR_TO_RAD_PER_SEC = math.radians(1.0) / 3600.0

# 固定纳入顺序 = 先验重要性优先级（见模块 docstring）
CONSTITUENTS = (
    ("M2", 28.9841042),
    ("S2", 30.0),
    ("N2", 28.4397295),
    ("K1", 15.0410686),
    ("O1", 13.9430356),
    ("K2", 30.0821373),
)
PRIORITY = [name for name, _ in CONSTITUENTS]
FREQ_DEG_PER_HOUR = {name: f for name, f in CONSTITUENTS}
OMEGA_RAD_PER_SEC = {
    name: f * _DEG_PER_HOUR_TO_RAD_PER_SEC for name, f in CONSTITUENTS
}

HOURS_PER_DAY = 24.0


@dataclass
class Included:
    name: str
    omega: float  # rad/s


@dataclass
class Excluded:
    name: str
    masked_by: Optional[str]
    reason: str  # "rayleigh"：记录长度不足；"insufficient_samples"：点数不足
    # reason == "rayleigh" 时填写（小时）：
    beat_period_hours: float = 0.0
    record_hours: float = 0.0
    deficit_hours: float = 0.0
    # reason == "insufficient_samples" 时填写：
    have_samples: int = 0
    need_samples: int = 0


def _beat_hours(fa: float, fb: float) -> float:
    """瑞利临界记录长度（小时）：360·R / |f_a - f_b|。

    频率单位是度/平太阳时，频率差为度/时，一个会合周期对应 360°，
    故临界时长 = 360·R/Δf 小时（不是 1/Δf）。
    """
    return 360.0 * config.RAYLEIGH_FACTOR / abs(fa - fb)


def select_constituents(record_seconds: float, n_samples: int):
    """按记录跨度与样本点数决定纳入哪些分潮。

    返回 (included, excluded)：
      included: 第一项恒为平均流 Included("Z0", 0.0)；
      excluded: 每条注明原因——"rayleigh"（被谁遮住、临界时长、尚缺时长，
                小时）或 "insufficient_samples"（现有多少点、至少还需多少点）。

    两道门槛依次执行：
      1) 瑞利判据（跨度 T），见模块 docstring；
      2) 自由度门槛：平均流 1 个系数、每个分潮 cos/sin 2 个系数，N 个点
         最多纳入 floor((N-1)/2) 个分潮；超出时按优先级从低到高剔除
         （先丢 K2，再 O1、K1……）。跨度够、点太稀（如每小时设定但只有
         几个点）时由这道门兜底，保证正规方程不奇异。
    """
    record_hours = record_seconds / 3600.0
    included: List[Included] = [Included("Z0", 0.0)]
    excluded: List[Excluded] = []

    for name in PRIORITY:
        masked: Optional[Excluded] = None
        for other in PRIORITY:
            if other == name:
                continue
            # 只与更高优先级候选比较（先验上更重要的那些）
            if PRIORITY.index(other) > PRIORITY.index(name):
                continue
            beat = _beat_hours(FREQ_DEG_PER_HOUR[name], FREQ_DEG_PER_HOUR[other])
            if record_hours + 1e-9 < beat:
                # 与某个更高优先级分潮分不开；取“最分不开”的那个
                if masked is None or beat > masked.beat_period_hours:
                    masked = Excluded(
                        name=name,
                        masked_by=other,
                        reason="rayleigh",
                        beat_period_hours=beat,
                        record_hours=record_hours,
                        deficit_hours=beat - record_hours,
                    )
        if masked is None:
            included.append(Included(name, OMEGA_RAD_PER_SEC[name]))
        else:
            excluded.append(masked)

    # 自由度门槛：1（平均流）+ 2·k 个系数 <= N
    max_harmonics = max(0, (n_samples - 1) // 2)
    harmonics = included[1:]
    if len(harmonics) > max_harmonics:
        keep = harmonics[:max_harmonics]
        drop = harmonics[max_harmonics:]
        included = [included[0]] + keep
        need = 1 + 2 * (len(keep) + 1)
        for d in drop:
            excluded.append(Excluded(
                name=d.name,
                masked_by=None,
                reason="insufficient_samples",
                have_samples=n_samples,
                need_samples=need,
            ))

    return included, excluded
