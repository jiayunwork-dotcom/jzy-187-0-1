"""批次数据的业务校验。

校验规则（每条错误都定位到 samples[i].t / .u / .v 这样的具体字段）：

1. 时刻必须能解析、带时区，且不早于测站布放时间；
2. 同一批内不允许出现重复时刻；
3. 与该测站历史批次之间不允许出现重复时刻；
4. 采样必须等间隔，且间隔必须是测站设定采样间隔的整数倍
   （允许中间缺测，但不允许不等间隔采样或与测站设定不符）。

返回按时刻排序、去重前的合法记录列表 [(t, u, v), ...]（已排序）。
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from typing import List, Tuple

from .errors import ValidationError

_UTC = timezone.utc

# 间隔判为整数倍的容差：相对 1e-6 或绝对 1e-3 秒，取大者
_RTOL = 1e-6
_ATOL = 1e-3

Record = Tuple[float, float, float]


def _multiple_ok(delta: float, dt_nom: float) -> bool:
    k = round(delta / dt_nom)
    if k <= 0:
        return False
    return abs(delta - k * dt_nom) <= max(_ATOL, _RTOL * dt_nom)


def validate_batch_records(
    conn: sqlite3.Connection,
    station: sqlite3.Row,
    raw_samples,
) -> List[Record]:
    """raw_samples：pydantic SampleIn 列表（字段顺序即用户提交顺序）。"""
    errors: List[dict] = []
    records: List[Record] = []

    for i, s in enumerate(raw_samples):
        # parse_iso 自身抛 ValidationError（单条即中断）；这里先做一遍
        # 轻量解析，把所有坏时刻一次性收齐再报。
        from datetime import datetime

        try:
            dt = datetime.fromisoformat(s.t)
        except ValueError:
            errors.append({
                "field": f"samples[{i}].t",
                "msg": f"时间格式无法解析：{s.t!r}，需要 ISO 8601",
            })
            continue
        if dt.tzinfo is None:
            errors.append({
                "field": f"samples[{i}].t",
                "msg": f"时间 {s.t!r} 缺少时区信息（必须带时区）",
            })
            continue
        t = dt.astimezone(_UTC).timestamp()
        if t < station["deployed_at"]:
            errors.append({
                "field": f"samples[{i}].t",
                "msg": "采样时刻早于测站布放时间，不允许纳入该测站",
            })
        records.append((t, float(s.u), float(s.v)))

    if errors:
        raise ValidationError(errors)

    # 规则 2：批内重复时刻（按提交顺序，报后出现的那一个）
    seen: dict[float, int] = {}
    for i, (t, _u, _v) in enumerate(records):
        if t in seen:
            raise ValidationError([{
                "field": f"samples[{i}].t",
                "msg": f"同一时刻在本批中重复出现（首次出现于 samples[{seen[t]}]）",
            }])
        seen[t] = i

    records.sort(key=lambda r: r[0])

    # 规则 3：与历史批次重复时刻
    station_id = station["code"]
    ts = [r[0] for r in records]
    placeholders = ",".join("?" for _ in ts)
    dup_rows = conn.execute(
        f"SELECT t, batch_no FROM samples WHERE station_id = ? "
        f"AND t IN ({placeholders})",
        [station_id, *ts],
    ).fetchall()
    if dup_rows:
        # seen 记录每个时刻在用户原提交顺序中的下标
        row = dup_rows[0]
        raise ValidationError([{
            "field": f"samples[{seen[row['t']]}].t",
            "msg": f"该时刻在历史批次 {row['batch_no']!r} 中已存在",
        }])

    # 规则 4：等间隔，且每个采样时刻都落在以布放时刻为原点、采样间隔为
    # 步长的网格上，即 (t - t_dep) 是 sample_dt 的整数倍。逐点判网格既挡住
    # “不等间隔/间隔与测站设定不符”，也挡住不同批次各自等间隔、却互相错位
    # 半格的情形；缺测只体现为跨步更大，仍然合法。
    dt_nom = float(station["sample_dt"])
    t_dep = float(station["deployed_at"])
    prev_t = None
    for t, _u, _v in records:
        k = round((t - t_dep) / dt_nom)
        on_grid = abs((t - t_dep) - k * dt_nom) <= max(_ATOL, _RTOL * dt_nom)
        if not on_grid:
            idx = seen[t]
            raise ValidationError([{
                "field": f"samples[{idx}].t",
                "msg": (
                    f"采样时刻 {t - t_dep:.6f}s（相对布放）不是测站设定采样 "
                    f"间隔 {dt_nom:g}s 的整数倍（与测站设定不符）"
                ),
            }])
        if prev_t is not None:
            delta = t - prev_t
            if not _multiple_ok(delta, dt_nom):
                idx = seen[t]
                raise ValidationError([{
                    "field": f"samples[{idx}].t",
                    "msg": (
                        f"相邻采样间隔 {delta:.6f}s 不是测站设定采样间隔 "
                        f"{dt_nom:g}s 的整数倍（不等间隔采样）"
                    ),
                }])
        prev_t = t

    return records
