"""业务编排：测站/批次入库、触发全量重拟合、版本存档与回查。

更新策略：全量重拟合（refit-from-scratch）
-----------------------------------------
每到一批，把该站从布放至今**所有**批次的全部观测点取出来，重新做一次
调和拟合并存一版。选择它的理由与代价（详见 README）：

* 理由：结果与“一次性拿全部数据从头拟合”**逐位相同**——同一批点、同一
  排序、同样的正规方程，浮点运算顺序一致，天然满足振幅相对差 ≤ 1e-9、
  相位差 ≤ 1e-7，不依赖增量累积量的数值等价性；逻辑也最简单、可审计。
* 代价：计算量随站点累计样本数 O(N·k²) 线性增长（求解正规方程），存储
  上每版完整留一份结果 JSON。对“几天一批、单机几台仪器”的运维节奏完全
  够用；若未来 N 很大，可再切换为存储 X^Ty、X^TX 累积量的增量方案。
"""
from __future__ import annotations

import json
import math
import sqlite3
import time as _time
from typing import List, Optional

from . import constituents as ctab
from .errors import ConflictError, NotFoundError
from .fit import fit_harmonics
from .timeutil import to_iso
from .validation import Record, validate_batch_records


# ---------------------------------------------------------------- stations

def create_station(conn: sqlite3.Connection, data) -> dict:
    from .timeutil import parse_iso

    deployed = parse_iso(data.deployed_at, "deployed_at")
    exists = conn.execute(
        "SELECT 1 FROM stations WHERE code = ?", (data.code,)
    ).fetchone()
    if exists:
        raise ConflictError("code", f"测站编号 {data.code!r} 已存在")

    conn.execute(
        "INSERT INTO stations(code, latitude, longitude, sample_dt, "
        "deployed_at, created_at) VALUES (?, ?, ?, ?, ?, ?)",
        (data.code, data.latitude, data.longitude, data.sample_interval,
         deployed, _time.time()),
    )
    return get_station(conn, data.code)


def get_station(conn: sqlite3.Connection, code: str) -> dict:
    row = conn.execute("SELECT * FROM stations WHERE code = ?", (code,)).fetchone()
    if row is None:
        raise NotFoundError("station", f"测站 {code!r} 不存在")
    return _station_dict(row)


def list_stations(conn: sqlite3.Connection) -> List[dict]:
    rows = conn.execute("SELECT * FROM stations ORDER BY code").fetchall()
    return [_station_dict(r) for r in rows]


def _station_dict(row: sqlite3.Row) -> dict:
    return {
        "code": row["code"],
        "latitude": row["latitude"],
        "longitude": row["longitude"],
        "sample_interval_s": row["sample_dt"],
        "deployed_at": to_iso(row["deployed_at"]),
    }


# ----------------------------------------------------------------- batches

def upload_batch(conn: sqlite3.Connection, data) -> dict:
    station = conn.execute(
        "SELECT * FROM stations WHERE code = ?", (data.station,)
    ).fetchone()
    if station is None:
        raise NotFoundError("station", f"测站 {data.station!r} 不存在")

    # 幂等：同一批次号重复上传不产生新版本，直接返回首次处理时存档的版本。
    vrow = conn.execute(
        "SELECT id FROM versions WHERE station_id = ? AND batch_no = ?",
        (data.station, data.batch_no),
    ).fetchone()
    if vrow is not None:
        result = get_version(conn, data.station, vrow["id"])
        result["duplicate"] = True
        result["batch_no"] = data.batch_no
        return result

    records: List[Record] = validate_batch_records(conn, station, data.samples)

    seq_row = conn.execute(
        "SELECT COALESCE(MAX(arrival_seq), 0) + 1 AS next FROM batches "
        "WHERE station_id = ?",
        (data.station,),
    ).fetchone()
    arrival_seq = int(seq_row["next"])
    now = _time.time()

    conn.execute(
        "INSERT INTO batches(batch_no, station_id, arrival_seq, received_at, "
        "t_start, t_end, n_samples) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (data.batch_no, data.station, arrival_seq, now,
         records[0][0], records[-1][0], len(records)),
    )
    conn.executemany(
        "INSERT INTO samples(station_id, t, u, v, batch_no) VALUES (?, ?, ?, ?, ?)",
        [(data.station, t, u, v, data.batch_no) for (t, u, v) in records],
    )

    result = _refit_and_archive(conn, station, data.batch_no, now)
    result["duplicate"] = False
    return result


# --------------------------------------------------------------- analysis

def _load_all_samples(conn: sqlite3.Connection, station_id: str):
    rows = conn.execute(
        "SELECT t, u, v FROM samples WHERE station_id = ? ORDER BY t ASC",
        (station_id,),
    ).fetchall()
    return [r["t"] for r in rows], [r["u"] for r in rows], [r["v"] for r in rows]


def run_analysis(conn: sqlite3.Connection, station_id: str) -> dict:
    station = conn.execute(
        "SELECT * FROM stations WHERE code = ?", (station_id,)
    ).fetchone()
    if station is None:
        raise NotFoundError("station", f"测站 {station_id!r} 不存在")
    times, u, v = _load_all_samples(conn, station_id)
    if not times:
        raise NotFoundError("samples", f"测站 {station_id!r} 尚无任何数据批次")
    return _compute(station, times, u, v)


def _compute(station: sqlite3.Row, times, u, v) -> dict:
    span_s = times[-1] - times[0]
    included, excluded = ctab.select_constituents(span_s, len(times))

    names = [c.name for c in included if c.name != "Z0"]
    omegas = [c.omega for c in included if c.name != "Z0"]
    u0, v0, fitted = fit_harmonics(times, u, v, omegas,
                                   t_dep=float(station["deployed_at"]))

    constituent_results = []
    for name, item in zip(names, fitted):
        constituent_results.append({
            "name": name,
            "included": True,
            "frequency_deg_per_hour": ctab.FREQ_DEG_PER_HOUR[name],
            "omega_rad_per_s": ctab.OMEGA_RAD_PER_SEC[name],
            **item,
        })
    for ex in excluded:
        entry = {
            "name": ex.name,
            "included": False,
            "frequency_deg_per_hour": ctab.FREQ_DEG_PER_HOUR[ex.name],
            "omega_rad_per_s": ctab.OMEGA_RAD_PER_SEC[ex.name],
            "reason": ex.reason,
        }
        if ex.reason == "rayleigh":
            entry.update({
                "masked_by": ex.masked_by,
                "beat_period_hours": ex.beat_period_hours,
                "record_hours": ex.record_hours,
                "deficit_hours": ex.deficit_hours,
                "deficit_days": ex.deficit_hours / ctab.HOURS_PER_DAY,
            })
        else:
            entry.update({
                "masked_by": None,
                "have_samples": ex.have_samples,
                "need_samples": ex.need_samples,
                "deficit_samples": max(0, ex.need_samples - ex.have_samples),
            })
        constituent_results.append(entry)

    # 固定输出顺序：纳入者按优先级，未纳入者附在后面也按优先级
    order = {n: i for i, n in enumerate(ctab.PRIORITY)}
    constituent_results.sort(
        key=lambda r: (0 if r["included"] else 1, order[r["name"]])
    )

    mean_speed = math.hypot(u0, v0)
    # 速度近零时方向角无物理意义，避免残差把 atan2 顶成 ±180°
    if mean_speed < 1e-10:
        mean_dir_rad = 0.0
    else:
        mean_dir_rad = math.atan2(v0, u0)
    return {
        "station": station["code"],
        "generated_at": to_iso(_time.time()),
        "record": {
            "first_time": to_iso(times[0]),
            "last_time": to_iso(times[-1]),
            "span_seconds": float(span_s),
            "span_hours": float(span_s / 3600.0),
            "n_samples": len(times),
        },
        "criterion": {
            "name": "conservative_rayleigh",
            "rayleigh_factor": ctab.config.RAYLEIGH_FACTOR,
            "rule": "门槛1：T >= R·360°/|f_a-f_b|（度/时），候选分潮须与全部"
                    "更高优先级候选分潮可分辨，否则被其中最不可分辨者遮住；"
                    "门槛2：N 点最多拟合 floor((N-1)/2) 个分潮，不足按"
                    "优先级从低到高剔除",
        },
        "mean_flow": {
            "u_cm_s": float(u0),
            "v_cm_s": float(v0),
            "speed_cm_s": float(mean_speed),
            "direction_rad": float(mean_dir_rad),
            "direction_deg": float(math.degrees(mean_dir_rad)),
        },
        "constituents": constituent_results,
    }


def _refit_and_archive(conn, station, batch_no: str, now: float) -> dict:
    times, u, v = _load_all_samples(conn, station["code"])
    result = _compute(station, times, u, v)
    result["batch_no"] = batch_no

    ver_row = conn.execute(
        "SELECT COALESCE(MAX(version_no), 0) + 1 AS next FROM versions "
        "WHERE station_id = ?",
        (station["code"],),
    ).fetchone()
    version_no = int(ver_row["next"])
    result["version_no"] = version_no

    conn.execute(
        "INSERT INTO versions(station_id, version_no, batch_no, archived_at, "
        "result_json) VALUES (?, ?, ?, ?, ?)",
        (station["code"], version_no, batch_no, now,
         json.dumps(result, ensure_ascii=False, sort_keys=True)),
    )
    return result


# ---------------------------------------------------------------- versions

def _load_version_row(conn: sqlite3.Connection, station_id: str,
                      version_id: Optional[int], version_no: Optional[int]):
    if version_id is not None:
        row = conn.execute(
            "SELECT * FROM versions WHERE station_id = ? AND id = ?",
            (station_id, version_id),
        ).fetchone()
        label = version_id
    else:
        row = conn.execute(
            "SELECT * FROM versions WHERE station_id = ? AND version_no = ?",
            (station_id, version_no),
        ).fetchone()
        label = version_no
    if row is None:
        raise NotFoundError("version",
                            f"测站 {station_id!r} 不存在版本 {label!r}")
    return row


def get_version(conn, station_id: str, version_id: int) -> dict:
    row = _load_version_row(conn, station_id, version_id=version_id,
                            version_no=None)
    result = json.loads(row["result_json"])
    result["archived_at"] = to_iso(row["archived_at"])
    return result


def get_version_by_no(conn, station_id: str, version_no: int) -> dict:
    get_station(conn, station_id)  # 站不存在先报 station
    row = _load_version_row(conn, station_id, version_id=None,
                            version_no=version_no)
    result = json.loads(row["result_json"])
    result["archived_at"] = to_iso(row["archived_at"])
    return result


def list_versions(conn: sqlite3.Connection, station_id: str) -> List[dict]:
    get_station(conn, station_id)
    rows = conn.execute(
        "SELECT version_no, batch_no, archived_at FROM versions "
        "WHERE station_id = ? ORDER BY version_no",
        (station_id,),
    ).fetchall()
    return [{
        "version_no": r["version_no"],
        "batch_no": r["batch_no"],
        "archived_at": to_iso(r["archived_at"]),
    } for r in rows]
