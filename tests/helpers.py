"""测试辅助：独立于生产代码的合成数据生成与“理论真值”换算。

约定与服务端一致（这部分用复数重新实现一遍，当作独立 oracle，
不复用 app 里的换算函数，避免“用被测代码验证被测代码”）：

    y(t) = A cos(ω(t-t_dep) + g)

对 u、v 复振幅取 U = A_u e^{i g_u}、V = A_v e^{i g_v}，
旋转分量 W+ = (U + iV)/2（逆时针），W- = (U - iV)/2（顺时针）。
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import numpy as np

from app.constituents import FREQ_DEG_PER_HOUR

DT_HOURLY = 3600.0
EPOCH = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)


def omega(name: str) -> float:
    return math.radians(FREQ_DEG_PER_HOUR[name]) / 3600.0


def iso(t: datetime) -> str:
    return t.isoformat()


def make_station(client, code="S1", sample_interval=DT_HOURLY,
                 deployed_at=EPOCH, lat=30.0, lon=122.0):
    r = client.post("/stations", json={
        "code": code,
        "latitude": lat,
        "longitude": lon,
        "sample_interval": sample_interval,
        "deployed_at": iso(deployed_at),
    })
    assert r.status_code == 201, r.text
    return r.json()


def grid_times(n: int, dt: float = DT_HOURLY, start=EPOCH):
    return [start + timedelta(seconds=dt * k) for k in range(n)]


def component(t_abs, w, a, phase, t_dep_epoch=EPOCH):
    """A cos(ω(t - t_dep) + phase)。"""
    tt = np.array([(t - t_dep_epoch).total_seconds() for t in t_abs],
                  dtype=np.float64)
    return a * np.cos(w * tt + phase)


def signal(t_abs, consts, mean=(0.0, 0.0), t_dep_epoch=EPOCH):
    """consts: {name: (Au, gu, Av, gv)}；返回 u, v。"""
    u = np.full(len(t_abs), mean[0], dtype=np.float64)
    v = np.full(len(t_abs), mean[1], dtype=np.float64)
    for name, (au, gu, av, gv) in consts.items():
        w = omega(name)
        u += component(t_abs, w, au, gu, t_dep_epoch)
        v += component(t_abs, w, av, gv, t_dep_epoch)
    return u, v


def batch_payload(batch_no, station, t_abs, u, v):
    return {
        "batch_no": batch_no,
        "station": station,
        "samples": [
            {"t": iso(t), "u": float(uu), "v": float(vv)}
            for t, uu, vv in zip(t_abs, u, v)
        ],
    }


def upload(client, batch_no, station, t_abs, u, v):
    r = client.post("/batches",
                    json=batch_payload(batch_no, station, t_abs, u, v))
    assert r.status_code == 201, r.text
    return r.json()


def get_const(result, name):
    for c in result["constituents"]:
        if c["name"] == name:
            return c
    raise KeyError(name)


# ----------------------------------------------------- 独立的椭圆 oracle

def ellipse_oracle(au, gu, av, gv):
    """从 u、v 振幅相位（模型 y=A cos(x+g)）解析算椭圆要素。"""
    U = au * complex(math.cos(gu), math.sin(gu))
    V = av * complex(math.cos(gv), math.sin(gv))
    wp = (U + 1j * V) / 2.0
    wm = (U - 1j * V) / 2.0
    ap, am = abs(wp), abs(wm)
    major = ap + am
    minor = ap - am
    incl = ((math.atan2(wp.imag, wp.real) - math.atan2(wm.imag, wm.real))
            / 2.0) % math.pi
    phase = ((math.atan2(wp.imag, wp.real) + math.atan2(wm.imag, wm.real))
             / 2.0 + math.pi) % (2 * math.pi) - math.pi
    return {
        "major": major,
        "minor": minor,
        "incl": incl,
        "phase": phase,
        "u_amp": au, "u_phase": gu,
        "v_amp": av, "v_phase": gv,
    }


def wrap_pi(x):
    return (x + math.pi) % (2 * math.pi) - math.pi


def rel_err(a, b):
    denom = max(abs(a), abs(b), 1e-300)
    return abs(a - b) / denom
