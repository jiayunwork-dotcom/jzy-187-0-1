"""最小二乘拟合与椭圆换算的数值关系测试（直接测核心函数）。"""
from __future__ import annotations

import math

import numpy as np

from app.constituents import FREQ_DEG_PER_HOUR
from app.fit import fit_harmonics, wrap_pi

from .helpers import (EPOCH, component, ellipse_oracle, grid_times, omega,
                      signal)


def _run(t_abs, consts, mean=(0.0, 0.0), names=None, t_dep=EPOCH):
    u, v = signal(t_abs, consts, mean=mean, t_dep_epoch=t_dep)
    times = np.array([(t - t_dep).total_seconds() for t in t_abs]) + \
        t_dep.timestamp()
    om = [omega(n) for n in (names or list(consts.keys()))]
    return fit_harmonics(times, u, v, om, t_dep=t_dep.timestamp())


def test_pure_m2_recovery():
    # 单频、无噪声，30 天逐时数据；只拟 M2
    au, gu = 80.0, 0.7
    av, gv = 45.0, 2.1
    t_abs = grid_times(30 * 24)
    u0, v0, res = _run(
        t_abs, {"M2": (au, gu, av, gv)}, names=["M2"]
    )
    r = res[0]
    assert abs(u0) < 1e-9 and abs(v0) < 1e-9
    assert abs(r["u"]["amplitude_cm_s"] - au) < 1e-6
    assert abs(r["v"]["amplitude_cm_s"] - av) < 1e-6
    assert abs(wrap_pi(r["u"]["phase_rad"] - gu)) < 1e-7
    assert abs(wrap_pi(r["v"]["phase_rad"] - gv)) < 1e-7

    truth = ellipse_oracle(au, gu, av, gv)
    e = r["ellipse"]
    assert abs(e["semi_major_axis_cm_s"] - truth["major"]) < 1e-6
    assert abs(e["signed_semi_minor_axis_cm_s"] - truth["minor"]) < 1e-6
    assert abs(e["inclination_rad"] - truth["incl"]) < 1e-7
    assert abs(wrap_pi(e["phase_rad"] - truth["phase"])) < 1e-7


def test_ccw_circle_minor_equals_major_positive():
    # 纯逆时针圆（东-北系）：t=0 流速向东，之后向北转 →
    # u=A cos x，v=A sin x = A cos(x-π/2)，v 相位滞后 u 90°
    t_abs = grid_times(30 * 24)
    u0, v0, res = _run(t_abs, {"M2": (50.0, 0.0, 50.0, -math.pi / 2)},
                       names=["M2"])
    e = res[0]["ellipse"]
    assert abs(e["semi_major_axis_cm_s"] - 50.0) < 1e-8
    assert abs(e["signed_semi_minor_axis_cm_s"] - 50.0) < 1e-8


def test_cw_circle_minor_equals_negative_major():
    # 顺时针圆：t=0 向东，之后向南转 → v=-A sin x = A cos(x+π/2)
    t_abs = grid_times(30 * 24)
    _, _, res = _run(t_abs, {"M2": (50.0, 0.0, 50.0, math.pi / 2)},
                     names=["M2"])
    e = res[0]["ellipse"]
    assert abs(e["semi_major_axis_cm_s"] - 50.0) < 1e-8
    assert abs(e["signed_semi_minor_axis_cm_s"] + 50.0) < 1e-8


def test_coordinate_rotation_axes_invariant_angle_rotates():
    # 一般椭圆参数；把数据随坐标系整体主动旋转 α：
    # u'=cosα u - sinα v，v'=sinα u + cosα v
    # （等价于把 (u,v) 向量逆时针转 α；若只是换用被动旋转的新坐标轴，
    # 方向角符号相反地变 -α——这里取主动旋转，期望 θ 正好增加 α）。
    au, gu, av, gv = 60.0, 0.5, 35.0, 1.9
    alpha = math.radians(37.0)
    truth = ellipse_oracle(au, gu, av, gv)

    t_abs = grid_times(40 * 24)
    u, v = signal(t_abs, {"M2": (au, gu, av, gv)})
    c, s = math.cos(alpha), math.sin(alpha)
    up = c * u - s * v
    vp = s * u + c * v
    times = np.array([(t - EPOCH).total_seconds() for t in t_abs]) + \
        EPOCH.timestamp()
    _, _, out0 = fit_harmonics(times, u, v, [omega("M2")],
                               t_dep=EPOCH.timestamp())
    _, _, out1 = fit_harmonics(times, up, vp, [omega("M2")],
                               t_dep=EPOCH.timestamp())
    e0, e1 = out0[0]["ellipse"], out1[0]["ellipse"]

    # 长短半轴不变
    assert abs(e1["semi_major_axis_cm_s"] - e0["semi_major_axis_cm_s"]) < 1e-8
    assert abs(e1["signed_semi_minor_axis_cm_s"]
               - e0["signed_semi_minor_axis_cm_s"]) < 1e-8
    # 方向角正好转过 α（mod π）
    assert abs((e1["inclination_rad"] - e0["inclination_rad"] - alpha + math.pi)
               % (2 * math.pi) - math.pi) < 1e-9
    # 与解析真值一致
    assert abs(e0["inclination_rad"] - truth["incl"]) < 1e-9


def test_constant_offset_changes_only_mean_flow():
    au, gu, av, gv = 55.0, 1.2, 30.0, 2.4
    cu, cv = 12.3, -7.5
    t_abs = grid_times(30 * 24)
    u0a, v0a, ra = _run(t_abs, {"M2": (au, gu, av, gv)}, names=["M2"])
    u0b, v0b, rb = _run(t_abs, {"M2": (au, gu, av, gv)},
                        mean=(cu, cv), names=["M2"])

    assert abs(u0b - cu) < 1e-9 and abs(v0b - cv) < 1e-9
    assert abs(u0a) < 1e-9 and abs(v0a) < 1e-9
    for ka, kb in zip(ra, rb):
        for key in ("u", "v"):
            assert abs(ka[key]["amplitude_cm_s"] - kb[key]["amplitude_cm_s"]) < 1e-10
            assert abs(wrap_pi(ka[key]["phase_rad"] - kb[key]["phase_rad"])) < 1e-12
        for key in ("semi_major_axis_cm_s", "signed_semi_minor_axis_cm_s",
                    "inclination_rad", "phase_rad"):
            assert abs(ka["ellipse"][key] - kb["ellipse"][key]) < 1e-10


def test_missing_segments_dont_break_recovery():
    # 中间挖掉两段缺测，纯 M2 仍应精确恢复
    au, gu, av, gv = 40.0, 0.3, 60.0, 2.9
    t_abs = grid_times(30 * 24)
    keep = np.ones(len(t_abs), dtype=bool)
    keep[100:150] = False
    keep[400:430] = False
    t_use = [t for t, k in zip(t_abs, keep) if k]
    u, v = signal(t_use, {"M2": (au, gu, av, gv)})
    times = np.array([(t - EPOCH).total_seconds() for t in t_use]) + \
        EPOCH.timestamp()
    _, _, res = fit_harmonics(times, u, v, [omega("M2")],
                              t_dep=EPOCH.timestamp())
    r = res[0]
    assert abs(r["u"]["amplitude_cm_s"] - au) < 1e-7
    assert abs(wrap_pi(r["u"]["phase_rad"] - gu)) < 1e-9
