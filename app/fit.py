"""潮流调和最小二乘拟合与潮流椭圆换算（核心计算全部手写）。

只允许 NumPy 做底层线性代数：用 NumPy 组装并求解正规方程组
``X^T X beta = X^T y``，最小二乘逻辑（模型、矩阵、系数换算）不使用
``np.linalg.lstsq`` 或任何潮汐库；椭圆换算也在这里亲手实现。

调和模型（每个纳入分潮两根基函数，平均流一根常数项）::

    u(t) = u0 + Σ_c [a_u,c cos(ωc(t-t0)) + b_u,c sin(ωc(t-t0))]
    v(t) = v0 + Σ_c [a_v,c cos(ωc(t-t0)) + b_v,c sin(ωc(t-t0))]

相位约定（在结果中固定并写明）：时间原点取**测站布放时刻 t_dep**，即
写成 ``y = A cos(ω(t-t_dep) + g)`` 时，g 为输出相位，范围 (-π, π]。
拟合时为了数值稳定以样本时间重心 t0 为原点，解完后用解析旋转
``Q_dep = Q_fit · exp(-iω(t0-t_dep))`` 精确换回布放时刻相位，浮点误差
仅 O(eps)。

复系数定义（对实信号 y = a cos x + b sin x）::

    Q = a - i b = A e^{i g_abs},   A = |Q|, g_abs = atan2(-b, a)

潮流椭圆（u、v 复系数记 U、V）::

    W+ = (U + iV)/2  逆时针旋转分量，振幅 A+ = |W+|
    W- = (U - iV)/2  顺时针旋转分量，振幅 A- = |W-|
    长半轴 MAJ = A+ + A-
    带符号短半轴 MIN = A+ - A-   （>0 逆时针，<0 顺时针）
    方向角 θ = ((arg W+) - (arg W-))/2，模 π 到 [0, π)
    位相   g = ((arg W+) + (arg W-))/2，模 2π 折到 (-π, π]

方向角为数学约定：从东（+u 轴）起逆时针为正，取值 [0, π)。
若把流速矢量随参考系一起整体旋转 α（主动旋转，u'=cosα·u-sinα·v，
v'=sinα·u+cosα·v），长短半轴与位相 g 不变，方向角正好转过 α（模 π）。
"""
from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import numpy as np


def wrap_pi(angle: float) -> float:
    """折到 (-π, π]。"""
    w = (angle + math.pi) % (2.0 * math.pi) - math.pi
    if w == -math.pi:
        w = math.pi
    return w


def _complex_coeff(a: float, b: float) -> complex:
    """实拟合系数 a cos x + b sin x -> 复系数 a - i b。"""
    return complex(a, -b)


def fit_harmonics(
    times: Sequence[float],
    u: Sequence[float],
    v: Sequence[float],
    omegas: Sequence[float],
    t_dep: float,
) -> Tuple[float, float, List[dict]]:
    """对给定分潮做最小二乘调和拟合。

    参数
    ----
    times, u, v : 等间隔网格上的观测（允许缺测，直接只给有观测的点），
                  u/v 单位 cm/s，times 为 Unix UTC 秒。
    omegas      : 纳入拟合的分潮角频率（rad/s），不含平均流（自动加）。
    t_dep       : 布放时刻，相位以此为时间原点。

    返回
    ----
    u0, v0, [每个分潮一条 {u,v,ellipse} 结果，顺序同 omegas]
    """
    t = np.asarray(times, dtype=np.float64)
    yu = np.asarray(u, dtype=np.float64)
    yv = np.asarray(v, dtype=np.float64)
    n = t.size
    # select_constituents 已保证 n >= 1 + 2·k（必要时按优先级剔除分潮），
    # 这里只做防御性兜底。
    if n < 1 + 2 * len(omegas):
        raise ValueError("样本点数少于待估参数数，无法拟合")

    # 以样本时间重心为拟合原点，保证自变量有界、数值稳定。
    t0 = float(t.mean())
    tau = t - t0

    # 组装设计矩阵：第 0 列常数项，其后每个分潮两列 cos / sin。
    ncols = 1 + 2 * len(omegas)
    x = np.empty((n, ncols), dtype=np.float64)
    x[:, 0] = 1.0
    for j, w in enumerate(omegas):
        arg = w * tau
        x[:, 1 + 2 * j] = np.cos(arg)
        x[:, 2 + 2 * j] = np.sin(arg)

    # 手写最小二乘：解正规方程 X^T X β = X^T y（u、v 同时解）。
    gram = x.T @ x
    rhs = x.T @ np.column_stack((yu, yv))
    try:
        beta = np.linalg.solve(gram, rhs)
    except np.linalg.LinAlgError as exc:
        raise ValueError(f"正规方程奇异，分潮可能未能分开：{exc}") from exc

    u0 = float(beta[0, 0])
    v0 = float(beta[0, 1])

    shift = t0 - t_dep  # 拟合原点相对布放时刻的偏移（秒）
    results: List[dict] = []
    for j, w in enumerate(omegas):
        au = float(beta[1 + 2 * j, 0])
        bu = float(beta[2 + 2 * j, 0])
        av = float(beta[1 + 2 * j, 1])
        bv = float(beta[2 + 2 * j, 1])

        # 复系数（拟合原点）-> 解析旋转到布放时刻：
        # Q_dep = Q_fit · exp(-iω(t0-t_dep))
        rot = complex(math.cos(-w * shift), math.sin(-w * shift))
        uu = _complex_coeff(au, bu) * rot
        vv = _complex_coeff(av, bv) * rot

        u_amp, u_phase = abs(uu), wrap_pi(math.atan2(uu.imag, uu.real))
        v_amp, v_phase = abs(vv), wrap_pi(math.atan2(vv.imag, vv.real))

        wp = (uu + 1j * vv) / 2.0
        wm = (uu - 1j * vv) / 2.0
        ap, am = abs(wp), abs(wm)
        major = ap + am
        minor = ap - am  # 逆时针为正
        # 长轴方向恒为 (φ+ - φ-)/2，与哪个旋转分量占优无关；
        # 交换二者反而会把 θ 变成 π-θ，破坏“坐标旋转 α 角、方向角加 α”。
        # 正圆（ap=0 或 am=0）没有方向角，arg 为 0 属正常的退化情形。
        phi_p = math.atan2(wp.imag, wp.real)
        phi_m = math.atan2(wm.imag, wm.real)
        incl = wrap_pi((phi_p - phi_m) / 2.0) % math.pi
        phase = wrap_pi((phi_p + phi_m) / 2.0)

        results.append({
            "u": {"amplitude_cm_s": float(u_amp), "phase_rad": float(u_phase)},
            "v": {"amplitude_cm_s": float(v_amp), "phase_rad": float(v_phase)},
            "ellipse": {
                "semi_major_axis_cm_s": float(major),
                "signed_semi_minor_axis_cm_s": float(minor),
                "inclination_rad": float(incl),
                "inclination_deg": float(math.degrees(incl)),
                "phase_rad": float(phase),
            },
        })

    return u0, v0, results
