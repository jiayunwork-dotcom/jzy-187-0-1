# 坐底式海流计潮流调和分析服务

近海坐底式海流计（ADCP/海流计）观测数据的后端服务：测站建档、批次数据
上传（允许乱序、重复、缺测）、每批自动做潮流调和最小二乘分析并**按批次
存档**，任何历史版本都能回查。只对外提供 HTTP/JSON 接口，数据与结果落
SQLite（放在挂载卷上）。

* Python 3.11（固定）、FastAPI、NumPy
* 最小二乘拟合、分潮可分辨判别、潮流椭圆换算均为手写；**不使用任何现成
  潮汐分析库**（不依赖 T_TIDE/utide/pytides 等），NumPy 仅用于组装矩阵
  与求解线性方程组（`np.linalg.solve`，不用 `lstsq`）
* 容器基于 `python:3.11-slim`

---

## 1. 数据模型

| 对象 | 字段 |
|---|---|
| 测站 station | 编号 `code`、纬度、经度、采样间隔（秒）、布放时间（ISO 8601 带时区） |
| 批次 batch | 全局唯一批次号 `batch_no`、所属测站、等间隔时刻上的 (u, v)（cm/s） |
| 样本 sample | (测站, 时刻) 唯一；缺测段就是没有行 |
| 版本 version | 每接收一个**新**批次存档一版完整分析结果（JSON），版本号站内递增 |

时间一律 Unix UTC 秒落库；HTTP 层用带时区的 ISO 8601，缺时区直接判错。

## 2. 候选分潮与可分辨判别（写入代码 `app/constituents.py`）

候选：**M2、S2、N2、K2、K1、O1** 加平均流 Z0。频率写死（度/平太阳时）：

| | M2 | S2 | N2 | K2 | K1 | O1 |
|---|---|---|---|---|---|---|
| 度/时 | 28.9841042 | 30.0 | 28.4397295 | 30.0821373 | 15.0410686 | 13.9430356 |

记录长度 T = 末观测时刻 − 首观测时刻（与中间缺测段无关；缺测只降低估计
稳定性，不改变频率分辨力）。

**门槛 1——保守版瑞利判据（Rayleigh criterion）**

分潮 c 纳入的条件是对**所有先验优先级更高的候选分潮**都有

```
T ≥ R · 360° / |f_c − f_j|,    R = 1（可用环境变量 CURRENT_RAYLEIGH_FACTOR 调）
```

即至少覆盖一个会合周期。固定先验优先级 **M2 > S2 > N2 > K1 > O1 > K2**
（近岸海域能量排序的常用选择）。要求与所有更高优先级*候选*（而非仅已
纳入者）可分辨，是为了防止链式遮蔽：例如短记录里 S2 没进，不能因此让与
它只差 0.082 度/时的 K2 混进来。被遮住的分潮在结果里给出
`masked_by`、会合周期 `beat_period_hours` 与**还差多长**
（`deficit_hours` / `deficit_days`）。

关键会合周期：

| 分潮对 | 临界记录长 |
|---|---|
| S2–K2 | 182.6 天（短记录绝不会把 S2、K2 一起拟合） |
| M2–N2 | 27.6 天 |
| M2–S2 | 14.8 天 |
| K1–O1 | 13.7 天 |

**门槛 2——自由度**

平均流 1 个系数、每个分潮 cos/sin 2 个系数，N 个点最多纳入
`floor((N−1)/2)` 个分潮；跨度够但点太稀时，按优先级从低到高剔除，结果
里以 `reason="insufficient_samples"` 注明 `have_samples` /
`need_samples` / `deficit_samples`。

## 3. 拟合与椭圆换算（`app/fit.py`）

每个分潮 c 两根基函数，加平均流常数项：

```
u(t) = u0 + Σ_c [a_u,c cos ωc(t−t0) + b_u,c sin ωc(t−t0)]
v(t) = v0 + Σ_c [a_v,c cos ωc(t−t0) + b_v,c sin ωc(t−t0)]
```

组装设计矩阵 X 后手写正规方程 `XᵀX β = Xᵀy`（u、v 同时解）。拟合时以样本
时间重心为原点保证数值稳定，解完用一次解析复数旋转把相位精确换到
**布放时刻**为时间原点（浮点误差仅 O(eps)）：

```
y = A cos(ω(t − t_dep) + g)
Q = a − i b = A·e^{ig}（g ∈ (−π, π]）
```

潮流椭圆（U、V 为 u、v 的复系数）：

```
W+ = (U + iV)/2   逆时针旋转分量        W− = (U − iV)/2   顺时针旋转分量
长半轴 MAJ = |W+| + |W−|
带符号短半轴 MIN = |W+| − |W−|   （>0 逆时针，<0 顺时针）
方向角 θ = (arg W+ − arg W−)/2 ∈ [0, π)，从东向北为正
位相   g = (arg W+ + arg W−)/2 ∈ (−π, π]
```

流速矢量随参考系整体旋转 α 时，长/短半轴与位相不变，方向角正好加 α
（模 π）。

## 4. 更新策略：全量重拟合（理由与代价）

每到一个**新**批次，把该站**全部历史样本**（按时刻排序）取出，重新做一次
完全相同的拟合并存一版。重复批次号直接幂等返回首版结果，不重算、不新增
版本。

* **为什么选它**：天然与“一次性拿全部数据从头拟合”**逐位相同**——同样的
  点、同样的排序、同样的正规方程与浮点运算顺序，结果不依赖增量累积量的
  数值等价性；实现简单、可审计、可随时复现任意一版（版本本身就是冻结
  JSON）。
* **代价**：每批计算量随累计样本数线性增长（O(N·k²)，k 为纳入分潮数），
  每版多存一份结果 JSON。对“几天一批、单站逐时、数台仪器”的运维规模可
  忽略。若将来 N 很大，可另增“只累积 XᵀX、Xᵀy”的增量实现，但必须解决
  时间重心/相位原点不一致带来的数值偏差，才能满足下面的容差——当前刻意
  不引入该复杂度。

等价性指标（pytest 中显式断言）：振幅相对差 ≤ **1e-9**，相位差 ≤
**1e-7**；乱序到达的最终结果与按时间顺序送入一致。

## 5. 输入校验（错误全部定位到具体字段，HTTP 422）

* 采样时刻无法解析 / 不带时区 → `samples[i].t`
* 时刻早于测站布放时间 → `samples[i].t`
* 同一时刻批内重复、或与历史批次重复 → `samples[i].t`
* 不等间隔、或不是测站设定采样间隔的整数倍、或批次整体不在以布放时刻为
  原点的采样网格上（允许缺测跨步）→ `samples[i].t`
* u / v 为 NaN/Infinity（非有限数）→ `samples[i].u` / `.v`
* 测站不存在 → 404，字段 `station`
* 批次号重复 → **不报错**，幂等返回原版本（响应带 `duplicate: true`）

## 6. HTTP 接口

| 方法 路径 | 说明 |
|---|---|
| `POST /stations` | 建测站（编号重复 409） |
| `GET /stations` / `GET /stations/{code}` | 列/查测站 |
| `POST /batches` | 上传一批；返回该批形成的最新版分析结果 |
| `GET /stations/{code}/versions` | 版本清单 |
| `GET /stations/{code}/versions/{n}` | 回查第 n 版（任何一版都在） |
| `GET /stations/{code}/analysis` | 不落档地重算当前结果 |
| `GET /health` | 存活探针 |

请求示例：

```jsonc
// POST /stations
{"code": "A1", "latitude": 30.0, "longitude": 122.5,
 "sample_interval": 3600, "deployed_at": "2025-01-01T00:00:00+00:00"}

// POST /batches
{"batch_no": "A1-20250105", "station": "A1",
 "samples": [{"t": "2025-01-01T00:00:00+00:00", "u": 12.3, "v": -4.1},
             {"t": "2025-01-01T01:00:00+00:00", "u": 11.8, "v": -3.9}]}
```

结果中每个纳入分潮含 `u/v` 的振幅（cm/s）与相位（rad），以及
`ellipse`：`semi_major_axis_cm_s`、`signed_semi_minor_axis_cm_s`、
`inclination_rad/deg`、`phase_rad`；未纳入分潮 `included=false` 并给出
`reason` / `masked_by` / 缺口。另有 `mean_flow`（平均流 u/v、流速大小与
方向）与本版 `record`（首末时刻、跨度、样本数）。

## 7. 运行与测试

```bash
# 本地
pip install -r requirements.txt
CURRENT_DATA_DIR=./data uvicorn app.main:app --host 0.0.0.0 --port 8000

# 测试
python -m pytest          # 29 个用例，覆盖所有上述物理关系与校验

# 容器（只暴露 8000，SQLite 在挂载卷 /data）
docker build -t current-harmonics .
docker run -p 8000:8000 -v $(pwd)/data:/data current-harmonics
```

环境变量：`CURRENT_DATA_DIR`（SQLite 目录，默认 `/data`）、
`CURRENT_RAYLEIGH_FACTOR`（瑞利判据因子，默认 1）。
