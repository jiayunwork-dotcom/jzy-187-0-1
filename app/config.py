"""运行期配置。

数据目录（SQLite 文件所在位置）由环境变量 ``CURRENT_DATA_DIR`` 指定，
默认是容器内挂载点 ``/data``。所有持久化状态只有一个文件
``harmonics.db``，放在挂载卷上，容器重建后各版结果仍在。
"""
from __future__ import annotations

import os

DATA_DIR = os.environ.get("CURRENT_DATA_DIR", "/data")
DB_PATH = os.path.join(DATA_DIR, "harmonics.db")

# 瑞利判据因子 R：两个频率为 f_a、f_b 的分潮要被分开，
# 需要记录长度 >= R / |f_a - f_b|。R 可由环境变量覆盖。
RAYLEIGH_FACTOR = float(os.environ.get("CURRENT_RAYLEIGH_FACTOR", "1.0"))
