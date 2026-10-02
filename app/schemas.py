"""HTTP 请求模型（Pydantic v2）。

时间统一用 ISO 8601 字符串，且必须带时区（不带时区的时刻无法可靠换算，
直接判错）。u、v 必须是有限实数，NaN/Inf 一律拒绝。
"""
from __future__ import annotations

import math
from typing import List

from pydantic import BaseModel, Field, field_validator


class StationIn(BaseModel):
    code: str = Field(min_length=1, description="测站编号")
    latitude: float = Field(ge=-90.0, le=90.0)
    longitude: float = Field(ge=-180.0, le=180.0)
    sample_interval: float = Field(gt=0.0, description="采样间隔，秒")
    deployed_at: str = Field(description="布放时间，ISO 8601 且必须带时区")


class SampleIn(BaseModel):
    t: str = Field(description="采样时刻，ISO 8601 且必须带时区")
    u: float
    v: float

    @field_validator("u", "v")
    @classmethod
    def _finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("必须是有限实数（不接受 NaN/Infinity）")
        return value


class BatchIn(BaseModel):
    batch_no: str = Field(min_length=1, description="批次唯一编号")
    station: str = Field(min_length=1)
    samples: List[SampleIn] = Field(min_length=1)
