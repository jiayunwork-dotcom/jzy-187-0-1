"""时间换算工具：ISO 8601（必须带时区）<-> Unix UTC 秒。"""
from __future__ import annotations

from datetime import datetime, timezone

from .errors import ValidationError


def parse_iso(value: str, field: str) -> float:
    """把带时区的 ISO 8601 字符串解析为 Unix UTC 秒。

    不带时区、格式非法都在 field 指向的位置报 422。
    """
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        raise ValidationError([{"field": field,
                                "msg": f"时间格式无法解析：{value!r}，需要 ISO 8601"}])
    if dt.tzinfo is None:
        raise ValidationError([{"field": field,
                                "msg": f"时间 {value!r} 缺少时区信息（必须带时区）"}])
    return dt.astimezone(timezone.utc).timestamp()


def to_iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
