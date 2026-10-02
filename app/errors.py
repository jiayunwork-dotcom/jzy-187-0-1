"""自定义错误：在路由里翻译成对应的 HTTP 状态码。

* ValidationError  -> 422，detail 为 [{field, msg}, ...]，field 指到具体字段
* NotFoundError    -> 404
* ConflictError    -> 409（如测站编号已存在）
"""
from __future__ import annotations

from typing import Any, List


class ServiceError(Exception):
    status_code = 400

    def __init__(self, errors: Any):
        if isinstance(errors, str):
            errors = [{"field": "", "msg": errors}]
        self.errors = errors
        super().__init__(str(errors))


class ValidationError(ServiceError):
    status_code = 422

    def __init__(self, errors: List[dict] | str):
        super().__init__(errors)


class NotFoundError(ServiceError):
    status_code = 404

    def __init__(self, field: str, msg: str):
        super().__init__([{"field": field, "msg": msg}])


class ConflictError(ServiceError):
    status_code = 409

    def __init__(self, field: str, msg: str):
        super().__init__([{"field": field, "msg": msg}])
