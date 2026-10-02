"""FastAPI 应用：只对外提供 HTTP/JSON 接口。

路由
----
POST   /stations                 建测站
GET    /stations                 列测站
GET    /stations/{code}          查测站
POST   /batches                  上传一批数据（幂等：批次号唯一），返回本批后新版结果
GET    /stations/{code}/versions                版本清单（任何一版都可回查）
GET    /stations/{code}/versions/{version_no}   按版本号回查某一版
GET    /stations/{code}/analysis                不新增数据，重算当前结果（不存档）
GET    /health                   存活探针
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse

from . import service
from .database import get_conn, init_db
from .errors import ServiceError
from .schemas import BatchIn, StationIn


@asynccontextmanager
async def lifespan(_: FastAPI):
    # 启动即建表；SQLite 文件在挂载卷上，重启后历史版本原样可读
    init_db()
    yield


app = FastAPI(title="坐底式海流计潮流调和分析服务", version="1.0.0",
              lifespan=lifespan)


@app.exception_handler(ServiceError)
async def _service_error_handler(_: Request, exc: ServiceError):
    return JSONResponse(status_code=exc.status_code,
                        content={"detail": exc.errors})


def db():
    with get_conn() as conn:
        yield conn


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/stations", status_code=201)
def create_station(data: StationIn, conn=Depends(db)):
    return service.create_station(conn, data)


@app.get("/stations")
def list_stations(conn=Depends(db)):
    return service.list_stations(conn)


@app.get("/stations/{code}")
def get_station(code: str, conn=Depends(db)):
    return service.get_station(conn, code)


@app.post("/batches", status_code=201)
def upload_batch(data: BatchIn, conn=Depends(db)):
    return service.upload_batch(conn, data)


@app.get("/stations/{code}/versions")
def list_versions(code: str, conn=Depends(db)):
    return service.list_versions(conn, code)


@app.get("/stations/{code}/versions/{version_no}")
def get_version(code: str, version_no: int, conn=Depends(db)):
    return service.get_version_by_no(conn, code, version_no)


@app.get("/stations/{code}/analysis")
def get_analysis(code: str, conn=Depends(db)):
    return service.run_analysis(conn, code)
