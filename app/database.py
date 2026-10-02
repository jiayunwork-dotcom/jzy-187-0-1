"""SQLite 持久化层。

存储模型（3NF 风格，全部时间戳以 Unix 秒 float64 落库）：

* ``stations``  测站：编号、经纬度、采样间隔、布放时间
* ``batches``   数据批次：唯一批次号、所属测站、到达顺序、时间覆盖范围
* ``samples``   样本点：(测站, 时刻) 唯一，缺测段直接不落行
* ``versions``  每次形成新结果后存档的一版分析结果（JSON）

样本按 (station_id, t) 唯一：既防止同批重复时刻，也防止跨批重复时刻。
批次号全局唯一。版本号 (station_id, version_no) 唯一，每批递增。
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from typing import Iterator

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS stations (
    code        TEXT PRIMARY KEY,
    latitude    REAL NOT NULL,
    longitude   REAL NOT NULL,
    sample_dt   REAL NOT NULL,            -- 采样间隔，秒
    deployed_at REAL NOT NULL,            -- 布放时间，Unix UTC 秒
    created_at  REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS batches (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    batch_no     TEXT NOT NULL UNIQUE,
    station_id   TEXT NOT NULL REFERENCES stations(code),
    arrival_seq  INTEGER NOT NULL,        -- 到达本站的先后序号
    received_at  REAL NOT NULL,
    t_start      REAL NOT NULL,
    t_end        REAL NOT NULL,
    n_samples    INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS samples (
    station_id TEXT NOT NULL REFERENCES stations(code),
    t          REAL NOT NULL,
    u          REAL NOT NULL,
    v          REAL NOT NULL,
    batch_no   TEXT NOT NULL REFERENCES batches(batch_no),
    PRIMARY KEY (station_id, t)
);
CREATE INDEX IF NOT EXISTS idx_samples_station_t
    ON samples(station_id, t);

CREATE TABLE IF NOT EXISTS versions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    station_id  TEXT NOT NULL REFERENCES stations(code),
    version_no  INTEGER NOT NULL,
    batch_no    TEXT NOT NULL,
    archived_at REAL NOT NULL,
    result_json TEXT NOT NULL,
    UNIQUE (station_id, version_no)
);
CREATE INDEX IF NOT EXISTS idx_versions_station
    ON versions(station_id, version_no);
"""


def connect() -> sqlite3.Connection:
    conn = sqlite3.connect(config.DB_PATH, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    return conn


@contextmanager
def get_conn() -> Iterator[sqlite3.Connection]:
    conn = connect()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    import os

    os.makedirs(config.DATA_DIR, exist_ok=True)
    with get_conn() as conn:
        conn.executescript(SCHEMA)
