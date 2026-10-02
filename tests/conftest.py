"""pytest 全局夹具：把数据目录指到临时目录，再拉起 TestClient。"""
from __future__ import annotations

import os
import sys
import tempfile

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

_TMP = tempfile.mkdtemp(prefix="current-test-")
os.environ["CURRENT_DATA_DIR"] = _TMP


@pytest.fixture()
def client():
    # 每次测试前清库，保证互不干扰；库文件本身留在临时卷上。
    from app import database

    if os.path.exists(database.config.DB_PATH):
        os.remove(database.config.DB_PATH)
    for suffix in ("-wal", "-shm"):
        p = database.config.DB_PATH + suffix
        if os.path.exists(p):
            os.remove(p)

    from app.main import app

    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="session")
def data_dir():
    return _TMP
