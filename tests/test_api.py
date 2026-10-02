"""端到端 API 测试：更新等价性、乱序、重复、持久化、字段级报错。"""
from __future__ import annotations

from datetime import timedelta

from .helpers import (EPOCH, get_const, grid_times, make_station, signal,
                      upload, wrap_pi)


def _float_equal(a, b, *, rel=1e-11, abs_=1e-11):
    """递归比较 JSON：float 用容差，其余要求完全相等。"""
    if isinstance(a, dict):
        assert set(a) == set(b), (a.keys(), b.keys())
        for k in a:
            _float_equal(a[k], b[k], rel=rel, abs_=abs_)
    elif isinstance(a, list):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            _float_equal(x, y, rel=rel, abs_=abs_)
    elif isinstance(a, float) or isinstance(b, float):
        assert abs(a - b) <= max(abs_, rel * max(abs(a), abs(b))), (a, b)
    else:
        assert a == b, (a, b)


def _strip_meta(result):
    """比对时去掉与“怎么送”有关的版本元信息，只留分析结果本体。"""
    return {
        "mean_flow": result["mean_flow"],
        "constituents": result["constituents"],
        "record": result["record"],
    }


# ------------------------------------------------- 等价性：分批/乱序/重复

def test_incremental_matches_one_shot(client):
    make_station(client)
    n = 20 * 24
    t_abs = grid_times(n)
    u, v = signal(t_abs, {"M2": (60.0, 0.9, 35.0, 2.2),
                          "S2": (20.0, 1.5, 12.0, 0.3),
                          "K1": (15.0, 0.4, 10.0, 2.0)})

    # 一次性送入
    one = upload(client, "B-all", "S1", t_abs, u, v)

    # 另一站：分成 5 批，覆盖同样的数据
    make_station(client, code="S2")
    cuts = list(range(0, n, n // 5)) + [n]
    last = None
    for i in range(5):
        a, b = cuts[i], cuts[i + 1]
        last = upload(client, f"B{i}", "S2", t_abs[a:b], u[a:b], v[a:b])

    assert last["version_no"] == 5
    _float_equal(_strip_meta(last), _strip_meta(one))

    # 显式重算接口也一致
    r = client.get("/stations/S2/analysis")
    assert r.status_code == 200
    _float_equal(_strip_meta(r.json()), _strip_meta(one))

    # 需求门槛：振幅相对差 ≤ 1e-9，相位差 ≤ 1e-7
    by_name = {c["name"]: c for c in last["constituents"] if c["included"]}
    for c in (c for c in one["constituents"] if c["included"]):
        if c["name"] == "Z0":
            continue
        d = by_name[c["name"]]
        for comp in ("u", "v"):
            amp_ref = c[comp]["amplitude_cm_s"]
            if amp_ref > 1e-8:
                rel = abs(d[comp]["amplitude_cm_s"] - amp_ref) / amp_ref
                assert rel <= 1e-9, (c["name"], comp, rel)
            assert abs(wrap_pi(d[comp]["phase_rad"] - c[comp]["phase_rad"])) <= 1e-7


def test_out_of_order_arrival_gives_same_final_result(client):
    make_station(client, code="ORD")
    n = 18 * 24
    t_abs = grid_times(n)
    u, v = signal(t_abs, {"M2": (50.0, 0.2, 25.0, 1.1),
                          "S2": (18.0, 2.0, 10.0, 0.8),
                          "K1": (12.0, 1.0, 8.0, 2.6)})
    cuts = [0, 100, 200, 300, n]
    ordered = None
    for i in range(4):
        a, b = cuts[i], cuts[i + 1]
        ordered = upload(client, f"O{i}", "ORD", t_abs[a:b], u[a:b], v[a:b])

    make_station(client, code="REV")
    for i in [3, 1, 0, 2]:
        a, b = cuts[i], cuts[i + 1]
        rev = upload(client, f"R{i}", "REV", t_abs[a:b], u[a:b], v[a:b])

    _float_equal(_strip_meta(rev), _strip_meta(ordered))


def test_duplicate_batch_upload_is_idempotent(client):
    make_station(client, code="DUP")
    t_abs = grid_times(24 * 15)
    u, v = signal(t_abs, {"M2": (44.0, 0.6, 30.0, 1.7)})
    first = upload(client, "D1", "DUP", t_abs, u, v)
    versions_after_first = client.get("/stations/DUP/versions").json()
    assert len(versions_after_first) == 1

    again = client.post("/batches", json={
        "batch_no": "D1", "station": "DUP",
        "samples": [{"t": t.isoformat(), "u": float(uu), "v": float(vv)}
                    for t, uu, vv in zip(t_abs, u, v)],
    })
    assert again.status_code == 201
    body = again.json()
    assert body["duplicate"] is True
    _float_equal(_strip_meta(body), _strip_meta(first))
    assert len(client.get("/stations/DUP/versions").json()) == 1


def test_versions_archived_and_replayable(client):
    make_station(client, code="ARC")
    t_abs = grid_times(40 * 24)
    u, v = signal(t_abs, {"M2": (50.0, 0.4, 20.0, 2.2),
                          "S2": (15.0, 0.9, 9.0, 0.1)})
    half = 20 * 24
    v1 = upload(client, "A1", "ARC", t_abs[:half], u[:half], v[:half])
    v2 = upload(client, "A2", "ARC", t_abs[half:], u[half:], v[half:])
    assert v1["version_no"] == 1 and v2["version_no"] == 2

    listing = client.get("/stations/ARC/versions").json()
    assert [x["version_no"] for x in listing] == [1, 2]
    assert [x["batch_no"] for x in listing] == ["A1", "A2"]

    r1 = client.get("/stations/ARC/versions/1").json()
    assert r1["record"]["n_samples"] == half
    r2 = client.get("/stations/ARC/versions/2").json()
    assert r2["record"]["n_samples"] == 40 * 24
    # 任何一版都能回查，且第 1 版不被后来的版本覆盖
    assert r1["version_no"] == 1
    assert r1["batch_no"] == "A1"


def test_persistence_across_restart(client, data_dir):
    make_station(client, code="PERSIST")
    t_abs = grid_times(15 * 24)
    u, v = signal(t_abs, {"M2": (48.0, 1.1, 26.0, 0.5)})
    before = upload(client, "P1", "PERSIST", t_abs, u, v)

    # 模拟服务重启：新 TestClient 重新跑 startup，但库文件还在挂载卷上
    from fastapi.testclient import TestClient
    from app.main import app

    with TestClient(app) as c2:
        r = c2.get("/stations/PERSIST/versions/1")
        assert r.status_code == 200
        _float_equal(_strip_meta(r.json()), _strip_meta(before))
        rl = c2.get("/stations/PERSIST/versions")
        assert len(rl.json()) == 1


# ------------------------------------------------- 端到端物理关系

def test_short_record_s2_and_k2_not_both_fit(client):
    make_station(client, code="SHORT")
    # 30 天：S2 可纳入，K2 必须注明被 S2 遮住
    t_abs = grid_times(30 * 24)
    u, v = signal(t_abs, {"M2": (50.0, 0.3, 30.0, 2.0),
                          "S2": (20.0, 0.9, 12.0, 1.4),
                          "K2": (8.0, 1.2, 5.0, 0.3)})
    res = upload(client, "S", "SHORT", t_abs, u, v)
    s2 = get_const(res, "S2")
    k2 = get_const(res, "K2")
    assert s2["included"] is True
    assert k2["included"] is False
    assert k2["masked_by"] == "S2"
    assert k2["deficit_days"] > 100.0


# --------------------------------------------------------- 字段级 422 报错

def _post_batch(client, **over):
    payload = {
        "batch_no": "X", "station": "E", "samples": [
            {"t": (EPOCH + timedelta(seconds=3600 * i)).isoformat(),
             "u": 1.0, "v": 0.0} for i in range(5)
        ],
    }
    payload.update(over)
    return client.post("/batches", json=payload)


def test_error_station_missing(client):
    r = _post_batch(client)
    assert r.status_code == 404
    assert r.json()["detail"][0]["field"] == "station"


def test_error_nonfinite_velocity(client):
    make_station(client, code="E")
    payload = {
        "batch_no": "X", "station": "E",
        "samples": [
            {"t": EPOCH.isoformat(), "u": 1.0, "v": 0.0},
            {"t": (EPOCH + timedelta(hours=1)).isoformat(),
             "u": "NaN", "v": 0.0},
        ],
    }
    r = client.post("/batches", json=payload)
    assert r.status_code == 422
    locs = [tuple(e["loc"]) for e in r.json()["detail"]]
    assert any(loc == ("body", "samples", 1, "u") for loc in locs)


def test_error_duplicate_time_within_batch(client):
    make_station(client, code="E")
    h = timedelta(hours=1)
    r = client.post("/batches", json={
        "batch_no": "DUPW", "station": "E",
        "samples": [
            {"t": EPOCH.isoformat(), "u": 1.0, "v": 0.0},
            {"t": (EPOCH + h).isoformat(), "u": 1.0, "v": 0.0},
            {"t": (EPOCH + h).isoformat(), "u": 2.0, "v": 0.0},
        ],
    })
    assert r.status_code == 422
    err = r.json()["detail"][0]
    assert err["field"] == "samples[2].t"
    assert "重复" in err["msg"]


def test_error_duplicate_time_across_batches(client):
    make_station(client, code="E")
    h = timedelta(hours=1)
    times = [EPOCH + i * h for i in range(4)]
    upload(client, "H1", "E", times[:2], [1.0, 1.0], [0.0, 0.0])
    r = client.post("/batches", json={
        "batch_no": "H2", "station": "E",
        "samples": [
            {"t": times[1].isoformat(), "u": 9.0, "v": 0.0},
            {"t": times[3].isoformat(), "u": 9.0, "v": 0.0},
        ],
    })
    assert r.status_code == 422
    err = r.json()["detail"][0]
    assert err["field"] == "samples[0].t"
    assert "H1" in err["msg"]


def test_error_unequal_or_mismatched_interval(client):
    make_station(client, code="E", sample_interval=3600.0)
    r = client.post("/batches", json={
        "batch_no": "G", "station": "E",
        "samples": [
            {"t": EPOCH.isoformat(), "u": 1.0, "v": 0.0},
            {"t": (EPOCH + timedelta(minutes=70))
             .isoformat(), "u": 1.0, "v": 0.0},
        ],
    })
    assert r.status_code == 422
    err = r.json()["detail"][0]
    assert err["field"] == "samples[1].t"
    assert "整数倍" in err["msg"]


def test_error_off_grid_even_if_spacing_matches(client):
    # 两批各自间隔都是 3600s，但第二批整体错位半个步长：必须拒绝
    make_station(client, code="GRID", sample_interval=3600.0)
    t1 = [EPOCH, EPOCH + timedelta(hours=1)]
    upload(client, "G1", "GRID", t1, [1.0, 1.0], [0.0, 0.0])
    r = client.post("/batches", json={
        "batch_no": "G2", "station": "GRID",
        "samples": [
            {"t": (EPOCH + timedelta(hours=2, minutes=30)).isoformat(),
             "u": 1.0, "v": 0.0},
            {"t": (EPOCH + timedelta(hours=3, minutes=30)).isoformat(),
             "u": 1.0, "v": 0.0},
        ],
    })
    assert r.status_code == 422
    assert r.json()["detail"][0]["field"] == "samples[0].t"


def test_gap_middle_is_allowed_and_analyzed(client):
    # 中间缺掉一天：相邻跨 2 个步长，合法且 M2 仍能精确恢复
    make_station(client, code="GAP")
    t_all = grid_times(20 * 24)
    u_all, v_all = signal(t_all, {"M2": (40.0, 0.8, 25.0, 2.1)})
    part1 = list(range(5 * 24))
    part2 = list(range(6 * 24, 20 * 24))  # 第 5～6 天整体缺测
    idx = part1 + part2
    res = upload(client, "G", "GAP", [t_all[i] for i in idx],
                 u_all[idx], v_all[idx])
    m2 = get_const(res, "M2")
    assert m2["included"] is True
    assert abs(m2["u"]["amplitude_cm_s"] - 40.0) < 1e-6
    assert abs(wrap_pi(m2["u"]["phase_rad"] - 0.8)) < 1e-7


def test_sparse_samples_excluded_for_insufficient_points(client):
    # 跨度 30 天但只有 3 个点：最多拟合 1 个分潮，S2 须注明点数不足
    make_station(client, code="SPARSE")
    ts = [EPOCH, EPOCH + timedelta(days=10), EPOCH + timedelta(days=30)]
    r = client.post("/batches", json={
        "batch_no": "SP", "station": "SPARSE",
        "samples": [{"t": t.isoformat(), "u": 1.0, "v": 0.0} for t in ts],
    })
    assert r.status_code == 201, r.text
    res = r.json()
    s2 = get_const(res, "S2")
    assert s2["included"] is False
    assert s2["reason"] == "insufficient_samples"
    assert s2["have_samples"] == 3
    assert s2["need_samples"] > 3


def test_error_infinite_velocity(client):
    make_station(client, code="INF")
    r = client.post("/batches", json={
        "batch_no": "I", "station": "INF",
        "samples": [
            {"t": EPOCH.isoformat(), "u": "Infinity", "v": 0.0},
        ],
    })
    assert r.status_code == 422
    locs = [tuple(e["loc"]) for e in r.json()["detail"]]
    assert any(loc == ("body", "samples", 0, "u") for loc in locs)


def test_error_batch_before_deployment(client):
    make_station(client, code="LATE",
                 deployed_at=EPOCH + timedelta(days=1))
    r = client.post("/batches", json={
        "batch_no": "EARLY", "station": "LATE",
        "samples": [
            {"t": EPOCH.isoformat(), "u": 1.0, "v": 0.0},
        ],
    })
    assert r.status_code == 422
    err = r.json()["detail"][0]
    assert err["field"] == "samples[0].t"
    assert "布放" in err["msg"]


def test_error_naive_timestamp_rejected(client):
    make_station(client, code="TZ")
    r = client.post("/batches", json={
        "batch_no": "N", "station": "TZ",
        "samples": [{"t": "2025-01-01T00:00:00", "u": 1.0, "v": 0.0}],
    })
    assert r.status_code == 422
    # parse_iso 的字段化错误（而非 pydantic）
    err = r.json()["detail"][0]
    assert err["field"] == "samples[0].t"
    assert "时区" in err["msg"]
