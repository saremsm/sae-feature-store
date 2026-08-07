"""Tests for store.partition: tiny flat set from a fake dump, both layouts, both
code paths (single COPY and --per-bucket-passes), and the --check CLI."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import duckdb
import numpy as np
import pyarrow.parquet as pq
import pytest

from store import partition as part_mod
from store import schema
from store.partition import (
    BUCKET_MAP_FILENAME,
    STATS_FILENAME,
    PartitionConfig,
    bucket_bounds,
    bucket_expr_sql,
    check_dataset,
    main,
    write_partitions,
)

from .conftest import make_flat

N_FEATURES = 64
N_TOKENS = 1200
MEAN_L0 = 8
N_BUCKETS = 4
# DuckDB clamps Parquet ROW_GROUP_SIZE to its 2048-row vector granularity.
ROW_GROUP_SIZE = 2048


@pytest.fixture()
def flat(tmp_path: Path) -> Path:
    d = tmp_path / "flat"
    make_flat(
        d,
        n_features=N_FEATURES,
        n_tokens=N_TOKENS,
        n_files=3,
        mean_l0=MEAN_L0,
        seed=5,
    )
    return d


def _run(flat: Path, out: Path, *extra: str) -> int:
    argv = [
        "--flat", str(flat), "--out", str(out),
        "--n-buckets", str(N_BUCKETS),
        "--row-group-size", str(ROW_GROUP_SIZE),
        "--threads", "2", "--memory-limit", "1GB",
        *extra,
    ]
    return main(argv)


def _read_all(path_glob: str, order: str) -> dict[str, np.ndarray]:
    con = duckdb.connect()
    try:
        return con.execute(
            f"SELECT CAST(bucket AS BIGINT) AS bucket, token_idx, feature, "
            f"value FROM read_parquet('{path_glob}', hive_partitioning=1) "
            f"ORDER BY {order}"
        ).fetchnumpy()
    finally:
        con.close()


def _per_feature_counts(files_glob: str) -> dict[int, int]:
    con = duckdb.connect()
    try:
        rows = con.execute(
            f"SELECT feature, count(*) FROM read_parquet('{files_glob}') "
            f"GROUP BY 1"
        ).fetchall()
    finally:
        con.close()
    return {int(f): int(c) for f, c in rows}


#


@pytest.mark.parametrize(
    "n_buckets,domain",
    [(8, 64), (128, 6144), (7, 100), (100, 7), (1, 5), (13, 13)],
)
def test_bucket_bounds_cover_domain_once(n_buckets: int, domain: int) -> None:
    bounds = bucket_bounds(n_buckets, domain)
    assert len(bounds) == n_buckets
    # contiguous + disjoint + exact cover of [0, domain)
    assert bounds[0][0] == 0
    assert bounds[-1][1] == domain
    for (a_lo, a_hi), (b_lo, b_hi) in zip(bounds, bounds[1:]):
        assert a_lo <= a_hi == b_lo <= b_hi
    # formula assignment agrees with interval membership for every key
    for k in range(domain):
        b = k * n_buckets // domain
        lo, hi = bounds[b]
        assert lo <= k < hi


def test_bucket_bounds_rejects_bad_args() -> None:
    with pytest.raises(ValueError):
        bucket_bounds(0, 10)
    with pytest.raises(ValueError):
        bucket_bounds(4, 0)


#


def test_partition_roundtrip(flat: Path, tmp_path: Path) -> None:
    out = tmp_path / "bucketed"
    assert _run(flat, out) == 0

    meta_in = json.loads((flat / schema.META_FILENAME).read_text())
    meta_out = json.loads((out / schema.META_FILENAME).read_text())
    bmap = json.loads((out / BUCKET_MAP_FILENAME).read_text())
    stats = json.loads((out / STATS_FILENAME).read_text())

    # meta copied forward, partition params added, nothing lost
    for k, v in meta_in.items():
        assert meta_out[k] == v
    assert meta_out["partition"]["n_buckets"] == N_BUCKETS
    assert meta_out["partition"]["layout"] == "feature"
    assert meta_out["partition"]["source_rows"] == meta_in["n_rows"]

    # bucket_map covers all features exactly once
    assert bmap["domain"] == N_FEATURES
    assert [b["lo"] for b in bmap["buckets"]][0] == 0
    assert bmap["buckets"][-1]["hi"] == N_FEATURES
    assert bmap["bucket_expr"] == bucket_expr_sql(
        "feature", N_BUCKETS, N_FEATURES
    )

    # total rows unchanged; per-feature counts preserved vs the source
    data = _read_all(
        (out / "bucket=*" / "*.parquet").as_posix(),
        "feature, token_idx",
    )
    assert data["feature"].size == meta_in["n_rows"]
    src_counts = _per_feature_counts((flat / "rows-*.parquet").as_posix())
    feats, cnts = np.unique(data["feature"], return_counts=True)
    dst_counts = {int(f): int(c) for f, c in zip(feats, cnts)}
    assert dst_counts == src_counts

    # every feature's rows live in exactly one bucket, the mapped one
    for f, b in zip(data["feature"], data["bucket"]):
        assert int(b) == int(f) * N_BUCKETS // N_FEATURES

    # stats.json: per-bucket rows/features/min-max present and consistent
    assert stats["total_rows"] == meta_in["n_rows"]
    assert stats["n_features"] == len(src_counts)
    assert sum(b["rows"] for b in stats["buckets"]) == stats["total_rows"]
    assert all(b["bytes"] > 0 for b in stats["buckets"] if b["rows"])
    for b in stats["buckets"]:
        if b["rows"]:
            assert 1 <= b["min_rows_per_feature"] <= b["max_rows_per_feature"]
    assert stats["mean_row_groups_touched_per_feature"] >= 1.0


def test_partition_sorted_and_rowgroup_stats(
    flat: Path, tmp_path: Path
) -> None:
    out = tmp_path / "bucketed"
    assert _run(flat, out) == 0
    files = sorted(out.glob("bucket=*/*.parquet"))
    assert files
    saw_multi_group = False
    for f in files:
        pf = pq.ParquetFile(f)
        assert pf.schema_arrow.names == ["token_idx", "feature", "value"]
        md = pf.metadata
        saw_multi_group = saw_multi_group or md.num_row_groups > 1
        fi = pf.schema_arrow.names.index("feature")
        prev_max = None
        for g in range(md.num_row_groups):
            st = md.row_group(g).column(fi).statistics
            assert st is not None and st.has_min_max
            if prev_max is not None:
                # non-overlapping up to the shared boundary feature
                assert int(st.min) >= prev_max
            prev_max = int(st.max)
        tb = pf.read(columns=["feature", "token_idx"])
        key = (
            tb["feature"].to_numpy().astype(np.uint64) << np.uint64(32)
        ) | tb["token_idx"].to_numpy().astype(np.uint64)
        assert bool(np.all(key[1:] >= key[:-1])), f"{f} not sorted"
    assert saw_multi_group, "test should exercise multiple row groups"


def test_per_bucket_passes_identical(flat: Path, tmp_path: Path) -> None:
    out_a = tmp_path / "single"
    out_b = tmp_path / "passes"
    assert _run(flat, out_a) == 0
    assert _run(flat, out_b, "--per-bucket-passes") == 0

    dirs_a = sorted(d.name for d in out_a.glob("bucket=*"))
    dirs_b = sorted(d.name for d in out_b.glob("bucket=*"))
    assert dirs_a == dirs_b

    for d in dirs_a:
        # identical rows in identical order (sort key is unique per row)
        a = pq.read_table(sorted((out_a / d).glob("*.parquet")))
        b = pq.read_table(sorted((out_b / d).glob("*.parquet")))
        assert a.schema.names == b.schema.names
        for col in a.schema.names:
            np.testing.assert_array_equal(
                a[col].to_numpy(), b[col].to_numpy(), err_msg=f"{d}/{col}"
            )

    ja = json.loads((out_a / BUCKET_MAP_FILENAME).read_text())
    jb = json.loads((out_b / BUCKET_MAP_FILENAME).read_text())
    assert ja == jb


def test_rerun_clears_stale_partitions(flat: Path, tmp_path: Path) -> None:
    out = tmp_path / "bucketed"
    assert _run(flat, out) == 0
    stale = out / f"bucket={N_BUCKETS + 3}" / "data_0.parquet"
    stale.parent.mkdir()
    # a valid but wrongly-placed parquet file
    src = next(out.glob("bucket=*/*.parquet"))
    stale.write_bytes(src.read_bytes())
    assert check_dataset(out).ok is False  # stale dir breaks the dataset
    assert _run(flat, out) == 0  # rerun clears it
    assert not stale.parent.exists()
    assert check_dataset(out).ok is True


#


def test_token_layout_variant(flat: Path, tmp_path: Path) -> None:
    out = tmp_path / "bucketed_by_token"
    assert _run(flat, out, "--layout", "token") == 0

    bmap = json.loads((out / BUCKET_MAP_FILENAME).read_text())
    assert bmap["layout"] == "token"
    assert bmap["key"] == "token_idx"
    assert bmap["domain"] == N_TOKENS
    assert bmap["buckets"][-1]["hi"] == N_TOKENS

    data = _read_all(
        (out / "bucket=*" / "*.parquet").as_posix(),
        "token_idx, feature",
    )
    meta_in = json.loads((flat / schema.META_FILENAME).read_text())
    assert data["token_idx"].size == meta_in["n_rows"]
    # every token's rows live in exactly one bucket, the mapped one
    for t, b in zip(data["token_idx"], data["bucket"]):
        assert int(b) == int(t) * N_BUCKETS // N_TOKENS

    stats = json.loads((out / STATS_FILENAME).read_text())
    assert stats["n_tokens"] == N_TOKENS  # every token has rows (topk-like)
    for b in stats["buckets"]:
        if b["rows"]:
            # our fake dump gives every token exactly MEAN_L0 rows
            assert b["min_rows_per_token"] == MEAN_L0
            assert b["max_rows_per_token"] == MEAN_L0

    # files sorted by (token_idx, feature)
    for f in sorted(out.glob("bucket=*/*.parquet")):
        tb = pq.read_table(f, columns=["token_idx", "feature"])
        key = (
            tb["token_idx"].to_numpy().astype(np.uint64) << np.uint64(32)
        ) | tb["feature"].to_numpy().astype(np.uint64)
        assert bool(np.all(key[1:] >= key[:-1]))


def test_many_partitions_many_threads_stay_sorted(tmp_path: Path) -> None:
    """Regression: DuckDB 1.5.5's multi-threaded hive-partitioned COPY does not
    preserve the ORDER BY inside partition files."""
    flat = tmp_path / "flat"
    make_flat(flat, n_features=6144, n_tokens=800, n_files=2, mean_l0=32,
              seed=9)
    out = tmp_path / "bucketed"
    assert main([
        "--flat", str(flat), "--out", str(out),
        "--n-buckets", "128", "--row-group-size", "2048",
        "--threads", "4", "--memory-limit", "1GB",
    ]) == 0
    for f in sorted(out.glob("bucket=*/*.parquet")):
        assert f.name == "data_0.parquet"
        tb = pq.read_table(f, columns=["feature", "token_idx"])
        key = (
            tb["feature"].to_numpy().astype(np.uint64) << np.uint64(32)
        ) | tb["token_idx"].to_numpy().astype(np.uint64)
        assert bool(np.all(key[1:] >= key[:-1])), f"{f} not sorted"


def test_token_layout_per_bucket_passes_identical(
    flat: Path, tmp_path: Path
) -> None:
    out_a = tmp_path / "tok_single"
    out_b = tmp_path / "tok_passes"
    assert _run(flat, out_a, "--layout", "token") == 0
    assert _run(flat, out_b, "--layout", "token", "--per-bucket-passes") == 0
    assert sorted(d.name for d in out_a.glob("bucket=*")) == sorted(
        d.name for d in out_b.glob("bucket=*")
    )
    for d in sorted(d.name for d in out_a.glob("bucket=*")):
        a = pq.read_table(sorted((out_a / d).glob("*.parquet")))
        b = pq.read_table(sorted((out_b / d).glob("*.parquet")))
        for col in a.schema.names:
            np.testing.assert_array_equal(
                a[col].to_numpy(), b[col].to_numpy(), err_msg=f"{d}/{col}"
            )


#


def test_check_cli_passes_and_detects_corruption(
    flat: Path, tmp_path: Path
) -> None:
    out = tmp_path / "bucketed"
    assert _run(flat, out) == 0
    assert main(["--check", str(out)]) == 0

    # corrupt one partition file: reverse its rows (breaks sort order)
    victim = sorted(out.glob("bucket=*/*.parquet"))[0]
    tb = pq.read_table(victim)
    rev = tb.take(np.arange(tb.num_rows - 1, -1, -1))
    victim.unlink()
    pq.write_table(rev, victim, row_group_size=ROW_GROUP_SIZE)
    rep = check_dataset(out)
    assert not rep.ok
    assert any("not sorted" in e for e in rep.errors)
    assert main(["--check", str(out)]) == 1


def test_check_detects_row_loss_and_misplaced_rows(
    flat: Path, tmp_path: Path
) -> None:
    out = tmp_path / "bucketed"
    assert _run(flat, out) == 0

    # move a file into the wrong bucket dir -> rows misplaced + one-bucket
    files = sorted(out.glob("bucket=*/*.parquet"))
    a = files[0]
    wrong_dir = out / f"bucket={N_BUCKETS - 1}"
    moved = wrong_dir / "smuggled.parquet"
    a.replace(moved)
    rep = check_dataset(out)
    assert not rep.ok
    assert any("does not match the bucket expression" in e for e in rep.errors)
    moved.replace(a)

    # delete a file -> total row count changes
    b = files[1]
    b.unlink()
    rep = check_dataset(out)
    assert not rep.ok
    assert any("total rows changed" in e for e in rep.errors)


def test_check_requires_dataset_files(tmp_path: Path) -> None:
    rep = check_dataset(tmp_path / "nope")
    assert not rep.ok
    assert main(["--check", str(tmp_path / "nope")]) == 1


#


def test_module_cli_subprocess(flat: Path, tmp_path: Path) -> None:
    out = tmp_path / "bucketed"
    repo_root = Path(__file__).resolve().parents[1]
    cmd = [
        sys.executable, "-m", "store.partition",
        "--flat", str(flat), "--out", str(out),
        "--n-buckets", str(N_BUCKETS),
        "--row-group-size", str(ROW_GROUP_SIZE),
        "--threads", "2", "--memory-limit", "1GB",
    ]
    r = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    r2 = subprocess.run(
        [sys.executable, "-m", "store.partition", "--check", str(out)],
        cwd=repo_root, capture_output=True, text=True,
    )
    assert r2.returncode == 0, r2.stderr
    assert (out / STATS_FILENAME).exists()
    assert (out / BUCKET_MAP_FILENAME).exists()


def test_missing_flat_meta_is_actionable(tmp_path: Path) -> None:
    empty = tmp_path / "flat"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="meta.json"):
        write_partitions(
            PartitionConfig(flat=empty, out=tmp_path / "out")
        )
