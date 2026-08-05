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
    PartitionConfig,
    bucket_bounds,
    bucket_expr_sql,
    check_dataset,
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


def test_missing_flat_meta_is_actionable(tmp_path: Path) -> None:
    empty = tmp_path / "flat"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="meta.json"):
        write_partitions(
            PartitionConfig(flat=empty, out=tmp_path / "out")
        )
