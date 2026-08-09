"""Tests for store.queries: all methods agree on both canonical queries over a tiny
store (flat + both bucketed layouts)"""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pytest

from store import queries
from store.partition import (
    STATS_FILENAME,
    PartitionConfig,
    check_dataset,
    write_partitions,
)

from .conftest import make_flat

N_FEATURES = 64
N_TOKENS = 900
MEAN_L0 = 6
N_BUCKETS = 8
ROW_GROUP_SIZE = 2048  # DuckDB clamps to its 2048-row vector granularity


def build_tiny_store(root: Path, *, seed: int = 11) -> dict:
    """work-root shaped store: flat/ + bucketed/ + bucketed_by_token/ with bucket
    maps, stats.json, and forwarded meta.json, exactly as the real dump+partition
    pipeline leaves them."""
    meta = make_flat(
        root / queries.FLAT_SUBDIR,
        n_features=N_FEATURES,
        n_tokens=N_TOKENS,
        n_files=3,
        mean_l0=MEAN_L0,
        seed=seed,
    )
    for layout, sub in (
        ("feature", queries.BUCKETED_SUBDIR),
        ("token", queries.TOKEN_BUCKETED_SUBDIR),
    ):
        cfg = PartitionConfig(
            flat=root / queries.FLAT_SUBDIR,
            out=root / sub,
            layout=layout,
            n_buckets=N_BUCKETS,
            row_group_size=ROW_GROUP_SIZE,
            threads=2,
            memory_limit="1GB",
        )
        write_partitions(cfg)
        rep = check_dataset(
            cfg.out, threads=2, memory_limit="1GB", write_stats=True
        )
        assert rep.ok, rep.errors
    return meta


@pytest.fixture(scope="module")
def store(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("store")
    build_tiny_store(root)
    return root


def _feature_methods(root: Path, feature: int) -> dict[str, queries.QueryResult]:
    return {
        "bucketed": queries.tokens_for_feature(root, feature, "bucketed"),
        "token_bucketed_scan": queries.tokens_for_feature(
            root, feature, "token"
        ),
        "flat_pyarrow": queries.flat_scan_tokens_for_feature(root, feature),
        "flat_duckdb": queries.duckdb_scan_tokens_for_feature(root, feature),
    }


def _token_methods(root: Path, token: int) -> dict[str, queries.QueryResult]:
    return {
        "token_bucketed": queries.features_for_token(root, token, "token"),
        "feature_bucketed_scan": queries.features_for_token(
            root, token, "bucketed"
        ),
        "flat_pyarrow": queries.flat_scan_features_for_token(root, token),
        "flat_duckdb": queries.duckdb_scan_features_for_token(root, token),
    }


@pytest.mark.parametrize("feature", [0, 7, 31, N_FEATURES - 1])
def test_all_methods_agree_tokens_for_feature(store: Path, feature: int):
    results = _feature_methods(store, feature)
    queries.assert_same_rows({n: r.table for n, r in results.items()})
    ref = results["flat_duckdb"].table
    assert ref.num_rows > 0  # every feature fires in the tiny store
    assert set(results["bucketed"].table.column_names) == {
        "token_idx", "value",
    }


@pytest.mark.parametrize("token", [0, 123, N_TOKENS - 1])
def test_all_methods_agree_features_for_token(store: Path, token: int):
    results = _token_methods(store, token)
    queries.assert_same_rows({n: r.table for n, r in results.items()})
    ref = results["flat_duckdb"].table
    assert ref.num_rows == MEAN_L0  # every token has exactly mean_l0 rows
    assert set(results["token_bucketed"].table.column_names) == {
        "feature", "value",
    }


def test_assert_same_rows_catches_mismatch(store: Path):
    a = queries.tokens_for_feature(store, 3).table
    b = queries.tokens_for_feature(store, 4).table
    with pytest.raises(AssertionError, match="result mismatch"):
        queries.assert_same_rows({"a": a, "b": b})


def test_bucketed_reads_one_partition(store: Path):
    res = queries.tokens_for_feature(store, 9, "bucketed")
    # exactly the one file of feature 9's bucket.
    assert res.files_touched == 1
    flat = queries.flat_scan_tokens_for_feature(store, 9)
    assert res.bytes_read is not None and flat.bytes_read is not None
    assert res.bytes_read < flat.bytes_read
    assert flat.files_touched == 3  # flat has no useful feature pruning


def test_token_bucketed_reads_one_partition(store: Path):
    res = queries.features_for_token(store, 100, "token")
    assert res.files_touched == 1
    scan = queries.features_for_token(store, 100, "bucketed")
    assert scan.files_touched >= 1  # every feature bucket gets filtered


def test_result_unpacks_as_spec_tuple(store: Path):
    table, elapsed_s, bytes_read = queries.tokens_for_feature(store, 2)
    assert isinstance(table, pa.Table)
    assert elapsed_s >= 0.0
    assert isinstance(bytes_read, int)


def test_duckdb_reports_upper_bound_bytes(store: Path):
    res = queries.duckdb_scan_tokens_for_feature(store, 2)
    flat_total = sum(
        f.stat().st_size for f in (store / queries.FLAT_SUBDIR).glob(
            "rows-*.parquet"
        )
    )
    assert res.bytes_read == flat_total
    assert res.row_groups_touched is None  # DuckDB does not expose it


def test_bucket_for_key_bounds(store: Path):
    bmap = queries.load_bucket_map(store / queries.BUCKETED_SUBDIR)
    assert queries.bucket_for_key(0, bmap) == 0
    assert queries.bucket_for_key(N_FEATURES - 1, bmap) == N_BUCKETS - 1
    with pytest.raises(ValueError, match="outside the key domain"):
        queries.bucket_for_key(N_FEATURES, bmap)
    with pytest.raises(ValueError, match="outside the key domain"):
        queries.bucket_for_key(-1, bmap)


def test_missing_layout_dir_is_actionable(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="store.partition"):
        queries.tokens_for_feature(tmp_path, 0, "bucketed")
    with pytest.raises(ValueError, match="unknown layout"):
        queries.resolve_layout_dir(tmp_path, "nope")


def test_stats_json_present_for_bench(store: Path):
    # bench's sampling cross-checks against stats.json.
    stats = json.loads(
        (store / queries.BUCKETED_SUBDIR / STATS_FILENAME).read_text()
    )
    assert stats["n_features"] == N_FEATURES


def test_queries_cli_smoke(store: Path, capsys: pytest.CaptureFixture):
    rc = queries.main(
        [
            "--store", str(store), "--query", "tokens-for-feature",
            "--key", "5", "--limit", "3",
        ]
    )
    assert rc == 0
    assert "token_idx" in capsys.readouterr().out
