"""Tests for store.bench: end-to-end run over a tiny store writes both outputs, the
kill-point gate fires and still writes them."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from store import bench, queries

from .test_queries import N_FEATURES, N_TOKENS, build_tiny_store


@pytest.fixture(scope="module")
def store(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("bench_store")
    build_tiny_store(root, seed=23)
    return root


def _run(store: Path, out: Path, *extra: str) -> int:
    return bench.main(
        [
            "--store", str(store),
            "--n-features", "3", "--n-tokens", "3", "--trials", "1",
            "--out", str(out), "--seed", "0",
            *extra,
        ]
    )


def test_sampling_is_stratified_by_tercile(store: Path):
    picked = bench.sample_features(
        store / queries.BUCKETED_SUBDIR, 6, seed=1
    )
    assert len(picked) == 6
    terciles = [p["tercile"] for p in picked]
    assert set(terciles) == set(bench.TERCILES)
    assert all(t in bench.TERCILES for t in terciles)
    # bottom-tercile picks really are rarer than top-tercile picks
    bot = min(p["rows"] for p in picked if p["tercile"] == "bottom")
    top = max(p["rows"] for p in picked if p["tercile"] == "top")
    assert bot <= top
    # deterministic under the seed
    assert picked == bench.sample_features(
        store / queries.BUCKETED_SUBDIR, 6, seed=1
    )


def test_sampling_cross_checks_stats(store: Path, tmp_path: Path):
    # a stale stats.json must be caught, not silently used
    src = store / queries.BUCKETED_SUBDIR
    stats = json.loads((src / "stats.json").read_text())
    stats["total_rows"] += 1
    bad = tmp_path / "bucketed_bad"
    bad.mkdir()
    for f in src.glob("bucket=*/*.parquet"):
        dst = bad / f.parent.name
        dst.mkdir(exist_ok=True)
        dst.joinpath(f.name).write_bytes(f.read_bytes())
    (bad / "stats.json").write_text(json.dumps(stats))
    with pytest.raises(AssertionError, match="stale stats.json"):
        bench.sample_features(bad, 3, seed=0)


def test_sample_tokens_uniform_and_seeded(store: Path):
    meta = json.loads(
        (store / queries.BUCKETED_SUBDIR / "meta.json").read_text()
    )
    toks = bench.sample_tokens(meta, 5, seed=3)
    assert len(toks) == len(set(toks)) == 5
    assert all(0 <= t < N_TOKENS for t in toks)
    assert toks == bench.sample_tokens(meta, 5, seed=3)


@pytest.mark.skipif(
    not hasattr(os, "posix_fadvise"),
    reason="posix_fadvise unavailable on this platform",
)
def test_fadvise_fallback_path_executes(store: Path):
    files = bench.store_files(store)
    assert files
    assert bench.drop_caches(files, method="fadvise") == "fadvise"


def test_drop_caches_auto_without_root(
    store: Path, monkeypatch: pytest.MonkeyPatch
):
    # force the root-only global drop to fail.
    monkeypatch.setattr(bench, "_drop_caches_global", lambda: False)
    method = bench.drop_caches(bench.store_files(store), method="auto")
    expected = "fadvise" if hasattr(os, "posix_fadvise") else "none"
    assert method == expected


def test_drop_caches_none_is_noop(store: Path):
    assert bench.drop_caches([], method="none") == "none"


def test_drop_caches_pinned_method_errors_when_unavailable(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(bench, "_drop_caches_global", lambda: False)
    with pytest.raises(PermissionError, match="requires root"):
        bench.drop_caches([], method="drop_caches")


def test_aggregate_percentiles():
    samples = [
        bench.Sample(
            key=0, trial=i, elapsed_s=float(i + 1), rows=10,
            bytes_read=100, files_touched=1, row_groups_touched=None,
        )
        for i in range(4)
    ]
    agg = bench.aggregate(samples)
    assert agg["n_samples"] == 4
    assert agg["p50_s"] == pytest.approx(np.percentile([1, 2, 3, 4], 50))
    assert agg["mean_s"] == pytest.approx(2.5)
    assert agg["bytes_read_mean"] == pytest.approx(100.0)
    assert agg["row_groups_touched_mean"] is None


def test_method_matrix_requires_feature_layout(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="store.partition"):
        bench.method_matrix(tmp_path)


def test_method_matrix_token_layout_optional(store: Path, tmp_path: Path):
    # copy only flat + bucketed; the token-layout methods must be skipped
    partial = tmp_path / "partial"
    for sub in (queries.FLAT_SUBDIR, queries.BUCKETED_SUBDIR):
        for f in (store / sub).rglob("*"):
            if f.is_file():
                dst = partial / f.relative_to(store)
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_bytes(f.read_bytes())
    matrix = bench.method_matrix(partial)
    assert "token_bucketed_scan" not in matrix["tokens_for_feature"]
    assert "token_bucketed" not in matrix["features_for_token"]
    assert "bucketed" in matrix["tokens_for_feature"]
