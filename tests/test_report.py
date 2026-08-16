"""Tests for ``store.report`` ."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from store import report

REPO_ROOT = Path(__file__).resolve().parents[1]


#


def _agg(
    bytes_read: float,
    rows: float,
    files: float,
    row_groups: float | None,
) -> dict[str, Any]:
    return {
        "agg": {
            "n_samples": 4,
            "p50_s": 0.01,
            "p90_s": 0.02,
            "p99_s": 0.03,
            "mean_s": 0.015,
            "rows_mean": rows,
            "bytes_read_mean": bytes_read,
            "files_touched_mean": files,
            "row_groups_touched_mean": row_groups,
        },
        "samples": [],
    }


def _method(bytes_read: float, rows: float, files: float, rg: float | None) -> dict[str, Any]:
    return {
        "cold": _agg(bytes_read, rows, files, rg),
        "warm": _agg(bytes_read, rows, files, rg),
    }


def make_bench(flat_bytes: int = 10_000) -> dict[str, Any]:
    return {
        "format_version": 1,
        "created_at": "2026-08-15T00:00:00+00:00",
        "store": "work",
        "args": {"n_features": 2, "n_tokens": 2, "trials": 2, "seed": 0},
        "cache_drop_method": "fadvise",
        "env": {
            "python": "3.10.12",
            "duckdb": "1.5.5",
            "pyarrow": "25.0.1",
            "cpu_model": "TestCPU",
            "cpu_count": 4,
            "ram_bytes": 8 * 2**30,
        },
        "sampling": {"features": [], "tokens": []},
        "results": {
            "tokens_for_feature": {
                "bucketed": _method(64.0, 400.0, 1.0, 1.0),
                "flat_pyarrow": _method(float(flat_bytes), 400.0, 2.0, 4.0),
                "flat_duckdb": _method(float(flat_bytes), 400.0, 2.0, None),
                "token_bucketed_scan": _method(6_000.0, 400.0, 4.0, 6.0),
            },
            "features_for_token": {
                "token_bucketed": _method(48.0, 4.0, 1.0, 1.0),
                "feature_bucketed_scan": _method(8_000.0, 4.0, 4.0, 6.0),
                "flat_pyarrow": _method(1_000.0, 4.0, 1.0, 1.0),
                "flat_duckdb": _method(float(flat_bytes), 4.0, 2.0, None),
            },
        },
        "killpoint": {
            "passed": True,
            "threshold": 5.0,
            "message": "KILL-POINT synthetic: PASS",
        },
    }


def make_stats(layout: str = "feature", total_bytes: int = 8_000) -> dict[str, Any]:
    if layout == "feature":
        return {
            "format_version": 1,
            "layout": "feature",
            "key": "feature",
            "n_buckets": 4,
            "domain": 10,
            "total_rows": 4_000,
            "total_bytes": total_bytes,
            "n_features": 10,
            "n_files": 4,
            "n_row_groups": 8,
            "mean_row_groups_touched_per_feature": 1.0,
            "boundary_shared_row_groups": 0,
            "buckets": [],
        }
    return {
        "format_version": 1,
        "layout": "token",
        "key": "token_idx",
        "n_buckets": 4,
        "domain": 1_000,
        "total_rows": 4_000,
        "total_bytes": 6_000,
        "n_tokens": 1_000,
        "n_files": 4,
        "n_row_groups": 4,
        "mean_row_groups_touched_per_token": 1.0,
        "boundary_shared_row_groups": 0,
        "buckets": [],
    }


def make_meta() -> dict[str, Any]:
    return {
        "format_version": 1,
        "sae_checkpoint": {"path": "/x/results/frontier/fake_ckpt/checkpoint.pt", "sha256": "ab" * 32},
        "sae_config": {"n_features": 10, "activation": "topk", "k": 4},
        "hook_name": "blocks.8.hook_resid_post",
        "model_name": "gpt2",
        "shard": {"path": "/x/data/holdout.bin"},
        "n_tokens_encoded": 1_000,
        "n_rows": 4_000,
        "mean_l0": 4.0,
        "partition": {
            "layout": "feature",
            "n_buckets": 4,
            "row_group_size": 100,
            "bucket_expr": "(CAST(feature AS BIGINT) * 4) // 10",
        },
    }


def _write_all(tmp_path: Path, **overrides: dict[str, Any] | None) -> dict[str, str]:
    payloads: dict[str, Any] = {
        "bench": make_bench(),
        "stats": make_stats("feature"),
        "token_stats": make_stats("token"),
        "meta": make_meta(),
    }
    payloads.update({k: v for k, v in overrides.items() if v is not None})
    paths: dict[str, str] = {}
    for name, payload in payloads.items():
        p = tmp_path / f"{name}.json"
        if payload is not _MISSING:
            p.write_text(json.dumps(payload), encoding="utf-8")
        paths[name] = p.as_posix()
    return paths


_MISSING = object()


#


def test_flat_bytes_disagreement_rejected() -> None:
    bench = make_bench()
    bench["results"]["features_for_token"]["flat_duckdb"]["cold"]["agg"][
        "bytes_read_mean"
    ] = 9_999.0
    with pytest.raises(report.ReportError, match="disagrees across query shapes"):
        report.flat_bytes_from_bench(bench)


def _real_paths() -> dict[str, Path]:
    return {
        "bench": REPO_ROOT / "results" / "bench.json",
        "stats": REPO_ROOT / "work" / "bucketed" / "stats.json",
        "token_stats": REPO_ROOT / "work" / "bucketed_by_token" / "stats.json",
        "meta": REPO_ROOT / "work" / "bucketed" / "meta.json",
    }


