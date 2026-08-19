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


def test_build_report_dense_and_ratio_arithmetic() -> None:
    text = report.build_report(make_bench(), make_stats(), make_meta(), make_stats("token"))
    # dense = 1,000 tokens x 10 features x 2 B = 20,000 B, shown with its arithmetic
    assert "1,000 tokens x 10 features x 2 B (fp16) = **20,000 B**" in text
    # flat bytes recovered from flat_duckdb cost accounting = 10,000
    assert "| flat (token order) | 2 | - | 10,000 |" in text
    # feature layout: 8,000 B / 4,000 rows = 2.00 B/row.
    assert "| 2.00 | 2.5x | 0.80x |" in text
    # token layout: 6,000/4,000 = 1.50 B/row; dense/6,000 = 3.3x; 0.60x of flat
    assert "| 1.50 | 3.3x | 0.60x |" in text
    # killpoint line propagated
    assert "KILL-POINT synthetic: PASS" in text


def test_build_report_scaleup_extrapolation() -> None:
    text = report.build_report(
        make_bench(), make_stats(), make_meta(), make_stats("token")
    )
    assert "(EXTRAPOLATED)" in text
    # 4 rows/token x 1e12 tokens = 4e12 rows
    assert "| rows | 4.00e+12 | 4 rows/token x 1,000,000,000,000 tokens |" in text
    # feature layout: 2.0 B/row x 4e12 rows = 8e12 B = 8.0 TB
    assert "| feature-bucketed storage | 8.0 TB | 2.0000 B/row x 4.00e+12 rows |" in text
    # sort input: 4e12 rows x 12 B = 48 TB
    assert "| ingest sort input (uncompressed) | 48.0 TB |" in text
    # rows per feature: 4e12 / 10 = 4e11; row groups: 4e11 / 100 = 4e9
    assert "| rows per feature (mean) | 4.00e+11 |" in text
    assert "| row groups per feature (mean) | 4,000,000,000 |" in text


def test_latency_tables_have_all_methods_in_order() -> None:
    text = report.build_report(make_bench(), make_stats(), make_meta(), None)
    lines = text.splitlines()
    start = lines.index("### tokens_for_feature (cold)")
    rows = [l for l in lines[start : start + 10] if l.startswith("| ")]
    labels = [l.split("|")[1].strip() for l in rows if not l.startswith("| method")]
    assert labels == [
        "feature-bucketed",
        "token-bucketed scan",
        "flat scan (pyarrow)",
        "flat scan (DuckDB)",
    ]
    # DuckDB has no row-group accounting -> "-"
    duck = [l for l in rows if "flat scan (DuckDB)" in l][0]
    assert duck.rstrip().endswith("| - |")


def test_missing_token_stats_omits_rows_only() -> None:
    text = report.build_report(make_bench(), make_stats(), make_meta(), None)
    scale = text.split("## Query latency")[0]
    assert "| token-bucketed " not in scale  # no scale-table row
    assert "| token-bucketed storage |" not in text  # no scale-up row
    assert "| feature-bucketed |" in scale
    # the latency tables still show the measured token_bucketed method
    assert "### features_for_token (cold)" in text
    assert "| token-bucketed | 0.0100" in text


def test_report_is_ascii_and_deterministic() -> None:
    args = (make_bench(), make_stats(), make_meta(), make_stats("token"))
    text = report.build_report(*args)
    text.encode("ascii")  # raises if any non-ASCII sneaks in
    assert text == report.build_report(*args)


#


def test_row_count_mismatch_rejected() -> None:
    bad = make_stats()
    bad["total_rows"] = 4_001
    with pytest.raises(report.ReportError, match="row-count mismatch"):
        report.build_report(make_bench(), bad, make_meta(), None)


def test_wrong_layout_stats_rejected() -> None:
    with pytest.raises(report.ReportError, match="feature-bucketed stats"):
        report.build_report(make_bench(), make_stats("token"), make_meta(), None)
    with pytest.raises(report.ReportError, match="token-bucketed stats"):
        report.build_report(make_bench(), make_stats(), make_meta(), make_stats())


def test_flat_bytes_disagreement_rejected() -> None:
    bench = make_bench()
    bench["results"]["features_for_token"]["flat_duckdb"]["cold"]["agg"][
        "bytes_read_mean"
    ] = 9_999.0
    with pytest.raises(report.ReportError, match="disagrees across query shapes"):
        report.flat_bytes_from_bench(bench)


def test_cold_warning_when_cache_drop_none() -> None:
    bench = make_bench()
    bench["cache_drop_method"] = "none"
    text = report.build_report(bench, make_stats(), make_meta(), None)
    assert "NOT cold" in text


#


def test_main_writes_markdown_to_stdout(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    paths = _write_all(tmp_path)
    rc = report.main(
        [
            "--bench", paths["bench"],
            "--stats", paths["stats"],
            "--token-stats", paths["token_stats"],
            "--meta", paths["meta"],
        ]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert out.startswith("<!-- Generated by `python -m store.report`")
    assert "## Query latency" in out


def test_main_missing_token_stats_is_nonfatal(tmp_path: Path) -> None:
    paths = _write_all(tmp_path, token_stats=_MISSING)  # type: ignore[arg-type]
    rc = report.main(
        [
            "--bench", paths["bench"],
            "--stats", paths["stats"],
            "--token-stats", paths["token_stats"],
            "--meta", paths["meta"],
        ]
    )
    assert rc == 0


def test_main_missing_bench_fails_with_actionable_error(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    paths = _write_all(tmp_path, bench=_MISSING)  # type: ignore[arg-type]
    rc = report.main(
        [
            "--bench", paths["bench"],
            "--stats", paths["stats"],
            "--token-stats", paths["token_stats"],
            "--meta", paths["meta"],
        ]
    )
    assert rc == 1
    assert any("not found" in r.getMessage() for r in caplog.records)


def test_main_mismatched_artifacts_exit_nonzero(tmp_path: Path) -> None:
    bad = make_stats()
    bad["total_rows"] = 1
    paths = _write_all(tmp_path, stats=bad)
    rc = report.main(
        [
            "--bench", paths["bench"],
            "--stats", paths["stats"],
            "--token-stats", paths["token_stats"],
            "--meta", paths["meta"],
        ]
    )
    assert rc == 1


def test_target_tokens_flag(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    paths = _write_all(tmp_path)
    rc = report.main(
        [
            "--bench", paths["bench"],
            "--stats", paths["stats"],
            "--token-stats", paths["token_stats"],
            "--meta", paths["meta"],
            "--target-tokens", str(10**9),
        ]
    )
    out = capsys.readouterr().out
    assert rc == 0
    # 4 rows/token x 1e9 tokens = 4e9 rows
    assert "| rows | 4.00e+09 |" in out


#


def _real_paths() -> dict[str, Path]:
    return {
        "bench": REPO_ROOT / "results" / "bench.json",
        "stats": REPO_ROOT / "work" / "bucketed" / "stats.json",
        "token_stats": REPO_ROOT / "work" / "bucketed_by_token" / "stats.json",
        "meta": REPO_ROOT / "work" / "bucketed" / "meta.json",
    }


def test_readme_tables_match_regenerated_report() -> None:
    paths = _real_paths()
    if not all(p.is_file() for p in paths.values()):
        pytest.skip("measured work/ artifacts not present (fresh clone)")
    bench = json.loads(paths["bench"].read_text(encoding="utf-8"))
    stats = json.loads(paths["stats"].read_text(encoding="utf-8"))
    token_stats = json.loads(paths["token_stats"].read_text(encoding="utf-8"))
    meta = json.loads(paths["meta"].read_text(encoding="utf-8"))
    generated = report.build_report(bench, stats, meta, token_stats)
    readme_lines = set(
        (REPO_ROOT / "README.md").read_text(encoding="utf-8").splitlines()
    )
    missing = [
        line
        for line in generated.splitlines()
        if line.strip() and not line.startswith("<!--") and line not in readme_lines
    ]
    assert not missing, f"README.md has drifted from store.report output: {missing[:5]}"
    # spot-check the headline measured numbers made it through
    assert any("1,027,192,256" in l for l in readme_lines)
    assert any("23.1x" in l and "KILL-POINT" in l for l in readme_lines)
