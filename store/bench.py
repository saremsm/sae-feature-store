"""Benchmark the two canonical queries under a cold/warm protocol. Without root the
fallback is ``os.posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED)`` on every file in
the store; on platforms without fadvise (Windows dev box) nothing is dropped."""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import duckdb
import numpy as np
import pyarrow as pa

from . import queries, schema
from .partition import STATS_FILENAME, _sql_path

log = logging.getLogger("store.bench")

TERCILES = ("bottom", "middle", "top")
QUERY_TOKENS_FOR_FEATURE = "tokens_for_feature"
QUERY_FEATURES_FOR_TOKEN = "features_for_token"

DEFAULT_KILL_THRESHOLD = 5.0


#


def store_files(store: Path) -> list[Path]:
    """Every parquet file the benchmark can possibly touch."""
    out: list[Path] = []
    for sub in (queries.FLAT_SUBDIR, queries.BUCKETED_SUBDIR,
                queries.TOKEN_BUCKETED_SUBDIR):
        d = store / sub
        if d.is_dir():
            out.extend(sorted(d.rglob("*.parquet")))
    return out


#


def feature_counts(bucketed: Path) -> tuple[np.ndarray, np.ndarray]:
    """(features, rows_per_feature) measured over the bucketed set with one DuckDB
    pass on the ``feature`` column."""
    glob = f"{_sql_path(bucketed)}/bucket=*/*.parquet"
    con = duckdb.connect()
    try:
        res = con.execute(
            f"SELECT CAST(feature AS BIGINT) AS f, count(*) AS c "
            f"FROM read_parquet('{glob}') GROUP BY 1 ORDER BY 1"
        ).fetchnumpy()
    finally:
        con.close()
    return res["f"].astype(np.int64), res["c"].astype(np.int64)


def sample_tokens(meta: dict[str, Any], n: int, seed: int) -> list[int]:
    """Uniform token_idx sample over [0, n_tokens_encoded)."""
    n_tokens = int(meta["n_tokens_encoded"])
    rng = np.random.default_rng(seed + 1)
    take = min(n, n_tokens)
    return sorted(
        int(t) for t in rng.choice(n_tokens, size=take, replace=False)
    )


#


def _read_proc(path: str) -> str | None:
    try:
        return Path(path).read_text()
    except OSError:
        return None


def capture_env() -> dict[str, Any]:
    env: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "duckdb": duckdb.__version__,
        "pyarrow": pa.__version__,
        "numpy": np.__version__,
        "cpu_model": platform.processor() or None,
        "cpu_count": os.cpu_count(),
        "ram_bytes": None,
        "lsblk": None,
    }
    cpuinfo = _read_proc("/proc/cpuinfo")
    if cpuinfo:
        for line in cpuinfo.splitlines():
            if line.lower().startswith("model name"):
                env["cpu_model"] = line.split(":", 1)[1].strip()
                break
    meminfo = _read_proc("/proc/meminfo")
    if meminfo:
        for line in meminfo.splitlines():
            if line.startswith("MemTotal:"):
                env["ram_bytes"] = int(line.split()[1]) * 1024
                break
    if shutil.which("lsblk"):
        try:
            env["lsblk"] = subprocess.run(
                ["lsblk", "-d", "-o", "NAME,ROTA,MODEL"],
                capture_output=True, text=True, timeout=10, check=False,
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError) as exc:
            env["lsblk"] = f"unavailable: {exc}"
    return env


#

MethodFn = Callable[[Path, int], queries.QueryResult]


def method_matrix(store: Path) -> dict[str, dict[str, MethodFn]]:
    """query type -> {method name -> fn(store, key)}. Bucketed-layout methods appear
    only if their dataset exists on disk."""
    have_feature = (
        store / queries.BUCKETED_SUBDIR / queries.BUCKET_MAP_FILENAME
    ).exists()
    have_token = (
        store / queries.TOKEN_BUCKETED_SUBDIR / queries.BUCKET_MAP_FILENAME
    ).exists()
    if not have_feature:
        raise FileNotFoundError(
            f"no bucketed dataset under {store / queries.BUCKETED_SUBDIR} - "
            f"run `python -m store.partition` first"
        )
    if not have_token:
        log.warning(
            "no token-bucketed dataset under %s; the token-layout methods "
            "are skipped (run `python -m store.partition --layout token`)",
            store / queries.TOKEN_BUCKETED_SUBDIR,
        )
    tff: dict[str, MethodFn] = {
        "bucketed": lambda s, k: queries.tokens_for_feature(s, k, "bucketed"),
        "flat_pyarrow": queries.flat_scan_tokens_for_feature,
        "flat_duckdb": queries.duckdb_scan_tokens_for_feature,
    }
    fft: dict[str, MethodFn] = {
        "feature_bucketed_scan": lambda s, k: queries.features_for_token(
            s, k, "bucketed"
        ),
        "flat_pyarrow": queries.flat_scan_features_for_token,
        "flat_duckdb": queries.duckdb_scan_features_for_token,
    }
    if have_token:
        tff["token_bucketed_scan"] = lambda s, k: queries.tokens_for_feature(
            s, k, "token"
        )
        fft["token_bucketed"] = lambda s, k: queries.features_for_token(
            s, k, "token"
        )
    return {QUERY_TOKENS_FOR_FEATURE: tff, QUERY_FEATURES_FOR_TOKEN: fft}


def cross_check(
    store: Path,
    methods: dict[str, MethodFn],
    keys: list[int],
    qtype: str,
) -> None:
    """Assert every method returns the identical row set for every key."""
    for key in keys:
        tables = {name: fn(store, key).table for name, fn in methods.items()}
        try:
            queries.assert_same_rows(tables)
        except AssertionError as exc:
            raise AssertionError(f"{qtype} key={key}: {exc}") from exc
    log.info(
        "cross-check OK: %s, %d keys x %d methods agree",
        qtype, len(keys), len(methods),
    )


@dataclass
class Sample:
    key: int
    trial: int
    elapsed_s: float
    rows: int
    bytes_read: int | None
    files_touched: int | None
    row_groups_touched: int | None


def run_matrix(
    store: Path,
    matrix: dict[str, dict[str, MethodFn]],
    keys_by_query: dict[str, list[int]],
    trials: int,
    cache_drop: str,
    evict_files: list[Path],
) -> tuple[dict[str, dict[str, dict[str, list[Sample]]]], str]:
    """cold+warm samples for every query type x method, plus the cache-drop method
    actually used."""
    out: dict[str, dict[str, dict[str, list[Sample]]]] = {
        q: {m: {"cold": [], "warm": []} for m in methods}
        for q, methods in matrix.items()
    }
    used = "none"
    total = sum(
        len(keys_by_query[q]) * len(m) * trials for q, m in matrix.items()
    )
    done = 0
    for qtype, methods in matrix.items():
        for key in keys_by_query[qtype]:
            for trial in range(trials):
                for mname, fn in methods.items():
                    used = drop_caches(evict_files, cache_drop)
                    cold = fn(store, key)
                    warm = fn(store, key)
                    for temp, res in (("cold", cold), ("warm", warm)):
                        out[qtype][mname][temp].append(
                            Sample(
                                key=key, trial=trial,
                                elapsed_s=res.elapsed_s,
                                rows=res.table.num_rows,
                                bytes_read=res.bytes_read,
                                files_touched=res.files_touched,
                                row_groups_touched=res.row_groups_touched,
                            )
                        )
                    done += 1
                    if done % 25 == 0 or done == total:
                        log.info("bench: %d/%d measurements", done, total)
    return out, used


def aggregate(samples: list[Sample]) -> dict[str, Any]:
    el = np.array([s.elapsed_s for s in samples], dtype=np.float64)
    p50, p90, p99 = (
        float(v) for v in np.percentile(el, [50.0, 90.0, 99.0])
    )

    def _mean(vals: list[int | None]) -> float | None:
        known = [v for v in vals if v is not None]
        return float(np.mean(known)) if known else None

    return {
        "n_samples": len(samples),
        "p50_s": p50,
        "p90_s": p90,
        "p99_s": p99,
        "mean_s": float(el.mean()),
        "rows_mean": float(np.mean([s.rows for s in samples])),
        "bytes_read_mean": _mean([s.bytes_read for s in samples]),
        "files_touched_mean": _mean([s.files_touched for s in samples]),
        "row_groups_touched_mean": _mean(
            [s.row_groups_touched for s in samples]
        ),
    }


#


def _fmt_bytes(v: float | None) -> str:
    if v is None:
        return "-"
    for unit in ("B", "KiB", "MiB", "GiB"):
        if abs(v) < 1024 or unit == "GiB":
            return f"{v:,.1f} {unit}"
        v /= 1024
    return f"{v:,.1f} GiB"  # pragma: no cover


