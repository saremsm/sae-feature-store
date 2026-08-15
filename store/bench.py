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


def _drop_caches_global() -> bool:
    """``sync`` + ``echo 3 > /proc/sys/vm/drop_caches``. Root only."""
    try:
        if hasattr(os, "sync"):
            os.sync()
        with open("/proc/sys/vm/drop_caches", "w") as fh:
            fh.write("3\n")
        return True
    except OSError:
        return False


def _fadvise_dontneed(files: list[Path]) -> bool:
    """Best-effort page-cache eviction of ``files`` via POSIX_FADV_DONTNEED (works
    without root; dirty pages are synced first so they can drop)."""
    if not hasattr(os, "posix_fadvise"):
        return False
    if hasattr(os, "sync"):
        os.sync()
    for f in files:
        fd = os.open(f, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)
    return True


def drop_caches(files: list[Path], method: str = "auto") -> str:
    """Evict ``files`` (ideally everything) from the page cache."""
    if method == "none":
        return "none"
    if method in ("auto", "drop_caches") and _drop_caches_global():
        return "drop_caches"
    if method == "drop_caches":
        raise PermissionError(
            "--cache-drop drop_caches requires root "
            "(sudo -E python -m store.bench ...)"
        )
    if method in ("auto", "fadvise") and _fadvise_dontneed(files):
        return "fadvise"
    if method == "fadvise":
        raise OSError("os.posix_fadvise is unavailable on this platform")
    return "none"


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


def sample_features(
    bucketed: Path, n: int, seed: int
) -> list[dict[str, Any]]:
    """Stratified feature sample: split features by rows-per-feature into terciles
    and draw ~n/3 from each (uniform within tercile, without replacement)."""
    feats, counts = feature_counts(bucketed)
    stats_path = bucketed / STATS_FILENAME
    if stats_path.exists():
        stats = json.loads(stats_path.read_text())
        if int(stats["total_rows"]) != int(counts.sum()) or int(
            stats["n_features"]
        ) != int(feats.size):
            raise AssertionError(
                f"measured per-feature counts (rows={int(counts.sum())}, "
                f"features={int(feats.size)}) disagree with {stats_path} "
                f"(rows={stats['total_rows']}, "
                f"features={stats['n_features']}) - stale stats.json?"
            )
    else:
        log.warning("no %s; skipping totals cross-check", stats_path)

    order = np.argsort(counts, kind="stable")
    thirds = np.array_split(order, 3)  # bottom, middle, top by frequency
    rng = np.random.default_rng(seed)
    per = [n // 3 + (1 if i < n % 3 else 0) for i in range(3)]
    picked: list[dict[str, Any]] = []
    for tercile, idxs, want in zip(TERCILES, thirds, per):
        take = min(want, idxs.size)
        sel = rng.choice(idxs, size=take, replace=False)
        for i in sorted(int(s) for s in sel):
            picked.append(
                {
                    "feature": int(feats[i]),
                    "rows": int(counts[i]),
                    "tercile": tercile,
                }
            )
    return picked


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


def killpoint(
    results: dict[str, dict[str, dict[str, dict[str, Any]]]],
    store: Path,
    threshold: float,
    cache_drop_method: str,
) -> tuple[bool, str]:
    """(passed, message). The gate: bucketed cold p99 for tokens_for_feature must be
    >= ``threshold`` x faster than the flat pyarrow scan's cold p99."""
    tff = results[QUERY_TOKENS_FOR_FEATURE]
    bucketed = tff["bucketed"]["cold"]["agg"]
    flat = tff["flat_pyarrow"]["cold"]["agg"]
    b_p99, f_p99 = bucketed["p99_s"], flat["p99_s"]
    speedup = (f_p99 / b_p99) if b_p99 > 0 else float("inf")
    header = (
        f"KILL-POINT tokens_for_feature cold p99: bucketed {b_p99:.4f}s vs "
        f"flat scan {f_p99:.4f}s -> {speedup:.1f}x (threshold "
        f">= {threshold:.1f}x)"
    )
    if speedup >= threshold:
        return True, f"{header}: PASS"

    lines = [f"{header}: FAIL", "Diagnosis:"]
    bmap = queries.load_bucket_map(store / queries.BUCKETED_SUBDIR)
    n_buckets, domain = int(bmap["n_buckets"]), int(bmap["domain"])
    stats_path = store / queries.BUCKETED_SUBDIR / STATS_FILENAME
    stats = (
        json.loads(stats_path.read_text()) if stats_path.exists() else None
    )

    rg_b = bucketed["row_groups_touched_mean"]
    if stats is not None:
        total_rg = int(stats["n_row_groups"])
        rows_per_rg = (
            stats["total_rows"] / total_rg if total_rg else float("nan")
        )
        rows_per_q = bucketed["rows_mean"]
        if rg_b is not None and total_rg and rg_b >= total_rg / n_buckets:
            lines.append(
                f"- feature filter may not be pushing down: a query touches "
                f"{rg_b:.1f} row groups on average, i.e. ~every row group "
                f"in its bucket ({total_rg / n_buckets:.1f}); check that "
                f"partition files carry feature min/max statistics and are "
                f"sorted (python -m store.partition --check)."
            )
        if rows_per_q and rows_per_rg / max(rows_per_q, 1.0) > 10.0:
            lines.append(
                f"- row groups too large: ~{rows_per_rg:,.0f} rows/group vs "
                f"~{rows_per_q:,.0f} rows returned per query - each hit "
                f"decompresses >>10x the needed rows; re-partition with a "
                f"smaller --row-group-size."
            )
    feats_per_bucket = domain / n_buckets
    b_bytes, f_bytes = (
        bucketed["bytes_read_mean"], flat["bytes_read_mean"],
    )
    if (
        b_bytes is not None and f_bytes and b_bytes / f_bytes > 1.0 / 32.0
    ):
        lines.append(
            f"- bucket too coarse: a bucketed query reads "
            f"{b_bytes / f_bytes:.1%} of the flat scan's bytes "
            f"({feats_per_bucket:.0f} features/bucket); raise --n-buckets."
        )
    if cache_drop_method == "none":
        lines.append(
            "- cold trials were NOT cold (cache_drop_method=none): both "
            "sides ran from page cache, which compresses the gap; re-run "
            "with root (drop_caches) or on a platform with posix_fadvise."
        )
    if len(lines) == 2:
        lines.append(
            "- no structural suspect found in the recorded bytes/row-group "
            "numbers; suspect small-store overheads (per-file open cost "
            "dominating) or measurement noise."
        )
    return False, "\n".join(lines)


#


def _fmt_bytes(v: float | None) -> str:
    if v is None:
        return "-"
    for unit in ("B", "KiB", "MiB", "GiB"):
        if abs(v) < 1024 or unit == "GiB":
            return f"{v:,.1f} {unit}"
        v /= 1024
    return f"{v:,.1f} GiB"  # pragma: no cover


def render_md(payload: dict[str, Any]) -> str:
    env = payload["env"]
    lines = [
        "# Bench results",
        "",
        f"- created: {payload['created_at']}",
        f"- store: `{payload['store']}`",
        f"- args: n_features={payload['args']['n_features']} "
        f"n_tokens={payload['args']['n_tokens']} "
        f"trials={payload['args']['trials']} seed={payload['args']['seed']}",
        f"- cache drop method: **{payload['cache_drop_method']}**"
        + (
            " (WARNING: cold trials are not cold)"
            if payload["cache_drop_method"] == "none"
            else ""
        ),
        f"- cpu: {env['cpu_model'] or 'unknown'} x{env['cpu_count']}, ram: "
        f"{_fmt_bytes(env['ram_bytes'])}",
        f"- versions: python {env['python']}, duckdb {env['duckdb']}, "
        f"pyarrow {env['pyarrow']}",
    ]
    if env.get("lsblk"):
        lines += ["", "```", env["lsblk"], "```"]
    for qtype, methods in payload["results"].items():
        for temp in ("cold", "warm"):
            lines += [
                "",
                f"## {qtype} ({temp})",
                "",
                "| method | p50 (s) | p90 (s) | p99 (s) | mean (s) | rows "
                "| bytes read | files | row groups |",
                "|---|---|---|---|---|---|---|---|---|",
            ]
            for mname, temps in methods.items():
                a = temps[temp]["agg"]
                rg = a["row_groups_touched_mean"]
                fl = a["files_touched_mean"]
                lines.append(
                    f"| {mname} | {a['p50_s']:.4f} | {a['p90_s']:.4f} | "
                    f"{a['p99_s']:.4f} | {a['mean_s']:.4f} | "
                    f"{a['rows_mean']:,.0f} | "
                    f"{_fmt_bytes(a['bytes_read_mean'])} | "
                    + (f"{fl:.1f} | " if fl is not None else "- | ")
                    + (f"{rg:.1f} |" if rg is not None else "- |")
                )
    kp = payload["killpoint"]
    lines += ["", "## Kill point", "", "```", kp["message"], "```", ""]
    return "\n".join(lines)


#


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m store.bench",
        description=(
            "Benchmark tokens_for_feature / features_for_token across "
            "layouts and flat baselines under a cold/warm protocol."
        ),
    )
    p.add_argument("--store", type=Path, default=Path("work"),
                   help="store root containing flat/, bucketed/, "
                        "bucketed_by_token/ (default: work/)")
    p.add_argument("--n-features", type=int, default=20,
                   help="features to sample, stratified by frequency "
                        "tercile (default 20)")
    p.add_argument("--n-tokens", type=int, default=20,
                   help="tokens to sample uniformly (default 20)")
    p.add_argument("--trials", type=int, default=5,
                   help="cold+warm trial pairs per key per method "
                        "(default 5)")
    p.add_argument("--out", type=Path,
                   default=Path("results") / "bench.json",
                   help="JSON output path; the markdown report lands next "
                        "to it with a .md suffix (default "
                        "results/bench.json)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cache-drop",
                   choices=["auto", "drop_caches", "fadvise", "none"],
                   default="auto",
                   help="cold-trial eviction method (default auto: "
                        "drop_caches if root, else fadvise, else none)")
    p.add_argument("--kill-threshold", type=float,
                   default=DEFAULT_KILL_THRESHOLD,
                   help="required bucketed-vs-flat cold p99 speedup for "
                        "tokens_for_feature (default 5.0; exit 2 below it)")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    store: Path = args.store
    bucketed = store / queries.BUCKETED_SUBDIR
    meta = json.loads((bucketed / schema.META_FILENAME).read_text())
    matrix = method_matrix(store)

    feat_sample = sample_features(bucketed, args.n_features, args.seed)
    tok_sample = sample_tokens(meta, args.n_tokens, args.seed)
    log.info(
        "sampled %d features (terciles: %s) and %d tokens",
        len(feat_sample),
        {t: sum(1 for f in feat_sample if f["tercile"] == t)
         for t in TERCILES},
        len(tok_sample),
    )
    keys_by_query = {
        QUERY_TOKENS_FOR_FEATURE: [f["feature"] for f in feat_sample],
        QUERY_FEATURES_FOR_TOKEN: tok_sample,
    }

    for qtype, methods in matrix.items():
        cross_check(store, methods, keys_by_query[qtype], qtype)

    evict = store_files(store)
    t0 = time.monotonic()
    raw, used = run_matrix(
        store, matrix, keys_by_query, args.trials, args.cache_drop, evict
    )
    log.info(
        "measurements done in %.1fs (cache drop: %s)",
        time.monotonic() - t0, used,
    )
    if used == "none":
        log.warning(
            "no cache eviction available - cold numbers are NOT cold"
        )

    results: dict[str, Any] = {}
    for qtype, methods in raw.items():
        results[qtype] = {}
        for mname, temps in methods.items():
            results[qtype][mname] = {
                temp: {
                    "agg": aggregate(samples),
                    "samples": [vars(s) for s in samples],
                }
                for temp, samples in temps.items()
            }

    ok, message = killpoint(results, store, args.kill_threshold, used)

    payload = schema.jsonable(
        {
            "format_version": schema.FORMAT_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(
                timespec="seconds"
            ),
            "store": str(store),
            "args": {
                "n_features": args.n_features,
                "n_tokens": args.n_tokens,
                "trials": args.trials,
                "seed": args.seed,
                "kill_threshold": args.kill_threshold,
            },
            "cache_drop_method": used,
            "env": capture_env(),
            "sampling": {"features": feat_sample, "tokens": tok_sample},
            "results": results,
            "killpoint": {
                "passed": ok,
                "threshold": args.kill_threshold,
                "message": message,
            },
        }
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1))
    md_path = args.out.with_suffix(".md")
    md_path.write_text(render_md(payload))
    log.info("wrote %s and %s", args.out, md_path)

    print(message)
    return 0 if ok else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
