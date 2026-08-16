"""Regenerate the README's numeric tables from the measured artifacts."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, TextIO

log = logging.getLogger("store.report")

DEFAULT_BENCH = "results/bench.json"
DEFAULT_STATS = "work/bucketed/stats.json"
DEFAULT_TOKEN_STATS = "work/bucketed_by_token/stats.json"
DEFAULT_META = "work/bucketed/meta.json"

#: Default scale-up target (README section 5).
DEFAULT_TARGET_TOKENS = 10**12

# : Display order + labels per query shape: matched layout first.
_METHOD_ORDER: dict[str, list[tuple[str, str]]] = {
    "tokens_for_feature": [
        ("bucketed", "feature-bucketed"),
        ("token_bucketed_scan", "token-bucketed scan"),
        ("flat_pyarrow", "flat scan (pyarrow)"),
        ("flat_duckdb", "flat scan (DuckDB)"),
    ],
    "features_for_token": [
        ("token_bucketed", "token-bucketed"),
        ("feature_bucketed_scan", "feature-bucketed scan"),
        ("flat_pyarrow", "flat scan (pyarrow)"),
        ("flat_duckdb", "flat scan (DuckDB)"),
    ],
}

_LATENCY_HEADER = (
    "| method | p50 (s) | p90 (s) | p99 (s) | mean (s) | rows | bytes read"
    " | files | row groups |"
)
_LATENCY_RULE = "|---|---|---|---|---|---|---|---|---|"


class ReportError(RuntimeError):
    """Inconsistent or missing artifacts; the report refuses to print."""


def _n_features(meta: dict[str, Any]) -> int:
    """The SAE width lives under ``sae_config`` in the forwarded meta."""
    return int(meta["sae_config"]["n_features"])


#


def fmt_int(n: float) -> str:
    """``1027192256 -> '1,027,192,256'`` (rounds float means)."""
    return f"{round(n):,}"


def fmt_bytes_binary(n: float) -> str:
    """Human-readable binary units, one decimal: ``7.1 MiB``, ``5.5 GiB``."""
    value = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(value) < 1024.0 or unit == "TiB":
            if unit == "B":
                return f"{value:.0f} B"
            return f"{value:.1f} {unit}"
        value /= 1024.0
    raise AssertionError("unreachable")


def fmt_tb(n: float) -> str:
    """Decimal terabytes, one decimal: ``213.7 TB``."""
    return f"{n / 1e12:,.1f} TB"


def _fmt_row_groups(v: float | None) -> str:
    return "-" if v is None else f"{v:.1f}"


def _ratio(numer: float, denom: float) -> float:
    if denom <= 0:
        raise ReportError(f"non-positive denominator in ratio: {denom!r}")
    return numer / denom


#


def _load_json(path: Path, what: str) -> dict[str, Any]:
    if not path.is_file():
        raise ReportError(
            f"{what} not found at {path} -- run the stage that produces it, or"
            f" point the matching --{what.split()[0]} flag at the right file"
        )
    with open(path, encoding="utf-8") as fh:
        loaded = json.load(fh)
    if not isinstance(loaded, dict):
        raise ReportError(f"{what} at {path} is not a JSON object")
    return loaded


def flat_bytes_from_bench(bench: dict[str, Any]) -> int:
    """Recover the flat rows-files' on-disk size from flat_duckdb accounting."""
    seen: dict[str, float] = {}
    for query, methods in bench.get("results", {}).items():
        agg = methods.get("flat_duckdb", {}).get("cold", {}).get("agg", {})
        value = agg.get("bytes_read_mean")
        if value is not None:
            seen[query] = float(value)
    if not seen:
        raise ReportError(
            "bench.json has no flat_duckdb results; cannot recover the flat"
            " layout's on-disk size"
        )
    values = set(seen.values())
    if len(values) != 1:
        raise ReportError(
            f"flat_duckdb bytes_read disagrees across query shapes: {seen!r}"
            " -- bench.json mixes runs over different flat sets"
        )
    return round(values.pop())


def flat_files_from_bench(bench: dict[str, Any]) -> int:
    """Flat rows-file count, from the same flat_duckdb full-scan accounting."""
    for methods in bench.get("results", {}).values():
        agg = methods.get("flat_duckdb", {}).get("cold", {}).get("agg", {})
        value = agg.get("files_touched_mean")
        if value is not None:
            return round(float(value))
    raise ReportError(
        "bench.json has no flat_duckdb results; cannot recover the flat"
        " layout's file count"
    )


def cross_check(
    meta: dict[str, Any],
    stats: dict[str, Any],
    token_stats: dict[str, Any] | None,
) -> None:
    """Fail loudly if the artifacts describe different datasets."""
    n_rows = int(meta["n_rows"])
    if int(stats["total_rows"]) != n_rows:
        raise ReportError(
            f"row-count mismatch: meta.json n_rows={n_rows:,} but feature"
            f" stats.json total_rows={int(stats['total_rows']):,} -- these"
            " artifacts come from different runs"
        )
    if token_stats is not None and int(token_stats["total_rows"]) != n_rows:
        raise ReportError(
            f"row-count mismatch: meta.json n_rows={n_rows:,} but token"
            f" stats.json total_rows={int(token_stats['total_rows']):,} --"
            " these artifacts come from different runs"
        )
    if stats.get("layout") != "feature":
        raise ReportError(
            f"--stats must point at the feature-bucketed stats.json (layout"
            f" field is {stats.get('layout')!r})"
        )
    if token_stats is not None and token_stats.get("layout") != "token":
        raise ReportError(
            f"--token-stats must point at the token-bucketed stats.json"
            f" (layout field is {token_stats.get('layout')!r})"
        )
    if _n_features(meta) != int(stats["domain"]):
        raise ReportError(
            f"feature-domain mismatch: meta n_features={_n_features(meta)}"
            f" vs stats domain={stats['domain']}"
        )


#


def section_provenance(bench: dict[str, Any], meta: dict[str, Any]) -> list[str]:
    env = bench.get("env", {})
    args = bench.get("args", {})
    cache = bench.get("cache_drop_method", "unknown")
    lines = [
        "## Provenance",
        "",
        f"- bench run: {bench.get('created_at', 'unknown')}; store"
        f" `{bench.get('store', '?')}`; args: {args.get('n_features', '?')}"
        f" features x {args.get('n_tokens', '?')} tokens x"
        f" {args.get('trials', '?')} trials, seed {args.get('seed', '?')}",
        f"- cache drop before each cold trial: **{cache}**",
        f"- host: {env.get('cpu_model', 'unknown CPU')} x"
        f"{env.get('cpu_count', '?')}, "
        f"{float(env.get('ram_bytes', 0)) / 2**30:.1f} GiB RAM",
        f"- versions: python {env.get('python', '?')}, duckdb"
        f" {env.get('duckdb', '?')}, pyarrow {env.get('pyarrow', '?')}",
        f"- source data: checkpoint"
        f" `{Path(str(meta['sae_checkpoint']['path'])).parent.name}`"
        f" (sha256 {str(meta['sae_checkpoint']['sha256'])[:12]}...), hook"
        f" `{meta['hook_name']}`, shard"
        f" `{Path(str(meta['shard']['path'])).name}`",
    ]
    if cache == "none":
        lines.append(
            "- **WARNING: cache_drop_method is 'none' -- the 'cold' numbers"
            " below are NOT cold.**"
        )
    return lines


