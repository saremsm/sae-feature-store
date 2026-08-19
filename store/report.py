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


def section_scale(
    meta: dict[str, Any],
    stats: dict[str, Any],
    token_stats: dict[str, Any] | None,
    flat_bytes: int,
    flat_files: int,
) -> list[str]:
    n_rows = int(meta["n_rows"])
    n_tokens = int(meta["n_tokens_encoded"])
    n_features = _n_features(meta)
    dense_bytes = n_tokens * n_features * 2  # fp16

    lines = [
        "## Scale",
        "",
        f"- rows: **{fmt_int(n_rows)}** over **{fmt_int(n_tokens)}** tokens"
        f" x **{fmt_int(n_features)}** features, mean L0"
        f" **{float(meta['mean_l0']):g}**",
        f"- dense baseline: {fmt_int(n_tokens)} tokens x {fmt_int(n_features)}"
        f" features x 2 B (fp16) = **{fmt_int(dense_bytes)} B**"
        f" ({fmt_bytes_binary(dense_bytes)})",
        "",
        "| layout | files | row groups | on-disk bytes | size | bytes/row"
        " | x smaller than dense | size vs flat |",
        "|---|---|---|---|---|---|---|---|",
    ]

    def row(
        label: str,
        nbytes: int,
        files: str,
        row_groups: str,
    ) -> str:
        return (
            f"| {label} | {files} | {row_groups} | {fmt_int(nbytes)}"
            f" | {fmt_bytes_binary(nbytes)} | {_ratio(nbytes, n_rows):.2f}"
            f" | {_ratio(dense_bytes, nbytes):.1f}x"
            f" | {_ratio(nbytes, flat_bytes):.2f}x |"
        )

    lines.append(
        f"| dense fp16 (hypothetical) | - | - | {fmt_int(dense_bytes)}"
        f" | {fmt_bytes_binary(dense_bytes)}"
        f" | {_ratio(dense_bytes, n_rows):.2f} | 1.0x"
        f" | {_ratio(dense_bytes, flat_bytes):.2f}x |"
    )
    lines.append(row("flat (token order)", flat_bytes, str(flat_files), "-"))
    lines.append(
        row(
            "feature-bucketed",
            int(stats["total_bytes"]),
            str(int(stats["n_files"])),
            str(int(stats["n_row_groups"])),
        )
    )
    if token_stats is not None:
        lines.append(
            row(
                "token-bucketed",
                int(token_stats["total_bytes"]),
                str(int(token_stats["n_files"])),
                str(int(token_stats["n_row_groups"])),
            )
        )
    else:
        log.warning("token-bucketed stats missing; omitting its scale row")
    return lines


def _latency_table(methods: dict[str, Any], query: str, temp: str) -> list[str]:
    order = list(_METHOD_ORDER.get(query, []))
    known = {key for key, _ in order}
    order += [(key, key) for key in methods if key not in known]

    lines = [_LATENCY_HEADER, _LATENCY_RULE]
    for key, label in order:
        if key not in methods:
            log.warning("bench.json has no %s/%s results; row omitted", query, key)
            continue
        agg = methods[key][temp]["agg"]
        lines.append(
            f"| {label} | {agg['p50_s']:.4f} | {agg['p90_s']:.4f}"
            f" | {agg['p99_s']:.4f} | {agg['mean_s']:.4f}"
            f" | {fmt_int(agg['rows_mean'])}"
            f" | {fmt_bytes_binary(agg['bytes_read_mean'])}"
            f" | {agg['files_touched_mean']:.0f}"
            f" | {_fmt_row_groups(agg.get('row_groups_touched_mean'))} |"
        )
    return lines


def section_latency(bench: dict[str, Any]) -> list[str]:
    lines = ["## Query latency"]
    for query in ("tokens_for_feature", "features_for_token"):
        if query not in bench.get("results", {}):
            raise ReportError(f"bench.json has no results for {query!r}")
        for temp in ("cold", "warm"):
            lines += ["", f"### {query} ({temp})", ""]
            lines += _latency_table(bench["results"][query], query, temp)
    kp = bench.get("killpoint")
    if kp is not None:
        lines += [
            "",
            "### Kill point",
            "",
            f"- {kp['message']}",
        ]
    return lines


def section_scaleup(
    meta: dict[str, Any],
    stats: dict[str, Any],
    token_stats: dict[str, Any] | None,
    flat_bytes: int,
    target_tokens: int,
) -> list[str]:
    n_rows = int(meta["n_rows"])
    n_tokens = int(meta["n_tokens_encoded"])
    n_features = _n_features(meta)
    mean_l0 = float(meta["mean_l0"])
    row_group_size = int(meta["partition"]["row_group_size"])
    n_buckets = int(meta["partition"]["n_buckets"])

    rows_t = mean_l0 * target_tokens
    bpr_feature = _ratio(int(stats["total_bytes"]), n_rows)
    size_feature = bpr_feature * rows_t
    bpr_flat = _ratio(flat_bytes, n_rows)
    dense_t = target_tokens * n_features * 2
    rows_per_feature = rows_t / n_features
    bytes_per_feature = size_feature / n_features
    row_groups_per_feature = rows_per_feature / row_group_size
    bucket_bytes_feature = size_feature / n_buckets
    # (token_idx u32, feature u32, value f32) = 12 B/row uncompressed
    sort_input = rows_t * 12
    scale_factor = target_tokens / n_tokens

    lines = [
        f"## Scale-up to {target_tokens:.0e} tokens (EXTRAPOLATED)",
        "",
        "All numbers in this table are **extrapolated** from the measured"
        f" bytes/row and mean L0 above; the scale factor over the measured"
        f" run is {target_tokens:,.0f} / {fmt_int(n_tokens)} ="
        f" **{scale_factor:,.0f}x** tokens.",
        "",
        "| quantity (extrapolated) | value | arithmetic |",
        "|---|---|---|",
        f"| rows | {rows_t:.2e} | {mean_l0:g} rows/token x"
        f" {target_tokens:,.0f} tokens |",
        f"| feature-bucketed storage | {fmt_tb(size_feature)} |"
        f" {bpr_feature:.4f} B/row x {rows_t:.2e} rows |",
        f"| flat storage | {fmt_tb(bpr_flat * rows_t)} | {bpr_flat:.4f} B/row"
        f" x {rows_t:.2e} rows |",
    ]
    if token_stats is not None:
        bpr_token = _ratio(int(token_stats["total_bytes"]), n_rows)
        lines.append(
            f"| token-bucketed storage | {fmt_tb(bpr_token * rows_t)} |"
            f" {bpr_token:.4f} B/row x {rows_t:.2e} rows |"
        )
    lines += [
        f"| dense fp16 baseline | {fmt_tb(dense_t)} | {target_tokens:,.0f}"
        f" x {fmt_int(n_features)} x 2 B |",
        f"| rows per feature (mean) | {rows_per_feature:.2e} | {rows_t:.2e}"
        f" / {fmt_int(n_features)} features |",
        f"| row groups per feature (mean) | {fmt_int(row_groups_per_feature)}"
        f" | {rows_per_feature:.2e} / {fmt_int(row_group_size)}-row groups |",
        f"| cold read per mean feature query |"
        f" {bytes_per_feature / 1e9:,.1f} GB | {fmt_tb(size_feature)} /"
        f" {fmt_int(n_features)} features |",
        f"| per-bucket file at {n_buckets} buckets |"
        f" {fmt_tb(bucket_bytes_feature)} | {fmt_tb(size_feature)} /"
        f" {n_buckets} |",
        f"| ingest sort input (uncompressed) | {fmt_tb(sort_input)} |"
        f" {rows_t:.2e} rows x 12 B |",
    ]
    return lines


def build_report(
    bench: dict[str, Any],
    stats: dict[str, Any],
    meta: dict[str, Any],
    token_stats: dict[str, Any] | None,
    target_tokens: int = DEFAULT_TARGET_TOKENS,
) -> str:
    """Assemble the full markdown document (also used by the tests)."""
    cross_check(meta, stats, token_stats)
    flat_bytes = flat_bytes_from_bench(bench)
    parts: list[list[str]] = [
        [
            "<!-- Generated by `python -m store.report`. Do not edit the"
            " numbers by hand. -->",
        ],
        section_provenance(bench, meta),
        section_scale(
            meta, stats, token_stats, flat_bytes, flat_files_from_bench(bench)
        ),
        section_latency(bench),
        section_scaleup(meta, stats, token_stats, flat_bytes, target_tokens),
    ]
    return "\n\n".join("\n".join(p) for p in parts) + "\n"


#


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m store.report",
        description="Regenerate the README numeric tables from"
        " bench.json/stats.json/meta.json (writes markdown to stdout).",
    )
    parser.add_argument("--bench", default=DEFAULT_BENCH, help="results/bench.json")
    parser.add_argument(
        "--stats", default=DEFAULT_STATS, help="feature-bucketed stats.json"
    )
    parser.add_argument(
        "--token-stats",
        default=DEFAULT_TOKEN_STATS,
        help="token-bucketed stats.json (optional; its rows are omitted with"
        " a warning if the file is missing)",
    )
    parser.add_argument(
        "--meta", default=DEFAULT_META, help="forwarded meta.json (bucketed)"
    )
    parser.add_argument(
        "--target-tokens",
        type=int,
        default=DEFAULT_TARGET_TOKENS,
        help="token count for the extrapolated scale-up section"
        " (default: 10^12)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None, out: TextIO | None = None) -> int:
    args = make_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        stream=sys.stderr,
    )
    out = sys.stdout if out is None else out
    try:
        bench = _load_json(Path(args.bench), "bench json")
        stats = _load_json(Path(args.stats), "stats json")
        meta = _load_json(Path(args.meta), "meta json")
        token_stats: dict[str, Any] | None = None
        token_path = Path(args.token_stats)
        if token_path.is_file():
            token_stats = _load_json(token_path, "token-stats json")
        else:
            log.warning(
                "token-bucketed stats not found at %s; its rows are omitted",
                token_path,
            )
        report = build_report(
            bench, stats, meta, token_stats, target_tokens=args.target_tokens
        )
    except (ReportError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        log.error("report failed: %s", exc)
        return 1
    out.write(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
