"""Partition the flat (token_idx, feature, value) rows by feature bucket."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pyarrow.parquet as pq

from . import schema

log = logging.getLogger("store.partition")

BUCKET_MAP_FILENAME = "bucket_map.json"
STATS_FILENAME = "stats.json"
FLAT_ROWS_GLOB = "rows-*.parquet"
PARTITION_GLOB = "bucket=*/*.parquet"

DEFAULT_N_BUCKETS = 128
DEFAULT_THREADS = 16
DEFAULT_MEMORY_LIMIT = "48GB"


#


@dataclass(frozen=True)
class Layout:
    """A bucketing layout: which column is the key, how big its domain."""

    name: str  # "feature" | "token"
    key: str  # column bucketed on
    secondary: str  # tie-break sort column
    key_label: str  # singular noun used in stats.json field names

    def domain(self, meta: dict[str, Any]) -> int:
        if self.name == "feature":
            n = int(meta["sae_config"]["n_features"])
        else:
            n = int(meta["n_tokens_encoded"])
        if n <= 0:
            raise ValueError(
                f"meta.json gives a non-positive key domain for layout "
                f"'{self.name}': {n}"
            )
        return n


LAYOUTS: dict[str, Layout] = {
    "feature": Layout("feature", "feature", "token_idx", "feature"),
    "token": Layout("token", "token_idx", "feature", "token"),
}


def bucket_bounds(n_buckets: int, domain: int) -> list[tuple[int, int]]:
    """[lo, hi) key range of every bucket under ``bucket = key*B // D``. bucket b
    holds exactly the keys k with ``b <= k*B/D < b+1``."""
    if n_buckets < 1:
        raise ValueError(f"n_buckets must be >= 1, got {n_buckets}")
    if domain < 1:
        raise ValueError(f"key domain must be >= 1, got {domain}")

    def ceil_div(a: int, b: int) -> int:
        return -(-a // b)

    return [
        (ceil_div(b * domain, n_buckets), ceil_div((b + 1) * domain, n_buckets))
        for b in range(n_buckets)
    ]


def bucket_expr_sql(key: str, n_buckets: int, domain: int) -> str:
    """The bucket expression, kept in one place so writer and checker agree."""
    return f"(CAST({key} AS BIGINT) * {n_buckets}) // {domain}"


#


@dataclass
class PartitionConfig:
    flat: Path
    out: Path
    layout: str = "feature"
    n_buckets: int = DEFAULT_N_BUCKETS
    row_group_size: int = schema.DEFAULT_ROW_GROUP_SIZE
    threads: int = DEFAULT_THREADS
    memory_limit: str = DEFAULT_MEMORY_LIMIT
    temp_dir: Path | None = None  # default: <out>/_duckdb_tmp
    per_bucket_passes: bool = False


#


def _sql_path(p: Path | str) -> str:
    """Path -> SQL string literal body: POSIX separators."""
    return Path(p).as_posix().replace("'", "''")


def _connect(cfg: PartitionConfig) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute(f"SET threads = {int(cfg.threads)}")
    con.execute(f"SET memory_limit = '{cfg.memory_limit}'")
    temp = cfg.temp_dir or (cfg.out / "_duckdb_tmp")
    temp.mkdir(parents=True, exist_ok=True)
    con.execute(f"SET temp_directory = '{_sql_path(temp)}'")
    # Guarantees the ORDER BY in COPY survives into the written files.
    con.execute("SET preserve_insertion_order = true")
    return con


def load_flat_meta(flat: Path) -> dict[str, Any]:
    meta_path = flat / schema.META_FILENAME
    if not meta_path.exists():
        raise FileNotFoundError(
            f"expected {meta_path} - point --flat at a directory written by "
            f"`python -m store.dump` (it must contain meta.json)"
        )
    return json.loads(meta_path.read_text())


def _flat_glob_sql(flat: Path) -> str:
    return f"read_parquet('{_sql_path(flat)}/{FLAT_ROWS_GLOB}')"


def _bucketed_glob_sql(out: Path) -> str:
    return (
        f"read_parquet('{_sql_path(out)}/{PARTITION_GLOB}', "
        f"hive_partitioning=1)"
    )


def _clear_partitions(out: Path) -> None:
    """Remove any existing bucket=* dirs so stale files from a previous run can
    never mix with the new write."""
    removed = 0
    for d in sorted(out.glob("bucket=*")):
        if d.is_dir():
            shutil.rmtree(d)
            removed += 1
    if removed:
        log.info("cleared %d existing bucket=* dirs under %s", removed, out)


#


def write_partitions(cfg: PartitionConfig) -> dict[str, Any]:
    """Partition the flat rows and write bucket_map.json + meta.json."""
    layout = LAYOUTS[cfg.layout]
    meta = load_flat_meta(cfg.flat)
    domain = layout.domain(meta)
    bounds = bucket_bounds(cfg.n_buckets, domain)
    expr = bucket_expr_sql(layout.key, cfg.n_buckets, domain)
    n_empty = sum(1 for lo, hi in bounds if lo == hi)
    if n_empty:
        log.warning(
            "n_buckets=%d > domain=%d: %d buckets are empty by construction",
            cfg.n_buckets, domain, n_empty,
        )

    cfg.out.mkdir(parents=True, exist_ok=True)
    _clear_partitions(cfg.out)
    con = _connect(cfg)
    try:
        src = _flat_glob_sql(cfg.flat)
        (source_rows,) = con.execute(f"SELECT count(*) FROM {src}").fetchone()
        source_rows = int(source_rows)
        if source_rows != int(meta.get("n_rows", -1)):
            log.warning(
                "flat set has %d rows but meta.json says n_rows=%s "
                "(partial dump?); the partition check will verify against "
                "the measured %d",
                source_rows, meta.get("n_rows"), source_rows,
            )
        (max_key,) = con.execute(
            f"SELECT max({layout.key}) FROM {src}"
        ).fetchone()
        if max_key is not None and int(max_key) >= domain:
            raise ValueError(
                f"max {layout.key} in the flat set is {int(max_key)}, "
                f">= the key domain {domain} from meta.json - the bucket "
                f"expression would overflow n_buckets"
            )

        t0 = time.monotonic()
        if cfg.per_bucket_passes:
            _copy_per_bucket(con, cfg, layout, domain, expr)
        else:
            _copy_single_pass(con, cfg, layout, expr)
            _finalize_partitions(con, cfg, layout)
        log.info(
            "partitioned %d rows into %s in %.1fs (%s)",
            source_rows, cfg.out, time.monotonic() - t0,
            "per-bucket passes" if cfg.per_bucket_passes else "single COPY",
        )
    finally:
        con.close()
        _cleanup_temp(cfg)

    _write_bucket_map(cfg, layout, domain, bounds, expr)
    fwd = _forward_meta(cfg, meta, layout, domain, source_rows)
    return fwd


def _copy_single_pass(
    con: duckdb.DuckDBPyConnection,
    cfg: PartitionConfig,
    layout: Layout,
    expr: str,
) -> None:
    """One global sort + hive-partitioned COPY (the default path)."""
    sql = (
        f"COPY (SELECT token_idx, feature, value, "
        f"CAST({expr} AS BIGINT) AS bucket "
        f"FROM {_flat_glob_sql(cfg.flat)} "
        f"ORDER BY bucket, {layout.key}, {layout.secondary}) "
        f"TO '{_sql_path(cfg.out)}' "
        f"(FORMAT PARQUET, PARTITION_BY (bucket), "
        f"ROW_GROUP_SIZE {int(cfg.row_group_size)}, COMPRESSION ZSTD, "
        f"OVERWRITE_OR_IGNORE)"
    )
    log.debug("COPY sql: %s", sql)
    con.execute(sql)


def _file_is_sorted(path: Path, layout: Layout) -> bool:
    """True iff the file is sorted by (key, secondary)."""
    tb = pq.read_table(path, columns=[layout.key, layout.secondary])
    k = tb[layout.key].to_numpy().astype(np.uint64)
    s = tb[layout.secondary].to_numpy().astype(np.uint64)
    packed = (k << np.uint64(32)) | s
    return bool(np.all(packed[1:] >= packed[:-1]))


def _finalize_partitions(
    con: duckdb.DuckDBPyConnection, cfg: PartitionConfig, layout: Layout
) -> None:
    """Make every partition a single file sorted by (key, secondary). This also
    makes the single-pass output byte-for-row identical to --per-bucket-passes:
    one sorted data_0.parquet per bucket."""
    rewritten = 0
    for part_dir in sorted(cfg.out.glob("bucket=*")):
        files = sorted(part_dir.glob("*.parquet"))
        if not files:
            continue
        if len(files) == 1 and _file_is_sorted(files[0], layout):
            if files[0].name != "data_0.parquet":
                files[0].rename(part_dir / "data_0.parquet")
            continue
        file_list = ", ".join(f"'{_sql_path(f)}'" for f in files)
        tmp = part_dir / "_sorting.parquet.tmp"
        con.execute(
            f"COPY (SELECT token_idx, feature, value "
            f"FROM read_parquet([{file_list}]) "
            f"ORDER BY {layout.key}, {layout.secondary}) "
            f"TO '{_sql_path(tmp)}' "
            f"(FORMAT PARQUET, ROW_GROUP_SIZE {int(cfg.row_group_size)}, "
            f"COMPRESSION ZSTD)"
        )
        for f in files:
            f.unlink()
        tmp.rename(part_dir / "data_0.parquet")
        rewritten += 1
    if rewritten:
        log.info(
            "finalize: consolidated/re-sorted %d partitions (DuckDB "
            "multi-threaded partitioned COPY does not preserve ORDER BY)",
            rewritten,
        )


def _copy_per_bucket(
    con: duckdb.DuckDBPyConnection,
    cfg: PartitionConfig,
    layout: Layout,
    domain: int,
    expr: str,
) -> None:
    """Fallback: one filtered pass per bucket. n_buckets scans of the flat set, but
    each sort holds only ~1/n_buckets of the rows, so it works when the global
    ORDER BY would exceed --memory-limit (the ~5B-row case)."""
    src = _flat_glob_sql(cfg.flat)
    for b, (lo, hi) in enumerate(bucket_bounds(cfg.n_buckets, domain)):
        if lo == hi:
            continue
        part_dir = cfg.out / f"bucket={b}"
        part_dir.mkdir(parents=True, exist_ok=True)
        dst = part_dir / "data_0.parquet"
        sql = (
            f"COPY (SELECT token_idx, feature, value FROM {src} "
            f"WHERE {expr} = {b} "
            f"ORDER BY {layout.key}, {layout.secondary}) "
            f"TO '{_sql_path(dst)}' "
            f"(FORMAT PARQUET, ROW_GROUP_SIZE {int(cfg.row_group_size)}, "
            f"COMPRESSION ZSTD)"
        )
        con.execute(sql)
        # The single-pass path never materializes empty partitions; match it.
        if pq.ParquetFile(dst).metadata.num_rows == 0:
            dst.unlink()
            part_dir.rmdir()
        if (b + 1) % 16 == 0 or b == cfg.n_buckets - 1:
            log.info("per-bucket passes: %d/%d done", b + 1, cfg.n_buckets)


def _cleanup_temp(cfg: PartitionConfig) -> None:
    temp = cfg.temp_dir or (cfg.out / "_duckdb_tmp")
    try:
        if temp.is_dir() and not any(temp.iterdir()):
            temp.rmdir()
    except OSError:  # pragma: no cover - best effort
        pass


def _write_bucket_map(
    cfg: PartitionConfig,
    layout: Layout,
    domain: int,
    bounds: list[tuple[int, int]],
    expr: str,
) -> None:
    payload = {
        "format_version": schema.FORMAT_VERSION,
        "layout": layout.name,
        "key": layout.key,
        "secondary_sort": layout.secondary,
        "n_buckets": cfg.n_buckets,
        "domain": domain,
        "bucket_expr": expr,
        "buckets": [
            {"bucket": b, "lo": lo, "hi": hi}
            for b, (lo, hi) in enumerate(bounds)
        ],
    }
    (cfg.out / BUCKET_MAP_FILENAME).write_text(json.dumps(payload, indent=1))
    log.info("wrote %s", cfg.out / BUCKET_MAP_FILENAME)


def _forward_meta(
    cfg: PartitionConfig,
    meta: dict[str, Any],
    layout: Layout,
    domain: int,
    source_rows: int,
) -> dict[str, Any]:
    fwd = dict(meta)
    fwd["partition"] = schema.jsonable(
        {
            "layout": layout.name,
            "key": layout.key,
            "secondary_sort": layout.secondary,
            "n_buckets": cfg.n_buckets,
            "domain": domain,
            "bucket_expr": bucket_expr_sql(layout.key, cfg.n_buckets, domain),
            "row_group_size": cfg.row_group_size,
            "threads": cfg.threads,
            "memory_limit": cfg.memory_limit,
            "per_bucket_passes": cfg.per_bucket_passes,
            "source_flat": str(cfg.flat),
            "source_rows": source_rows,
            "created_at": datetime.now(timezone.utc).isoformat(
                timespec="seconds"
            ),
        }
    )
    (cfg.out / schema.META_FILENAME).write_text(json.dumps(fwd, indent=1))
    log.info("wrote %s", cfg.out / schema.META_FILENAME)
    return fwd


#


