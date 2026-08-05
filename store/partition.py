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


@dataclass
class BucketStat:
    bucket: int
    lo: int
    hi: int
    rows: int
    bytes: int
    n_keys: int
    min_rows_per_key: int
    max_rows_per_key: int
    n_files: int
    n_row_groups: int


@dataclass
class CheckReport:
    ok: bool = True
    errors: list[str] = field(default_factory=list)
    layout: str = ""
    key: str = ""
    n_buckets: int = 0
    domain: int = 0
    total_rows: int = 0
    total_bytes: int = 0
    n_keys: int = 0
    n_files: int = 0
    n_row_groups: int = 0
    mean_row_groups_touched_per_key: float = 0.0
    boundary_shared_row_groups: int = 0
    buckets: list[BucketStat] = field(default_factory=list)

    def error(self, msg: str) -> None:
        self.ok = False
        self.errors.append(msg)
        log.error("check: %s", msg)


def check_dataset(
    out: Path,
    *,
    threads: int = DEFAULT_THREADS,
    memory_limit: str = DEFAULT_MEMORY_LIMIT,
    write_stats: bool = False,
) -> CheckReport:
    """Run every post-check against a bucketed dataset; optionally (re)write
    stats.json. Never raises on a failed invariant - failures land in
    ``report.errors`` (the CLI exits non-zero on any)."""
    rep = CheckReport()
    out = Path(out)
    map_path = out / BUCKET_MAP_FILENAME
    meta_path = out / schema.META_FILENAME
    if not map_path.exists() or not meta_path.exists():
        rep.error(
            f"{out} is not a bucketed dataset (missing "
            f"{BUCKET_MAP_FILENAME} or {schema.META_FILENAME})"
        )
        return rep

    bmap = json.loads(map_path.read_text())
    meta = json.loads(meta_path.read_text())
    layout = LAYOUTS[bmap["layout"]]
    n_buckets = int(bmap["n_buckets"])
    domain = int(bmap["domain"])
    rep.layout, rep.key = layout.name, layout.key
    rep.n_buckets, rep.domain = n_buckets, domain

    # Guard against a hand-edited bucket_map.json drifting from the formula.
    expect = bucket_bounds(n_buckets, domain)
    got = [(int(b["lo"]), int(b["hi"])) for b in bmap["buckets"]]
    if got != expect:
        rep.error("bucket_map.json ranges do not match the bucket formula")
        return rep
    if bmap.get("bucket_expr") != bucket_expr_sql(layout.key, n_buckets, domain):
        rep.error("bucket_map.json bucket_expr does not match the formula")
        return rep

    files = sorted(out.glob(PARTITION_GLOB))
    if not files:
        rep.error(f"no partition files under {out / PARTITION_GLOB}")
        return rep
    rep.n_files = len(files)

    expected_rows = int(
        meta.get("partition", {}).get("source_rows", meta.get("n_rows", -1))
    )
    expr = bucket_expr_sql(layout.key, n_buckets, domain)

    con = duckdb.connect()
    con.execute(f"SET threads = {int(threads)}")
    con.execute(f"SET memory_limit = '{memory_limit}'")
    try:
        rel = _bucketed_glob_sql(out)

        (total, max_key) = con.execute(
            f"SELECT count(*), max({layout.key}) FROM {rel}"
        ).fetchone()
        rep.total_rows = int(total)
        if rep.total_rows != expected_rows:
            rep.error(
                f"total rows changed: bucketed has {rep.total_rows}, "
                f"source had {expected_rows}"
            )
        if max_key is not None and int(max_key) >= domain:
            rep.error(
                f"max {layout.key} {int(max_key)} >= domain {domain}"
            )

        (mismatch,) = con.execute(
            f"SELECT count(*) FROM {rel} "
            f"WHERE CAST(bucket AS BIGINT) <> {expr}"
        ).fetchone()
        if int(mismatch):
            rep.error(
                f"{int(mismatch)} rows live in a partition that does not "
                f"match the bucket expression"
            )

        (multi,) = con.execute(
            f"SELECT count(*) FROM ("
            f"SELECT {layout.key} FROM {rel} GROUP BY 1 "
            f"HAVING count(DISTINCT bucket) > 1)"
        ).fetchone()
        if int(multi):
            rep.error(
                f"{int(multi)} {layout.key_label}s have rows in more than "
                f"one bucket"
            )

        per_key = con.execute(
            f"SELECT CAST(bucket AS BIGINT) AS b, "
            f"CAST({layout.key} AS BIGINT) AS k, count(*) AS c "
            f"FROM {rel} GROUP BY 1, 2 ORDER BY 1, 2"
        ).fetchnumpy()
        kb = per_key["b"].astype(np.int64)
        kk = per_key["k"].astype(np.int64)
        kc = per_key["c"].astype(np.int64)
        rep.n_keys = int(kk.size)
    finally:
        con.close()

    # -- per-file scan: sort order + row-group statistics -------------------
    per_bucket_files: dict[int, list[Path]] = {}
    for f in files:
        b = int(f.parent.name.split("=", 1)[1])
        per_bucket_files.setdefault(b, []).append(f)

    key_col = layout.key
    sec_col = layout.secondary
    intervals: dict[int, list[tuple[int, int]]] = {}
    total_bytes = 0
    n_row_groups = 0
    boundary_shared = 0
    files_stats: dict[int, tuple[int, int]] = {}  # bucket -> (n_files, n_rg)

    for b, bfiles in sorted(per_bucket_files.items()):
        if not 0 <= b < n_buckets:
            rep.error(f"partition dir bucket={b} outside [0, {n_buckets})")
            continue
        lo, hi = expect[b]
        ivals: list[tuple[int, int]] = []
        n_rg_b = 0
        for f in bfiles:
            total_bytes += f.stat().st_size
            pf = pq.ParquetFile(f)
            names = pf.schema_arrow.names
            if "bucket" in names:
                rep.error(f"{f}: partition column 'bucket' stored in file")
            if key_col not in names or sec_col not in names:
                rep.error(
                    f"{f}: missing column(s) {key_col}/{sec_col}; "
                    f"schema is {names}"
                )
                continue
            key_i = names.index(key_col)
            md = pf.metadata
            n_rg_b += md.num_row_groups
            prev_max: int | None = None
            prev_last: int | None = None
            for g in range(md.num_row_groups):
                st = md.row_group(g).column(key_i).statistics
                if st is None or not st.has_min_max:
                    rep.error(f"{f} row group {g}: no min/max statistics")
                    continue
                gmin, gmax = int(st.min), int(st.max)
                ivals.append((gmin, gmax))
                if prev_max is not None:
                    if gmin < prev_max:
                        rep.error(
                            f"{f}: row groups {g - 1} and {g} overlap on "
                            f"{key_col} ({prev_max} > {gmin})"
                        )
                    elif gmin == prev_max:
                        boundary_shared += 1
                prev_max = gmax
                # data-level sort check on (key, secondary)
                tb = pf.read_row_group(g, columns=[key_col, sec_col])
                k = tb[key_col].to_numpy().astype(np.uint64)
                s = tb[sec_col].to_numpy().astype(np.uint64)
                packed = (k << np.uint64(32)) | s
                if packed.size and not bool(np.all(packed[1:] >= packed[:-1])):
                    rep.error(
                        f"{f} row group {g}: not sorted by "
                        f"({key_col}, {sec_col})"
                    )
                if packed.size:
                    if prev_last is not None and int(packed[0]) < prev_last:
                        rep.error(
                            f"{f}: rows not sorted across row groups "
                            f"{g - 1} -> {g}"
                        )
                    prev_last = int(packed[-1])
                if k.size and (int(k.min()) < lo or int(k.max()) >= hi):
                    rep.error(
                        f"{f} row group {g}: {key_col} outside bucket {b} "
                        f"range [{lo}, {hi})"
                    )
        intervals[b] = ivals
        n_row_groups += n_rg_b
        files_stats[b] = (len(bfiles), n_rg_b)

    rep.total_bytes = total_bytes
    rep.n_row_groups = n_row_groups
    rep.boundary_shared_row_groups = boundary_shared

    # -- per-bucket stats + mean row-groups-touched-per-key -----------------
    touched_total = 0
    for b in range(n_buckets):
        lo, hi = expect[b]
        sel = kb == b
        keys_b = kk[sel]
        counts_b = kc[sel]
        if keys_b.size and (
            int(keys_b.min()) < lo or int(keys_b.max()) >= hi
        ):
            rep.error(
                f"bucket {b}: contains {layout.key_label}s outside "
                f"[{lo}, {hi})"
            )
        nf, nrg = files_stats.get(b, (0, 0))
        bts = sum(
            f.stat().st_size for f in per_bucket_files.get(b, [])
        )
        rep.buckets.append(
            BucketStat(
                bucket=b, lo=lo, hi=hi,
                rows=int(counts_b.sum()) if counts_b.size else 0,
                bytes=bts,
                n_keys=int(keys_b.size),
                min_rows_per_key=int(counts_b.min()) if counts_b.size else 0,
                max_rows_per_key=int(counts_b.max()) if counts_b.size else 0,
                n_files=nf,
                n_row_groups=nrg,
            )
        )
        for gmin, gmax in intervals.get(b, []):
            touched_total += int(
                np.searchsorted(keys_b, gmax, side="right")
                - np.searchsorted(keys_b, gmin, side="left")
            )
    if rep.n_keys:
        rep.mean_row_groups_touched_per_key = touched_total / rep.n_keys

    sum_bucket_rows = sum(s.rows for s in rep.buckets)
    if sum_bucket_rows != rep.total_rows:
        rep.error(
            f"per-bucket rows sum to {sum_bucket_rows}, "
            f"total is {rep.total_rows}"
        )

    if write_stats:
        _write_stats(out, layout, rep)
    else:
        _cross_check_stats(out, layout, rep)

    log.info(
        "check %s: %s | layout=%s rows=%d bytes=%d %ss=%d files=%d "
        "row_groups=%d mean_row_groups_touched_per_%s=%.3f "
        "boundary_shared=%d",
        out, "OK" if rep.ok else f"FAILED ({len(rep.errors)} errors)",
        rep.layout, rep.total_rows, rep.total_bytes, layout.key_label,
        rep.n_keys, rep.n_files, rep.n_row_groups, layout.key_label,
        rep.mean_row_groups_touched_per_key, rep.boundary_shared_row_groups,
    )
    return rep


def _stats_payload(layout: Layout, rep: CheckReport) -> dict[str, Any]:
    kl = layout.key_label
    return {
        "format_version": schema.FORMAT_VERSION,
        "layout": rep.layout,
        "key": rep.key,
        "n_buckets": rep.n_buckets,
        "domain": rep.domain,
        "total_rows": rep.total_rows,
        "total_bytes": rep.total_bytes,
        f"n_{kl}s": rep.n_keys,
        "n_files": rep.n_files,
        "n_row_groups": rep.n_row_groups,
        f"mean_row_groups_touched_per_{kl}": (
            rep.mean_row_groups_touched_per_key
        ),
        "boundary_shared_row_groups": rep.boundary_shared_row_groups,
        "buckets": [
            {
                "bucket": s.bucket, "lo": s.lo, "hi": s.hi,
                "rows": s.rows, "bytes": s.bytes,
                f"{kl}s": s.n_keys,
                f"min_rows_per_{kl}": s.min_rows_per_key,
                f"max_rows_per_{kl}": s.max_rows_per_key,
                "files": s.n_files, "row_groups": s.n_row_groups,
            }
            for s in rep.buckets
        ],
    }


