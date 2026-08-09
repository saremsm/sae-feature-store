"""The two canonical queries against the bucketed store, plus baselines."""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

import duckdb
import pyarrow as pa
import pyarrow.dataset as pads
import pyarrow.parquet as pq

from .partition import BUCKET_MAP_FILENAME, FLAT_ROWS_GLOB, _sql_path

log = logging.getLogger("store.queries")

# Sub-directories of the store root (``--store work/``).
FLAT_SUBDIR = "flat"
BUCKETED_SUBDIR = "bucketed"
TOKEN_BUCKETED_SUBDIR = "bucketed_by_token"

#: layout name -> (sub-directory, expected bucket_map layout field)
LAYOUT_DIRS: dict[str, tuple[str, str]] = {
    "bucketed": (BUCKETED_SUBDIR, "feature"),
    "feature": (BUCKETED_SUBDIR, "feature"),
    "token": (TOKEN_BUCKETED_SUBDIR, "token"),
}

_OUT_SCHEMAS: dict[str, pa.Schema] = {
    # tokens_for_feature -> (token_idx, value)
    "token_idx": pa.schema(
        [
            pa.field("token_idx", pa.uint32()),
            pa.field("value", pa.float32()),
        ]
    ),
    # features_for_token -> (feature, value)
    "feature": pa.schema(
        [
            pa.field("feature", pa.uint32()),
            pa.field("value", pa.float32()),
        ]
    ),
}


@dataclass
class QueryResult:
    """A query answer plus its honest cost accounting."""

    table: pa.Table
    elapsed_s: float
    bytes_read: int | None
    files_touched: int | None = None
    row_groups_touched: int | None = None
    method: str = ""

    def __iter__(self) -> Iterator[Any]:
        return iter((self.table, self.elapsed_s, self.bytes_read))


#


def resolve_layout_dir(store_dir: Path | str, layout: str) -> Path:
    """Map a store root + layout name to the bucketed dataset directory."""
    if layout not in LAYOUT_DIRS:
        raise ValueError(
            f"unknown layout {layout!r}; expected one of "
            f"{sorted(set(LAYOUT_DIRS))}"
        )
    sub, _ = LAYOUT_DIRS[layout]
    d = Path(store_dir) / sub
    if not (d / BUCKET_MAP_FILENAME).exists():
        raise FileNotFoundError(
            f"expected {d / BUCKET_MAP_FILENAME} - run "
            f"`python -m store.partition"
            f"{' --layout token' if sub == TOKEN_BUCKETED_SUBDIR else ''}` "
            f"first, or point --store at the work/ root"
        )
    return d


def load_bucket_map(dataset_dir: Path | str) -> dict[str, Any]:
    return json.loads(
        (Path(dataset_dir) / BUCKET_MAP_FILENAME).read_text()
    )


def bucket_for_key(key: int, bmap: dict[str, Any]) -> int:
    """The bucket holding ``key`` under ``bucket = key * B // domain``."""
    domain = int(bmap["domain"])
    n_buckets = int(bmap["n_buckets"])
    if not 0 <= key < domain:
        raise ValueError(
            f"{bmap['key']}={key} outside the key domain [0, {domain})"
        )
    b = key * n_buckets // domain
    lo, hi = (int(bmap["buckets"][b]["lo"]), int(bmap["buckets"][b]["hi"]))
    if not lo <= key < hi:  # pragma: no cover - guards a corrupt map
        raise AssertionError(
            f"bucket_map.json disagrees with the bucket formula for "
            f"key {key}: bucket {b} covers [{lo}, {hi})"
        )
    return b


def _partition_files(dataset_dir: Path, bucket: int) -> list[Path]:
    return sorted((dataset_dir / f"bucket={bucket}").glob("*.parquet"))


def _all_partition_files(dataset_dir: Path) -> list[Path]:
    return sorted(dataset_dir.glob("bucket=*/*.parquet"))


def _flat_files(store_dir: Path | str) -> list[Path]:
    flat = Path(store_dir) / FLAT_SUBDIR
    files = sorted(flat.glob(FLAT_ROWS_GLOB))
    if not files:
        raise FileNotFoundError(
            f"no {FLAT_ROWS_GLOB} under {flat} - run `python -m store.dump` "
            f"first, or point --store at the work/ root"
        )
    return files


#


def _touched_stats(
    files: list[Path], filter_col: str, key: int
) -> tuple[int, int, int]:
    """(bytes_read, files_touched, row_groups_touched) that a filtered scan of
    ``files`` on ``filter_col == key`` must visit."""
    total_bytes = 0
    files_touched = 0
    rgs_touched = 0
    for f in files:
        md = pq.ParquetFile(f).metadata
        names = [md.schema.column(i).name for i in range(md.num_columns)]
        ci = names.index(filter_col)
        hit = False
        for g in range(md.num_row_groups):
            rg = md.row_group(g)
            st = rg.column(ci).statistics
            if st is not None and st.has_min_max:
                if not int(st.min) <= key <= int(st.max):
                    continue
            rgs_touched += 1
            hit = True
            total_bytes += sum(
                rg.column(c).total_compressed_size
                for c in range(rg.num_columns)
            )
        if hit:
            files_touched += 1
    return total_bytes, files_touched, rgs_touched


#


def _empty_result(filter_col: str, method: str) -> QueryResult:
    out = _OUT_SCHEMAS["token_idx" if filter_col == "feature" else "feature"]
    return QueryResult(
        table=out.empty_table(), elapsed_s=0.0, bytes_read=0,
        files_touched=0, row_groups_touched=0, method=method,
    )


def _pyarrow_filtered(
    files: list[Path], filter_col: str, key: int, method: str
) -> QueryResult:
    """Timed pyarrow.dataset read of ``files`` with ``filter_col == key``,
    projecting the two output columns."""
    out_cols = list(
        _OUT_SCHEMAS[
            "token_idx" if filter_col == "feature" else "feature"
        ].names
    )
    if not files:
        return _empty_result(filter_col, method)
    t0 = time.perf_counter()
    dset = pads.dataset([str(f) for f in files], format="parquet")
    table = dset.to_table(
        columns=out_cols, filter=pads.field(filter_col) == key
    )
    elapsed = time.perf_counter() - t0
    nbytes, nfiles, nrgs = _touched_stats(files, filter_col, key)
    return QueryResult(
        table=table, elapsed_s=elapsed, bytes_read=nbytes,
        files_touched=nfiles, row_groups_touched=nrgs, method=method,
    )


#


def tokens_for_feature(
    store_dir: Path | str, feature: int, layout: str = "bucketed"
) -> QueryResult:
    """All ``(token_idx, value)`` rows of ``feature``."""
    d = resolve_layout_dir(store_dir, layout)
    bmap = load_bucket_map(d)
    if bmap["key"] == "feature":
        files = _partition_files(d, bucket_for_key(feature, bmap))
        method = "bucketed"
    else:
        files = _all_partition_files(d)
        method = "token_bucketed_scan"
    return _pyarrow_filtered(files, "feature", feature, method)


def features_for_token(
    store_dir: Path | str, token_idx: int, layout: str = "token"
) -> QueryResult:
    """All ``(feature, value)`` rows active at ``token_idx``."""
    d = resolve_layout_dir(store_dir, layout)
    bmap = load_bucket_map(d)
    if bmap["key"] == "token_idx":
        files = _partition_files(d, bucket_for_key(token_idx, bmap))
        method = "token_bucketed"
    else:
        files = _all_partition_files(d)
        method = "feature_bucketed_scan"
    return _pyarrow_filtered(files, "token_idx", token_idx, method)


#


def flat_scan_tokens_for_feature(
    store_dir: Path | str, feature: int
) -> QueryResult:
    """pyarrow full scan of work/flat with the feature filter."""
    return _pyarrow_filtered(
        _flat_files(store_dir), "feature", feature, "flat_pyarrow"
    )


def flat_scan_features_for_token(
    store_dir: Path | str, token_idx: int
) -> QueryResult:
    return _pyarrow_filtered(
        _flat_files(store_dir), "token_idx", token_idx, "flat_pyarrow"
    )


def _duckdb_flat(
    store_dir: Path | str,
    filter_col: str,
    key: int,
    con: duckdb.DuckDBPyConnection | None = None,
) -> QueryResult:
    files = _flat_files(store_dir)
    glob = f"{_sql_path(Path(store_dir) / FLAT_SUBDIR)}/{FLAT_ROWS_GLOB}"
    out_cols = _OUT_SCHEMAS[
        "token_idx" if filter_col == "feature" else "feature"
    ].names
    sql = (
        f"SELECT {', '.join(out_cols)} FROM read_parquet('{glob}') "
        f"WHERE {filter_col} = ?"
    )
    own = con is None
    if own:
        con = duckdb.connect()
    try:
        t0 = time.perf_counter()
        table = con.execute(sql, [key]).to_arrow_table()
        elapsed = time.perf_counter() - t0
    finally:
        if own:
            con.close()
    # DuckDB does not expose bytes/row-groups read.
    return QueryResult(
        table=table, elapsed_s=elapsed,
        bytes_read=sum(f.stat().st_size for f in files),
        files_touched=len(files), row_groups_touched=None,
        method="flat_duckdb",
    )


def duckdb_scan_tokens_for_feature(
    store_dir: Path | str,
    feature: int,
    con: duckdb.DuckDBPyConnection | None = None,
) -> QueryResult:
    """``SELECT token_idx, value FROM read_parquet('work/flat/rows-*.parquet')"""
    return _duckdb_flat(store_dir, "feature", feature, con)


def duckdb_scan_features_for_token(
    store_dir: Path | str,
    token_idx: int,
    con: duckdb.DuckDBPyConnection | None = None,
) -> QueryResult:
    return _duckdb_flat(store_dir, "token_idx", token_idx, con)


#


def canonical_table(table: pa.Table) -> pa.Table:
    """Schema-normalized (nullable dropped, columns ordered as written) and row-
    sorted copy, so results from different engines compare exactly."""
    names = sorted(table.column_names)
    target = pa.schema(
        [
            pa.field(
                n,
                pa.float32() if n == "value" else pa.uint32(),
            )
            for n in names
        ]
    )
    t = table.select(names).cast(target).combine_chunks()
    return t.sort_by([(n, "ascending") for n in names])


def assert_same_rows(results: dict[str, pa.Table]) -> None:
    """Assert every method returned the identical row (multi)set."""
    items = list(results.items())
    ref_name, ref = items[0]
    ref_c = canonical_table(ref)
    for name, other in items[1:]:
        other_c = canonical_table(other)
        if ref_c.num_rows != other_c.num_rows or not ref_c.equals(other_c):
            raise AssertionError(
                f"result mismatch: {ref_name} returned {ref_c.num_rows} "
                f"rows, {name} returned {other_c.num_rows} rows (or same "
                f"count, different contents)"
            )


#


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m store.queries",
        description="Run one canonical query and print rows + timing.",
    )
    p.add_argument("--store", type=Path, default=Path("work"),
                   help="store root containing flat/, bucketed/, "
                        "bucketed_by_token/ (default: work/)")
    p.add_argument("--query", choices=["tokens-for-feature",
                                       "features-for-token"], required=True)
    p.add_argument("--key", type=int, required=True,
                   help="feature index or token_idx")
    p.add_argument("--layout", choices=sorted(set(LAYOUT_DIRS)),
                   default=None,
                   help="default: the query's native layout")
    p.add_argument("--baseline", choices=["flat-pyarrow", "flat-duckdb"],
                   default=None, help="run a flat baseline instead")
    p.add_argument("--limit", type=int, default=10,
                   help="rows to print (default 10)")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    fns: dict[tuple[str, str | None], Callable[[], QueryResult]] = {
        ("tokens-for-feature", None): lambda: tokens_for_feature(
            args.store, args.key, args.layout or "bucketed"),
        ("tokens-for-feature", "flat-pyarrow"): lambda:
            flat_scan_tokens_for_feature(args.store, args.key),
        ("tokens-for-feature", "flat-duckdb"): lambda:
            duckdb_scan_tokens_for_feature(args.store, args.key),
        ("features-for-token", None): lambda: features_for_token(
            args.store, args.key, args.layout or "token"),
        ("features-for-token", "flat-pyarrow"): lambda:
            flat_scan_features_for_token(args.store, args.key),
        ("features-for-token", "flat-duckdb"): lambda:
            duckdb_scan_features_for_token(args.store, args.key),
    }
    res = fns[(args.query, args.baseline)]()
    log.info(
        "%s key=%d method=%s: %d rows in %.4fs | bytes_read=%s "
        "files=%s row_groups=%s",
        args.query, args.key, res.method, res.table.num_rows, res.elapsed_s,
        res.bytes_read, res.files_touched, res.row_groups_touched,
    )
    print(res.table.slice(0, max(args.limit, 0)))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
