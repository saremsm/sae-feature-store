# sae-feature-store

A columnar store for sparse autoencoder feature activations at the ~1B-row
scale, plus a benchmark of the two canonical query shapes. It encodes a
held-out token shard with a trained SAE from the sibling `sae-gpt2-small`
project, writes one `(token_idx uint32, feature uint32, value float32)` row
per active feature to Parquet (zstd), and materializes the same rows three
ways -- flat token-order staging, feature-bucketed, token-bucketed -- to
measure what each layout buys.

The data: GPT-2-small residual stream at layer 8 (`blocks.8.hook_resid_post`),
encoded by the `topk_x8_k32` checkpoint (top-k SAE, expansion 8, k=32,
6,144 features) over the held-out shard `holdout.bin` -- 252,754 sequences of
128 tokens from `monology/pile-uncopyrighted` docs [0, 20000), of which
**32,099,758** BOS-excluded tokens were encoded. Top-k with no dead features
means exactly k rows per token, so mean L0 is exactly **32.0** and the store
holds 32,099,758 x 32 = **1,027,192,256** rows.

All numeric tables below are generated from the measured artifacts
(`results/bench.json`, `work/bucketed/stats.json` + `meta.json`,
`work/bucketed_by_token/stats.json`) by `python -m store.report`; rebuild
them with `python -m store.report > README_tables.md`.

Provenance of `work/`: the committed metadata is from the 2026-08-17/18 rebuild of the store (dump then partition, same checkpoint, shard, and pipeline); `results/bench.json` was recorded against the 2026-08-15/16 build of the same inputs. The metas' `git.sae_repo` field records the dump host's checkout and does not resolve in the published history; the checkpoint is identified by its sha256 above.

## Provenance

- bench run: 2026-08-16T00:29:54+00:00; store `work`; args: 20 features x 20 tokens x 5 trials, seed 0
- cache drop before each cold trial: **drop_caches**
- host: Intel(R) Xeon(R) Platinum 8358 CPU @ 2.60GHz x30, 222.2 GiB RAM
- versions: python 3.10.12, duckdb 1.5.5, pyarrow 25.0.1
- source data: checkpoint `topk_x8_k32` (sha256 f78463dd1520...), hook `blocks.8.hook_resid_post`, shard `holdout.bin`

## Scale

- rows: **1,027,192,256** over **32,099,758** tokens x **6,144** features, mean L0 **32**
- dense baseline: 32,099,758 tokens x 6,144 features x 2 B (fp16) = **394,441,826,304 B** (367.4 GiB)

| layout | files | row groups | on-disk bytes | size | bytes/row | x smaller than dense | size vs flat |
|---|---|---|---|---|---|---|---|
| dense fp16 (hypothetical) | - | - | 394,441,826,304 | 367.4 GiB | 384.00 | 1.0x | 67.12x |
| flat (token order) | 20 | - | 5,876,266,151 | 5.5 GiB | 5.72 | 67.1x | 1.00x |
| feature-bucketed | 128 | 1091 | 6,860,123,529 | 6.4 GiB | 6.68 | 57.5x | 1.17x |
| token-bucketed | 128 | 1152 | 5,237,653,842 | 4.9 GiB | 5.10 | 75.3x | 0.89x |

Reading the table: sparsity is doing almost all the work -- storing only the
~0.52% of (token, feature) cells that are nonzero (32 of 6,144) shrinks the
data 57-75x versus a dense fp16 matrix, and zstd on top gets the flat layout
to 5.72 B/row against 12 B/row uncompressed (~2.1x). The feature-bucketed
copy is 1.17x the flat bytes because its `(feature, token_idx)` sort
scrambles `token_idx`, the widest and previously delta-friendly column; the
token-bucketed copy keeps the dump order and compresses best at 5.10 B/row.

## Layout

Both bucketed layouts are hive-partitioned into 128 directories
(`bucket=NN/data_0.parquet`), one file per bucket, 1,000,000-row groups,
zstd. The feature layout assigns `bucket = (feature * 128) // 6144`, so each
bucket holds a contiguous range of 6,144 / 128 = 48 features, and every file
is sorted by `(feature, token_idx)`. The token layout is the mirror image:
`bucket = (token_idx * 128) // 32099758`, files sorted by
`(token_idx, feature)`.

The point of bucket + sort + bounded row groups is that a point lookup never
opens what it does not need. Partition pruning via `bucket_map.json` picks
the single file whose key range contains the key, and inside that file the
sort makes each row group's min/max statistics for the key column tight and
non-overlapping, so the reader admits only the row group(s) that can contain
the key and predicate pushdown skips the rest without decoding them. At 48
features and ~8.0M rows (1,027,192,256 / 128) per bucket, a bucket spans
~8.5 row groups, so most features sit inside a single row group; the
measured population mean is **1.157 row groups touched per feature** (963
boundary-straddling pairs). The token layout hits exactly **1.000**: at k=32
rows per token, a 1,000,000-row group holds exactly 31,250 whole tokens, so
row-group cuts always land on token boundaries. Sort order is enforced by a
finalize pass after DuckDB's partitioned COPY (which does not preserve ORDER
BY inside partitions -- on the host, all 128 came back unsorted) and
re-verified by `python -m store.partition --check`.

## Query latency

### tokens_for_feature (cold)

| method | p50 (s) | p90 (s) | p99 (s) | mean (s) | rows | bytes read | files | row groups |
|---|---|---|---|---|---|---|---|---|
| feature-bucketed | 0.0532 | 0.0633 | 0.0807 | 0.0538 | 134,704 | 7.1 MiB | 1 | 1.1 |
| token-bucketed scan | 1.5539 | 1.5727 | 1.8511 | 1.5629 | 134,704 | 4.8 GiB | 128 | 1152.0 |
| flat scan (pyarrow) | 1.4643 | 1.4881 | 1.8632 | 1.4756 | 134,704 | 5.5 GiB | 20 | 1046.0 |
| flat scan (DuckDB) | 1.1239 | 1.2767 | 1.4078 | 1.1266 | 134,704 | 5.5 GiB | 20 | - |

### tokens_for_feature (warm)

| method | p50 (s) | p90 (s) | p99 (s) | mean (s) | rows | bytes read | files | row groups |
|---|---|---|---|---|---|---|---|---|
| feature-bucketed | 0.0149 | 0.0166 | 0.0200 | 0.0142 | 134,704 | 7.1 MiB | 1 | 1.1 |
| token-bucketed scan | 1.1949 | 1.2105 | 1.2228 | 1.1954 | 134,704 | 4.8 GiB | 128 | 1152.0 |
| flat scan (pyarrow) | 1.0155 | 1.0354 | 1.0523 | 1.0168 | 134,704 | 5.5 GiB | 20 | 1046.0 |
| flat scan (DuckDB) | 0.5349 | 0.6978 | 0.8425 | 0.5335 | 134,704 | 5.5 GiB | 20 | - |

### features_for_token (cold)

| method | p50 (s) | p90 (s) | p99 (s) | mean (s) | rows | bytes read | files | row groups |
|---|---|---|---|---|---|---|---|---|
| token-bucketed | 0.0255 | 0.0491 | 0.0514 | 0.0295 | 32 | 4.5 MiB | 1 | 1.0 |
| feature-bucketed scan | 1.5659 | 1.6017 | 1.6158 | 1.5686 | 32 | 5.7 GiB | 128 | 968.6 |
| flat scan (pyarrow) | 0.0390 | 0.0691 | 0.0737 | 0.0415 | 32 | 5.0 MiB | 1 | 1.0 |
| flat scan (DuckDB) | 0.1532 | 0.1565 | 0.1624 | 0.1531 | 32 | 5.5 GiB | 20 | - |

### features_for_token (warm)

| method | p50 (s) | p90 (s) | p99 (s) | mean (s) | rows | bytes read | files | row groups |
|---|---|---|---|---|---|---|---|---|
| token-bucketed | 0.0108 | 0.0128 | 0.0138 | 0.0109 | 32 | 4.5 MiB | 1 | 1.0 |
| feature-bucketed scan | 1.1437 | 1.1619 | 1.1740 | 1.1418 | 32 | 5.7 GiB | 128 | 968.6 |
| flat scan (pyarrow) | 0.0167 | 0.0229 | 0.0249 | 0.0173 | 32 | 5.0 MiB | 1 | 1.0 |
| flat scan (DuckDB) | 0.1198 | 0.1215 | 0.1252 | 0.1200 | 32 | 5.5 GiB | 20 | - |

### Kill point

- KILL-POINT tokens_for_feature cold p99: bucketed 0.0807s vs flat scan 1.8632s -> 23.1x (threshold >= 5.0x): PASS

Read together: each layout wins exactly the query it was sorted for,
and loses the other one badly. `tokens_for_feature` on the feature layout
touches 1 file / ~1.1 row groups / 7.1 MiB versus the flat scan's 20 files /
1,046 row groups / 5.5 GiB -- ~786x fewer bytes read for 27.5x cold p50
(1.4643 s -> 0.0532 s), 23.1x cold p99 (the kill-point gate, threshold 5x),
and ~68x warm p50; the speedup is smaller than the byte ratio because a
fixed footer-read + row-group-decode cost -- roughly the bucketed warm p50,
~15 ms -- does not shrink with the bytes. It also beats
the stronger flat baseline: DuckDB's parallel scan cuts the flat cold p99 to
1.4078 s, still 17.4x slower than the bucketed read. The shape the feature
layout is bad at is `features_for_token`: 32 rows cost a 128-file, 5.7 GiB
scan at 1.57 s cold -- ~61x slower than the token layout's 25 ms, and worse
than doing no layout work at all, since flat pyarrow answers the same query
in 39 ms cold. That flat number is not luck: the flat files are written in
token order, so `token_idx` row-group statistics prune 1,045 of 1,046 row
groups even without partitioning. The mirror holds too -- the feature query
over the token layout costs 1.55 s, as bad as a flat scan. Conclusions:
neither single layout serves both queries; the feature-bucketed copy is the
essential one (nothing else answers its query fast) while the token-bucketed
copy is cheap insurance over the already-token-ordered flat set (25 ms vs 39
ms cold, 1 footer vs 20).

## Scale-up to 1e+12 tokens (EXTRAPOLATED)

All numbers in this table are **extrapolated** from the measured bytes/row and mean L0 above; the scale factor over the measured run is 1,000,000,000,000 / 32,099,758 = **31,153x** tokens.

| quantity (extrapolated) | value | arithmetic |
|---|---|---|
| rows | 3.20e+13 | 32 rows/token x 1,000,000,000,000 tokens |
| feature-bucketed storage | 213.7 TB | 6.6785 B/row x 3.20e+13 rows |
| flat storage | 183.1 TB | 5.7207 B/row x 3.20e+13 rows |
| token-bucketed storage | 163.2 TB | 5.0990 B/row x 3.20e+13 rows |
| dense fp16 baseline | 12,288.0 TB | 1,000,000,000,000 x 6,144 x 2 B |
| rows per feature (mean) | 5.21e+09 | 3.20e+13 / 6,144 features |
| row groups per feature (mean) | 5,208 | 5.21e+09 / 1,000,000-row groups |
| cold read per mean feature query | 34.8 GB | 213.7 TB / 6,144 features |
| per-bucket file at 128 buckets | 1.7 TB | 213.7 TB / 128 |
| ingest sort input (uncompressed) | 384.0 TB | 3.20e+13 rows x 12 B |

How the two queries change (extrapolated): `tokens_for_feature` stops being
a point read. A mean feature's rows grow with its frequency -- 31,153x more
tokens means ~5.2e9 rows and ~35 GB compressed (extrapolated) for the mean
feature, spread over ~5,208 row groups -- so cold latency becomes
sequential-read bound and scales linearly with feature frequency; rare
features stay cheap. `features_for_token` stays
O(k): 32 rows in one row group in one file regardless of corpus size,
provided partition count grows so per-file row-group counts (and footers)
stay bounded -- its cost is a footer plus one row-group decode, roughly the
constant measured here.

What breaks, in order: (1) per-bucket file sizes -- at 128 buckets each
partition is a ~1.7 TB (extrapolated) Parquet file holding ~250,000
1M-row groups (3.20e+13 rows / 128 buckets / 10^6); footers alone become a
problem, so bucket count must scale with data. (2)
The ingest sort -- the measured run globally sorted 12.3 GB uncompressed
(1,027,192,256 x 12 B) inside a 48 GB memory limit; 3.2e13 rows are 384 TB
(extrapolated) of sort input, far beyond RAM, and the `--per-bucket-passes`
fallback becomes 128+ full scans of a ~183 TB flat set. (3) `token_idx`
itself -- one monotone counter over a single shard does not survive many
ingest batches, and mapping tokens back to documents needs an explicit
token -> (shard, doc, position) index rather than arithmetic.

The design that fixes it: partition first by document/time range -- each
ingest shard (say 10^9-10^10 tokens) is dumped, sorted, and feature-bucketed
independently, so every sort is the size measured here and ingest is
append-only with no global reshuffle. Within a range, keep the feature
buckets; across ranges, maintain a per-feature secondary index mapping
feature -> (range, file, row-group / byte extent), so `tokens_for_feature`
issues targeted range reads across shards instead of consulting thousands of
footers, and can read only the ranges a caller asks for. Add the token ->
document index as a small per-range table (the `tokens-*.parquet` sidecars
already hold the mapping), and tier the ranges: recent/hot on NVMe, the long
tail in object storage, with the secondary index deciding what to fetch.

## Reproduce

The four commands (GPU box; `--sae-repo` defaults to `~/sae-gpt2-small`,
override via flag or the `SAE_REPO` env var -- `store/sae_import.py` is the
only module that touches `sys.path`):

```bash
# 1. encode the holdout shard -> flat staging Parquet (GPU, ~8 min)
python -m store.dump \
  --checkpoint ~/sae-gpt2-small/results/frontier/topk_x8_k32/checkpoint.pt \
  --shard ~/sae-gpt2-small/data/holdout.bin \
  --n-tokens 200000000 --out work/flat/ \
  --batch-seqs 512 --rows-per-file 50000000

# 2. feature-bucketed layout (CPU)
python -m store.partition --flat work/flat/ --out work/bucketed/ \
  --n-buckets 128 --row-group-size 1000000 --threads 16 --memory-limit 48GB

# 3. token-bucketed layout (CPU)
python -m store.partition --flat work/flat/ --out work/bucketed_by_token/ \
  --layout token \
  --n-buckets 128 --row-group-size 1000000 --threads 16 --memory-limit 48GB

# 4. cold/warm benchmark (root so cold trials can drop_caches)
sudo -E python -m store.bench --store work/ \
  --n-features 20 --n-tokens 20 --trials 5 --out results/bench.json
```

Then `python -m store.partition --check work/bucketed/` (and
`.../bucketed_by_token/`) re-verifies an existing dataset, and
`python -m store.report > README_tables.md` regenerates every numeric table
in this README from `results/bench.json` + the `work/` stats/meta. Tests
(CPU-only, no GPU or network; green on Windows/Git Bash and Linux):
`python -m pytest -q`.

## Limitations

**Kill point:** if the bucketed layout's cold p99 for `tokens_for_feature`
is not >= 5x (`--kill-threshold`) faster than the flat pyarrow scan's cold
p99, the bench prints a diagnosis (bucket too coarse? row groups too large
vs rows returned? feature filter not pushing down? cold not actually cold?)
and exits 2. Other flags: `--seed`, `--cache-drop
auto|drop_caches|fadvise|none`.

Tests (CPU, no GPU or network needed; the one real-GPT-2 test skips itself
if GPT-2 can't be loaded):

```bash
python -m pytest -q
```

## Status

- **Dump** (verified on the host): import shim, schemas +
  token_idx mapping, sequential dump to staging Parquet, tests (27 CPU tests,
  incl. one real-GPT-2 end-to-end).

  Measured dump run (`topk_x8_k32`, the full holdout shard):

  - `work/flat/`: **1,027,192,256 rows** over **32,099,758 tokens**, mean L0
    exactly **32.000** (matches the checkpoint's `metrics.json` `l0: 32.0`;
    `dead_frac_eval: 0.0`, so topk yields exactly k rows per token).
  - 20 rows files (19 x 52,019,200 + one 38,827,456 tail; files cut at batch
    boundaries, so counts overshoot `--rows-per-file 50000000` slightly - by
    design, it keeps `--resume` restart points on whole sequences) + 20
    paired tokens files + `meta.json`.
  - **5.7 GB total, ~5.8 bytes/row** zstd-compressed (~286 MB per 52M-row
    file; 12 bytes/row uncompressed, ~2.1x compression).
  - Throughput on the host (A100-class, `--batch-seqs 512`): ~70k tok/s /
    ~2.23M rows/s, flat from start to finish; the full dump took ~7.7 min.
  - Adapter notes, verified against the real sibling repo: the shard opens
    via `data.TokenShard` (array at `._tokens`; a raw-memmap fallback exists
    and is tested to agree), the sidecar `n_seqs` is cross-checked, and the
    L0 reference is found nested under `metrics.json`'s `"metrics"` key.
    `store.dump` also tolerates the model and SAE living on different
    devices (transformer_lens auto-moves GPT-2 to CUDA regardless of
    `--device`).
- **Partition** (verified on the host): `store.partition`  - 
  feature-bucketed and token-bucketed hive layouts via DuckDB, bucket_map /
  stats / forwarded meta, `--per-bucket-passes` fallback, `--check`
  verifier. 19 new CPU tests (45 pass + 1 skippable real-GPT-2).

  Measured partition runs (both from the 1,027,192,256-row `work/flat/`, 128
  buckets, 1M row groups, `--threads 16 --memory-limit 48GB`; `--check`
  re-run after each and idempotent):

  - `work/bucketed/` (feature layout): **72.7 s** single COPY + finalize;
    **6.86 GB** (~6.7 bytes/row), 128 files, 1,091 row groups,
    **mean_row_groups_touched_per_feature = 1.157**, 963 boundary-shared
    row-group pairs (48 features/bucket, ~8.0M rows/bucket → ~8.5 row
    groups/bucket, so most features land inside one row group and the rest
    straddle one cut).
  - `work/bucketed_by_token/` (token layout): **57.8 s**; **5.24 GB**
    (~5.1 bytes/row), 128 files, 1,152 row groups,
    **mean_row_groups_touched_per_token = 1.000** with 0 boundary-shared
    pairs - not luck: at exactly k=32 rows/token, a 1,000,000-row group
    holds exactly 31,250 whole tokens, so row-group cuts always fall on
    token boundaries.
  - The finalize pass consolidated/re-sorted all 128 partitions in both
    runs - i.e. on the host, DuckDB 1.5.5's multi-threaded partitioned COPY
    left *every* partition unsorted, confirming the pass is load-bearing at
    scale, not just under the sandbox repro.
  - Size vs the 5.7 GB flat set: token layout compresses better (its
    global order matches the dump order, so `token_idx` stays
    delta-friendly), the feature layout ~20% worse (sorting by feature
    scrambles `token_idx`, the widest column). Expected, and the price of
    1-partition feature lookups.
- **Benchmark** (verified on the host): `store.queries` +
  `store.bench` - canonical queries with per-query cost accounting,
  cold/warm benchmark with stratified sampling, correctness cross-check,
  hardware capture, JSON + markdown reports, and the 5x kill-point gate.
  28 new CPU tests (73 pass + 1 skippable real-GPT-2), green on both the
  Windows dev box and the Lambda box.

  Measured benchmark run (`results/bench.{json,md}`; 20 features stratified by
  frequency tercile + 20 uniform tokens, 5 cold+warm trials each, root
  `drop_caches` before every cold trial; Xeon 8358 x30, 222 GB RAM,
  DuckDB 1.5.5 / pyarrow 25.0.1):

  - **Kill point PASS: 23.1x** - `tokens_for_feature` cold p99 0.081 s
    bucketed vs 1.863 s flat pyarrow scan (p50: 0.053 s vs 1.464 s; warm
    p50: 0.015 s vs 1.016 s, ~71x). Against the tougher flat baseline 
    (DuckDB, cold p99 1.408 s) the speedup is still 17.4x - both flat 
    engines lose the same way. All four methods returned identical
    row sets for every sampled key before timing.
  - The mechanism, from the recorded cost columns: a bucketed feature
    query touches **1 file / 1.1 row groups / 7.1 MiB** (matching the partition's
    population mean of 1.157 row groups per feature) vs the flat scan's
    20 files / 1,046 row groups / 5.5 GiB - pushdown prunes nothing on
    token-ordered flat files. ~790x fewer bytes buys ~23-27x latency; the
    remainder is fixed footer + one-row-group decode cost.
  - Mirror image confirmed: `features_for_token` on the token layout is
    26 ms cold / 11 ms warm (1 file / 1 row group / 4.5 MiB, exactly 32
    rows every time at k=32), vs **1.57 s** scanning the feature layout
    (~61x). And the feature query over the token layout costs 1.55 s  - 
    as bad as no layout. Neither layout serves both queries; each is
    ~20-60x on its own.
  - Predicted quirk, measured: flat pyarrow answers the *token* query in
    39 ms cold - the flat set is token-ordered, so `token_idx` statistics
    prune 1,045/1,046 row groups. The token layout still wins (1 footer
    vs 20) but is cheap insurance; the feature layout is the essential
    one. Caveat: `lsblk` shows `ROTA=1` on the virtio disks, which often
    misreports - cold absolutes stand on the drop_caches protocol, not
    on a device claim.
  - Single node, single NVMe-class local disk (`lsblk` reports `ROTA=1` on the
    virtio devices, which commonly misreports -- cold absolutes rest on the
    `drop_caches` protocol, not on a device claim). No network storage, no
    concurrency, no writer/reader contention.
  - One SAE width (6,144 features) and one sparsity regime (top-k, k=32, so L0
    is exactly 32 for every token). A ReLU/L1 SAE with heavy-tailed L0, or a
    much wider SAE, shifts the bucket-balance and bytes/row numbers.
  - Synthetic query mix: single-key point lookups only (20 features stratified
    by frequency tercile, 20 uniform tokens, 5 trials each). No range queries,
    no top-k-by-value, no joins against the token table, no concurrent load.
  - 32.1M tokens / 1.03B rows measured; every 10^12-token number above is
    extrapolated arithmetic, labeled as such, not a measurement.
  - The kill-point ratio (23.1x) compares against a full flat scan; a smarter
    baseline (e.g. flat data sorted by feature) would narrow it -- that
    baseline is what the feature-bucketed layout *is*.
