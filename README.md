# sae-feature-store

Columnar store for sparse autoencoder feature activations. Encodes held-out
tokens with a trained SAE from the sibling `sae-gpt2-small` project, writes
`(token_idx, feature, value)` rows to Parquet (zstd), and - in later stages  - 
buckets them by feature index and benchmarks the two canonical query shapes.

Target scale: >= 1B rows minimum (met: see Status), ~5B rows if a 200M-token
held-out shard is ever generated. The current holdout shard is 32,099,758
BOS-excluded stream tokens; `--n-tokens` beyond that caps with a warning. All
bulk output lives under `./work/` (gitignored) on local NVMe. Runs on Python
3.10+ (both dev machines use 3.10).

## Layout

```
store/
  sae_import.py   import shim for ~/sae-gpt2-small (the ONLY sys.path toucher)
  schema.py       Arrow schemas, layout constants, the token_idx mapping
  dump.py         Dump: sequential GPU encode -> work/flat/ staging Parquet
  partition.py    Partitioning: DuckDB bucketing of the flat rows by feature (or
                  token_idx) range -> work/bucketed/, work/bucketed_by_token/
tests/            CPU-only tests (fake SAE repo + fake model; one skippable
                  real-GPT-2 test)
work/             (gitignored) flat/ staging; bucketed/ + bucketed_by_token/
results/          committed benchmark outputs (numbers come only from real runs)
```

## The `--sae-repo` contract

The SAE project lives at `~/sae-gpt2-small` on the GPU box and is **not**
pip-installed. Every entry point takes `--sae-repo` (default: the `SAE_REPO`
env var, else `~/sae-gpt2-small`). Exactly one module  - 
`store/sae_import.py` - inserts that path into `sys.path` and imports
`SparseAutoencoder` (from `sparse_autoencoder.py`) plus `TokenShard` /
`ActivationLoader` (from `data.py`) and `evaluate` when present. No other
file touches `sys.path`. If the path or an expected symbol is missing you get
an actionable error naming the file it expected and telling you to pass
`--sae-repo`.

Tests never need the real repo: they either write a tiny fake repo to a temp
directory and point `--sae-repo` at it, or pass a pre-built `SAEModules`
namespace (and a fake model) directly into `store.dump.main(...)`.

The store only ever reads the shard **sequentially** - the SAE repo's
shuffling ring-buffer loader is deliberately unused - so `token_idx` is
monotone and reproducible run-to-run.

## Data model

Written by `store.dump` into `--out` (e.g. `work/flat/`):

- `rows-NNNNN.parquet` - one row per (token, active feature):
  `token_idx uint32, feature uint32, value float32`. ~50M rows per file,
  1M-row groups, zstd. `feature_bucket` is *not* stored; it is derived from
  `feature` in partitioning bucketing stage.
- `tokens-NNNNN.parquet` - one row per encoded token:
  `token_idx uint32, seq_idx uint32, pos uint16, token_id uint16`, so
  "features at token t" can be decoded back to text later.
- `meta.json` - checkpoint path + sha256, normalized SAE config (activation,
  k / l1_coeff, expansion, n_features, input_scale, plus the raw config
  dict), hook name, shard path + sidecar contents, n_tokens encoded, n_rows,
  mean L0, rows_per_file, row_group_size, created_at, git SHAs of both
  repos, and the completed-segment list `--resume` uses.

**token_idx** is the index of a token in the sequential, BOS-excluded stream
over the held-out shard (`[n_seqs, seq_len]`, BOS at position 0 of every
sequence):

```
token_idx = seq_idx * (seq_len - 1) + (pos - 1)        # 1 <= pos < seq_len
```

The mapping (and its inverse) is a pure function in `store/schema.py`
(`token_index` / `token_to_seq_pos`); `store.dump` asserts against it on
every batch, so the ROWS and TOKENS tables cannot disagree.

Written by `store.partition` into `--out` (e.g. `work/bucketed/`):

- `bucket=NN/data_*.parquet` - hive-partitioned rows, same three columns.
  `bucket` lives only in the directory name, never inside the files; read
  the dataset back with `read_parquet(..., hive_partitioning=1)`. Bucket
  `b` holds the contiguous feature range
  `[ceil(b*n_features/n_buckets), ceil((b+1)*n_features/n_buckets))` via
  `bucket = feature * n_buckets // n_features`, and every file is sorted by
  `(feature, token_idx)`, so a single-feature lookup touches exactly one
  partition and skips row groups via their min/max `feature` statistics.
  Adjacent row groups may share the one boundary feature that straddles a
  row-group cut; the check enforces `max(rg_i) <= min(rg_{i+1})`.
- `bucket_map.json` - layout, key column, `n_buckets`, key domain, the
  bucket SQL expression, and every bucket's `[lo, hi)` key range.
- `stats.json` - per-bucket rows, bytes, features, min/max rows per
  feature, files, row groups; plus totals and the measured mean
  row-groups-touched-per-feature.
- `meta.json` - the flat set's meta copied forward with a `partition` block
  added (layout, n_buckets, domain, bucket expr, row_group_size, threads,
  memory_limit, per_bucket_passes, source flat dir + measured
  `source_rows`, created_at).

`--layout token` produces the same on-disk shape with `token_idx` as the
bucketing key (domain = `n_tokens_encoded`, files sorted by
`(token_idx, feature)`, stats fields named per token), written to
`work/bucketed_by_token/`. **Tradeoff:** the feature-bucketed layout answers
"all tokens where feature f fires" from one partition but scatters "all
features at token t" across every bucket; the token-bucketed layout is the
mirror image. Neither layout is right for both canonical queries - that is
the point, and the benchmark benchmark will show it.

## Commands

Sanity smoke on the host (~1 min), then the full run:

```bash
python -m store.dump \
  --checkpoint ~/sae-gpt2-small/results/frontier/<best>/checkpoint.pt \
  --shard ~/sae-gpt2-small/data/holdout.bin \
  --n-tokens 2000000 --out work/flat_smoke/

python -m store.dump \
  --checkpoint ~/sae-gpt2-small/results/frontier/<best>/checkpoint.pt \
  --shard ~/sae-gpt2-small/data/holdout.bin \
  --n-tokens 200000000 --out work/flat/ \
  --batch-seqs 512 --rows-per-file 50000000
```

Useful flags: `--sae-repo` (see contract above), `--resume` (continue an
interrupted run; complete files are skipped based on `meta.json`),
`--encode-chunk` (tokens per `sae.encode` call - caps the dense activation
buffer; lower it if the SAE is wide and GPU memory is tight), `--device`,
`--layer` / `--hook-name` (default: the checkpoint's layer, else 8, at
`blocks.<layer>.hook_resid_post`).

The dump loads GPT-2-small via transformer_lens, runs to the residual hook
with `stop_at_layer` under `torch.no_grad()` (bf16 autocast on CUDA, fp32
into the SAE), and logs tokens/s and rows/s as it goes - this step is
GPU-forward-bound. At the end it hard-asserts `rows written == sum of
per-token L0` accumulated on the device, and warns (never fails) if mean L0
deviates >5% from the checkpoint's `metrics.json`.

partitioning partitioning on the host (CPU-only; ~1B rows fits the 48 GB sort budget
with huge headroom on the 222 GB machine):

```bash
python -m store.partition --flat work/flat/ --out work/bucketed/ \
  --n-buckets 128 --row-group-size 1000000 --threads 16 --memory-limit 48GB

python -m store.partition --flat work/flat/ --out work/bucketed_by_token/ \
  --layout token \
  --n-buckets 128 --row-group-size 1000000 --threads 16 --memory-limit 48GB

python -m store.partition --check work/bucketed/
python -m store.partition --check work/bucketed_by_token/
```

The default path is one DuckDB `COPY ... PARTITION_BY (bucket)` over a
global `ORDER BY bucket, feature, token_idx` (spill dir: `--temp-dir`,
default `<out>/_duckdb_tmp`, i.e. under `work/`), followed by a finalize
pass that leaves every partition as a single sorted `data_0.parquet`  - 
necessary because DuckDB's multi-threaded hive-partitioned COPY does *not*
preserve the ORDER BY inside partition files (measured on 1.5.5: several of
128 partitions come back unsorted at `--threads 4`+); partitions that
already arrive as one sorted file are detected cheaply and left untouched.
`--per-bucket-passes`
instead runs one `WHERE bucket = i` pass per bucket: n_buckets scans of the
flat set, but each sort holds only ~1/n_buckets of the rows. Use it only if
the global sort exceeds `--memory-limit` (the hypothetical ~5B-row / 200M
token case); at the measured 1.03B rows the global sort fits in RAM and the
single pass is faster. Both paths produce byte-for-row identical, identically
ordered partitions (tested). Reruns into an existing `--out` first clear old
`bucket=*` dirs, so stale files never mix in. `--check <dir>` re-verifies an
existing dataset: row totals vs the recorded source count, every feature in
exactly one (correct) bucket, per-file sort order, row-group statistics
present and non-overlapping, and stats.json still true; it exits non-zero on
any violation and logs the mean row-groups-touched-per-feature.

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
