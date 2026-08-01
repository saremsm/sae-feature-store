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
tests/            CPU-only tests (fake SAE repo + fake model; one skippable
                  real-GPT-2 test)
work/             (gitignored) flat/ staging output; bucketed layout in partitioning
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
