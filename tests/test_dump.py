"""Tests for store.dump: tiny CPU pipeline with a fake SAE repo and a fake model."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow.parquet as pq
import pytest
import torch

import store
from store import dump as dump_mod
from store import schema
from store.sae_import import load_sae_from_checkpoint, load_sae_modules

from .conftest import FakeModel, make_checkpoint, make_fake_repo, make_shard

D_MODEL = 16
N_FEATURES = 64
N_SEQS = 32
SEQ_LEN = 16
PER_SEQ = SEQ_LEN - 1
N_STREAM = N_SEQS * PER_SEQ  # 480
BATCH_SEQS = 8
ENCODE_CHUNK = 64  # < batch tokens (120), so chunking is exercised
ROWS_PER_FILE = 3000  # forces several files
LAYER = 0
HOOK = f"blocks.{LAYER}.hook_resid_post"


def _setup(root: Path, *, tag: str, seed: int = 11) -> SimpleNamespace:
    repo = make_fake_repo(root / "repo", tag=tag)
    mods = load_sae_modules(str(repo))
    cfg_cls = sys.modules["sparse_autoencoder"].SAEConfig
    ckpt = make_checkpoint(
        root / "checkpoint.pt",
        mods.SparseAutoencoder,
        cfg_cls,
        d_model=D_MODEL,
        n_features=N_FEATURES,
        seed=seed,
        layer=LAYER,
    )
    shard = make_shard(root / "holdout.bin", n_seqs=N_SEQS, seq_len=SEQ_LEN, seed=2)
    model = FakeModel(d_model=D_MODEL)
    sae = load_sae_from_checkpoint(mods, ckpt).sae
    return SimpleNamespace(
        root=root, repo=repo, mods=mods, ckpt=ckpt, shard=shard, model=model,
        sae=sae, tokens=np.fromfile(shard, dtype=np.uint16).reshape(N_SEQS, SEQ_LEN),
    )


def _argv(env: SimpleNamespace, out: Path, n_tokens: int, *extra: str) -> list[str]:
    return [
        "--checkpoint", str(env.ckpt),
        "--shard", str(env.shard),
        "--n-tokens", str(n_tokens),
        "--out", str(out),
        "--batch-seqs", str(BATCH_SEQS),
        "--rows-per-file", str(ROWS_PER_FILE),
        "--encode-chunk", str(ENCODE_CHUNK),
        "--sae-repo", str(env.repo),
        "--device", "cpu",
        "--layer", str(LAYER),
        *extra,
    ]


def _read_meta(out: Path) -> dict:
    import json

    return json.loads((out / schema.META_FILENAME).read_text())


def _read_all(out: Path, pattern: str) -> dict[str, np.ndarray]:
    files = sorted(out.glob(pattern))
    assert files, f"no files match {pattern} in {out}"
    tables = [pq.read_table(f) for f in files]
    cols = tables[0].column_names
    return {
        c: np.concatenate([t.column(c).to_numpy() for t in tables]) for c in cols
    }


def _expected_rows(env: SimpleNamespace, n_target: int) -> dict[str, np.ndarray]:
    """Independent re-derivation of the row stream with the same batch and chunk
    shapes the dump used (identical shapes => bitwise-identical floats on CPU)."""
    tok_idx: list[np.ndarray] = []
    feat: list[np.ndarray] = []
    vals: list[np.ndarray] = []
    base = 0
    for lo in range(0, env.tokens.shape[0], BATCH_SEQS):
        batch = env.tokens[lo : lo + BATCH_SEQS]
        flat = dump_mod.batch_residuals(
            env.model, batch, HOOK, LAYER, torch.device("cpu")
        )
        keep = min(flat.shape[0], n_target - base)
        flat = flat[:keep]
        with torch.no_grad():
            for clo in range(0, keep, ENCODE_CHUNK):
                h = env.sae.encode(flat[clo : clo + ENCODE_CHUNK])
                r, c = torch.nonzero(h > 0, as_tuple=True)
                tok_idx.append((base + clo + r).numpy().astype(np.uint32))
                feat.append(c.numpy().astype(np.uint32))
                vals.append(h[r, c].numpy().astype(np.float32))
        base += keep
        if base >= n_target:
            break
    return {
        "token_idx": np.concatenate(tok_idx),
        "feature": np.concatenate(feat),
        "value": np.concatenate(vals),
    }


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory) -> SimpleNamespace:
    """One full dump over the fake pipeline, shared by the read-only tests."""
    root = tmp_path_factory.mktemp("dump-pipeline")
    env = _setup(root, tag="pipeline")
    out = root / "flat"
    rc = dump_mod.main(
        argv=_argv(env, out, N_STREAM), modules=env.mods, model=env.model
    )
    assert rc == 0
    env.out = out
    env.meta = _read_meta(out)
    return env


def test_multiple_files_and_segments_recorded(pipeline):
    rows_files = sorted(p.name for p in pipeline.out.glob("rows-*.parquet"))
    tokens_files = sorted(p.name for p in pipeline.out.glob("tokens-*.parquet"))
    assert len(rows_files) >= 3, "test sizing should force several files"
    assert len(rows_files) == len(tokens_files)
    segs = pipeline.meta["segments"]
    assert [s["rows_file"] for s in segs] == rows_files
    assert [s["tokens_file"] for s in segs] == tokens_files
    assert segs[-1]["token_end"] == N_STREAM


def test_tokens_table_decodes_back_to_shard(pipeline):
    tok = _read_all(pipeline.out, "tokens-*.parquet")
    assert np.array_equal(
        tok["token_idx"], np.arange(N_STREAM, dtype=np.uint32)
    )
    mapped = schema.token_index(
        tok["seq_idx"].astype(np.int64), tok["pos"].astype(np.int64), SEQ_LEN
    )
    assert np.array_equal(mapped, tok["token_idx"].astype(np.int64))
    assert np.array_equal(
        pipeline.tokens[tok["seq_idx"], tok["pos"]], tok["token_id"]
    )
    assert np.all(tok["pos"] >= 1)  # BOS never appears


def test_meta_contents(pipeline):
    meta = pipeline.meta
    assert meta["format_version"] == schema.FORMAT_VERSION
    assert meta["n_tokens_requested"] == N_STREAM
    assert meta["n_tokens_encoded"] == N_STREAM
    assert meta["hook_name"] == HOOK
    assert meta["layer"] == LAYER
    assert meta["compression"] == "zstd"
    assert meta["rows_per_file"] == ROWS_PER_FILE
    assert meta["row_group_size"] == schema.DEFAULT_ROW_GROUP_SIZE
    cfg = meta["sae_config"]
    assert cfg["n_features"] == N_FEATURES
    assert cfg["d_model"] == D_MODEL
    assert cfg["activation"] == "relu"
    assert cfg["input_scale"] == 1.0
    ck = meta["sae_checkpoint"]
    assert ck["path"] == str(pipeline.ckpt.resolve())
    assert len(ck["sha256"]) == 64
    assert meta["shard"]["sidecar"]["dataset"] == "fake-holdout"
    assert meta["shard"]["seq_len"] == SEQ_LEN
    assert meta["progress"]["complete"] is True
    assert set(meta["git"]) == {"feature_store", "sae_repo"}
    assert meta["created_at"]


def test_trim_to_requested_n_tokens(tmp_path):
    env = _setup(tmp_path, tag="trim")
    out = tmp_path / "flat"
    n = 100  # not a multiple of PER_SEQ: forces a mid-sequence trim
    dump_mod.main(argv=_argv(env, out, n), modules=env.mods, model=env.model)
    meta = _read_meta(out)
    assert meta["n_tokens_encoded"] == n
    tok = _read_all(out, "tokens-*.parquet")
    assert np.array_equal(tok["token_idx"], np.arange(n, dtype=np.uint32))
    rows = _read_all(out, "rows-*.parquet")
    assert rows["token_idx"].max() < n
    assert rows["token_idx"].shape[0] == meta["n_rows"] == meta["l0_sum"]


def test_refuses_dirty_out_dir_without_resume(tmp_path):
    env = _setup(tmp_path, tag="dirty")
    out = tmp_path / "flat"
    dump_mod.main(argv=_argv(env, out, 30), modules=env.mods, model=env.model)
    with pytest.raises(SystemExit):
        dump_mod.main(argv=_argv(env, out, 30), modules=env.mods, model=env.model)


class _Interrupt(RuntimeError):
    pass


def test_reference_l0_reads_top_level_and_nested(tmp_path):
    """sae-gpt2-small writes {"metrics": {"l0": ...}}; older runs may write l0 at
    the top level. Both must be found; absence returns None."""
    import json as _json

    ckpt = tmp_path / "checkpoint.pt"
    ckpt.write_bytes(b"")
    metrics = tmp_path / "metrics.json"

    metrics.write_text(_json.dumps({"l0": 27.5}))
    assert dump_mod._reference_l0(ckpt) == 27.5

    metrics.write_text(
        _json.dumps({"run": "topk_x8_k32", "metrics": {"l0": 32.0, "fvu": 0.17}})
    )
    assert dump_mod._reference_l0(ckpt) == 32.0

    metrics.write_text(_json.dumps({"metrics": {"fvu": 0.17}}))
    assert dump_mod._reference_l0(ckpt) is None

    metrics.unlink()
    assert dump_mod._reference_l0(ckpt) is None


def test_resume_after_interruption_matches_uninterrupted(tmp_path):
    env = _setup(tmp_path, tag="resume")

    ref_out = tmp_path / "ref"
    dump_mod.main(argv=_argv(env, ref_out, N_STREAM), modules=env.mods, model=env.model)

    out = tmp_path / "resumed"

    def boom(seg: dump_mod.Segment) -> None:
        if seg.index == 0:
            raise _Interrupt("simulated crash after first segment")

    with pytest.raises(_Interrupt):
        dump_mod.main(
            argv=_argv(env, out, N_STREAM),
            modules=env.mods,
            model=env.model,
            after_flush=boom,
        )
    # partial state: meta exists, marked incomplete
    assert _read_meta(out)["progress"]["complete"] is False

    dump_mod.main(
        argv=_argv(env, out, N_STREAM, "--resume"),
        modules=env.mods,
        model=env.model,
    )

    ref_meta, meta = _read_meta(ref_out), _read_meta(out)
    assert meta["progress"]["complete"] is True
    assert meta["n_rows"] == ref_meta["n_rows"]
    assert meta["l0_sum"] == ref_meta["l0_sum"]
    for pattern in ("rows-*.parquet", "tokens-*.parquet"):
        a, b = _read_all(ref_out, pattern), _read_all(out, pattern)
        for col in a:
            assert np.array_equal(a[col], b[col]), (pattern, col)


def test_resume_on_complete_run_is_noop(tmp_path):
    env = _setup(tmp_path, tag="noop")
    out = tmp_path / "flat"
    dump_mod.main(argv=_argv(env, out, 60), modules=env.mods, model=env.model)
    before = _read_meta(out)
    dump_mod.main(
        argv=_argv(env, out, 60, "--resume"), modules=env.mods, model=env.model
    )
    assert _read_meta(out) == before


def test_cli_runs_as_module():
    root = Path(store.__file__).resolve().parents[1]
    proc = subprocess.run(
        [sys.executable, "-m", "store.dump", "--help"],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert proc.returncode == 0, proc.stderr
    assert "--sae-repo" in proc.stdout


def test_real_gpt2_end_to_end(tmp_path, gpt2_model):
    """The brief's canonical test: fake SAE (random weights, 64 features) + real
    GPT-2 on a 32-sequence shard. Skips if GPT-2 cannot be loaded."""
    repo = make_fake_repo(tmp_path / "repo", tag="gpt2")
    mods = load_sae_modules(str(repo))
    cfg_cls = sys.modules["sparse_autoencoder"].SAEConfig
    d_model = int(gpt2_model.cfg.d_model)
    ckpt = make_checkpoint(
        tmp_path / "checkpoint.pt", mods.SparseAutoencoder, cfg_cls,
        d_model=d_model, n_features=N_FEATURES, seed=5, layer=8,
    )
    shard = make_shard(tmp_path / "holdout.bin", n_seqs=32, seq_len=SEQ_LEN, seed=3)
    out = tmp_path / "flat"
    n_target = 32 * PER_SEQ
    dump_mod.main(
        argv=[
            "--checkpoint", str(ckpt), "--shard", str(shard),
            "--n-tokens", str(n_target), "--out", str(out),
            "--batch-seqs", "8", "--rows-per-file", str(ROWS_PER_FILE),
            "--encode-chunk", str(ENCODE_CHUNK), "--sae-repo", str(repo),
            "--device", "cpu",
        ],
        modules=mods,
        model=gpt2_model,
    )
    meta = _read_meta(out)
    assert meta["layer"] == 8  # taken from the checkpoint
    assert meta["hook_name"] == "blocks.8.hook_resid_post"
    assert meta["n_rows"] == meta["l0_sum"] > 0

    rows = _read_all(out, "rows-*.parquet")
    assert rows["token_idx"].shape[0] == meta["n_rows"]

    # Recompute the first batch with identical shapes and compare its rows.
    sae = load_sae_from_checkpoint(mods, ckpt).sae
    tokens = np.fromfile(shard, dtype=np.uint16).reshape(32, SEQ_LEN)
    flat = dump_mod.batch_residuals(
        gpt2_model, tokens[:8], "blocks.8.hook_resid_post", 8,
        torch.device("cpu"),
    )
    with torch.no_grad():
        first_chunk = sae.encode(flat[:ENCODE_CHUNK])
    r, c = torch.nonzero(first_chunk > 0, as_tuple=True)
    n = r.shape[0]
    assert np.array_equal(rows["token_idx"][:n], r.numpy().astype(np.uint32))
    assert np.array_equal(rows["feature"][:n], c.numpy().astype(np.uint32))
    assert np.allclose(
        rows["value"][:n], first_chunk[r, c].numpy(), rtol=0, atol=1e-6
    )

    tok = _read_all(out, "tokens-*.parquet")
    assert np.array_equal(tokens[tok["seq_idx"], tok["pos"]], tok["token_id"])
