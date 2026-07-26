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


