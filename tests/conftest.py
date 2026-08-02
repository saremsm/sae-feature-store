"""Shared test fixtures: fake SAE repos, fake shard files, a fake model."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

FAKE_MODULE_NAMES = ("sparse_autoencoder", "data", "evaluate")

BOS_TOKEN_ID = 50256

FAKE_SAE_SOURCE = '''\
"""Fake sparse_autoencoder.py matching the sae-gpt2-small contract."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

TAG = "{tag}"


@dataclass(frozen=True)
class SAEConfig:
    d_model: int
    n_features: int
    activation: str = "relu"
    expansion: float = 4.0
    input_scale: float = 1.0


class SparseAutoencoder(nn.Module):
    def __init__(self, config: SAEConfig):
        super().__init__()
        self.config = config
        self.n_features = config.n_features
        self.W_enc = nn.Parameter(torch.zeros(config.d_model, config.n_features))
        self.b_enc = nn.Parameter(torch.zeros(config.n_features))
        self.b_dec = nn.Parameter(torch.zeros(config.d_model))
        self.register_buffer("input_scale", torch.tensor(float(config.input_scale)))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu((x - self.b_dec) * self.input_scale @ self.W_enc + self.b_enc)
'''

FAKE_DATA_SOURCE = '''\
"""Fake data.py matching the sae-gpt2-small contract."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

TAG = "{tag}"


class TokenShard:
    """uint16 memmap of packed token ids + .json sidecar."""

    def __init__(self, path: str):
        p = Path(path)
        self.path = p
        side = p.with_suffix(".json")
        if not side.exists():
            side = Path(str(p) + ".json")
        self.meta = json.loads(side.read_text())
        flat = np.memmap(p, dtype=np.uint16, mode="r")
        self.tokens = np.asarray(flat).reshape(-1, int(self.meta["seq_len"]))


class ActivationLoader:
    """Shuffling ring-buffer loader; deliberately unused by the store."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "sae-feature-store never uses the shuffling loader"
        )
'''


@pytest.fixture(autouse=True)
def _isolate_import_state():
    """Restore sys.path and drop fake SAE modules after every test."""
    saved_path = list(sys.path)
    yield
    sys.path[:] = saved_path
    for name in FAKE_MODULE_NAMES:
        sys.modules.pop(name, None)


def make_fake_repo(dst: Path, *, tag: str = "fake", with_data: bool = True) -> Path:
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "sparse_autoencoder.py").write_text(FAKE_SAE_SOURCE.format(tag=tag))
    if with_data:
        (dst / "data.py").write_text(FAKE_DATA_SOURCE.format(tag=tag))
    return dst


def make_shard(
    path: Path,
    *,
    n_seqs: int,
    seq_len: int,
    seed: int = 0,
    max_token_id: int = 1000,
) -> Path:
    """Write a packed uint16 shard (BOS at pos 0 of every sequence) + sidecar."""
    rng = np.random.default_rng(seed)
    toks = rng.integers(1, max_token_id, size=(n_seqs, seq_len)).astype(np.uint16)
    toks[:, 0] = BOS_TOKEN_ID
    path.write_bytes(toks.reshape(-1).tobytes())
    sidecar = {
        "n_tokens": int(toks.size),
        "seq_len": int(seq_len),
        "doc_range": [0, int(n_seqs)],
        "dataset": "fake-holdout",
        "seed": int(seed),
    }
    path.with_suffix(".json").write_text(json.dumps(sidecar))
    return path


def make_checkpoint(
    path: Path,
    sae_cls: type,
    cfg_cls: type,
    *,
    d_model: int,
    n_features: int,
    seed: int = 0,
    layer: int = 0,
) -> Path:
    """Build a randomly-initialized fake SAE and torch.save its checkpoint."""
    torch.manual_seed(seed)
    cfg = cfg_cls(d_model=d_model, n_features=n_features)
    sae = sae_cls(cfg)
    with torch.no_grad():
        sae.W_enc.normal_(0.0, 1.0 / d_model**0.5)
        sae.b_enc.normal_(0.0, 0.05)
        sae.b_dec.normal_(0.0, 0.01)
    torch.save(
        {
            "state_dict": sae.state_dict(),
            "config": {
                "d_model": d_model,
                "n_features": n_features,
                "activation": "relu",
                "expansion": n_features / d_model,
                "input_scale": 1.0,
            },
            "layer": layer,
        },
        path,
    )
    return path


class FakeModel:
    """Minimal deterministic stand-in for HookedTransformer."""

    def __init__(self, d_model: int = 16, vocab: int = 50257, seed: int = 7):
        g = torch.Generator().manual_seed(seed)
        self.emb = torch.randn(vocab, d_model, generator=g)
        self.cfg = types.SimpleNamespace(d_model=d_model)
        self.calls: list[dict] = []

    def run_with_cache(
        self,
        tokens: torch.Tensor,
        *,
        names_filter=None,
        stop_at_layer=None,
        return_type=None,
        **kwargs,
    ):
        self.calls.append(
            {
                "shape": tuple(tokens.shape),
                "names_filter": names_filter,
                "stop_at_layer": stop_at_layer,
                "return_type": return_type,
            }
        )
        t = tokens.long().cpu()
        resid = self.emb[t]
        pos_scale = 1.0 + 0.01 * torch.arange(t.shape[1], dtype=torch.float32)
        resid = resid * pos_scale[None, :, None]
        return None, {names_filter: resid}


def make_flat(
    out: Path,
    *,
    n_features: int,
    n_tokens: int,
    n_files: int = 3,
    mean_l0: int = 4,
    seed: int = 0,
    row_group_size: int = 64,
) -> dict:
    """Write a tiny fake-dump flat set: rows-NNNNN.parquet files in
    ``store.schema.ROWS_SCHEMA`` plus a meta.json shaped like the real one (via
    ``schema.build_meta``), for partition-stage tests."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    from store import schema

    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    per_file = -(-n_tokens // n_files)  # ceil
    segments = []
    n_rows = 0
    for i in range(n_files):
        t0, t1 = i * per_file, min((i + 1) * per_file, n_tokens)
        if t0 >= t1:
            break
        toks, feats, vals = [], [], []
        for t in range(t0, t1):
            f = rng.choice(n_features, size=mean_l0, replace=False)
            f.sort()
            toks.append(np.full(mean_l0, t, dtype=np.uint32))
            feats.append(f.astype(np.uint32))
            vals.append(
                (rng.random(mean_l0) + 0.01).astype(np.float32)
            )
        table = pa.table(
            {
                "token_idx": np.concatenate(toks),
                "feature": np.concatenate(feats),
                "value": np.concatenate(vals),
            },
            schema=schema.ROWS_SCHEMA,
        )
        name = schema.ROWS_FILE_FMT.format(index=i)
        pq.write_table(
            table,
            out / name,
            row_group_size=row_group_size,
            compression=schema.COMPRESSION,
        )
        segments.append(
            {
                "index": i,
                "rows_file": name,
                "tokens_file": schema.TOKENS_FILE_FMT.format(index=i),
                "seq_start": 0,
                "seq_end": 0,
                "token_start": t0,
                "token_end": t1,
                "n_rows": int(table.num_rows),
            }
        )
        n_rows += int(table.num_rows)

    meta = schema.build_meta(
        sae_checkpoint={"path": "fake/checkpoint.pt", "sha256": "0" * 64},
        sae_config={
            "activation": "topk",
            "k": mean_l0,
            "l1_coeff": 0.0,
            "expansion": 1.0,
            "n_features": n_features,
            "d_model": 16,
            "input_scale": 1.0,
            "raw": {},
        },
        hook_name="blocks.0.hook_resid_post",
        layer=0,
        model_name="fake",
        shard={"path": "fake/holdout.bin", "n_seqs": 1, "seq_len": 2,
               "sidecar": {}},
        n_tokens_requested=n_tokens,
        n_tokens_encoded=n_tokens,
        n_rows=n_rows,
        l0_sum=n_rows,
        mean_l0=float(mean_l0),
        rows_per_file=per_file * mean_l0,
        row_group_size=row_group_size,
        segments=segments,
        progress={"seqs_done": 0, "tokens_done": n_tokens, "complete": True},
        git={"feature_store": None, "sae_repo": None},
    )
    (out / schema.META_FILENAME).write_text(json.dumps(meta, indent=1))
    return meta


@pytest.fixture(scope="session")
def gpt2_model():
    """Real GPT-2 via transformer_lens, or skip if it cannot be loaded (e.g. no
    network access to the HF hub and no local cache)."""
    try:
        from transformer_lens import HookedTransformer

        return HookedTransformer.from_pretrained("gpt2")
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"real GPT-2 unavailable: {type(exc).__name__}: {exc}")
