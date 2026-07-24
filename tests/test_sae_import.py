"""Tests for store.sae_import: the sys.path shim, checkpoint loader, and shard
adapter."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from store import sae_import
from store.sae_import import (
    SAEImportError,
    load_sae_from_checkpoint,
    load_sae_modules,
    open_token_shard,
    sae_n_features,
)

from .conftest import make_checkpoint, make_fake_repo, make_shard


def test_missing_repo_raises_actionable_error(tmp_path):
    missing = tmp_path / "nope"
    with pytest.raises(SAEImportError) as exc:
        load_sae_modules(str(missing))
    msg = str(exc.value)
    assert "sparse_autoencoder.py" in msg
    assert "--sae-repo" in msg


def test_dir_without_sae_module_raises(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(SAEImportError) as exc:
        load_sae_modules(str(tmp_path / "empty"))
    assert "sparse_autoencoder.py" in str(exc.value)


def test_loads_symbols_from_fake_repo(tmp_path):
    repo = make_fake_repo(tmp_path / "repo", tag="alpha")
    mods = load_sae_modules(str(repo))
    assert mods.SparseAutoencoder is not None
    assert mods.TokenShard is not None
    assert mods.ActivationLoader is not None
    assert mods.evaluate is None  # fake repo ships no evaluate.py
    assert mods.repo_path == repo.resolve()


def test_env_var_is_respected(tmp_path, monkeypatch):
    repo = make_fake_repo(tmp_path / "envrepo", tag="env")
    monkeypatch.setenv(sae_import.ENV_VAR, str(repo))
    mods = load_sae_modules(None)
    assert mods.repo_path == repo.resolve()


def test_explicit_arg_overrides_env(tmp_path, monkeypatch):
    good = make_fake_repo(tmp_path / "good", tag="good")
    monkeypatch.setenv(sae_import.ENV_VAR, str(tmp_path / "bogus"))
    mods = load_sae_modules(str(good))
    assert mods.repo_path == good.resolve()
    # ... and a bogus explicit arg fails even when the env var is valid.
    monkeypatch.setenv(sae_import.ENV_VAR, str(good))
    with pytest.raises(SAEImportError):
        load_sae_modules(str(tmp_path / "bogus"))


def test_missing_data_module_is_optional_but_requirable(tmp_path):
    repo = make_fake_repo(tmp_path / "nodata", tag="nodata", with_data=False)
    mods = load_sae_modules(str(repo))
    assert mods.TokenShard is None
    assert mods.ActivationLoader is None
    with pytest.raises(SAEImportError) as exc:
        load_sae_modules(str(repo), require=("SparseAutoencoder", "TokenShard"))
    assert "data.py" in str(exc.value)
    assert "TokenShard" in str(exc.value)


def test_switching_repos_reloads_modules(tmp_path):
    repo_a = make_fake_repo(tmp_path / "a", tag="tag-a")
    repo_b = make_fake_repo(tmp_path / "b", tag="tag-b")
    mods_a = load_sae_modules(str(repo_a))
    assert __import__("sparse_autoencoder").TAG == "tag-a"
    mods_b = load_sae_modules(str(repo_b))
    assert __import__("sparse_autoencoder").TAG == "tag-b"
    assert mods_a.SparseAutoencoder is not mods_b.SparseAutoencoder


def test_load_sae_from_checkpoint_roundtrip(tmp_path):
    repo = make_fake_repo(tmp_path / "repo", tag="ckpt")
    mods = load_sae_modules(str(repo))
    cfg_cls = __import__("sparse_autoencoder").SAEConfig
    ckpt = make_checkpoint(
        tmp_path / "checkpoint.pt",
        mods.SparseAutoencoder,
        cfg_cls,
        d_model=16,
        n_features=64,
        seed=3,
        layer=5,
    )
    loaded = load_sae_from_checkpoint(mods, ckpt)
    assert loaded.n_features == 64
    assert loaded.d_model == 16
    assert loaded.layer == 5
    assert sae_n_features(loaded.sae) == 64
    assert not loaded.sae.training
    # Weights must round-trip exactly.
    ref = torch.load(ckpt, map_location="cpu", weights_only=True)["state_dict"]
    for name, tensor in loaded.sae.state_dict().items():
        assert torch.equal(tensor, ref[name]), name


def test_load_sae_from_missing_checkpoint(tmp_path):
    repo = make_fake_repo(tmp_path / "repo")
    mods = load_sae_modules(str(repo))
    with pytest.raises(SAEImportError) as exc:
        load_sae_from_checkpoint(mods, tmp_path / "absent.pt")
    assert "checkpoint" in str(exc.value)


def test_open_token_shard_tokenshard_and_memmap_agree(tmp_path):
    shard_path = make_shard(tmp_path / "holdout.bin", n_seqs=8, seq_len=16, seed=1)

    repo = make_fake_repo(tmp_path / "with_data", tag="wd")
    via_ts = open_token_shard(load_sae_modules(str(repo)), shard_path)
    assert via_ts.source == "TokenShard"

    repo2 = make_fake_repo(tmp_path / "no_data", tag="nd", with_data=False)
    via_mm = open_token_shard(load_sae_modules(str(repo2)), shard_path)
    assert via_mm.source == "memmap"

    assert via_ts.tokens.shape == via_mm.tokens.shape == (8, 16)
    assert np.array_equal(np.asarray(via_ts.tokens), np.asarray(via_mm.tokens))
    assert via_ts.seq_len == 16
    assert via_ts.n_stream_tokens == 8 * 15
    assert via_ts.sidecar["dataset"] == "fake-holdout"


def test_open_token_shard_missing_file(tmp_path):
    repo = make_fake_repo(tmp_path / "repo")
    mods = load_sae_modules(str(repo))
    with pytest.raises(SAEImportError) as exc:
        open_token_shard(mods, tmp_path / "absent.bin")
    assert "shard" in str(exc.value)


_PRIVATE_ATTR_DATA_SOURCE = '''\
"""data.py variant mirroring the real sae-gpt2-small TokenShard, which keeps
its array in the private attribute ``_tokens``."""
import json
from pathlib import Path

import numpy as np


class TokenShard:
    def __init__(self, path):
        p = Path(path)
        self.meta = json.loads(p.with_suffix(".json").read_text())
        self.seq_len = int(self.meta["seq_len"])
        flat = np.memmap(p, dtype=np.uint16, mode="r")
        self._tokens = np.asarray(flat).reshape(-1, self.seq_len)
        self.n_seqs = self._tokens.shape[0]
'''


def test_open_token_shard_private_tokens_attr(tmp_path):
    """The real TokenShard stores the array at ``_tokens``; the adapter must find it
    there instead of falling back to memmap."""
    shard_path = make_shard(tmp_path / "holdout.bin", n_seqs=6, seq_len=16, seed=4)
    repo = make_fake_repo(tmp_path / "repo", tag="priv")
    (repo / "data.py").write_text(_PRIVATE_ATTR_DATA_SOURCE)
    mods = load_sae_modules(str(repo))
    shard = open_token_shard(mods, shard_path)
    assert shard.source == "TokenShard"
    expected = np.fromfile(shard_path, dtype=np.uint16).reshape(6, 16)
    assert np.array_equal(np.asarray(shard.tokens), expected)


def test_open_token_shard_rejects_n_seqs_mismatch(tmp_path):
    """A sidecar claiming a different n_seqs than the file holds must fail loudly
    rather than silently mis-mapping token_idx."""
    import json as _json

    shard_path = make_shard(tmp_path / "holdout.bin", n_seqs=8, seq_len=16, seed=5)
    side = shard_path.with_suffix(".json")
    sidecar = _json.loads(side.read_text())
    sidecar["n_seqs"] = 9  # lie
    side.write_text(_json.dumps(sidecar))
    repo = make_fake_repo(tmp_path / "repo", tag="mm", with_data=False)
    mods = load_sae_modules(str(repo))
    with pytest.raises(SAEImportError) as exc:
        open_token_shard(mods, shard_path)
    assert "n_seqs" in str(exc.value)
