"""Import shim for the sibling ``sae-gpt2-small`` repo."""

from __future__ import annotations

import dataclasses
import importlib
import inspect
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import torch

log = logging.getLogger("store.sae_import")

DEFAULT_SAE_REPO: str = "~/sae-gpt2-small"
ENV_VAR: str = "SAE_REPO"

# : Module names we import from the SAE repo.
_MODULE_NAMES: tuple[str, ...] = ("sparse_autoencoder", "data", "evaluate")

_EXPECTED_LOCATION: dict[str, str] = {
    "SparseAutoencoder": "sparse_autoencoder.py (class SparseAutoencoder)",
    "TokenShard": "data.py (class TokenShard)",
    "ActivationLoader": "data.py (class ActivationLoader)",
    "evaluate": "evaluate.py (module)",
}

DEFAULT_REQUIRED: tuple[str, ...] = ("SparseAutoencoder",)


class SAEImportError(RuntimeError):
    """Raised when the SAE repo or an expected symbol cannot be loaded."""


@dataclass(frozen=True)
class SAEModules:
    """Namespace of symbols loaded from the SAE repo."""

    SparseAutoencoder: type
    TokenShard: type | None
    ActivationLoader: type | None
    evaluate: ModuleType | None
    repo_path: Path


def resolve_sae_repo(sae_repo: str | None = None) -> Path:
    """CLI arg > ``SAE_REPO`` env var > ``~/sae-gpt2-small``."""
    raw = sae_repo or os.environ.get(ENV_VAR) or DEFAULT_SAE_REPO
    return Path(os.path.expanduser(raw)).resolve()


def _activate_repo(repo: Path) -> None:
    """Put ``repo`` first on sys.path and evict stale same-named modules."""
    entry = str(repo)
    if entry in sys.path:
        sys.path.remove(entry)
    sys.path.insert(0, entry)
    for name in _MODULE_NAMES:
        mod = sys.modules.get(name)
        if mod is None:
            continue
        file = getattr(mod, "__file__", None)
        if file is None or Path(file).resolve().parent != repo:
            del sys.modules[name]
    importlib.invalidate_caches()


def load_sae_modules(
    sae_repo: str | None = None,
    *,
    require: tuple[str, ...] = DEFAULT_REQUIRED,
) -> SAEModules:
    """Import the SAE repo and return its public symbols."""
    repo = resolve_sae_repo(sae_repo)
    hint = (
        f"expected {repo}/sparse_autoencoder.py -- pass --sae-repo "
        f"(or set {ENV_VAR}) to point at your sae-gpt2-small checkout"
    )
    if not repo.is_dir():
        raise SAEImportError(f"SAE repo directory not found: {repo}; {hint}")
    if not (repo / "sparse_autoencoder.py").is_file():
        raise SAEImportError(
            f"{repo} exists but has no sparse_autoencoder.py; {hint}"
        )

    _activate_repo(repo)

    try:
        sa_mod = importlib.import_module("sparse_autoencoder")
    except Exception as exc:  # pragma: no cover - import-time repo bugs
        raise SAEImportError(
            f"failed to import {repo}/sparse_autoencoder.py: {exc!r}"
        ) from exc

    symbols: dict[str, Any] = {
        "SparseAutoencoder": getattr(sa_mod, "SparseAutoencoder", None),
        "TokenShard": None,
        "ActivationLoader": None,
        "evaluate": None,
    }

    if (repo / "data.py").is_file():
        try:
            data_mod = importlib.import_module("data")
        except Exception as exc:
            raise SAEImportError(
                f"failed to import {repo}/data.py: {exc!r}"
            ) from exc
        symbols["TokenShard"] = getattr(data_mod, "TokenShard", None)
        symbols["ActivationLoader"] = getattr(data_mod, "ActivationLoader", None)
    else:
        log.warning(
            "no data.py in %s; TokenShard/ActivationLoader unavailable "
            "(shard reads will fall back to a raw memmap)",
            repo,
        )

    if (repo / "evaluate.py").is_file():
        try:
            symbols["evaluate"] = importlib.import_module("evaluate")
        except Exception as exc:
            raise SAEImportError(
                f"failed to import {repo}/evaluate.py: {exc!r}"
            ) from exc

    missing = [name for name in require if symbols.get(name) is None]
    if missing:
        wanted = "; ".join(
            f"{name} from {repo}/{_EXPECTED_LOCATION[name]}" for name in missing
        )
        raise SAEImportError(
            f"SAE repo at {repo} is missing expected symbols: {wanted} -- "
            f"pass --sae-repo (or set {ENV_VAR}) to a checkout that provides "
            f"them"
        )

    log.info("loaded SAE modules from %s", repo)
    return SAEModules(
        SparseAutoencoder=symbols["SparseAutoencoder"],
        TokenShard=symbols["TokenShard"],
        ActivationLoader=symbols["ActivationLoader"],
        evaluate=symbols["evaluate"],
        repo_path=repo,
    )


#

_STATE_KEYS: tuple[str, ...] = (
    "sae_state_dict",
    "state_dict",
    "model_state_dict",
    "model",
)
_CONFIG_KEYS: tuple[str, ...] = ("config", "sae_config")


@dataclass
class LoadedSAE:
    sae: torch.nn.Module
    config: dict[str, Any]  # raw config dict from the checkpoint
    n_features: int
    d_model: int
    layer: int | None  # layer index if the checkpoint stores one
    checkpoint_path: Path


def _pick(d: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for k in keys:
        if k in d:
            return d[k]
    return None


def _infer_dims(
    config: dict[str, Any], state: dict[str, torch.Tensor]
) -> tuple[int, int]:
    """(n_features, d_model) from config, falling back to state-dict shapes."""
    n_features = config.get("n_features")
    d_model = config.get("d_model")
    if n_features is None and "b_enc" in state:
        n_features = int(state["b_enc"].shape[0])
    if d_model is None and "b_dec" in state:
        d_model = int(state["b_dec"].shape[0])
    if (n_features is None or d_model is None) and "W_enc" in state:
        w = state["W_enc"]
        if w.ndim == 2:
            if n_features is not None and d_model is None:
                d_model = int(w.shape[0] if w.shape[1] == n_features else w.shape[1])
            elif d_model is not None and n_features is None:
                n_features = int(w.shape[1] if w.shape[0] == d_model else w.shape[0])
    if n_features is None or d_model is None:
        raise SAEImportError(
            "could not determine n_features/d_model from the checkpoint "
            "config or state dict (looked for config keys n_features/d_model "
            "and state keys b_enc/b_dec/W_enc)"
        )
    return int(n_features), int(d_model)


def _construct_sae(
    modules: SAEModules, config: dict[str, Any], n_features: int, d_model: int
) -> torch.nn.Module:
    """Build an SAE instance sized from the checkpoint config. Prefers ``SAEConfig``
    from the same module (filtering the checkpoint's config dict to its dataclass
    fields); otherwise filters by the constructor's own keyword parameters."""
    full = dict(config)
    full.setdefault("n_features", n_features)
    full.setdefault("d_model", d_model)

    sa_mod = sys.modules.get(modules.SparseAutoencoder.__module__)
    cfg_cls = getattr(sa_mod, "SAEConfig", None) if sa_mod else None
    if cfg_cls is not None and dataclasses.is_dataclass(cfg_cls):
        names = {f.name for f in dataclasses.fields(cfg_cls)}
        kwargs = {k: v for k, v in full.items() if k in names}
        try:
            return modules.SparseAutoencoder(cfg_cls(**kwargs))
        except Exception as exc:
            raise SAEImportError(
                f"SparseAutoencoder(SAEConfig(**{sorted(kwargs)})) failed: "
                f"{exc!r}"
            ) from exc

    try:
        params = inspect.signature(modules.SparseAutoencoder).parameters
        kwargs = {k: v for k, v in full.items() if k in params}
        return modules.SparseAutoencoder(**kwargs)
    except Exception as exc:
        raise SAEImportError(
            f"could not construct SparseAutoencoder from checkpoint config "
            f"(keys: {sorted(full)}): {exc!r}"
        ) from exc


def load_sae_from_checkpoint(
    modules: SAEModules,
    checkpoint: str | Path,
    device: torch.device | str = "cpu",
) -> LoadedSAE:
    """torch.load the checkpoint, size + build the SAE, load weights, eval."""
    path = Path(checkpoint).expanduser().resolve()
    if not path.is_file():
        raise SAEImportError(
            f"checkpoint not found: {path} -- expected something like "
            f"{modules.repo_path}/results/frontier/<name>/checkpoint.pt"
        )
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        log.warning(
            "weights_only load of %s failed; retrying with weights_only=False",
            path,
        )
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        raise SAEImportError(
            f"checkpoint {path} did not contain a dict (got {type(ckpt)!r})"
        )

    state = _pick(ckpt, _STATE_KEYS)
    if state is None and all(torch.is_tensor(v) for v in ckpt.values()):
        state = ckpt  # bare state dict
    if not isinstance(state, dict):
        raise SAEImportError(
            f"checkpoint {path} has no state dict under any of {_STATE_KEYS}"
        )

    raw_cfg = _pick(ckpt, _CONFIG_KEYS)
    config: dict[str, Any] = dict(raw_cfg) if isinstance(raw_cfg, dict) else {}

    n_features, d_model = _infer_dims(config, state)
    sae = _construct_sae(modules, config, n_features, d_model)

    try:
        sae.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        log.warning(
            "strict state-dict load failed (%s); retrying strict=False", exc
        )
        result = sae.load_state_dict(state, strict=False)
        if result.missing_keys:
            log.warning("missing keys: %s", result.missing_keys)
        if result.unexpected_keys:
            log.warning("unexpected keys: %s", result.unexpected_keys)

    sae.eval()
    sae.requires_grad_(False)
    sae.to(device)

    layer = ckpt.get("layer")
    return LoadedSAE(
        sae=sae,
        config=config,
        n_features=n_features,
        d_model=d_model,
        layer=int(layer) if isinstance(layer, int) else None,
        checkpoint_path=path,
    )


def sae_n_features(sae: torch.nn.Module) -> int:
    """``sae.n_features``, falling back to ``sae.config.n_features``."""
    n = getattr(sae, "n_features", None)
    if n is None:
        n = getattr(getattr(sae, "config", None), "n_features", None)
    if n is None:
        raise SAEImportError(
            "SAE exposes neither .n_features nor .config.n_features"
        )
    return int(n)


#

_ARRAY_ATTRS: tuple[str, ...] = (
    "tokens",
    "_tokens",  # sae-gpt2-small's data.TokenShard stores the memmap here
    "ids",
    "token_ids",
    "data",
    "array",
)
_SIDECAR_ATTRS: tuple[str, ...] = ("meta", "sidecar", "info", "header")


@dataclass
class ShardData:
    """Sequential view of a token shard: ``[n_seqs, seq_len]`` uint16 ids."""

    tokens: np.ndarray
    sidecar: dict[str, Any]
    path: Path
    source: str  # "TokenShard" or "memmap"

    @property
    def n_seqs(self) -> int:
        return int(self.tokens.shape[0])

    @property
    def seq_len(self) -> int:
        return int(self.tokens.shape[1])

    @property
    def n_stream_tokens(self) -> int:
        """Number of BOS-excluded tokens in the sequential stream."""
        return self.n_seqs * (self.seq_len - 1)


def _read_sidecar(path: Path) -> dict[str, Any]:
    for cand in (path.with_suffix(".json"), Path(str(path) + ".json")):
        if cand.is_file():
            with cand.open() as fh:
                loaded = json.load(fh)
            if not isinstance(loaded, dict):
                raise SAEImportError(f"sidecar {cand} is not a JSON object")
            return loaded
    raise SAEImportError(
        f"no sidecar found for shard {path} (looked for "
        f"{path.with_suffix('.json').name} and {path.name}.json)"
    )


def _shape_tokens(flat_or_2d: np.ndarray, seq_len: int, origin: str) -> np.ndarray:
    arr = np.asarray(flat_or_2d)
    if not np.issubdtype(arr.dtype, np.integer):
        raise SAEImportError(
            f"{origin}: token array has non-integer dtype {arr.dtype}"
        )
    if arr.ndim == 2:
        if arr.shape[1] != seq_len:
            raise SAEImportError(
                f"{origin}: 2-D token array width {arr.shape[1]} != sidecar "
                f"seq_len {seq_len}"
            )
        return arr
    if arr.ndim != 1:
        raise SAEImportError(f"{origin}: token array has ndim {arr.ndim}")
    n = arr.shape[0]
    if n % seq_len != 0:
        n_trim = (n // seq_len) * seq_len
        log.warning(
            "%s: %d tokens not divisible by seq_len %d; trimming to %d",
            origin,
            n,
            seq_len,
            n_trim,
        )
        arr = arr[:n_trim]
    return arr.reshape(-1, seq_len)


def _via_token_shard(
    modules: SAEModules, path: Path
) -> tuple[np.ndarray, dict[str, Any]] | None:
    if modules.TokenShard is None:
        return None
    shard_obj: Any = None
    for ctor in (
        lambda: modules.TokenShard(str(path)),  # type: ignore[misc]
        lambda: modules.TokenShard(path=str(path)),  # type: ignore[misc]
    ):
        try:
            shard_obj = ctor()
            break
        except TypeError:
            continue
        except Exception as exc:
            log.warning("TokenShard(%s) raised %r; falling back", path, exc)
            return None
    if shard_obj is None:
        log.warning(
            "TokenShard constructor did not accept a path; falling back"
        )
        return None

    arr: np.ndarray | None = None
    for attr in _ARRAY_ATTRS:
        cand = getattr(shard_obj, attr, None)
        if cand is not None:
            try:
                arr = np.asarray(cand)
            except Exception:
                continue
            if arr.ndim in (1, 2):
                break
            arr = None
    sidecar: dict[str, Any] | None = None
    for attr in _SIDECAR_ATTRS:
        cand = getattr(shard_obj, attr, None)
        if isinstance(cand, dict):
            sidecar = cand
            break
    if sidecar is None:
        sidecar = _read_sidecar(path)
    if arr is None:
        log.warning(
            "TokenShard exposes no token array under %s; falling back",
            _ARRAY_ATTRS,
        )
        return None
    return arr, sidecar


def open_token_shard(modules: SAEModules, shard_path: str | Path) -> ShardData:
    """Open a shard for SEQUENTIAL iteration. Prefers the SAE repo's
    ``data.TokenShard``; falls back to a raw uint16 memmap plus the ``.json``
    sidecar if ``TokenShard`` is unavailable or its API cannot be probed."""
    path = Path(shard_path).expanduser().resolve()
    if not path.is_file():
        raise SAEImportError(
            f"shard not found: {path} -- expected something like "
            f"{modules.repo_path}/data/holdout.bin"
        )

    via = _via_token_shard(modules, path)
    if via is not None:
        arr, sidecar = via
        source = "TokenShard"
    else:
        sidecar = _read_sidecar(path)
        arr = np.memmap(path, dtype=np.uint16, mode="r")
        source = "memmap"

    seq_len = sidecar.get("seq_len")
    if not isinstance(seq_len, int) or seq_len < 2:
        raise SAEImportError(
            f"shard sidecar for {path} must contain integer seq_len >= 2, "
            f"got {seq_len!r}"
        )
    tokens = _shape_tokens(arr, seq_len, f"shard {path}")
    side_n_seqs = sidecar.get("n_seqs")
    if isinstance(side_n_seqs, int) and side_n_seqs != tokens.shape[0]:
        raise SAEImportError(
            f"shard {path}: sidecar says n_seqs={side_n_seqs} but the token "
            f"array holds {tokens.shape[0]} sequences of seq_len {seq_len}"
        )
    log.info(
        "opened shard %s via %s: %d seqs x seq_len %d (%d stream tokens)",
        path,
        source,
        tokens.shape[0],
        seq_len,
        tokens.shape[0] * (seq_len - 1),
    )
    return ShardData(tokens=tokens, sidecar=sidecar, path=path, source=source)
