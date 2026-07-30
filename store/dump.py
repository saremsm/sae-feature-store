"""Sequential GPU encode of a token shard -> staging Parquet. ``token_idx`` follows
:func:`store.schema.token_index` exactly and is asserted per batch."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from . import schema
from .sae_import import (
    LoadedSAE,
    SAEModules,
    ShardData,
    load_sae_from_checkpoint,
    load_sae_modules,
    open_token_shard,
)

log = logging.getLogger("store.dump")

DEFAULT_LAYER = 8
_METRICS_L0_KEYS = ("l0", "mean_l0", "final_l0", "L0", "l0_mean", "eval_l0")


class ResidualModel(Protocol):
    """The slice of HookedTransformer that dump needs (fakes implement it)."""

    def run_with_cache(
        self, tokens: torch.Tensor, **kwargs: Any
    ) -> tuple[Any, Any]: ...


#


@dataclass
class DumpConfig:
    checkpoint: Path
    shard: Path
    out: Path
    n_tokens: int
    batch_seqs: int = 512
    rows_per_file: int = schema.DEFAULT_ROWS_PER_FILE
    row_group_size: int = schema.DEFAULT_ROW_GROUP_SIZE
    encode_chunk: int = 16_384  # tokens per sae.encode() call (caps h memory)
    sae_repo: str | None = None
    resume: bool = False
    device: str = "auto"
    model_name: str = "gpt2"
    layer: int | None = None  # default: checkpoint's layer, else 8
    hook_name: str | None = None  # default: blocks.{layer}.hook_resid_post
    log_every: int = 20


#


@dataclass
class Segment:
    """One completed (rows file, tokens file) pair."""

    index: int
    rows_file: str
    tokens_file: str
    seq_start: int
    seq_end: int
    token_start: int
    token_end: int
    n_rows: int

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class _Buffers:
    rows: list[dict[str, np.ndarray]] = field(default_factory=list)
    tokens: list[dict[str, np.ndarray]] = field(default_factory=list)
    n_rows: int = 0


class SegmentWriter:
    """Buffers batches and writes paired rows-/tokens- Parquet files."""

    def __init__(
        self,
        out_dir: Path,
        rows_per_file: int,
        row_group_size: int,
        next_index: int,
        seq_cursor: int,
        token_cursor: int,
    ) -> None:
        self.out_dir = out_dir
        self.rows_per_file = rows_per_file
        self.row_group_size = row_group_size
        self.next_index = next_index
        self._buf = _Buffers()
        self._seq_start = seq_cursor
        self._token_start = token_cursor
        self._seq_end = seq_cursor
        self._token_end = token_cursor

    @property
    def pending_rows(self) -> int:
        return self._buf.n_rows

    def add(
        self,
        rows: dict[str, np.ndarray],
        tokens: dict[str, np.ndarray],
        seq_end: int,
        token_end: int,
    ) -> Segment | None:
        self._buf.rows.append(rows)
        self._buf.tokens.append(tokens)
        self._buf.n_rows += int(rows["token_idx"].shape[0])
        self._seq_end = seq_end
        self._token_end = token_end
        if self._buf.n_rows >= self.rows_per_file:
            return self.flush()
        return None

    def _concat(
        self, parts: list[dict[str, np.ndarray]], sch: pa.Schema
    ) -> pa.Table:
        cols = {
            f.name: np.concatenate([p[f.name] for p in parts])
            if parts
            else np.empty(0, dtype=f.type.to_pandas_dtype())
            for f in sch
        }
        return pa.Table.from_pydict(cols, schema=sch)

    def flush(self) -> Segment | None:
        """Write the pending buffers as one rows file + one tokens file."""
        if self._token_end == self._token_start:
            return None  # nothing buffered since the last flush
        idx = self.next_index
        rows_name = schema.ROWS_FILE_FMT.format(index=idx)
        tokens_name = schema.TOKENS_FILE_FMT.format(index=idx)
        rows_tbl = self._concat(self._buf.rows, schema.ROWS_SCHEMA)
        tokens_tbl = self._concat(self._buf.tokens, schema.TOKENS_SCHEMA)
        for name, tbl in ((rows_name, rows_tbl), (tokens_name, tokens_tbl)):
            pq.write_table(
                tbl,
                self.out_dir / name,
                row_group_size=self.row_group_size,
                compression=schema.COMPRESSION,
            )
        seg = Segment(
            index=idx,
            rows_file=rows_name,
            tokens_file=tokens_name,
            seq_start=self._seq_start,
            seq_end=self._seq_end,
            token_start=self._token_start,
            token_end=self._token_end,
            n_rows=self._buf.n_rows,
        )
        log.info(
            "wrote %s (%d rows) + %s (%d tokens)",
            rows_name,
            rows_tbl.num_rows,
            tokens_name,
            tokens_tbl.num_rows,
        )
        self.next_index += 1
        self._buf = _Buffers()
        self._seq_start = self._seq_end
        self._token_start = self._token_end
        return seg


#


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while blob := fh.read(chunk):
            h.update(blob)
    return h.hexdigest()


def git_sha(path: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def _resolve_device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


def _load_real_model(name: str, device: torch.device) -> ResidualModel:
    # transformer_lens is imported lazily so tests injecting a fake model never
    from transformer_lens import HookedTransformer

    log.info("loading %s via transformer_lens ...", name)
    model = HookedTransformer.from_pretrained(name)
    model = model.to(device)
    model.eval()
    return model


def batch_residuals(
    model: ResidualModel,
    toks: np.ndarray,
    hook_name: str,
    layer: int,
    device: torch.device,
) -> torch.Tensor:
    """Forward one batch of sequences to the residual hook. Returns float32
    ``[n_seqs * (seq_len - 1), d_model]`` with the BOS position dropped, on
    ``device`` (where the SAE lives)."""
    model_device = device
    params = getattr(model, "parameters", None)
    if callable(params):
        try:
            model_device = next(iter(params())).device
        except (StopIteration, TypeError):
            pass
    t = torch.from_numpy(np.ascontiguousarray(toks).astype(np.int64)).to(
        model_device
    )
    ctx = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if model_device.type == "cuda"
        else contextlib.nullcontext()
    )
    with torch.no_grad(), ctx:
        _, cache = model.run_with_cache(
            t,
            names_filter=hook_name,
            stop_at_layer=layer + 1,
            return_type=None,
        )
    resid = cache[hook_name]
    resid = resid[:, 1:, :]  # exclude BOS (position 0)
    return resid.reshape(-1, resid.shape[-1]).float().to(device)


def _normalize_sae_config(loaded: LoadedSAE) -> dict[str, Any]:
    raw = loaded.config
    input_scale = raw.get("input_scale")
    if input_scale is None:
        buf = getattr(loaded.sae, "input_scale", None)
        if torch.is_tensor(buf) and buf.numel() == 1:
            input_scale = float(buf.item())
    expansion = raw.get("expansion", raw.get("expansion_factor"))
    if expansion is None and loaded.d_model:
        expansion = loaded.n_features / loaded.d_model
    return {
        "activation": raw.get("activation", "relu"),
        "k": raw.get("k"),
        "l1_coeff": raw.get("l1_coeff", raw.get("l1_coefficient")),
        "expansion": expansion,
        "n_features": loaded.n_features,
        "d_model": loaded.d_model,
        "input_scale": input_scale,
        "raw": raw,
    }


def _reference_l0(checkpoint: Path) -> float | None:
    metrics_path = checkpoint.parent / "metrics.json"
    if not metrics_path.is_file():
        log.warning("no metrics.json next to %s; skipping L0 check", checkpoint)
        return None
    try:
        with metrics_path.open() as fh:
            metrics = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("could not read %s: %r", metrics_path, exc)
        return None
    if isinstance(metrics, dict):
        # The value may live at the top level or nested one deep under "metrics".
        scopes = [metrics]
        sub = metrics.get("metrics")
        if isinstance(sub, dict):
            scopes.append(sub)
        for scope in scopes:
            for key in _METRICS_L0_KEYS:
                val = scope.get(key)
                if isinstance(val, (int, float)):
                    return float(val)
    log.warning(
        "metrics.json at %s has no L0 under %s", metrics_path, _METRICS_L0_KEYS
    )
    return None


#


@dataclass
class ResumeState:
    segments: list[Segment]
    seq_cursor: int
    token_cursor: int
    n_rows: int
    l0_sum: int
    complete: bool


def _fresh_state() -> ResumeState:
    return ResumeState([], 0, 0, 0, 0, False)


def _load_resume_state(cfg: DumpConfig, ckpt_sha: str) -> ResumeState:
    meta_path = cfg.out / schema.META_FILENAME
    if not meta_path.is_file():
        log.info("--resume: no %s yet; starting fresh", schema.META_FILENAME)
        return _fresh_state()
    with meta_path.open() as fh:
        meta = json.load(fh)

    def _mismatch(what: str, want: Any, have: Any) -> None:
        raise SystemExit(
            f"--resume mismatch on {what}: existing run used {have!r}, this "
            f"invocation uses {want!r}. Use a fresh --out (or delete "
            f"{cfg.out}) to start over."
        )

    if meta.get("sae_checkpoint", {}).get("sha256") != ckpt_sha:
        _mismatch(
            "checkpoint sha256",
            ckpt_sha,
            meta.get("sae_checkpoint", {}).get("sha256"),
        )
    if meta.get("shard", {}).get("path") != str(cfg.shard):
        _mismatch("shard path", str(cfg.shard), meta.get("shard", {}).get("path"))
    for key, want in (
        ("n_tokens_requested", cfg.n_tokens),
        ("rows_per_file", cfg.rows_per_file),
        ("row_group_size", cfg.row_group_size),
    ):
        if meta.get(key) != want:
            _mismatch(key, want, meta.get(key))
    run_args = meta.get("progress", {}).get("run_args", {})
    for key, want in (
        ("batch_seqs", cfg.batch_seqs),
        ("encode_chunk", cfg.encode_chunk),
    ):
        if run_args.get(key) is not None and run_args.get(key) != want:
            log.warning(
                "--resume: %s changed (%r -> %r); values near the resume "
                "boundary may differ at float precision",
                key,
                run_args.get(key),
                want,
            )

    segments = [Segment(**{str(k): v for k, v in s.items()}) for s in meta["segments"]]
    progress = meta["progress"]
    state = ResumeState(
        segments=segments,
        seq_cursor=int(progress["seqs_done"]),
        token_cursor=int(progress["tokens_done"]),
        n_rows=int(meta["n_rows"]),
        l0_sum=int(meta["l0_sum"]),
        complete=bool(progress["complete"]),
    )

    # Drop any files not recorded as complete segments (partial writes).
    known = {s.rows_file for s in segments} | {s.tokens_file for s in segments}
    for f in sorted(cfg.out.glob("rows-*.parquet")) + sorted(
        cfg.out.glob("tokens-*.parquet")
    ):
        if f.name not in known:
            log.warning("--resume: removing partial file %s", f.name)
            f.unlink()
    log.info(
        "--resume: %d segments complete, continuing at seq %d / token %d",
        len(segments),
        state.seq_cursor,
        state.token_cursor,
    )
    return state


#


def _write_meta(cfg: DumpConfig, meta: dict[str, Any]) -> None:
    tmp = cfg.out / (schema.META_FILENAME + ".tmp")
    with tmp.open("w") as fh:
        json.dump(meta, fh, indent=2)
        fh.write("\n")
    os.replace(tmp, cfg.out / schema.META_FILENAME)


