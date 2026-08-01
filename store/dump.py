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


def dump(
    cfg: DumpConfig,
    modules: SAEModules | None = None,
    model: ResidualModel | None = None,
    after_flush: Callable[[Segment], None] | None = None,
) -> dict[str, Any]:
    """Run the dump; returns the final meta dict."""
    device = _resolve_device(cfg.device)
    log.info("device: %s", device)

    if modules is None:
        modules = load_sae_modules(cfg.sae_repo)

    cfg.checkpoint = Path(cfg.checkpoint).expanduser().resolve()
    cfg.shard = Path(cfg.shard).expanduser().resolve()
    cfg.out = Path(cfg.out)
    cfg.out.mkdir(parents=True, exist_ok=True)

    ckpt_sha = sha256_file(cfg.checkpoint)
    loaded = load_sae_from_checkpoint(modules, cfg.checkpoint, device=device)
    sae = loaded.sae
    log.info(
        "SAE: n_features=%d d_model=%d config=%s",
        loaded.n_features,
        loaded.d_model,
        {k: v for k, v in loaded.config.items() if k != "raw"},
    )

    shard = open_token_shard(modules, cfg.shard)
    per_seq = shard.seq_len - 1
    available = shard.n_stream_tokens
    n_target = min(cfg.n_tokens, available)
    if cfg.n_tokens > available:
        log.warning(
            "--n-tokens %d exceeds the %d BOS-excluded tokens in the shard; "
            "encoding %d",
            cfg.n_tokens,
            available,
            available,
        )

    layer = cfg.layer if cfg.layer is not None else (
        loaded.layer if loaded.layer is not None else DEFAULT_LAYER
    )
    hook_name = cfg.hook_name or f"blocks.{layer}.hook_resid_post"

    if model is None:
        model = _load_real_model(cfg.model_name, device)
    model_d = getattr(getattr(model, "cfg", None), "d_model", None)
    if model_d is not None and int(model_d) != loaded.d_model:
        raise SystemExit(
            f"model d_model={model_d} != SAE d_model={loaded.d_model}: wrong "
            f"checkpoint/model pair"
        )

    if cfg.resume:
        state = _load_resume_state(cfg, ckpt_sha)
        if state.complete:
            log.info("--resume: run already complete; nothing to do")
            with (cfg.out / schema.META_FILENAME).open() as fh:
                return json.load(fh)
    else:
        existing = list(cfg.out.glob("rows-*.parquet"))
        if existing or (cfg.out / schema.META_FILENAME).is_file():
            raise SystemExit(
                f"{cfg.out} already contains dump output; pass --resume to "
                f"continue it or point --out somewhere clean"
            )
        state = _fresh_state()

    writer = SegmentWriter(
        cfg.out,
        cfg.rows_per_file,
        cfg.row_group_size,
        next_index=len(state.segments),
        seq_cursor=state.seq_cursor,
        token_cursor=state.token_cursor,
    )

    def current_meta(complete: bool) -> dict[str, Any]:
        tokens_done = state.token_cursor
        return schema.build_meta(
            sae_checkpoint={"path": str(cfg.checkpoint), "sha256": ckpt_sha},
            sae_config=_normalize_sae_config(loaded),
            hook_name=hook_name,
            layer=layer,
            model_name=cfg.model_name,
            shard={
                "path": str(cfg.shard),
                "n_seqs": shard.n_seqs,
                "seq_len": shard.seq_len,
                "source": shard.source,
                "sidecar": shard.sidecar,
            },
            n_tokens_requested=cfg.n_tokens,
            n_tokens_encoded=tokens_done,
            n_rows=state.n_rows,
            l0_sum=state.l0_sum,
            mean_l0=(state.n_rows / tokens_done) if tokens_done else None,
            rows_per_file=cfg.rows_per_file,
            row_group_size=cfg.row_group_size,
            segments=[s.to_dict() for s in state.segments],
            progress={
                "seqs_done": state.seq_cursor,
                "tokens_done": tokens_done,
                "complete": complete,
                "run_args": {
                    "batch_seqs": cfg.batch_seqs,
                    "encode_chunk": cfg.encode_chunk,
                    "device": str(device),
                },
            },
            git={
                "feature_store": git_sha(Path(__file__).resolve().parents[1]),
                "sae_repo": git_sha(modules.repo_path),
            },
        )

    t_start = time.monotonic()
    t_mark = t_start
    tokens_mark = rows_mark = 0
    tokens_run = rows_run = 0
    n_batches = 0

    def _flush_segment(seg: Segment | None) -> None:
        if seg is None:
            return
        state.segments.append(seg)
        _write_meta(cfg, current_meta(complete=False))
        if after_flush is not None:
            after_flush(seg)

    while state.token_cursor < n_target and state.seq_cursor < shard.n_seqs:
        seq_lo = state.seq_cursor
        seq_hi = min(seq_lo + cfg.batch_seqs, shard.n_seqs)
        batch = np.asarray(shard.tokens[seq_lo:seq_hi])
        b = batch.shape[0]

        flat = batch_residuals(model, batch, hook_name, layer, device)
        n_batch_tokens = flat.shape[0]
        base = state.token_cursor
        keep = min(n_batch_tokens, n_target - base)
        flat = flat[:keep]

        rows_parts: list[dict[str, np.ndarray]] = []
        batch_rows = 0
        with torch.no_grad():
            for lo in range(0, keep, cfg.encode_chunk):
                hi = min(lo + cfg.encode_chunk, keep)
                h = sae.encode(flat[lo:hi])
                mask = h > 0
                state.l0_sum += int(mask.sum().item())
                r, c = torch.nonzero(mask, as_tuple=True)
                vals = h[r, c]
                rows_parts.append(
                    {
                        "token_idx": (base + lo + r).cpu().numpy().astype(np.uint32),
                        "feature": c.cpu().numpy().astype(np.uint32),
                        "value": vals.cpu().numpy().astype(np.float32),
                    }
                )
                batch_rows += int(r.shape[0])

        rows_batch = {
            name: np.concatenate([p[name] for p in rows_parts])
            for name in ("token_idx", "feature", "value")
        }

        seq_ids = np.repeat(
            np.arange(seq_lo, seq_hi, dtype=np.int64), per_seq
        )[:keep]
        poss = np.tile(np.arange(1, shard.seq_len, dtype=np.int64), b)[:keep]
        mapped = schema.token_index(seq_ids, poss, shard.seq_len)
        expected = np.arange(base, base + keep, dtype=np.int64)
        if not np.array_equal(mapped, expected):
            raise AssertionError(
                "token_idx mapping drifted from schema.token_index -- this is "
                "a bug in store.dump"
            )
        tokens_batch = {
            "token_idx": expected.astype(np.uint32),
            "seq_idx": seq_ids.astype(np.uint32),
            "pos": poss.astype(np.uint16),
            "token_id": batch[:, 1:].reshape(-1)[:keep].astype(np.uint16),
        }

        state.n_rows += batch_rows
        state.token_cursor = base + keep
        state.seq_cursor = seq_hi
        tokens_run += keep
        rows_run += batch_rows
        n_batches += 1

        _flush_segment(
            writer.add(
                rows_batch,
                tokens_batch,
                seq_end=state.seq_cursor,
                token_end=state.token_cursor,
            )
        )

        if n_batches % cfg.log_every == 0:
            now = time.monotonic()
            dt = max(now - t_mark, 1e-9)
            total_dt = max(now - t_start, 1e-9)
            log.info(
                "batch %d | %d/%d tokens | interval %.0f tok/s %.0f rows/s | "
                "run avg %.0f tok/s %.0f rows/s",
                n_batches,
                state.token_cursor,
                n_target,
                (tokens_run - tokens_mark) / dt,
                (rows_run - rows_mark) / dt,
                tokens_run / total_dt,
                rows_run / total_dt,
            )
            t_mark = now
            tokens_mark = tokens_run
            rows_mark = rows_run

    _flush_segment(writer.flush())

    # --- end-of-run invariants ------------------------------------------
    if state.n_rows != state.l0_sum:
        raise AssertionError(
            f"rows written ({state.n_rows}) != sum of per-token L0 on device "
            f"({state.l0_sum}); dump output is inconsistent"
        )
    mean_l0 = state.n_rows / state.token_cursor if state.token_cursor else 0.0
    ref_l0 = _reference_l0(cfg.checkpoint)
    if ref_l0 is not None and ref_l0 > 0:
        rel = abs(mean_l0 - ref_l0) / ref_l0
        if rel > 0.05:
            log.warning(
                "mean L0 %.3f deviates %.1f%% from checkpoint metrics.json "
                "L0 %.3f",
                mean_l0,
                100 * rel,
                ref_l0,
            )
        else:
            log.info(
                "mean L0 %.3f within 5%% of metrics.json L0 %.3f",
                mean_l0,
                ref_l0,
            )

    meta = current_meta(complete=True)
    _write_meta(cfg, meta)
    total_dt = max(time.monotonic() - t_start, 1e-9)
    log.info(
        "done: %d tokens -> %d rows (mean L0 %.3f) in %d files; %.0f tok/s "
        "%.0f rows/s overall (this run)",
        state.token_cursor,
        state.n_rows,
        mean_l0,
        len(state.segments),
        tokens_run / total_dt,
        rows_run / total_dt,
    )
    return meta


#


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m store.dump",
        description="Sequential SAE encode of a token shard to staging "
        "Parquet (rows-*.parquet + tokens-*.parquet + meta.json).",
    )
    p.add_argument("--checkpoint", required=True, type=Path,
                   help="SAE checkpoint .pt (e.g. ~/sae-gpt2-small/results/"
                        "frontier/<name>/checkpoint.pt)")
    p.add_argument("--shard", required=True, type=Path,
                   help="uint16 token shard .bin with a .json sidecar")
    p.add_argument("--n-tokens", required=True, type=int,
                   help="BOS-excluded tokens to encode (capped at shard size)")
    p.add_argument("--out", required=True, type=Path,
                   help="output directory, e.g. work/flat/")
    p.add_argument("--batch-seqs", type=int, default=512,
                   help="sequences per model forward (default 512)")
    p.add_argument("--rows-per-file", type=int,
                   default=schema.DEFAULT_ROWS_PER_FILE,
                   help="approx rows per Parquet file (default 50M)")
    p.add_argument("--row-group-size", type=int,
                   default=schema.DEFAULT_ROW_GROUP_SIZE,
                   help="Parquet row-group size (default 1M)")
    p.add_argument("--encode-chunk", type=int, default=16_384,
                   help="tokens per sae.encode call; caps dense-h memory "
                        "(default 16384)")
    p.add_argument("--sae-repo", default=None,
                   help="path to sae-gpt2-small (default: $SAE_REPO, else "
                        "~/sae-gpt2-small)")
    p.add_argument("--resume", action="store_true",
                   help="continue an interrupted run in --out, skipping "
                        "segments recorded complete in meta.json")
    p.add_argument("--device", default="auto",
                   help="cuda | cpu | auto (default auto)")
    p.add_argument("--model", dest="model_name", default="gpt2",
                   help="transformer_lens model name (default gpt2)")
    p.add_argument("--layer", type=int, default=None,
                   help="residual layer (default: checkpoint's layer, else 8)")
    p.add_argument("--hook-name", default=None,
                   help="override hook (default blocks.<layer>.hook_resid_post)")
    p.add_argument("--log-every", type=int, default=20,
                   help="log throughput every N batches (default 20)")
    return p


def main(
    argv: list[str] | None = None,
    *,
    modules: SAEModules | None = None,
    model: ResidualModel | None = None,
    after_flush: Callable[[Segment], None] | None = None,
) -> int:
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(name)s %(levelname)s %(message)s",
        )
    args = build_parser().parse_args(argv)
    cfg = DumpConfig(
        checkpoint=args.checkpoint,
        shard=args.shard,
        out=args.out,
        n_tokens=args.n_tokens,
        batch_seqs=args.batch_seqs,
        rows_per_file=args.rows_per_file,
        row_group_size=args.row_group_size,
        encode_chunk=args.encode_chunk,
        sae_repo=args.sae_repo,
        resume=args.resume,
        device=args.device,
        model_name=args.model_name,
        layer=args.layer,
        hook_name=args.hook_name,
        log_every=args.log_every,
    )
    dump(cfg, modules=modules, model=model, after_flush=after_flush)
    return 0


if __name__ == "__main__":
    sys.exit(main())
