"""Arrow schemas, layout constants, and the token_idx mapping. The shard is a packed
``[n_seqs, seq_len]`` array of GPT-2 token ids with a BOS token at position 0 of
every sequence."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np
import pyarrow as pa

#

FORMAT_VERSION: int = 1

COMPRESSION: str = "zstd"
DEFAULT_ROW_GROUP_SIZE: int = 1_000_000
DEFAULT_ROWS_PER_FILE: int = 50_000_000

ROWS_FILE_FMT: str = "rows-{index:05d}.parquet"
TOKENS_FILE_FMT: str = "tokens-{index:05d}.parquet"
META_FILENAME: str = "meta.json"

#

# : One row per (token, active feature) pair.
ROWS_SCHEMA: pa.Schema = pa.schema(
    [
        pa.field("token_idx", pa.uint32(), nullable=False),
        pa.field("feature", pa.uint32(), nullable=False),
        pa.field("value", pa.float32(), nullable=False),
    ]
)

# : One row per encoded token (whether or not any feature fired)
TOKENS_SCHEMA: pa.Schema = pa.schema(
    [
        pa.field("token_idx", pa.uint32(), nullable=False),
        pa.field("seq_idx", pa.uint32(), nullable=False),
        pa.field("pos", pa.uint16(), nullable=False),
        pa.field("token_id", pa.uint16(), nullable=False),
    ]
)

#

IndexLike = int | np.integer | np.ndarray


def token_index(seq_idx: IndexLike, pos: IndexLike, seq_len: int) -> Any:
    """Map (seq_idx, pos) -> token_idx in the sequential BOS-excluded stream.
    ``pos`` is the position within the sequence; position 0 is the BOS and is
    excluded, so valid positions are ``1 <= pos < seq_len``."""
    if seq_len < 2:
        raise ValueError(f"seq_len must be >= 2, got {seq_len}")
    seq_a = np.asarray(seq_idx, dtype=np.int64)
    pos_a = np.asarray(pos, dtype=np.int64)
    if np.any(seq_a < 0):
        raise ValueError("seq_idx must be >= 0")
    if np.any(pos_a < 1) or np.any(pos_a >= seq_len):
        raise ValueError(
            f"pos must be in [1, {seq_len - 1}] (pos 0 is BOS, excluded)"
        )
    out = seq_a * (seq_len - 1) + (pos_a - 1)
    if out.ndim == 0:
        return int(out)
    return out


def token_to_seq_pos(token_idx: IndexLike, seq_len: int) -> Any:
    """Inverse of :func:`token_index`: token_idx -> (seq_idx, pos)."""
    if seq_len < 2:
        raise ValueError(f"seq_len must be >= 2, got {seq_len}")
    t = np.asarray(token_idx, dtype=np.int64)
    if np.any(t < 0):
        raise ValueError("token_idx must be >= 0")
    seq_a = t // (seq_len - 1)
    pos_a = t % (seq_len - 1) + 1
    if t.ndim == 0:
        return int(seq_a), int(pos_a)
    return seq_a, pos_a


#


def jsonable(obj: Any) -> Any:
    """Best-effort coercion of config-ish objects to JSON-safe values."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    return str(obj)


def build_meta(
    *,
    sae_checkpoint: dict[str, Any],
    sae_config: dict[str, Any],
    hook_name: str,
    layer: int | None,
    model_name: str,
    shard: dict[str, Any],
    n_tokens_requested: int,
    n_tokens_encoded: int,
    n_rows: int,
    l0_sum: int,
    mean_l0: float | None,
    rows_per_file: int,
    row_group_size: int,
    segments: list[dict[str, Any]],
    progress: dict[str, Any],
    git: dict[str, Any],
    created_at: str | None = None,
) -> dict[str, Any]:
    """Assemble the meta.json dict."""
    return jsonable(
        {
            "format_version": FORMAT_VERSION,
            "created_at": created_at
            or datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "sae_checkpoint": sae_checkpoint,
            "sae_config": sae_config,
            "hook_name": hook_name,
            "layer": layer,
            "model_name": model_name,
            "shard": shard,
            "n_tokens_requested": n_tokens_requested,
            "n_tokens_encoded": n_tokens_encoded,
            "n_rows": n_rows,
            "l0_sum": l0_sum,
            "mean_l0": mean_l0,
            "rows_per_file": rows_per_file,
            "row_group_size": row_group_size,
            "compression": COMPRESSION,
            "segments": segments,
            "progress": progress,
            "git": git,
        }
    )
