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


