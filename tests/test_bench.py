"""Tests for store.bench: end-to-end run over a tiny store writes both outputs, the
kill-point gate fires and still writes them."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from store import bench, queries

from .test_queries import N_FEATURES, N_TOKENS, build_tiny_store


@pytest.fixture(scope="module")
def store(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("bench_store")
    build_tiny_store(root, seed=23)
    return root


def _run(store: Path, out: Path, *extra: str) -> int:
    return bench.main(
        [
            "--store", str(store),
            "--n-features", "3", "--n-tokens", "3", "--trials", "1",
            "--out", str(out), "--seed", "0",
            *extra,
        ]
    )


def test_sample_tokens_uniform_and_seeded(store: Path):
    meta = json.loads(
        (store / queries.BUCKETED_SUBDIR / "meta.json").read_text()
    )
    toks = bench.sample_tokens(meta, 5, seed=3)
    assert len(toks) == len(set(toks)) == 5
    assert all(0 <= t < N_TOKENS for t in toks)
    assert toks == bench.sample_tokens(meta, 5, seed=3)


def test_method_matrix_requires_feature_layout(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="store.partition"):
        bench.method_matrix(tmp_path)


def test_method_matrix_token_layout_optional(store: Path, tmp_path: Path):
    # copy only flat + bucketed; the token-layout methods must be skipped
    partial = tmp_path / "partial"
    for sub in (queries.FLAT_SUBDIR, queries.BUCKETED_SUBDIR):
        for f in (store / sub).rglob("*"):
            if f.is_file():
                dst = partial / f.relative_to(store)
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_bytes(f.read_bytes())
    matrix = bench.method_matrix(partial)
    assert "token_bucketed_scan" not in matrix["tokens_for_feature"]
    assert "token_bucketed" not in matrix["features_for_token"]
    assert "bucketed" in matrix["tokens_for_feature"]
