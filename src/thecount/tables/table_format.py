"""Shared binary constants and bounded-row iteration for COUNT tables."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np

import hashlib


MAGIC = b"COUN"  # four-byte tag at the start of every packed table
FLOAT32_VERSION = 1
UINT8_VERSION = 2
SCORE_ENCODING_LINEAR_UINT8 = 1
HEADER_SIZE = 64
SECTION_HEADER_SIZE = 64

U8 = np.dtype("u1")
U32 = np.dtype("<u4")
F32 = np.dtype("<f4")
U64 = np.dtype("<u8")

RAW_DOC_SEPARATOR_TOKEN = np.uint32(0xFFFFFFFF)
RAW_EOF_PAD_TOKEN = np.uint32(0xFFFFFFFE)
DEFAULT_SENTINEL_EOS_TOKEN_ID = 100257


def sha256_file(path: Path, chunk_bytes: int = 16 * 1024 * 1024) -> str:
    """Return the SHA-256 digest of *path* without loading it into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest()


def row_slices(rows: int, chunk_rows: int) -> Iterator[slice]:
    """Yield bounded contiguous slices covering ``range(rows)``."""

    for start in range(0, rows, chunk_rows):
        yield slice(start, min(rows, start + chunk_rows))


def struct_dtype(plen: int) -> np.dtype:
    """The reader's field-lexicographic comparison dtype for a prefix width."""

    return np.dtype([(f"f{j}", "<u4") for j in range(plen)])


def table_identity(table_dir: Path) -> tuple[Path, str]:
    """Return the sole supported table payload and its SHA-256 digest."""

    candidates = [
        path
        for path in (
            table_dir / "forward_index_topk.bin",
            table_dir / "history_vector_topk.bin",
        )
        if path.is_file()
    ]
    if len(candidates) != 1:
        raise ValueError(
            f"expected exactly one supported table in {table_dir}, got {candidates}"
        )
    path = candidates[0]
    return path, sha256_file(path)


def table_config(table_dir: Path) -> tuple[int, int, int]:
    """Read the runtime width, order, and vocabulary size from the manifest."""

    manifest_path = table_dir / "build_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read build manifest {manifest_path}: {exc}") from exc
    config = manifest.get("config") if isinstance(manifest, dict) else None
    if not isinstance(config, dict):
        raise ValueError(f"build manifest has no config object: {manifest_path}")
    values = (config.get("K"), config.get("max_order"), config.get("vocab_size"))
    if not all(
        isinstance(value, int) and not isinstance(value, bool) for value in values
    ):
        raise ValueError("build manifest must contain integer K/max_order/vocab_size")
    K, max_order, vocab_size = values
    if K <= 0 or max_order <= 0 or vocab_size <= K:
        raise ValueError("build manifest has invalid K/max_order/vocab_size")
    return K, max_order, vocab_size
