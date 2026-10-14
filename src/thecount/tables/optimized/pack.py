#!/usr/bin/env python3
"""Pack the stupid-backoff scoring output into a training COUNT table.

The scored rows written by ``thecount.tables.stupid_backoff_scoring`` store one dense
stupid-backoff row per explicit history:

* ``order1.toks.bin`` / ``order1.lps.bin``: one ``uint32`` / ``float32``
  row of width ``kmax``;
* for order ``n >= 2``, ``order<n>.pref.bin`` stores ``uint32`` history
  keys of width ``n - 1``, ``order<n>.ctot.bin`` stores history totals,
  and ``order<n>.toks.bin`` / ``order<n>.lps.bin`` store dense rows of
  width ``kmax``.

This packer filters histories by total count, crops rows to a requested K,
and writes an COUNT table layout consumed by
``olmo_core.data.count_table.CountTable``. The scored rows are sorted
byte-lexicographically over little-endian prefix bytes, but the production
reader searches numerically by ``uint32`` token tuple. Therefore every
retained order is explicitly reordered into numeric tuple order while its
token and score rows move under the same permutation.

A count source may store document boundaries as raw ``0xFFFFFFFF`` /
``0xFFFFFFFE`` token IDs. Those IDs are finite continuation candidates, not
dense-row padding. Candidate slots are mapped to the tokenizer's
EOS token before validation and packing. Prefixes remain strict: a retained
history containing an out-of-vocabulary raw sentinel is rejected rather than
silently merging distinct source histories.

Memory and temporary-storage bounds
-----------------------------------
Input, filtered rows, and output arrays are memory-mapped. Copying and
validation use ``--chunk-rows`` rows at a time. Numeric sorting requires one
in-memory ``int64`` permutation per retained order (8 bytes per retained
history) plus NumPy's sort workspace. Temporary disk use is one filtered
order at a time in addition to the partial table output.

The resulting ``build_manifest.json`` records the exact source
``scores_meta.json`` and its SHA256, the configuration, per-order row and byte
counts, and the output file's SHA256.

Example
-------

.. code-block:: bash

   PYTHONPATH=src:. python -m thecount.tables.pack \
     --input-dir /path/to/scored-rows \
     --output-dir /path/to/sb-tau4096-k16-o5 \
     --max-order 5 --K 16 --min-history-total 4096 \
     --vocab-size 100278

Use ``--dry-run`` without an output path to validate the source and print
the retained row counts and projected output size without writing anything.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import struct
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from thecount.tables.table_format import (
    DEFAULT_SENTINEL_EOS_TOKEN_ID,
    F32,
    FLOAT32_VERSION,
    HEADER_SIZE,
    MAGIC,
    RAW_DOC_SEPARATOR_TOKEN,
    RAW_EOF_PAD_TOKEN,
    SCORE_ENCODING_LINEAR_UINT8,
    SECTION_HEADER_SIZE,
    U32,
    U64,
    UINT8_VERSION,
    row_slices,
    sha256_file,
)


# Widest supported source row (scoring runs export kmax between 64 and 512).
MAX_SOURCE_K = 512
MASS_QUANTILE_SAMPLE_ROWS = 131_072
F64 = np.dtype("<f8")


class ScoredRowsFormatError(ValueError):
    """The scored rows are incomplete or internally inconsistent."""


@dataclass(frozen=True)
class OrderPlan:
    order: int
    input_rows: int
    output_rows: int
    input_bytes: dict[str, int]
    output_bytes: dict[str, int]


@dataclass(frozen=True)
class PackPlan:
    input_dir: Path
    source_manifest_path: Path
    source_manifest_sha256: str
    source_manifest: dict[str, Any]
    source_kmax: int
    max_order: int
    emit_orders: tuple[int, ...] | None
    K: int
    min_history_total: float
    vocab_size: int
    sentinel_eos_token_id: int
    ctot_dtype: str
    logprob_bytes: int
    chunk_rows: int
    orders: tuple[OrderPlan, ...]
    projected_output_bytes: int


@dataclass(frozen=True)
class SectionLayout:
    order: int
    rows: int
    prefixes_offset: int
    tokens_offset: int
    logprobs_offset: int
    finite_mask_offset: int


def estimate_table_bytes(
    order_rows: Iterable[tuple[int, int]],
    *,
    K: int,
    logprob_bytes: int,
) -> int:
    """Return the exact COUNT table file size before writing it.

    ``order_rows`` supplies ``(ngram_order, retained_history_rows)`` pairs.
    For uint8 COUNT table v2 this includes the row-wise packed finite mask as well as
    score codes and all fixed headers.
    """
    pairs = list(order_rows)
    if K <= 0:
        raise ValueError(f"K must be positive, got {K}")
    if logprob_bytes != 1:
        raise ValueError(f"logprob_bytes must be 1 (uint8 scores), got {logprob_bytes}")
    if len({order for order, _ in pairs}) != len(pairs):
        raise ValueError("order_rows contains a duplicate ngram order")
    total = HEADER_SIZE + len(pairs) * SECTION_HEADER_SIZE
    for order, rows in pairs:
        if order <= 0 or rows < 0:
            raise ValueError(
                f"order_rows values must be positive orders and non-negative "
                f"row counts, got {(order, rows)}"
            )
        total += rows * (order - 1) * U32.itemsize
        total += rows * K * U32.itemsize
        total += rows * K * logprob_bytes
        if logprob_bytes == 1:
            total += rows * ((K + 7) // 8)
    return total


def select_k_for_budget(
    order_rows: Iterable[tuple[int, int]],
    *,
    source_kmax: int,
    budget_bytes: int,
    logprob_bytes: int,
) -> dict[str, int]:
    """Choose the widest row whose exact COUNT table size does not exceed a budget."""

    rows = tuple(order_rows)
    if source_kmax <= 0 or budget_bytes <= 0:
        raise ValueError("source_kmax and budget_bytes must be positive")
    choices = tuple(
        (K, estimate_table_bytes(rows, K=K, logprob_bytes=logprob_bytes))
        for K in range(1, source_kmax + 1)
    )
    under = tuple(choice for choice in choices if choice[1] <= budget_bytes)
    if not under:
        raise ValueError("even K=1 exceeds the requested budget")
    selected_k, selected_bytes = max(under, key=lambda choice: choice[1])
    closest_k, closest_bytes = min(
        choices, key=lambda choice: abs(choice[1] - budget_bytes)
    )
    return {
        "K": selected_k,
        "bytes": selected_bytes,
        "delta_bytes": selected_bytes - budget_bytes,
        "closest_K": closest_k,
        "closest_bytes": closest_bytes,
    }


def _require_exact_size(path: Path, expected: int, label: str) -> int:
    try:
        actual = path.stat().st_size
    except FileNotFoundError as exc:
        raise ScoredRowsFormatError(f"missing {label}: {path}") from exc
    if actual != expected:
        raise ScoredRowsFormatError(
            f"{label} has {actual:,} bytes; expected exactly {expected:,}: {path}"
        )
    return actual


def _metadata_rows(metadata: dict[str, Any], order: int) -> int:
    orders = metadata.get("orders")
    if not isinstance(orders, dict):
        raise ScoredRowsFormatError("scores_meta.json must contain an 'orders' object")
    entry = orders.get(str(order))
    if not isinstance(entry, dict):
        raise ScoredRowsFormatError(f"scores_meta.json is missing metadata for order {order}")
    rows = entry.get("rows")
    if not isinstance(rows, int) or isinstance(rows, bool) or rows < 0:
        raise ScoredRowsFormatError(
            f"scores_meta.json orders.{order}.rows must be a non-negative integer"
        )
    return rows


def _normalize_declared_ctot_dtype(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).lower().replace("_", "").replace("-", "")
    aliases = {
        "uint64": "uint64",
        "u8": "uint64",
        "<u8": "uint64",
        "float64": "float64",
        "f8": "float64",
        "<f8": "float64",
    }
    return aliases.get(normalized)


def _validate_metadata(
    metadata: dict[str, Any],
    *,
    max_order: int,
    K: int,
    min_history_total: float,
    vocab_size: int,
    sentinel_eos_token_id: int,
    ctot_dtype: str,
    logprob_bytes: int,
) -> int:
    source_kmax = metadata.get("kmax")
    if (
        not isinstance(source_kmax, int)
        or isinstance(source_kmax, bool)
        or not 1 <= source_kmax <= MAX_SOURCE_K
    ):
        raise ScoredRowsFormatError(
            f"scores_meta.json kmax must be an integer in [1, {MAX_SOURCE_K}]"
        )
    if not 1 <= K <= source_kmax:
        raise ScoredRowsFormatError(f"K must be in [1, source kmax={source_kmax}], got {K}")
    if not 1 <= max_order:
        raise ScoredRowsFormatError(f"max_order must be positive, got {max_order}")
    if vocab_size <= K:
        raise ScoredRowsFormatError(
            f"vocab_size must be greater than K for the training residual, got "
            f"vocab_size={vocab_size}, K={K}"
        )
    if (
        not isinstance(sentinel_eos_token_id, int)
        or isinstance(sentinel_eos_token_id, bool)
        or not 0 <= sentinel_eos_token_id < vocab_size
    ):
        raise ScoredRowsFormatError(
            "sentinel_eos_token_id must be an integer model token in "
            f"[0, vocab_size), got {sentinel_eos_token_id}"
        )
    if logprob_bytes != 1:
        raise ScoredRowsFormatError(f"logprob_bytes must be 1 (uint8 scores), got {logprob_bytes}")
    if not math.isfinite(min_history_total) or min_history_total < 0:
        raise ScoredRowsFormatError(
            f"min_history_total must be finite and non-negative, got {min_history_total}"
        )

    source_tau = metadata.get("tau")
    if source_tau is not None:
        try:
            source_tau_f = float(source_tau)
        except (TypeError, ValueError) as exc:
            raise ScoredRowsFormatError("scores_meta.json tau must be numeric") from exc
        if not math.isfinite(source_tau_f) or source_tau_f < 0:
            raise ScoredRowsFormatError("scores_meta.json tau must be finite and non-negative")
        if min_history_total < source_tau_f:
            raise ScoredRowsFormatError(
                f"requested history-total threshold {min_history_total:g} is below "
                f"the scored rows' threshold {source_tau_f:g}; missing rows cannot be recovered"
            )

    declared = metadata.get("ctot_dtype")
    if declared is None and isinstance(metadata.get("dtypes"), dict):
        declared = metadata["dtypes"].get("ctot")
    if declared is not None:
        normalized = _normalize_declared_ctot_dtype(declared)
        if normalized is None:
            raise ScoredRowsFormatError(
                f"scores_meta.json declares unsupported ctot dtype {declared!r}"
            )
        if normalized != ctot_dtype:
            raise ScoredRowsFormatError(
                f"scores_meta.json declares ctot dtype {normalized}, but "
                f"--ctot-dtype is {ctot_dtype}"
            )

    # Require every section that will enter the output, including order 1.
    for order in range(1, max_order + 1):
        _metadata_rows(metadata, order)
        entry = metadata["orders"][str(order)]
        if order >= 2 and "missing_parent" in entry:
            missing_parent = entry["missing_parent"]
            if (
                not isinstance(missing_parent, int)
                or isinstance(missing_parent, bool)
                or missing_parent < 0
            ):
                raise ScoredRowsFormatError(
                    f"scores_meta.json orders.{order}.missing_parent must be "
                    f"a non-negative integer"
                )
            if missing_parent:
                raise ScoredRowsFormatError(
                    f"order {order} reports {missing_parent} histories without "
                    f"their backoff parent; scored rows are not complete"
                )
    if _metadata_rows(metadata, 1) != 1:
        raise ScoredRowsFormatError("order 1 must contain exactly one empty-prefix row")
    source_alpha = metadata.get("alpha")
    if source_alpha is not None:
        try:
            source_alpha_f = float(source_alpha)
        except (TypeError, ValueError) as exc:
            raise ScoredRowsFormatError("scores_meta.json alpha must be numeric") from exc
        if not math.isfinite(source_alpha_f) or not 0.0 < source_alpha_f <= 1.0:
            raise ScoredRowsFormatError(
                "scores_meta.json alpha must be finite and in the interval (0, 1]"
            )
    return source_kmax


def _ctot_np_dtype(ctot_dtype: str) -> np.dtype:
    if ctot_dtype == "uint64":
        return U64
    if ctot_dtype == "float64":
        return F64
    raise AssertionError(f"unhandled ctot dtype {ctot_dtype}")


def _validate_ctot_chunk(values: np.ndarray, *, order: int, ctot_dtype: str) -> None:
    if ctot_dtype == "float64":
        if not np.isfinite(values).all():
            raise ScoredRowsFormatError(f"order {order} ctot contains NaN or infinity")
        if np.any(values < 0):
            raise ScoredRowsFormatError(f"order {order} ctot contains a negative total")


def _count_retained(
    path: Path,
    rows: int,
    *,
    order: int,
    threshold: float,
    ctot_dtype: str,
    chunk_rows: int,
) -> int:
    if rows == 0:
        return 0
    dtype = _ctot_np_dtype(ctot_dtype)
    totals = np.memmap(path, dtype=dtype, mode="r", shape=(rows,))
    retained = 0
    try:
        for span in row_slices(rows, chunk_rows):
            values = np.asarray(totals[span])
            _validate_ctot_chunk(values, order=order, ctot_dtype=ctot_dtype)
            retained += int(np.count_nonzero(values >= threshold))
    finally:
        del totals
    return retained


def plan_scored_rows(
    input_dir: Path,
    *,
    max_order: int,
    K: int,
    min_history_total: float,
    vocab_size: int,
    sentinel_eos_token_id: int = DEFAULT_SENTINEL_EOS_TOKEN_ID,
    ctot_dtype: str = "uint64",
    logprob_bytes: int = 1,
    chunk_rows: int = 131_072,
    emit_orders: frozenset[int] | None = None,
) -> PackPlan:
    """Validate source metadata and exact file sizes, then count kept rows.

    ``emit_orders`` restricts which orders receive rows in the packed table.
    Orders outside the set keep their sections, empty, so the reader's
    fallback walks every order and accrues its backoff penalty per order of
    history shortening — keeping a candidate's score identical whether it is
    found through a packed row at a lower order or was inherited into a
    higher-order row at build time. Order 1 is required: it is the floor
    every backoff chain terminates in. Source rows are built from all orders
    regardless, so within-row backoff content is unaffected by the subset.
    """
    if emit_orders is not None:
        emit_orders = frozenset(int(o) for o in emit_orders)
        if 1 not in emit_orders:
            raise ValueError("emit_orders must include order 1, the backoff floor")
        bad = sorted(o for o in emit_orders if not 1 <= o <= max_order)
        if bad:
            raise ValueError(f"emit_orders outside 1..{max_order}: {bad}")
    input_dir = Path(input_dir).resolve()
    if chunk_rows <= 0:
        raise ScoredRowsFormatError(f"chunk_rows must be positive, got {chunk_rows}")
    manifest_path = input_dir / "scores_meta.json"
    try:
        manifest_bytes = manifest_path.read_bytes()
    except FileNotFoundError as exc:
        raise ScoredRowsFormatError(f"missing source manifest: {manifest_path}") from exc
    try:
        metadata = json.loads(manifest_bytes)
    except json.JSONDecodeError as exc:
        raise ScoredRowsFormatError(f"invalid JSON in {manifest_path}: {exc}") from exc
    if not isinstance(metadata, dict):
        raise ScoredRowsFormatError("scores_meta.json must contain a JSON object")

    source_kmax = _validate_metadata(
        metadata,
        max_order=max_order,
        K=K,
        min_history_total=min_history_total,
        vocab_size=vocab_size,
        sentinel_eos_token_id=sentinel_eos_token_id,
        ctot_dtype=ctot_dtype,
        logprob_bytes=logprob_bytes,
    )
    if metadata.get("key_transform") is not None:
        raise ScoredRowsFormatError(
            "scores_meta.json declares a key_transform; this packer only "
            "handles rows keyed by raw token ids"
        )
    order_plans: list[OrderPlan] = []
    for order in range(1, max_order + 1):
        input_rows = _metadata_rows(metadata, order)
        input_bytes: dict[str, int]
        if order == 1:
            toks_bytes = _require_exact_size(
                input_dir / "order1.toks.bin",
                source_kmax * U32.itemsize,
                "order 1 tokens",
            )
            lps_bytes = _require_exact_size(
                input_dir / "order1.lps.bin",
                source_kmax * F32.itemsize,
                "order 1 log scores",
            )
            input_bytes = {"tokens": toks_bytes, "logprobs": lps_bytes}
            output_rows = 1
        else:
            prefix_bytes = _require_exact_size(
                input_dir / f"order{order}.pref.bin",
                input_rows * (order - 1) * U32.itemsize,
                f"order {order} prefixes",
            )
            ctot_bytes = _require_exact_size(
                input_dir / f"order{order}.ctot.bin",
                input_rows * _ctot_np_dtype(ctot_dtype).itemsize,
                f"order {order} history totals",
            )
            toks_bytes = _require_exact_size(
                input_dir / f"order{order}.toks.bin",
                input_rows * source_kmax * U32.itemsize,
                f"order {order} tokens",
            )
            lps_bytes = _require_exact_size(
                input_dir / f"order{order}.lps.bin",
                input_rows * source_kmax * F32.itemsize,
                f"order {order} log scores",
            )
            input_bytes = {
                "prefixes": prefix_bytes,
                "history_totals": ctot_bytes,
                "tokens": toks_bytes,
                "logprobs": lps_bytes,
            }
            if emit_orders is not None and order not in emit_orders:
                output_rows = 0
            else:
                output_rows = _count_retained(
                    input_dir / f"order{order}.ctot.bin",
                    input_rows,
                    order=order,
                    threshold=min_history_total,
                    ctot_dtype=ctot_dtype,
                    chunk_rows=chunk_rows,
                )

        output_bytes = {
            "prefixes": output_rows * (order - 1) * U32.itemsize,
            "tokens": output_rows * K * U32.itemsize,
            "logprobs": output_rows * K * logprob_bytes,
            "finite_mask": (
                output_rows * ((K + 7) // 8) if logprob_bytes == 1 else 0
            ),
        }
        order_plans.append(
            OrderPlan(
                order=order,
                input_rows=input_rows,
                output_rows=output_rows,
                input_bytes=input_bytes,
                output_bytes=output_bytes,
            )
        )

    projected = estimate_table_bytes(
        ((order.order, order.output_rows) for order in order_plans),
        K=K,
        logprob_bytes=logprob_bytes,
    )
    expected_from_sections = HEADER_SIZE + len(order_plans) * SECTION_HEADER_SIZE
    expected_from_sections += sum(
        sum(order.output_bytes.values()) for order in order_plans
    )
    if projected != expected_from_sections:
        raise AssertionError(
            f"size estimator mismatch: {projected} != {expected_from_sections}"
        )
    return PackPlan(
        input_dir=input_dir,
        source_manifest_path=manifest_path,
        source_manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        source_manifest=metadata,
        source_kmax=source_kmax,
        max_order=max_order,
        emit_orders=(
            tuple(sorted(emit_orders)) if emit_orders is not None else None
        ),
        K=K,
        min_history_total=float(min_history_total),
        vocab_size=vocab_size,
        sentinel_eos_token_id=sentinel_eos_token_id,
        ctot_dtype=ctot_dtype,
        logprob_bytes=logprob_bytes,
        chunk_rows=chunk_rows,
        orders=tuple(order_plans),
        projected_output_bytes=projected,
    )


def _layouts(plan: PackPlan) -> list[SectionLayout]:
    cursor = HEADER_SIZE + len(plan.orders) * SECTION_HEADER_SIZE
    layouts: list[SectionLayout] = []
    for order_plan in plan.orders:
        prefix_offset = cursor
        cursor += order_plan.output_bytes["prefixes"]
        token_offset = cursor
        cursor += order_plan.output_bytes["tokens"]
        logprob_offset = cursor
        cursor += order_plan.output_bytes["logprobs"]
        finite_mask_offset = cursor if plan.logprob_bytes == 1 else 0
        cursor += order_plan.output_bytes["finite_mask"]
        layouts.append(
            SectionLayout(
                order=order_plan.order,
                rows=order_plan.output_rows,
                prefixes_offset=prefix_offset,
                tokens_offset=token_offset,
                logprobs_offset=logprob_offset,
                finite_mask_offset=finite_mask_offset,
            )
        )
    if cursor != plan.projected_output_bytes:
        raise AssertionError(
            f"internal layout mismatch: {cursor} != {plan.projected_output_bytes}"
        )
    return layouts


def _format_version(plan: PackPlan) -> int:
    return UINT8_VERSION if plan.logprob_bytes == 1 else FLOAT32_VERSION


def _write_header(
    path: Path,
    plan: PackPlan,
    layouts: list[SectionLayout],
    quantization_lows: dict[int, float | None],
) -> None:
    with path.open("r+b") as handle:
        header = struct.pack(
            "<4sIIII",
            MAGIC,
            _format_version(plan),
            plan.K,
            len(layouts),
            plan.vocab_size,
        )
        handle.write(header + b"\x00" * (HEADER_SIZE - len(header)))
        for layout in layouts:
            section = bytearray(SECTION_HEADER_SIZE)
            struct.pack_into(
                "<IIQQQQ",
                section,
                0,
                layout.order,
                0,
                layout.rows,
                layout.prefixes_offset,
                layout.tokens_offset,
                layout.logprobs_offset,
            )
            if plan.logprob_bytes == 1:
                quantization_low = quantization_lows.get(layout.order)
                if quantization_low is None:
                    quantization_low = 0.0
                struct.pack_into(
                    "<IffQ",
                    section,
                    40,
                    SCORE_ENCODING_LINEAR_UINT8,
                    quantization_low,
                    0.0,
                    layout.finite_mask_offset,
                )
            handle.write(section)


def _validate_scores_and_tokens(
    tokens: np.ndarray,
    scores: np.ndarray,
    *,
    order: int,
    vocab_size: int,
) -> None:
    if np.any(tokens >= vocab_size):
        bad = int(tokens[tokens >= vocab_size][0])
        raise ScoredRowsFormatError(
            f"order {order} contains token id {bad} outside vocab_size={vocab_size}"
        )
    if np.isnan(scores).any() or np.isposinf(scores).any():
        raise ScoredRowsFormatError(f"order {order} log scores contain NaN or +infinity")
    finite = np.isfinite(scores)
    if finite.any() and np.any(scores[finite] > np.float32(1e-6)):
        bad = float(scores[finite][scores[finite] > np.float32(1e-6)][0])
        raise ScoredRowsFormatError(
            f"order {order} contains positive log10 score {bad}; expected log probabilities"
        )


def _remap_candidate_sentinels(
    tokens: np.ndarray,
    *,
    eos_token_id: int,
) -> dict[str, int]:
    """Map raw document-boundary candidates to the model's EOS token."""
    counts: dict[str, int] = {}
    for source_id in (RAW_DOC_SEPARATOR_TOKEN, RAW_EOF_PAD_TOKEN):
        mask = tokens == source_id
        count = int(np.count_nonzero(mask))
        counts[str(int(source_id))] = count
        if count:
            tokens[mask] = np.uint32(eos_token_id)
    counts["total"] = sum(counts.values())
    return counts


def _add_remap_counts(total: dict[str, int], update: dict[str, int]) -> None:
    for key, value in update.items():
        total[key] = total.get(key, 0) + value


def _round_logprobs_in_place(
    scores: np.ndarray,
    *,
    logprob_bytes: int,
    chunk_rows: int,
) -> float | None:
    """Apply the per-order 256-level finite-value grid, if requested.

    The rounded float32 values remain in this temporary array for diagnostics.
    COUNT table v2 encodes those values as uint8 in :func:`_encode_logprob_rows`.
    """
    if logprob_bytes == 4 or scores.size == 0:
        return None
    finite_min: float | None = None
    for span in row_slices(len(scores), chunk_rows):
        chunk = np.asarray(scores[span])
        finite = np.isfinite(chunk)
        if finite.any():
            if np.any(chunk[finite] > 0.0):
                bad = float(chunk[finite][chunk[finite] > 0.0][0])
                raise ScoredRowsFormatError(
                    f"physical uint8 score encoding requires non-positive "
                    f"log10 scores, got {bad}"
                )
            value = float(chunk[finite].min())
            finite_min = value if finite_min is None else min(finite_min, value)
    if finite_min is None or finite_min == 0.0:
        return finite_min
    lo = finite_min
    scale = -lo
    for span in row_slices(len(scores), chunk_rows):
        chunk = np.asarray(scores[span]).copy()
        finite = np.isfinite(chunk)
        if finite.any():
            values = chunk[finite].astype(np.float64)
            q = np.rint((values - lo) / scale * 255.0)
            values = q / 255.0 * scale + lo
            chunk[finite] = values.astype(F32)
            scores[span] = chunk
    if isinstance(scores, np.memmap):
        scores.flush()
    return lo


def _encode_logprob_rows(
    scores: np.ndarray,
    *,
    quantization_low: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Encode rounded float32 rows as uint8 codes plus packed finite bits."""
    finite = np.isfinite(scores)
    codes = np.zeros(scores.shape, dtype=np.uint8)
    if finite.any():
        if quantization_low is None:
            raise AssertionError("finite quantized scores require a calibration minimum")
        if quantization_low < 0.0:
            values = np.asarray(scores[finite], dtype=np.float64)
            encoded = np.rint(
                (values - quantization_low) / (-quantization_low) * 255.0
            )
            codes[finite] = np.clip(encoded, 0.0, 255.0).astype(np.uint8)
    finite_mask = np.packbits(finite, axis=1, bitorder="little")
    return codes, finite_mask


def _row_mass_diagnostics(
    scores: np.ndarray,
    *,
    chunk_rows: int,
    sample_rows: int = MASS_QUANTILE_SAMPLE_ROWS,
) -> dict[str, Any]:
    """Summarize retained mass ``sum(10**stored_log10_score)`` per row.

    Mean, minimum, and maximum scan every row. Quantiles use all rows when
    small enough and otherwise a deterministic evenly spaced bounded sample.
    """
    rows = len(scores)
    if rows == 0:
        return {
            "row_count": 0,
            "mass_definition": "sum(10**stored_log10_score) over finite K slots",
            "mean": None,
            "min": None,
            "max": None,
            "quantiles": {},
            "quantile_method": "no rows",
            "quantile_sample_rows": 0,
        }

    total = 0.0
    minimum = math.inf
    maximum = -math.inf
    for span in row_slices(rows, chunk_rows):
        values = np.asarray(scores[span], dtype=np.float64)
        masses = np.power(10.0, values).sum(axis=1, dtype=np.float64)
        total += float(masses.sum(dtype=np.float64))
        minimum = min(minimum, float(masses.min()))
        maximum = max(maximum, float(masses.max()))

    sample_count = min(rows, sample_rows)
    if sample_count == rows:
        sampled_scores = np.asarray(scores, dtype=np.float64)
        method = "exact all rows"
    else:
        indices = np.linspace(0, rows - 1, sample_count, dtype=np.int64)
        sampled_scores = np.asarray(scores[indices], dtype=np.float64)
        method = (
            f"deterministic evenly spaced sample capped at "
            f"{MASS_QUANTILE_SAMPLE_ROWS} rows"
        )
    sample_masses = np.power(10.0, sampled_scores).sum(axis=1, dtype=np.float64)
    probabilities = (0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0)
    quantile_values = np.quantile(sample_masses, probabilities)
    quantiles = {
        f"p{int(probability * 100):02d}": float(value)
        for probability, value in zip(probabilities, quantile_values)
    }
    return {
        "row_count": rows,
        "mass_definition": "sum(10**stored_log10_score) over finite K slots",
        "mean": total / rows,
        "min": minimum,
        "max": maximum,
        "quantiles": quantiles,
        "quantile_method": method,
        "quantile_sample_rows": sample_count,
    }


def _map_or_empty(path: Path, dtype: np.dtype, shape: tuple[int, ...]) -> np.ndarray:
    if math.prod(shape) == 0:
        return np.empty(shape, dtype=dtype)
    return np.memmap(path, dtype=dtype, mode="r", shape=shape)


def _new_temp_memmap(
    path: Path, dtype: np.dtype, shape: tuple[int, ...]
) -> np.ndarray:
    elements = math.prod(shape)
    if elements == 0:
        return np.empty(shape, dtype=dtype)
    with path.open("wb") as handle:
        handle.truncate(elements * dtype.itemsize)
    return np.memmap(path, dtype=dtype, mode="r+", shape=shape)


def _map_output(
    path: Path, dtype: np.dtype, offset: int, shape: tuple[int, ...]
) -> np.ndarray:
    if math.prod(shape) == 0:
        return np.empty(shape, dtype=dtype)
    return np.memmap(path, dtype=dtype, mode="r+", offset=offset, shape=shape)


def _strict_numeric_order(prefixes: np.ndarray, previous: np.ndarray | None) -> np.ndarray:
    """Validate strict uint32 tuple order and return the final row."""
    if len(prefixes) == 0:
        if previous is None:
            raise AssertionError("empty chunk cannot establish a final prefix")
        return previous
    if previous is not None:
        combined = np.concatenate((previous.reshape(1, -1), prefixes[:1]), axis=0)
        _strict_numeric_order(combined, None)
    if len(prefixes) > 1:
        left = prefixes[:-1]
        right = prefixes[1:]
        undecided = np.ones(len(left), dtype=bool)
        descending = np.zeros(len(left), dtype=bool)
        for column in range(prefixes.shape[1]):
            descending |= undecided & (right[:, column] < left[:, column])
            undecided &= right[:, column] == left[:, column]
        if descending.any():
            row = int(np.flatnonzero(descending)[0])
            raise ScoredRowsFormatError(
                f"numeric prefix sort failed between {left[row].tolist()} and "
                f"{right[row].tolist()}"
            )
        if undecided.any():
            row = int(np.flatnonzero(undecided)[0])
            raise ScoredRowsFormatError(
                f"duplicate explicit prefix after filtering: {left[row].tolist()}"
            )
    return np.asarray(prefixes[-1]).copy()


def _numeric_permutation(prefixes: np.ndarray) -> np.ndarray:
    rows, width = prefixes.shape
    if rows == 0:
        return np.empty(0, dtype=np.int64)
    names = [f"f{column}" for column in range(width)]
    structured_dtype = np.dtype([(name, U32) for name in names])
    structured = prefixes.view(structured_dtype).reshape(rows)
    return np.argsort(structured, order=names, kind="stable")


def _copy_order1(
    plan: PackPlan,
    output_path: Path,
    layout: SectionLayout,
) -> tuple[float | None, dict[str, Any], dict[str, int]]:
    source_tokens = np.memmap(
        plan.input_dir / "order1.toks.bin",
        dtype=U32,
        mode="r",
        shape=(1, plan.source_kmax),
    )
    source_scores = np.memmap(
        plan.input_dir / "order1.lps.bin",
        dtype=F32,
        mode="r",
        shape=(1, plan.source_kmax),
    )
    tokens = np.asarray(source_tokens[:, : plan.K]).copy()
    scores = np.asarray(source_scores[:, : plan.K]).copy()
    remap_counts = _remap_candidate_sentinels(
        tokens, eos_token_id=plan.sentinel_eos_token_id
    )
    _validate_scores_and_tokens(tokens, scores, order=1, vocab_size=plan.vocab_size)
    quantization_low = _round_logprobs_in_place(
        scores,
        logprob_bytes=plan.logprob_bytes,
        chunk_rows=plan.chunk_rows,
    )
    mass_diagnostics = _row_mass_diagnostics(scores, chunk_rows=plan.chunk_rows)
    output_tokens = _map_output(
        output_path, U32, layout.tokens_offset, (1, plan.K)
    )
    output_tokens[:] = tokens
    if plan.logprob_bytes == 4:
        output_scores = _map_output(
            output_path, F32, layout.logprobs_offset, (1, plan.K)
        )
        output_scores[:] = scores
        output_mask = None
    else:
        output_scores = _map_output(
            output_path, np.dtype("u1"), layout.logprobs_offset, (1, plan.K)
        )
        output_mask = _map_output(
            output_path,
            np.dtype("u1"),
            layout.finite_mask_offset,
            (1, (plan.K + 7) // 8),
        )
        codes, finite_mask = _encode_logprob_rows(
            scores, quantization_low=quantization_low
        )
        output_scores[:] = codes
        output_mask[:] = finite_mask
    if isinstance(output_tokens, np.memmap):
        output_tokens.flush()
    if isinstance(output_scores, np.memmap):
        output_scores.flush()
    if isinstance(output_mask, np.memmap):
        output_mask.flush()
    del source_tokens, source_scores, output_tokens, output_scores, output_mask
    return quantization_low, mass_diagnostics, remap_counts


def _filter_order_to_temp(
    plan: PackPlan,
    order_plan: OrderPlan,
    temp_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[Path], dict[str, int]]:
    order = order_plan.order
    input_rows = order_plan.input_rows
    output_rows = order_plan.output_rows
    width = order - 1
    temp_paths = [
        temp_dir / f"order{order}.pref.u32",
        temp_dir / f"order{order}.toks.u32",
        temp_dir / f"order{order}.lps.f32",
    ]
    filtered_prefixes = _new_temp_memmap(temp_paths[0], U32, (output_rows, width))
    filtered_tokens = _new_temp_memmap(temp_paths[1], U32, (output_rows, plan.K))
    filtered_scores = _new_temp_memmap(temp_paths[2], F32, (output_rows, plan.K))
    remap_counts = {
        str(int(RAW_DOC_SEPARATOR_TOKEN)): 0,
        str(int(RAW_EOF_PAD_TOKEN)): 0,
        "total": 0,
    }
    if input_rows == 0:
        return (
            filtered_prefixes,
            filtered_tokens,
            filtered_scores,
            temp_paths,
            remap_counts,
        )

    source_prefixes = _map_or_empty(
        plan.input_dir / f"order{order}.pref.bin",
        U32,
        (input_rows, width),
    )
    source_totals = _map_or_empty(
        plan.input_dir / f"order{order}.ctot.bin",
        _ctot_np_dtype(plan.ctot_dtype),
        (input_rows,),
    )
    source_tokens = _map_or_empty(
        plan.input_dir / f"order{order}.toks.bin",
        U32,
        (input_rows, plan.source_kmax),
    )
    source_scores = _map_or_empty(
        plan.input_dir / f"order{order}.lps.bin",
        F32,
        (input_rows, plan.source_kmax),
    )
    cursor = 0
    try:
        for span in row_slices(input_rows, plan.chunk_rows):
            totals = np.asarray(source_totals[span])
            selected = totals >= plan.min_history_total
            count = int(np.count_nonzero(selected))
            if count == 0:
                continue
            end = cursor + count
            prefix_chunk = np.asarray(source_prefixes[span][selected])
            token_chunk = np.asarray(source_tokens[span, : plan.K][selected])
            score_chunk = np.asarray(source_scores[span, : plan.K][selected])
            if np.any(prefix_chunk >= plan.vocab_size):
                bad = int(prefix_chunk[prefix_chunk >= plan.vocab_size][0])
                raise ScoredRowsFormatError(
                    f"order {order} contains prefix token id {bad} outside "
                    f"vocab_size={plan.vocab_size}"
                )
            _add_remap_counts(
                remap_counts,
                _remap_candidate_sentinels(
                    token_chunk, eos_token_id=plan.sentinel_eos_token_id
                ),
            )
            _validate_scores_and_tokens(
                token_chunk,
                score_chunk,
                order=order,
                vocab_size=plan.vocab_size,
            )
            filtered_prefixes[cursor:end] = prefix_chunk
            filtered_tokens[cursor:end] = token_chunk
            filtered_scores[cursor:end] = score_chunk
            cursor = end
    finally:
        del source_prefixes, source_totals, source_tokens, source_scores
    if cursor != output_rows:
        raise ScoredRowsFormatError(
            f"order {order} retained-row count changed between planning and copy: "
            f"{cursor:,} != {output_rows:,}"
        )
    for array in (filtered_prefixes, filtered_tokens, filtered_scores):
        if isinstance(array, np.memmap):
            array.flush()
    return (
        filtered_prefixes,
        filtered_tokens,
        filtered_scores,
        temp_paths,
        remap_counts,
    )


def _sort_and_write_order(
    plan: PackPlan,
    order_plan: OrderPlan,
    output_path: Path,
    layout: SectionLayout,
    temp_dir: Path,
) -> tuple[float | None, dict[str, Any], dict[str, int]]:
    order = order_plan.order
    rows = order_plan.output_rows
    width = order - 1
    prefixes, tokens, scores, temp_paths, remap_counts = _filter_order_to_temp(
        plan, order_plan, temp_dir
    )
    try:
        quantization_low = _round_logprobs_in_place(
            scores,
            logprob_bytes=plan.logprob_bytes,
            chunk_rows=plan.chunk_rows,
        )
        mass_diagnostics = _row_mass_diagnostics(
            scores, chunk_rows=plan.chunk_rows
        )
        permutation = _numeric_permutation(prefixes)
        output_prefixes = _map_output(
            output_path, U32, layout.prefixes_offset, (rows, width)
        )
        output_tokens = _map_output(
            output_path, U32, layout.tokens_offset, (rows, plan.K)
        )
        output_score_dtype = F32 if plan.logprob_bytes == 4 else np.dtype("u1")
        output_scores = _map_output(
            output_path, output_score_dtype, layout.logprobs_offset, (rows, plan.K)
        )
        output_mask = (
            _map_output(
                output_path,
                np.dtype("u1"),
                layout.finite_mask_offset,
                (rows, (plan.K + 7) // 8),
            )
            if plan.logprob_bytes == 1
            else None
        )
        previous: np.ndarray | None = None
        try:
            for span in row_slices(rows, plan.chunk_rows):
                indices = permutation[span]
                prefix_chunk = np.asarray(prefixes[indices])
                token_chunk = np.asarray(tokens[indices])
                score_chunk = np.asarray(scores[indices])
                previous = _strict_numeric_order(prefix_chunk, previous)
                # The same permutation is deliberately applied to all three
                # arrays; tests also assert exact row association end to end.
                output_prefixes[span] = prefix_chunk
                output_tokens[span] = token_chunk
                if plan.logprob_bytes == 4:
                    output_scores[span] = score_chunk
                else:
                    codes, finite_mask = _encode_logprob_rows(
                        score_chunk, quantization_low=quantization_low
                    )
                    output_scores[span] = codes
                    assert output_mask is not None
                    output_mask[span] = finite_mask
            for array in (
                output_prefixes,
                output_tokens,
                output_scores,
                output_mask,
            ):
                if isinstance(array, np.memmap):
                    array.flush()
        finally:
            del output_prefixes, output_tokens, output_scores, output_mask
        del permutation
    finally:
        del prefixes, tokens, scores
        for path in temp_paths:
            path.unlink(missing_ok=True)
    return quantization_low, mass_diagnostics, remap_counts


def _manifest(
    plan: PackPlan,
    *,
    output_path: Path,
    output_sha256: str,
    quantization_lows: dict[int, float | None],
    mass_diagnostics: dict[int, dict[str, Any]],
    remap_counts: dict[int, dict[str, int]],
) -> dict[str, Any]:
    per_order: dict[str, Any] = {}
    for order in plan.orders:
        per_order[str(order.order)] = {
            "input_rows": order.input_rows,
            "output_rows": order.output_rows,
            "filtered_rows": order.input_rows - order.output_rows,
            "input_bytes": {
                **order.input_bytes,
                "total": sum(order.input_bytes.values()),
            },
            "output_bytes": {
                **order.output_bytes,
                "data_total": sum(order.output_bytes.values()),
                "section_header": SECTION_HEADER_SIZE,
            },
            "logprob_quantization_min": quantization_lows.get(order.order),
            "logprob_quantization_max": (
                0.0 if plan.logprob_bytes == 1 else None
            ),
            "retained_row_mass": mass_diagnostics[order.order],
            "source_sentinel_candidate_slots_remapped": remap_counts[order.order],
        }
    return {
        "format": "count-table",
        "format_version": _format_version(plan),
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source": {
            "input_dir": str(plan.input_dir),
            "manifest_path": str(plan.source_manifest_path),
            "manifest_sha256": plan.source_manifest_sha256,
            "manifest": plan.source_manifest,
        },
        "config": {
            "score_family": plan.source_manifest.get(
                "score_family", "stupid_backoff"
            ),
            "stupid_backoff_alpha": plan.source_manifest.get("alpha"),
            "max_order": plan.max_order,
            "emitted_orders": (
                list(plan.emit_orders)
                if plan.emit_orders is not None
                else list(range(1, plan.max_order + 1))
            ),
            "K": plan.K,
            "source_kmax": plan.source_kmax,
            "min_history_total": plan.min_history_total,
            "vocab_size": plan.vocab_size,
            "source_sentinel_mapping": {
                "source_token_ids": [
                    int(RAW_DOC_SEPARATOR_TOKEN),
                    int(RAW_EOF_PAD_TOKEN),
                ],
                "target_eos_token_id": plan.sentinel_eos_token_id,
                "scope": "candidate slots only; retained prefixes stay strict",
            },
            "ctot_dtype": plan.ctot_dtype,
            "logprob_bytes": plan.logprob_bytes,
            "logprob_encoding": (
                "linear_uint8_per_order_with_rowwise_little_endian_finite_mask"
                if plan.logprob_bytes == 1
                else "float32"
            ),
            "logprob_storage_note": (
                (
                    "COUNT table v2 stores one uint8 linear-quantization code per "
                    "score plus a row-wise packed finite mask "
                    "(ceil(K/8) bytes per row); lookup reconstructs only "
                    "selected rows as float32"
                )
                if plan.logprob_bytes == 1
                else "COUNT table v1 stores each score directly as float32"
            ),
            "chunk_rows": plan.chunk_rows,
            "source_prefix_order": plan.source_manifest.get(
                "prefix_order",
                "byte-lexicographic little-endian uint32 bytes",
            ),
            "output_prefix_order": "numeric lexicographic uint32 tuples",
            "stored_log_score": (
                "uint8 per-order linear code over [minimum finite score, 0], "
                "plus packed finite mask"
                if plan.logprob_bytes == 1
                else "float32 log10 source score"
            ),
            "row_mass_diagnostic": (
                "sum(10**stored_log10_score) over finite retained slots; "
                "the COUNT embedding consumes this unnormalized scale"
            ),
        },
        "orders": per_order,
        "output": {
            "path": str(output_path.resolve()),
            "bytes": plan.projected_output_bytes,
            "sha256": output_sha256,
        },
    }


def pack_scored_rows(
    plan: PackPlan,
    *,
    output_path: Path,
    manifest_path: Path,
    force: bool = False,
) -> dict[str, Any]:
    """Build the table atomically and return the written manifest."""
    output_path = Path(output_path)
    manifest_path = Path(manifest_path)
    if output_path.resolve() == manifest_path.resolve():
        raise ValueError("output_path and manifest_path must be different")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    for path, label in ((output_path, "output"), (manifest_path, "manifest")):
        if path.exists() and not force:
            raise FileExistsError(f"{label} already exists (pass --force to replace): {path}")

    layouts = _layouts(plan)
    partial_output = output_path.with_name(f".{output_path.name}.{os.getpid()}.partial")
    partial_manifest = manifest_path.with_name(
        f".{manifest_path.name}.{os.getpid()}.partial"
    )
    if partial_output.exists() or partial_manifest.exists():
        raise FileExistsError("stale packer partial file exists; inspect and remove it first")

    try:
        with partial_output.open("xb") as handle:
            handle.truncate(plan.projected_output_bytes)
        with tempfile.TemporaryDirectory(
            prefix=f".{output_path.name}.pack-", dir=output_path.parent
        ) as tmp:
            temp_dir = Path(tmp)
            quantization_lows: dict[int, float | None] = {}
            mass_diagnostics: dict[int, dict[str, Any]] = {}
            remap_counts: dict[int, dict[str, int]] = {}
            for order_plan, layout in zip(plan.orders, layouts):
                if order_plan.order != 1 and order_plan.output_rows == 0:
                    # An order outside the emitted subset, or one the
                    # threshold emptied: its section exists in the header
                    # with zero rows, and there is nothing to write.
                    quantization_lows[order_plan.order] = None
                    mass_diagnostics[order_plan.order] = {"retained_rows": 0}
                    remap_counts[order_plan.order] = {}
                    continue
                if order_plan.order == 1:
                    quantization_low, diagnostics, order_remap_counts = _copy_order1(
                        plan, partial_output, layout
                    )
                else:
                    (
                        quantization_low,
                        diagnostics,
                        order_remap_counts,
                    ) = _sort_and_write_order(
                        plan,
                        order_plan,
                        partial_output,
                        layout,
                        temp_dir,
                    )
                quantization_lows[order_plan.order] = quantization_low
                mass_diagnostics[order_plan.order] = diagnostics
                remap_counts[order_plan.order] = order_remap_counts
        _write_header(partial_output, plan, layouts, quantization_lows)
        with partial_output.open("rb") as handle:
            os.fsync(handle.fileno())
        output_sha256 = sha256_file(partial_output)
        manifest = _manifest(
            plan,
            output_path=output_path,
            output_sha256=output_sha256,
            quantization_lows=quantization_lows,
            mass_diagnostics=mass_diagnostics,
            remap_counts=remap_counts,
        )
        partial_manifest.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(partial_output, output_path)
        os.replace(partial_manifest, manifest_path)
        return manifest
    except BaseException:
        # These PID-qualified files are deliberately left in place. Dense
        # builds can take hours and write tens of gigabytes; preserving the
        # partial bytes makes a failure inspectable and potentially
        # recoverable instead of silently discarding the only evidence.
        raise


def _plan_json(plan: PackPlan) -> dict[str, Any]:
    return {
        "format": "count-table",
        "format_version": _format_version(plan),
        "source_manifest": str(plan.source_manifest_path),
        "source_manifest_sha256": plan.source_manifest_sha256,
        "source_kmax": plan.source_kmax,
        "config": {
            "max_order": plan.max_order,
            "K": plan.K,
            "min_history_total": plan.min_history_total,
            "vocab_size": plan.vocab_size,
            "sentinel_eos_token_id": plan.sentinel_eos_token_id,
            "ctot_dtype": plan.ctot_dtype,
            "logprob_bytes": plan.logprob_bytes,
            "chunk_rows": plan.chunk_rows,
        },
        "orders": {
            str(order.order): {
                "input_rows": order.input_rows,
                "output_rows": order.output_rows,
                "filtered_rows": order.input_rows - order.output_rows,
                "input_bytes": sum(order.input_bytes.values()),
                "output_data_bytes": sum(order.output_bytes.values()),
            }
            for order in plan.orders
        },
        "projected_output_bytes": plan.projected_output_bytes,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    output = parser.add_mutually_exclusive_group()
    output.add_argument(
        "--output-dir",
        type=Path,
        help="write forward_index_topk.bin and build_manifest.json here",
    )
    output.add_argument(
        "--output-file",
        type=Path,
        help="write COUNT table bytes to this file (manifest defaults beside it)",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        help="override build-manifest path (only meaningful with --output-file)",
    )
    parser.add_argument("--max-order", required=True, type=int)
    parser.add_argument("--K", required=True, type=int)
    parser.add_argument("--min-history-total", required=True, type=float)
    parser.add_argument("--vocab-size", required=True, type=int)
    parser.add_argument(
        "--sentinel-eos-token-id",
        type=int,
        default=DEFAULT_SENTINEL_EOS_TOKEN_ID,
        help=(
            "model EOS token replacing raw document-boundary candidates "
            "0xFFFFFFFF and 0xFFFFFFFE"
        ),
    )
    parser.add_argument(
        "--ctot-dtype",
        choices=("uint64", "float64"),
        default="uint64",
        help=(
            "history-total storage type; the original scorer writes uint64 "
            "(float64 must be selected explicitly for a differently declared source)"
        ),
    )
    parser.add_argument("--chunk-rows", type=int, default=131_072)
    parser.add_argument(
        "--orders",
        default="",
        help=(
            "comma-separated subset of orders to emit rows for, e.g. '7,1' "
            "or '7,5,3,1'. Empty means every order up to --max-order. Order "
            "1 is required. Non-emitted orders keep empty sections so the "
            "reader's fallback accrues its penalty per order of history "
            "shortening."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate/count and print the build plan without writing output",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace existing output and manifest paths",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    plan = plan_scored_rows(
        args.input_dir,
        max_order=args.max_order,
        K=args.K,
        min_history_total=args.min_history_total,
        vocab_size=args.vocab_size,
        sentinel_eos_token_id=args.sentinel_eos_token_id,
        ctot_dtype=args.ctot_dtype,
        logprob_bytes=1,
        chunk_rows=args.chunk_rows,
        emit_orders=(
            frozenset(int(tok) for tok in args.orders.split(","))
            if args.orders.strip()
            else None
        ),
    )
    if args.dry_run:
        print(json.dumps(_plan_json(plan), indent=2, sort_keys=True))
        return
    if args.output_dir is None and args.output_file is None:
        raise SystemExit("--output-dir or --output-file is required unless --dry-run is used")
    if args.output_dir is not None:
        if args.manifest_path is not None:
            raise SystemExit("--manifest-path cannot be combined with --output-dir")
        output_path = args.output_dir / "forward_index_topk.bin"
        manifest_path = args.output_dir / "build_manifest.json"
    else:
        output_path = args.output_file
        manifest_path = args.manifest_path or output_path.with_name(
            f"{output_path.stem}.build_manifest.json"
        )

    manifest = pack_scored_rows(
        plan,
        output_path=output_path,
        manifest_path=manifest_path,
        force=args.force,
    )
    print(
        f"wrote {output_path} ({manifest['output']['bytes']:,} bytes, "
        f"sha256={manifest['output']['sha256']})"
    )
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
