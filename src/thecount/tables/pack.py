#!/usr/bin/env python3
"""Pack scored rows into the COUNT table the trainer reads.

Input is the output of ``thecount.tables.stupid_backoff_scoring``: for order 1
a single row of the ``kmax`` best unigrams, and for each order ``n >= 2`` the
history keys (``uint32[P, n-1]``), history totals, and dense rows of ``kmax``
tokens with log10 scores. The packer keeps the histories whose total is at
least ``--min-history-total``, keeps the first ``K`` candidates of each row
(rows are already ranked), maps raw document-boundary candidates
(``0xFFFFFFFF`` and ``0xFFFFFFFE``) to the model's EOS token, sorts each
order's histories into numeric token order for the reader's binary search,
and stores each score as one byte: a linear code between the order's lowest
finite score and zero, with a packed bit mask marking finite slots.

File layout (all little-endian): a 64-byte header (``COUN`` tag, format
version 2, ``K``, section count, vocabulary size), one 64-byte section
header per order (order, rows, offsets of the prefix, token, code, and mask
arrays, the score calibration), then the sections' arrays in order.
``build_manifest.json`` records the source metadata and its digest, the
configuration, per-order row and byte counts, a retained-mass diagnostic,
and the table's SHA-256.

This module is the plain implementation: it holds one order in memory at a
time. ``--implementation optimized`` selects ``thecount.tables.optimized``,
which memory-maps and chunks everything; the two write identical tables.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import struct
from pathlib import Path
from typing import Any

import numpy as np

from thecount.tables.table_format import (
    DEFAULT_SENTINEL_EOS_TOKEN_ID,
    F32,
    HEADER_SIZE,
    MAGIC,
    RAW_DOC_SEPARATOR_TOKEN,
    RAW_EOF_PAD_TOKEN,
    SCORE_ENCODING_LINEAR_UINT8,
    SECTION_HEADER_SIZE,
    U32,
    U64,
    UINT8_VERSION,
    sha256_file,
)

MAX_SOURCE_K = 512
MASS_QUANTILE_SAMPLE_ROWS = 131_072
SENTINELS = (RAW_DOC_SEPARATOR_TOKEN, RAW_EOF_PAD_TOKEN)


class ScoredRowsFormatError(ValueError):
    """The scored rows are incomplete or internally inconsistent."""


def read_metadata(input_dir: Path, *, max_order: int, K: int, min_history_total: float,
                  vocab_size: int, eos_token_id: int, ctot_dtype: str) -> tuple[dict, bytes, int]:
    """Load ``scores_meta.json`` and check the request against it."""
    raw = (input_dir / "scores_meta.json").read_bytes()
    meta = json.loads(raw)
    kmax = meta.get("kmax")
    if not isinstance(kmax, int) or not 1 <= kmax <= MAX_SOURCE_K:
        raise ScoredRowsFormatError(f"scores_meta.json kmax must be an integer in [1, {MAX_SOURCE_K}]")
    if not 1 <= K <= kmax:
        raise ScoredRowsFormatError(f"K must be in [1, source kmax={kmax}], got {K}")
    if vocab_size <= K:
        raise ScoredRowsFormatError("vocab_size must be greater than K")
    if not 0 <= eos_token_id < vocab_size:
        raise ScoredRowsFormatError("sentinel_eos_token_id must be a model token")
    if not math.isfinite(min_history_total) or min_history_total < 0:
        raise ScoredRowsFormatError("min_history_total must be finite and non-negative")
    if meta.get("tau") is not None and min_history_total < float(meta["tau"]):
        raise ScoredRowsFormatError("requested history-total threshold is below the scored rows' threshold")
    declared = meta.get("ctot_dtype")
    if declared is not None and str(declared).lower() != ctot_dtype:
        raise ScoredRowsFormatError(f"scores_meta.json declares ctot dtype {declared}, but --ctot-dtype is {ctot_dtype}")
    if meta.get("key_transform") is not None:
        raise ScoredRowsFormatError("this packer only handles rows keyed by raw token ids")
    for order in range(1, max_order + 1):
        entry = meta.get("orders", {}).get(str(order))
        if not isinstance(entry, dict) or not isinstance(entry.get("rows"), int):
            raise ScoredRowsFormatError(f"scores_meta.json is missing metadata for order {order}")
        if order >= 2 and entry.get("missing_parent"):
            raise ScoredRowsFormatError(f"order {order} reports histories without their backoff parent")
    if meta["orders"]["1"]["rows"] != 1:
        raise ScoredRowsFormatError("order 1 must contain exactly one empty-prefix row")
    return meta, raw, kmax


def load_order(input_dir: Path, order: int, rows: int, kmax: int, ctot_dtype: str):
    """Read one order's arrays, checking every file has exactly its expected size."""
    def read(name, dtype, shape):
        path = input_dir / name
        expected = math.prod(shape) * dtype.itemsize
        if not path.exists() or path.stat().st_size != expected:
            raise ScoredRowsFormatError(f"{name} must hold exactly {expected:,} bytes")
        return np.fromfile(path, dtype=dtype).reshape(shape)

    if order == 1:
        return (None, None,
                read("order1.toks.bin", U32, (1, kmax)),
                read("order1.lps.bin", F32, (1, kmax)))
    totals_dtype = U64 if ctot_dtype == "uint64" else np.dtype("<f8")
    return (read(f"order{order}.pref.bin", U32, (rows, order - 1)),
            read(f"order{order}.ctot.bin", totals_dtype, (rows,)),
            read(f"order{order}.toks.bin", U32, (rows, kmax)),
            read(f"order{order}.lps.bin", F32, (rows, kmax)))


def remap_sentinels(tokens: np.ndarray, eos_token_id: int) -> dict[str, int]:
    """Replace raw document-boundary candidates with EOS; return how many."""
    counts = {}
    for sentinel in SENTINELS:
        mask = tokens == sentinel
        counts[str(int(sentinel))] = int(mask.sum())
        tokens[mask] = np.uint32(eos_token_id)
    counts["total"] = sum(counts.values())
    return counts


def check_rows(tokens: np.ndarray, scores: np.ndarray, order: int, vocab_size: int) -> None:
    if np.any(tokens >= vocab_size):
        raise ScoredRowsFormatError(f"order {order} contains a token id outside vocab_size={vocab_size}")
    if np.isnan(scores).any() or np.isposinf(scores).any():
        raise ScoredRowsFormatError(f"order {order} log scores contain NaN or +infinity")
    finite = np.isfinite(scores)
    if finite.any() and np.any(scores[finite] > np.float32(1e-6)):
        raise ScoredRowsFormatError(f"order {order} contains a positive log10 score")


def quantize(scores: np.ndarray, chunk_rows: int) -> tuple[float | None, np.ndarray]:
    """Round finite scores onto the order's 256-level grid between its lowest
    finite score and zero. Returns the grid's low end and the rounded rows."""
    finite = np.isfinite(scores)
    if not finite.any():
        return None, scores
    low = float(scores[finite].min())
    if low == 0.0:
        return low, scores
    rounded = scores.copy()
    scale = -low
    for start in range(0, len(scores), chunk_rows):
        chunk = rounded[start:start + chunk_rows]
        mask = np.isfinite(chunk)
        values = chunk[mask].astype(np.float64)
        q = np.rint((values - low) / scale * 255.0)
        chunk[mask] = (q / 255.0 * scale + low).astype(F32)
    return low, rounded


def encode(scores: np.ndarray, low: float | None) -> tuple[np.ndarray, np.ndarray]:
    """Rounded float32 rows to one byte per score plus a packed finite mask."""
    finite = np.isfinite(scores)
    codes = np.zeros(scores.shape, dtype=np.uint8)
    if finite.any() and low < 0.0:
        values = scores[finite].astype(np.float64)
        codes[finite] = np.clip(np.rint((values - low) / (-low) * 255.0), 0.0, 255.0).astype(np.uint8)
    return codes, np.packbits(finite, axis=1, bitorder="little")


def mass_diagnostics(scores: np.ndarray, chunk_rows: int) -> dict[str, Any]:
    """Retained mass ``sum(10**score)`` per row: mean, min, max, quantiles."""
    rows = len(scores)
    if rows == 0:
        return {"row_count": 0, "mass_definition": "sum(10**stored_log10_score) over finite K slots",
                "mean": None, "min": None, "max": None, "quantiles": {},
                "quantile_method": "no rows", "quantile_sample_rows": 0}
    total, minimum, maximum = 0.0, math.inf, -math.inf
    for start in range(0, rows, chunk_rows):
        masses = np.power(10.0, scores[start:start + chunk_rows].astype(np.float64)).sum(axis=1, dtype=np.float64)
        total += float(masses.sum(dtype=np.float64))
        minimum, maximum = min(minimum, float(masses.min())), max(maximum, float(masses.max()))
    sample = min(rows, MASS_QUANTILE_SAMPLE_ROWS)
    if sample == rows:
        sampled, method = scores.astype(np.float64), "exact all rows"
    else:
        sampled = scores[np.linspace(0, rows - 1, sample, dtype=np.int64)].astype(np.float64)
        method = f"deterministic evenly spaced sample capped at {MASS_QUANTILE_SAMPLE_ROWS} rows"
    probabilities = (0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0)
    values = np.quantile(np.power(10.0, sampled).sum(axis=1, dtype=np.float64), probabilities)
    return {"row_count": rows, "mass_definition": "sum(10**stored_log10_score) over finite K slots",
            "mean": total / rows, "min": minimum, "max": maximum,
            "quantiles": {f"p{int(p * 100):02d}": float(v) for p, v in zip(probabilities, values)},
            "quantile_method": method, "quantile_sample_rows": sample}


def numeric_order(prefixes: np.ndarray) -> np.ndarray:
    """Stable permutation sorting histories as numeric token tuples."""
    if len(prefixes) == 0:
        return np.zeros(0, dtype=np.int64)
    names = [f"f{c}" for c in range(prefixes.shape[1])]
    structured = prefixes.view(np.dtype([(n, U32) for n in names])).reshape(-1)
    return np.argsort(structured, order=names, kind="stable")


def pack(input_dir: Path, output_path: Path, manifest_path: Path, *, max_order: int, K: int,
         min_history_total: float, vocab_size: int,
         sentinel_eos_token_id: int = DEFAULT_SENTINEL_EOS_TOKEN_ID, ctot_dtype: str = "uint64",
         chunk_rows: int = 131_072, emit_orders: frozenset[int] | None = None,
         force: bool = False) -> dict[str, Any]:
    """Build the table and its manifest; return the manifest."""
    input_dir = Path(input_dir).resolve()
    if emit_orders is not None and 1 not in emit_orders:
        raise ValueError("emit_orders must include order 1, the backoff floor")
    for path in (output_path, manifest_path):
        if path.exists() and not force:
            raise FileExistsError(f"already exists (pass --force to replace): {path}")
    meta, meta_bytes, kmax = read_metadata(
        input_dir, max_order=max_order, K=K, min_history_total=min_history_total,
        vocab_size=vocab_size, eos_token_id=sentinel_eos_token_id, ctot_dtype=ctot_dtype)

    sections = []  # per order: everything the header, arrays, and manifest need
    for order in range(1, max_order + 1):
        input_rows = meta["orders"][str(order)]["rows"]
        prefixes, totals, tokens, scores = load_order(input_dir, order, input_rows, kmax, ctot_dtype)
        input_bytes = ({"tokens": tokens.nbytes, "logprobs": scores.nbytes} if order == 1 else
                       {"prefixes": prefixes.nbytes, "history_totals": totals.nbytes,
                        "tokens": tokens.nbytes, "logprobs": scores.nbytes})
        emitted = emit_orders is None or order in emit_orders
        if order == 1:
            keep = np.ones(1, dtype=bool)
        elif emitted:
            if ctot_dtype == "float64" and (not np.isfinite(totals).all() or np.any(totals < 0)):
                raise ScoredRowsFormatError(f"order {order} ctot contains an invalid total")
            keep = totals >= min_history_total
        else:
            keep = np.zeros(input_rows, dtype=bool)
        tokens = tokens[keep, :K].copy()
        scores = scores[keep, :K].copy()
        prefixes = None if order == 1 else prefixes[keep].copy()
        rows = len(tokens)
        section = {"order": order, "input_rows": input_rows, "rows": rows, "input_bytes": input_bytes,
                   "output_bytes": {"prefixes": rows * (order - 1) * 4, "tokens": rows * K * 4,
                                    "logprobs": rows * K, "finite_mask": rows * ((K + 7) // 8)}}
        if order != 1 and rows == 0:
            section.update(low=None, mass={"retained_rows": 0}, remap={},
                           prefixes=np.zeros((0, order - 1), U32), tokens=tokens,
                           codes=np.zeros((0, K), np.uint8), mask=np.zeros((0, (K + 7) // 8), np.uint8))
            sections.append(section)
            continue
        if order != 1 and np.any(prefixes >= vocab_size):
            raise ScoredRowsFormatError(f"order {order} contains a prefix token id outside vocab_size={vocab_size}")
        remap = remap_sentinels(tokens, sentinel_eos_token_id)
        check_rows(tokens, scores, order, vocab_size)
        low, scores = quantize(scores, chunk_rows)
        mass = mass_diagnostics(scores, chunk_rows)
        if order != 1:
            permutation = numeric_order(prefixes)
            prefixes, tokens, scores = prefixes[permutation], tokens[permutation], scores[permutation]
            if len(prefixes) > 1 and np.any(np.all(prefixes[1:] == prefixes[:-1], axis=1)):
                raise ScoredRowsFormatError(f"duplicate explicit prefix after filtering in order {order}")
        codes, mask = encode(scores, low)
        section.update(low=low, mass=mass, remap=remap,
                       prefixes=prefixes if order != 1 else np.zeros((1, 0), U32),
                       tokens=tokens, codes=codes, mask=mask)
        sections.append(section)

    # Lay the sections out back to back after the headers and write the file.
    cursor = HEADER_SIZE + len(sections) * SECTION_HEADER_SIZE
    for section in sections:
        section["offsets"] = {}
        for name in ("prefixes", "tokens", "logprobs", "finite_mask"):
            section["offsets"][name] = cursor
            cursor += section["output_bytes"][name]
    total_bytes = cursor
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        header = struct.pack("<4sIIII", MAGIC, UINT8_VERSION, K, len(sections), vocab_size)
        handle.write(header + b"\x00" * (HEADER_SIZE - len(header)))
        for section in sections:
            block = bytearray(SECTION_HEADER_SIZE)
            offsets = section["offsets"]
            struct.pack_into("<IIQQQQ", block, 0, section["order"], 0, section["rows"],
                             offsets["prefixes"], offsets["tokens"], offsets["logprobs"])
            low = section["low"] if section["low"] is not None else 0.0
            struct.pack_into("<IffQ", block, 40, SCORE_ENCODING_LINEAR_UINT8, low, 0.0, offsets["finite_mask"])
            handle.write(block)
        for section in sections:
            for array in (section["prefixes"], section["tokens"], section["codes"], section["mask"]):
                handle.write(np.ascontiguousarray(array).tobytes())
    if output_path.stat().st_size != total_bytes:
        raise AssertionError("written table size does not match its layout")

    manifest = {
        "format": "count-table",
        "format_version": UINT8_VERSION,
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source": {"input_dir": str(input_dir), "manifest_path": str(input_dir / "scores_meta.json"),
                   "manifest_sha256": hashlib.sha256(meta_bytes).hexdigest(), "manifest": meta},
        "config": {
            "score_family": meta.get("score_family", "stupid_backoff"),
            "stupid_backoff_alpha": meta.get("alpha"),
            "max_order": max_order,
            "emitted_orders": sorted(emit_orders) if emit_orders is not None else list(range(1, max_order + 1)),
            "K": K, "source_kmax": kmax, "min_history_total": float(min_history_total),
            "vocab_size": vocab_size,
            "source_sentinel_mapping": {"source_token_ids": [int(s) for s in SENTINELS],
                                        "target_eos_token_id": sentinel_eos_token_id,
                                        "scope": "candidate slots only; retained prefixes stay strict"},
            "ctot_dtype": ctot_dtype, "logprob_bytes": 1,
            "logprob_encoding": "linear_uint8_per_order_with_rowwise_little_endian_finite_mask",
            "logprob_storage_note": ("COUNT table v2 stores one uint8 linear-quantization code per score plus a "
                                     "row-wise packed finite mask (ceil(K/8) bytes per row); lookup reconstructs "
                                     "only selected rows as float32"),
            "chunk_rows": chunk_rows,
            "source_prefix_order": meta.get("prefix_order", "byte-lexicographic little-endian uint32 bytes"),
            "output_prefix_order": "numeric lexicographic uint32 tuples",
            "stored_log_score": "uint8 per-order linear code over [minimum finite score, 0], plus packed finite mask",
            "row_mass_diagnostic": ("sum(10**stored_log10_score) over finite retained slots; the COUNT embedding "
                                    "consumes this unnormalized scale"),
        },
        "orders": {
            str(s["order"]): {
                "input_rows": s["input_rows"], "output_rows": s["rows"],
                "filtered_rows": s["input_rows"] - s["rows"],
                "input_bytes": {**s["input_bytes"], "total": sum(s["input_bytes"].values())},
                "output_bytes": {**s["output_bytes"], "data_total": sum(s["output_bytes"].values()),
                                 "section_header": SECTION_HEADER_SIZE},
                "logprob_quantization_min": s["low"], "logprob_quantization_max": 0.0,
                "retained_row_mass": s["mass"], "source_sentinel_candidate_slots_remapped": s["remap"],
            } for s in sections
        },
        "output": {"path": str(output_path.resolve()), "bytes": total_bytes, "sha256": sha256_file(output_path)},
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def add_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--input-dir", required=True, type=Path)
    output = parser.add_mutually_exclusive_group()
    output.add_argument("--output-dir", type=Path, help="write forward_index_topk.bin and build_manifest.json here")
    output.add_argument("--output-file", type=Path, help="write the table to this file (manifest beside it)")
    parser.add_argument("--manifest-path", type=Path, help="override the manifest path (with --output-file)")
    parser.add_argument("--max-order", required=True, type=int)
    parser.add_argument("--K", required=True, type=int)
    parser.add_argument("--min-history-total", required=True, type=float)
    parser.add_argument("--vocab-size", required=True, type=int)
    parser.add_argument("--sentinel-eos-token-id", type=int, default=DEFAULT_SENTINEL_EOS_TOKEN_ID,
                        help="model EOS token replacing raw document-boundary candidates")
    parser.add_argument("--ctot-dtype", choices=("uint64", "float64"), default="uint64",
                        help="history-total storage type; the scorer writes uint64")
    parser.add_argument("--chunk-rows", type=int, default=131_072,
                        help="rows per processing chunk (also sets the manifest's mass-diagnostic accumulation)")
    parser.add_argument("--orders", default="",
                        help="comma-separated subset of orders to emit rows for; empty means all. "
                             "Order 1 is required; other orders keep empty sections.")
    parser.add_argument("--implementation", choices=("simple", "optimized"), default="simple",
                        help="plain in-memory packer (default) or the memory-mapped chunked one")
    parser.add_argument("--dry-run", action="store_true", help="optimized only: print the build plan")
    parser.add_argument("--force", action="store_true", help="replace existing output and manifest")
    return parser


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    if args.output_dir is None and args.output_file is None:
        raise SystemExit("--output-dir or --output-file is required")
    if args.output_dir is not None:
        if args.manifest_path is not None:
            raise SystemExit("--manifest-path cannot be combined with --output-dir")
        return args.output_dir / "forward_index_topk.bin", args.output_dir / "build_manifest.json"
    return args.output_file, args.manifest_path or args.output_file.with_name(
        f"{args.output_file.stem}.build_manifest.json")


def run(args: argparse.Namespace) -> dict[str, Any] | None:
    """Dispatch on ``args.implementation``; the default is the plain packer."""
    emit_orders = frozenset(int(t) for t in args.orders.split(",")) if args.orders.strip() else None
    if getattr(args, "implementation", "simple") == "optimized":
        from thecount.tables.optimized import pack as optimized

        plan = optimized.plan_scored_rows(
            args.input_dir, max_order=args.max_order, K=args.K, min_history_total=args.min_history_total,
            vocab_size=args.vocab_size, sentinel_eos_token_id=args.sentinel_eos_token_id,
            ctot_dtype=args.ctot_dtype, logprob_bytes=1, chunk_rows=args.chunk_rows, emit_orders=emit_orders)
        if args.dry_run:
            print(json.dumps(optimized._plan_json(plan), indent=2, sort_keys=True))
            return None
        output_path, manifest_path = resolve_paths(args)
        return optimized.pack_scored_rows(plan, output_path=output_path, manifest_path=manifest_path,
                                          force=args.force)
    if args.dry_run:
        raise SystemExit("--dry-run is only available with --implementation optimized")
    output_path, manifest_path = resolve_paths(args)
    return pack(args.input_dir, output_path, manifest_path, max_order=args.max_order, K=args.K,
                min_history_total=args.min_history_total, vocab_size=args.vocab_size,
                sentinel_eos_token_id=args.sentinel_eos_token_id, ctot_dtype=args.ctot_dtype,
                chunk_rows=args.chunk_rows, emit_orders=emit_orders, force=args.force)


def main() -> None:
    manifest = run(add_arguments(argparse.ArgumentParser(description=__doc__)).parse_args())
    if manifest is not None:
        print(f"wrote {manifest['output']['path']} ({manifest['output']['bytes']:,} bytes, "
              f"sha256={manifest['output']['sha256']})")


if __name__ == "__main__":
    main()
