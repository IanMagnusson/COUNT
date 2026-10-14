#!/usr/bin/env python3
"""Count n-grams over a tokenized corpus into sorted, range-partitioned parts.

Every window of ``n`` consecutive tokens inside one shard is an n-gram
instance, except windows with the document separator strictly before their
last position (the separator may end an n-gram but never sits inside its
history). Instances are counted exactly, n-grams seen fewer than
``--min-count`` times are dropped, and the survivors are written sorted
byte-lexicographically on their little-endian ``uint32`` key bytes, split
into parts by the first key byte::

    order<k>.range<NNNNN>.bin    record = 4k key bytes + 8 count bytes

Concatenating an order's parts in name order yields one globally sorted
table, which is what ``thecount.tables.stupid_backoff_scoring`` reads.
``count_meta.json`` records the parameters and per-order totals.

This module is the plain implementation: it forms every window of a shard
with one array operation and counts with ``numpy.unique``, so an order's
windows must fit in memory. ``--implementation optimized`` selects
``thecount.counts.optimized``, which counts partitions in parallel worker
processes and sweeps the corpus in groups; the two write identical files.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

U32 = np.dtype("<u4")
U64 = np.dtype("<u8")


def shard_paths(corpus_dir: Path, pattern: str) -> list[Path]:
    paths = sorted(corpus_dir.glob(pattern))
    if not paths:
        raise SystemExit(f"no shards matching {pattern!r} under {corpus_dir}")
    return paths


def load_shards(paths: list[Path]) -> list[np.ndarray]:
    shards = []
    for p in paths:
        if p.stat().st_size % 4:
            raise SystemExit(f"{p} is not a whole number of uint32 values")
        shards.append(np.fromfile(p, dtype=U32))
    return shards


def shard_windows(shard: np.ndarray, order: int, separator: int) -> np.ndarray:
    """Every valid ``order``-token window of one shard, shape ``(R, order)``."""
    if len(shard) < order:
        return np.zeros((0, order), dtype=U32)
    windows = np.lib.stride_tricks.sliding_window_view(shard, order)
    if order >= 2:
        valid = ~(windows[:, :-1] == np.uint32(separator)).any(axis=1)
        windows = windows[valid]
    return np.ascontiguousarray(windows, dtype=U32)


def count_windows(windows: np.ndarray, min_count: int) -> tuple[np.ndarray, np.ndarray]:
    """Return byte-lexicographically sorted distinct keys ``uint8[m, 4n]`` and
    their counts ``uint64[m]``, dropping keys seen fewer than ``min_count`` times."""
    key_bytes = 4 * windows.shape[1]
    keys = windows.view(np.uint8).reshape(-1, key_bytes).view(f"S{key_bytes}").reshape(-1)
    distinct, counts = np.unique(keys, return_counts=True)
    if min_count > 1:
        keep = counts >= min_count
        distinct, counts = distinct[keep], counts[keep]
    return distinct.view(np.uint8).reshape(-1, key_bytes), counts.astype(U64)


def write_part(path: Path, keys: np.ndarray, counts: np.ndarray) -> None:
    key_bytes = keys.shape[1] if keys.size else 0
    records = np.empty(len(counts), dtype=np.dtype([("key", "u1", (key_bytes,)), ("count", "<u8")]))
    if len(counts):
        records["key"] = keys
        records["count"] = counts
    records.view(np.uint8).tofile(path)


def run_simple(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.partitions < 1 or 256 % args.partitions:
        raise SystemExit("--partitions must be a divisor of 256")
    shift = (256 // args.partitions).bit_length() - 1  # first byte -> part number
    t0 = time.time()
    paths = shard_paths(Path(args.corpus_dir), args.pattern)
    shards = load_shards(paths)
    total_tokens = int(sum(len(s) for s in shards))
    if max(int(s.max()) for s in shards if len(s)) >= args.vocab_size:
        raise SystemExit(f"a token id is >= vocab-size {args.vocab_size}")
    params = {
        "corpus_dir": str(Path(args.corpus_dir).resolve()),
        "pattern": args.pattern,
        "min_count": args.min_count,
        "partitions": args.partitions,
        "vocab_size": args.vocab_size,
        "doc_separator_token_id": args.doc_separator_token_id,
        "total_tokens": total_tokens,
    }
    meta: dict = {
        "corpus_dir": params["corpus_dir"],
        "shards": len(paths),
        "total_tokens": total_tokens,
        "max_order": args.max_order,
        "min_count": args.min_count,
        "partitions": args.partitions,
        "vocab_size": args.vocab_size,
        "doc_separator_token_id": args.doc_separator_token_id,
        "key_encoding": "little-endian uint32, byte-lex sorted",
        "windows_cross_shard_boundaries": False,
        "separator_rule": "invalid if separator strictly before final position",
        "orders": {},
    }
    meta_path = out_dir / "count_meta.json"
    done: dict = {}
    if args.resume:
        if meta_path.exists():
            done = json.loads(meta_path.read_text()).get("orders", {})
        for order_key, record in done.items():
            if record.get("params") != params:
                raise SystemExit(f"--resume refused: order {order_key} was counted with different parameters")
    elif list(out_dir.glob("order*.range*.bin")) or meta_path.exists():
        raise SystemExit(f"{out_dir} already contains a count; use a new --out-dir or --resume")

    for order in range(1, args.max_order + 1):
        if str(order) in done:
            meta["orders"][str(order)] = done[str(order)]
            continue
        t_order = time.time()
        for stale in out_dir.glob(f"order{order}.range*.bin"):
            stale.unlink()
        windows = np.concatenate(
            [shard_windows(s, order, args.doc_separator_token_id) for s in shards]
        ) if shards else np.zeros((0, order), dtype=U32)
        keys, counts = count_windows(windows, args.min_count)
        part_of_key = keys[:, 0] >> np.uint8(shift) if shift else keys[:, 0]
        for part in range(args.partitions):
            selected = part_of_key == part
            write_part(out_dir / f"order{order}.range{part:05d}.bin", keys[selected], counts[selected])
        meta["orders"][str(order)] = {
            "distinct": int(len(counts)),
            "instances": int(counts.sum()) if len(counts) else 0,
            "parts": args.partitions,
            "seconds": round(time.time() - t_order, 1),
            "params": params,
        }
        meta_path.write_text(json.dumps(meta, indent=2))
        print(f"[count] order{order}: {len(counts):,} distinct / "
              f"{meta['orders'][str(order)]['instances']:,} instances", flush=True)
    meta["total_seconds"] = round(time.time() - t0, 1)
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"[count] DONE -> {out_dir}", flush=True)


def run(args: argparse.Namespace) -> None:
    """Dispatch on ``args.implementation``; the default is the plain counter."""
    if getattr(args, "implementation", "simple") == "optimized":
        from thecount.counts.optimized import count_corpus as optimized

        return optimized.run(args)
    return run_simple(args)


def add_arguments(ap: argparse.ArgumentParser) -> argparse.ArgumentParser:
    ap.add_argument("--corpus-dir", required=True)
    ap.add_argument("--pattern", default="part-*.npy")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-order", type=int, default=6)
    ap.add_argument("--min-count", type=int, default=1,
                    help="n-gram count floor; 1 keeps every observed n-gram")
    ap.add_argument("--partitions", type=int, default=256,
                    help="range partitions on the first key byte; divides 256")
    ap.add_argument("--implementation", choices=("simple", "optimized"), default="simple",
                    help="plain in-memory counter (default) or the parallel chunked one")
    ap.add_argument("--workers", type=int, default=8, help="optimized only: parallel partitions")
    ap.add_argument("--groups", type=int, default=16,
                    help="optimized only: partition-array sweeps per order; divides --partitions")
    ap.add_argument("--resume", action="store_true",
                    help="skip orders already recorded in count_meta.json")
    ap.add_argument("--vocab-size", type=int, default=100278)
    ap.add_argument("--doc-separator-token-id", type=int, default=100257)
    return ap


def main() -> None:
    run(add_arguments(argparse.ArgumentParser(description=__doc__)).parse_args())


if __name__ == "__main__":
    main()
