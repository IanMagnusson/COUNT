#!/usr/bin/env python3
"""Count n-grams directly over a tokenized corpus into staged count parts.

The corpus is loaded into memory (4 bytes per token), every n-gram window up
to the maximum order is formed, and each order is range-partitioned, sorted,
and run-length counted. The count is exact.

Output is the exact input contract of ``thecount.tables.stupid_backoff_scoring``:

    order<k>.range<NNNNN>.bin   record = 4k key bytes + 8 count bytes

with keys stored as native little-endian ``uint32`` and records sorted
BYTE-LEXICOGRAPHICALLY on those key bytes (not in numeric tuple order). That is
the convention the stupid-backoff scorer expects; see ``recs_from_dict`` in
``tests/support.py``, which is the reference encoder this module reproduces.
Parts are range-partitioned on the first key byte -- the low byte of the first token -- so concatenating parts in name order
yields a globally byte-lex-sorted table, which is what the scorer's
forward-reading ownership protocol assumes.

Document boundaries: a window is INVALID if the separator appears strictly
before its final position. The separator may END an n-gram but may never sit
inside its history; `pack.py` additionally rejects sentinel prefixes.

For `[1, SEP, 2, 3]` at order 2 this keeps `(1, SEP)` and `(2, 3)`, and drops
`(SEP, 2)`.

Windows do not cross SHARD boundaries. Each shard is windowed independently,
which drops at most (max_order - 1) windows per shard, a negligible fraction
of any real corpus, and removes a class of carry bugs.

Usage:
    python -m thecount.counts.count_corpus \
        --corpus-dir <dir of headerless little-endian uint32 shards> \
        --out-dir <dir> --max-order 6 --min-count 1 --partitions 256
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
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


def load_corpus(paths: list[Path]) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate shards into one uint32 array plus the per-shard offsets."""
    sizes = []
    for p in paths:
        nbytes = p.stat().st_size
        if nbytes % 4:
            raise SystemExit(f"{p} is not a whole number of uint32 values")
        sizes.append(nbytes // 4)
    total = int(sum(sizes))
    tokens = np.empty(total, dtype=U32)
    bounds = np.zeros(len(paths) + 1, dtype=np.int64)
    cursor = 0
    for i, (p, n) in enumerate(zip(paths, sizes)):
        tokens[cursor:cursor + n] = np.fromfile(p, dtype=U32)
        cursor += n
        bounds[i + 1] = cursor
    assert cursor == total
    return tokens, bounds


def invalid_starts(bounds: np.ndarray, order: int) -> np.ndarray:
    """Window starts whose n-gram would run past the end of its own shard."""
    if order == 1:
        return np.zeros(0, dtype=np.int64)
    bad = [np.arange(max(int(bounds[i + 1]) - order + 1, int(bounds[i])),
                     int(bounds[i + 1]), dtype=np.int64)
           for i in range(len(bounds) - 1)]
    return np.concatenate(bad) if bad else np.zeros(0, dtype=np.int64)


def boundary_gap(tokens: np.ndarray, sep_id: int,
                 chunk: int = 1 << 27) -> np.ndarray:
    """Consecutive non-separator tokens ending at each position, capped at 255.

    ``gap[i] == 0`` when ``tokens[i]`` is the separator. A window starting at
    ``i`` with order ``n`` keeps the separator out of its history exactly when
    ``gap[i + n - 2] >= n - 1``. Capping at 255 is safe: the comparison is
    always against ``n - 1``, which is small.
    """
    total = int(tokens.size)
    gap = np.empty(total, dtype=np.uint8)
    carry = 0
    for start in range(0, total, chunk):
        stop = min(start + chunk, total)
        block = tokens[start:stop]
        idx = np.arange(stop - start, dtype=np.int64)
        seed = np.int64(-(carry + 1))
        last = np.maximum.accumulate(np.where(block == sep_id, idx, seed))
        d = idx - last
        np.minimum(d, 255, out=d)
        gap[start:stop] = d.astype(np.uint8)
        carry = int(d[-1])
    return gap


def count_partition(
    tokens: np.ndarray,
    starts: np.ndarray,
    order: int,
    min_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (raw key bytes (m, 4*order), counts (m,)) byte-lex sorted."""
    key_b = 4 * order
    if starts.size == 0:
        return np.zeros((0, key_b), np.uint8), np.zeros(0, U64)
    # gather the order-token window beginning at each start
    windows = tokens[starts[:, None] + np.arange(order, dtype=np.int64)]
    raw = np.ascontiguousarray(windows.astype(U32, copy=False)).view(np.uint8)
    raw = raw.reshape(-1, key_b)
    flat = raw.view(f"S{key_b}").reshape(-1)
    uniq, counts = np.unique(flat, return_counts=True)
    if min_count > 1:
        keep = counts >= min_count
        uniq, counts = uniq[keep], counts[keep]
    out = uniq.view(np.uint8).reshape(-1, key_b)
    return out, counts.astype(U64, copy=False)


def write_part(path: Path, keys: np.ndarray, counts: np.ndarray) -> int:
    key_b = keys.shape[1] if keys.size else 0
    rec = np.empty(len(counts), dtype=np.dtype([("g", "u1", (key_b,)),
                                                ("c", "<u8")]))
    if len(counts):
        rec["g"] = keys
        rec["c"] = counts
    rec.view(np.uint8).tofile(path)
    return len(counts)


# Worker state shared by fork. `tokens` is 4 bytes per corpus token and is
# read-only here, so fork's copy-on-write shares it rather than duplicating it
# per worker. Passing it through a Pool argument would pickle a copy per task.
_W: dict = {}


def _init_worker(tokens, order, min_count, out_dir, bad_set, gap):
    _W.update(tokens=tokens, order=order, min_count=min_count,
              out_dir=Path(out_dir), bad_set=bad_set, gap=gap)


def _count_one(task):
    """Count one range partition and write its part. Returns (rows, instances).

    The per-partition sort inside `count_partition` is single-threaded and is
    the dominant cost of the whole stage, so partitions -- which are wholly
    independent by construction -- are what gets parallelized.
    """
    p, starts = task
    order = _W["order"]
    if order >= 2:
        # a separator strictly before the final position invalidates the gram
        starts = starts[_W["gap"][starts + (order - 2)] >= np.uint8(order - 1)]
    if _W["bad_set"].size:
        starts = starts[~np.isin(starts, _W["bad_set"], assume_unique=True)]
    keys, counts = count_partition(_W["tokens"], starts, order,
                                   _W["min_count"])
    path = _W["out_dir"] / f"order{order}.range{p:05d}.bin"
    rows = write_part(path, keys, counts)
    return rows, (int(counts.sum()) if len(counts) else 0)


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = shard_paths(Path(args.corpus_dir), args.pattern)

    t0 = time.time()
    print(f"[count] loading {len(paths)} shards", flush=True)
    tokens, bounds = load_corpus(paths)
    total_tokens = int(tokens.size)
    print(f"[count] {total_tokens:,} tokens in {time.time() - t0:.1f}s",
          flush=True)
    if int(tokens.max()) >= args.vocab_size:
        raise SystemExit(
            f"token id {int(tokens.max())} >= vocab-size {args.vocab_size}")

    # first key byte == low byte of the first token: the range-partition key
    first_byte = (tokens & np.uint32(0xFF)).astype(np.uint8)
    # separator-gap array drives the boundary rule; one byte per token
    gap = boundary_gap(tokens, args.doc_separator_token_id)

    meta: dict = {
        "corpus_dir": str(Path(args.corpus_dir).resolve()),
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

    if args.partitions < 1 or 256 % args.partitions:
        raise SystemExit("--partitions must be a divisor of 256")
    # every divisor of 256 is a power of two, so the partition of a first byte
    # is a right shift -- no uint8 bound arithmetic, no last-partition overflow
    shift = (256 // args.partitions).bit_length() - 1
    # computed once: the shifted array is as large as the corpus, so
    # recomputing it per partition would dominate the run
    part_id = first_byte if shift == 0 else (first_byte >> np.uint8(shift))
    del first_byte

    # Sweep the partition array once per group of partitions rather than once
    # per partition, and split each group's positions in RAM: `groups` sweeps
    # per order instead of `partitions`. Fewer groups mean fewer sweeps but a
    # larger per-group position list.
    groups = min(args.groups, args.partitions)
    if args.partitions % groups:
        raise SystemExit("--groups must divide --partitions")
    per_group = args.partitions // groups
    gshift = per_group.bit_length() - 1
    group_id = part_id if per_group == 1 else (part_id >> np.uint8(gshift))

    meta_path = out_dir / "count_meta.json"
    # Parameters that change what a part CONTAINS. Persisted per completed order
    # so a resume cannot stamp new parameters over parts built with old ones.
    cur_params = {
        "corpus_dir": meta["corpus_dir"],
        "pattern": args.pattern,
        "min_count": args.min_count,
        "partitions": args.partitions,
        "vocab_size": args.vocab_size,
        "doc_separator_token_id": args.doc_separator_token_id,
        "total_tokens": total_tokens,
    }
    existing_parts = sorted(out_dir.glob("order*.range*.bin"))
    done: dict = {}
    if args.resume:
        if meta_path.exists():
            try:
                done = json.loads(meta_path.read_text()).get("orders", {})
            except json.JSONDecodeError:
                done = {}
        for order_key, record in sorted(done.items()):
            was = record.get("params")
            if was != cur_params:
                differing = sorted(
                    k for k in cur_params
                    if not isinstance(was, dict) or was.get(k) != cur_params[k])
                raise SystemExit(
                    f"--resume refused: order {order_key} was counted with "
                    f"different parameters ({', '.join(differing)}). Use a new "
                    f"--out-dir, or drop --resume to rebuild from scratch.")
    elif existing_parts or meta_path.exists():
        raise SystemExit(
            f"{out_dir} already contains a count ({len(existing_parts)} parts). "
            f"Refusing to write into it without --resume: stale parts from a "
            f"different --partitions would be merged by downstream "
            f"concatenation. Use a new --out-dir or remove it first.")

    for order in range(1, args.max_order + 1):
        if str(order) in done:
            meta["orders"][str(order)] = done[str(order)]
            print(f"[count] order{order}: already complete, skipping",
                  flush=True)
            continue
        t_order = time.time()
        # drop any pre-existing parts for THIS order so a shrunken
        # --partitions cannot leave stale high-numbered parts behind
        for stale in out_dir.glob(f"order{order}.range*.bin"):
            stale.unlink()
        bad_set = np.sort(invalid_starts(bounds, order))
        limit = total_tokens - order + 1
        rows_total = 0
        records_total = 0
        for g in range(groups):
            sel = group_id[:limit] == np.uint8(g)
            group_starts = np.flatnonzero(sel).astype(np.int64)
            del sel
            if per_group == 1:
                sub_bounds = np.array([0, group_starts.size])
            else:
                sub = part_id[group_starts] & np.uint8(per_group - 1)
                idx = np.argsort(sub, kind="stable")
                group_starts = group_starts[idx]
                sub_bounds = np.searchsorted(sub[idx],
                                             np.arange(per_group + 1))
                del sub, idx
            tasks = [(g * per_group + j,
                      group_starts[sub_bounds[j]:sub_bounds[j + 1]])
                     for j in range(per_group)]
            if args.workers > 1:
                # A process pool executor rather than multiprocessing.Pool:
                # when the kernel OOM-kills a worker mid-sort, Pool waits
                # forever for the lost result and the job looks alive while
                # doing nothing, whereas the executor raises BrokenProcessPool
                # and the stage fails where a launcher can see it.
                from concurrent.futures import ProcessPoolExecutor
                with ProcessPoolExecutor(
                    max_workers=min(args.workers, len(tasks)),
                    mp_context=mp.get_context("fork"),
                    initializer=_init_worker,
                    initargs=(tokens, order, args.min_count, str(out_dir),
                              bad_set, gap),
                ) as pool:
                    for rows, insts in pool.map(_count_one, tasks, chunksize=1):
                        rows_total += rows
                        records_total += insts
            else:
                _init_worker(tokens, order, args.min_count, str(out_dir),
                             bad_set, gap)
                for task in tasks:
                    rows, insts = _count_one(task)
                    rows_total += rows
                    records_total += insts
            del group_starts, tasks
        meta["orders"][str(order)] = {
            "distinct": rows_total,
            "instances": records_total,
            "parts": args.partitions,
            "seconds": round(time.time() - t_order, 1),
            "params": cur_params,
        }
        # written per order so a lost job resumes at order granularity
        meta_path.write_text(json.dumps(meta, indent=2))
        print(f"[count] order{order}: {rows_total:,} distinct / "
              f"{records_total:,} instances in "
              f"{time.time() - t_order:.1f}s", flush=True)

    meta["total_seconds"] = round(time.time() - t0, 1)
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"[count] DONE in {meta['total_seconds']}s -> {out_dir}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus-dir", required=True)
    ap.add_argument("--pattern", default="part-*.npy")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-order", type=int, default=6)
    ap.add_argument("--min-count", type=int, default=1,
                    help="n-gram count floor; 1 keeps every observed n-gram")
    ap.add_argument("--partitions", type=int, default=256,
                    help="range partitions on the first key byte; divides 256")
    ap.add_argument("--workers", type=int, default=8,
                    help="parallel range partitions; the per-partition sort is "
                         "single-threaded and dominates this stage")
    ap.add_argument("--groups", type=int, default=16,
                    help="partition-array sweeps per order; divides --partitions")
    ap.add_argument("--resume", action="store_true",
                    help="skip orders already recorded in count_meta.json")
    ap.add_argument("--vocab-size", type=int, default=100278)
    ap.add_argument("--doc-separator-token-id", type=int, default=100257)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
