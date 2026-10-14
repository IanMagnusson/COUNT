#!/usr/bin/env python3
"""Stupid-backoff scoring: one dense top-Kmax row per retained history.

For every history (the ``n - 1`` tokens before a position) whose total
continuation count is at least ``tau``, this writes the ``kmax`` best next
tokens with a log10 stupid-backoff score each. A token that followed the
history in the corpus scores ``log10(count / total)``; every other candidate
comes from the backoff parent's row (the same history with its oldest token
dropped) with ``log10(alpha)`` added. Orders are built from 1 upward because
each order's rows are merged from the previous order's rows.

This module is the plain implementation: it loads an order's count parts into
memory and loops over histories. It is meant to be read, and it is fast enough
for small corpora. ``--implementation optimized`` selects
``thecount.tables.optimized``, which streams range-partitioned parts through a
worker pool and vectorizes the merge; the two write identical files.

Input: byte-lexicographically sorted count parts, ``order<k>.range<NNNNN>.bin``
with records of ``4k`` key bytes (little-endian ``uint32`` tokens) and 8 count
bytes, listed per order in the JSON file passed as ``--parts``.

Output per order n (all little-endian, row-aligned, in input order):
  order<n>.pref.bin  uint32[P, n-1]   history keys
  order<n>.ctot.bin  uint64[P]        history total counts
  order<n>.toks.bin  uint32[P, kmax]  rank-ordered slot tokens (0 pads)
  order<n>.lps.bin   float32[P, kmax] slot log10 scores (-inf pads)
plus order1.toks/lps.bin (the single unigram row) and scores_meta.json.
"""
import argparse
import json
import math
import os
import time

import numpy as np


def read_records(paths, n):
    """Concatenate an order's count parts into keys ``uint32[R, n]`` and
    counts ``uint64[R]``, in file order."""
    raw = b"".join(open(p, "rb").read() for p in paths)
    rec = 4 * n + 8
    if len(raw) % rec:
        raise ValueError(f"order{n} parts do not hold whole records")
    m = np.frombuffer(raw, np.uint8).reshape(-1, rec)
    keys = np.ascontiguousarray(m[:, :4 * n]).view(np.uint32).reshape(-1, n)
    counts = np.ascontiguousarray(m[:, 4 * n:]).view("<u8").reshape(-1)
    return keys, counts


def unigram_row(keys, counts, kmax):
    """The order-1 row: the ``kmax`` most frequent tokens, scored
    ``log10(count / N)``. Equal counts keep the later record first."""
    c = counts.astype(np.float64)
    total = float(c.sum())
    ranked = sorted(range(len(c)), key=lambda i: (-c[i], -i))[:kmax]
    toks = np.zeros((1, kmax), np.uint32)
    lps = np.full((1, kmax), -np.inf, np.float32)
    for slot, i in enumerate(ranked):
        toks[0, slot] = keys[i, 0]
        lps[0, slot] = np.float32(np.log10(c[i] / total))
    return toks, lps, total


def history_runs(keys, n):
    """Yield ``(start, end)`` record ranges sharing one history, in order."""
    start = 0
    total = len(keys)
    while start < total:
        end = start + 1
        while end < total and np.array_equal(keys[end, :n - 1], keys[start, :n - 1]):
            end += 1
        yield start, end
        start = end


def score_row(direct_tokens, direct_counts, ctot, parent_toks, parent_lps, alpha, kmax):
    """Merge one history's direct continuations with its parent's row.

    Candidates are ranked by score, highest first. Among equal scores the
    later-added candidate wins (a parent slot over a direct continuation, and
    later slots or records over earlier ones), which is the order the
    optimized implementation and the reference builder produce. A token that
    appears both directly and in the parent row keeps its direct score.
    """
    candidates = {}  # token -> (log10 score, insertion priority)
    for prio, (tok, count) in enumerate(zip(direct_tokens, direct_counts)):
        candidates[int(tok)] = (np.log10(np.float64(count) / np.float64(ctot)), prio)
    if parent_toks is not None:
        base = len(direct_tokens)
        for slot, (tok, lp) in enumerate(zip(parent_toks, parent_lps)):
            score = np.float64(lp) + math.log10(alpha)
            if np.isfinite(score) and int(tok) not in candidates:
                candidates[int(tok)] = (score, base + slot)
    ranked = sorted(candidates.items(), key=lambda kv: (-kv[1][0], -kv[1][1]))[:kmax]
    toks = np.zeros(kmax, np.uint32)
    lps = np.full(kmax, -np.inf, np.float32)
    for slot, (tok, (score, _)) in enumerate(ranked):
        toks[slot] = tok
        lps[slot] = np.float32(score)
    return toks, lps


def score_order(keys, counts, n, parent, tau, alpha, kmax):
    """Build every retained order-n row. ``parent`` is the order-1 row
    ``(toks, lps)`` when n is 2, else the order-(n-1) arrays
    ``(pref, toks, lps)``. Returns ``(pref, ctot, toks, lps, missing)``."""
    if n == 2:
        parent_index = None
    else:
        parent_pref, parent_toks, parent_lps = parent
        parent_index = {bytes(row.tobytes()): i for i, row in enumerate(parent_pref)}
    prefs, ctots, toks, lps = [], [], [], []
    missing = 0
    for start, end in history_runs(keys, n):
        ctot = int(counts[start:end].sum(dtype=np.uint64))
        if ctot < tau:
            continue
        history = keys[start, :n - 1]
        if n == 2:
            row_toks, row_lps = parent[0][0], parent[1][0]
        else:
            row = parent_index.get(bytes(history[1:].tobytes()))
            if row is None:
                missing += 1
                row_toks = row_lps = None
            else:
                row_toks, row_lps = parent_toks[row], parent_lps[row]
        t, l = score_row(keys[start:end, n - 1], counts[start:end], ctot, row_toks, row_lps, alpha, kmax)
        prefs.append(history)
        ctots.append(ctot)
        toks.append(t)
        lps.append(l)
    if not prefs:
        return (np.zeros((0, n - 1), np.uint32), np.zeros(0, np.uint64),
                np.zeros((0, kmax), np.uint32), np.full((0, kmax), -np.inf, np.float32), missing)
    return (np.stack(prefs).astype(np.uint32), np.asarray(ctots, np.uint64),
            np.stack(toks), np.stack(lps), missing)


def run_simple(args):
    os.makedirs(args.out_dir, exist_ok=True)
    parts = json.load(open(args.parts))
    meta_path = os.path.join(args.out_dir, "scores_meta.json")
    meta = {"tau": args.tau, "alpha": args.alpha, "kmax": args.kmax, "orders": {}}
    if os.path.exists(meta_path) and args.resume:
        meta = json.load(open(meta_path))
    parent = None
    for n in range(args.min_order, args.max_order + 1):
        paths = parts[str(n)]
        if not paths:
            raise RuntimeError(f"no parts for order{n}")
        t0 = time.time()
        if args.resume and str(n) in meta["orders"]:
            print(f"[sb-scoring] order{n}: already done, skip", flush=True)
            parent = load_parent(args.out_dir, n, args.kmax)
            continue
        keys, counts = read_records(paths, n)
        if n == 1:
            toks, lps, total = unigram_row(keys, counts, args.kmax)
            toks.tofile(os.path.join(args.out_dir, "order1.toks.bin"))
            lps.tofile(os.path.join(args.out_dir, "order1.lps.bin"))
            meta["orders"]["1"] = {"records": int(len(keys)), "rows": 1,
                                   "total_count": total,
                                   "seconds": round(time.time() - t0, 1)}
            parent = (toks, lps)
        else:
            if parent is None:
                parent = load_parent(args.out_dir, n, args.kmax)
            pref, ctot, toks, lps, missing = score_order(
                keys, counts, n, parent, args.tau, args.alpha, args.kmax)
            for name, arr in (("pref", pref), ("ctot", ctot), ("toks", toks), ("lps", lps)):
                arr.tofile(os.path.join(args.out_dir, f"order{n}.{name}.bin"))
            meta["orders"][str(n)] = {"records": int(len(keys)), "rows": int(len(pref)),
                                      "missing_parent": int(missing),
                                      "seconds": round(time.time() - t0, 1)}
            print(f"[sb-scoring] order{n}: {len(pref):,} rows from {len(keys):,} records, "
                  f"missing_parent={missing}, {time.time() - t0:.0f}s", flush=True)
            if missing:
                print(f"[sb-scoring] WARNING order{n}: {missing} histories had no "
                      f"surviving backoff parent (direct-only rows)", flush=True)
            parent = (pref, toks, lps)
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=1)
    print("[sb-scoring] DONE", flush=True)


def load_parent(out_dir, n, kmax):
    """Read the order-(n-1) rows already written under ``out_dir``."""
    if n <= 1:
        return None
    if n == 2:
        toks = np.fromfile(os.path.join(out_dir, "order1.toks.bin"), np.uint32).reshape(1, kmax)
        lps = np.fromfile(os.path.join(out_dir, "order1.lps.bin"), np.float32).reshape(1, kmax)
        return (toks, lps)
    p = n - 1
    pref = np.fromfile(os.path.join(out_dir, f"order{p}.pref.bin"), np.uint32).reshape(-1, p - 1)
    toks = np.fromfile(os.path.join(out_dir, f"order{p}.toks.bin"), np.uint32).reshape(-1, kmax)
    lps = np.fromfile(os.path.join(out_dir, f"order{p}.lps.bin"), np.float32).reshape(-1, kmax)
    return (pref, toks, lps)


def run(args):
    """Dispatch on ``args.implementation``; the default is the plain scorer."""
    if getattr(args, "implementation", "simple") == "optimized":
        from thecount.tables.optimized import stupid_backoff_scoring as optimized

        return optimized.run(args)
    return run_simple(args)


def add_arguments(ap):
    ap.add_argument("--parts", required=True, help="JSON file {order: [count part paths]}")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--min-order", type=int, default=1)
    ap.add_argument("--max-order", type=int, default=6)
    ap.add_argument("--tau", type=int, default=4096,
                    help="min history total count; deeper tiers filter these scored rows by ctot")
    ap.add_argument("--alpha", type=float, default=0.4)
    ap.add_argument("--kmax", type=int, default=64,
                    help="stored slots per row (row policies are prefix cuts of this)")
    ap.add_argument("--implementation", choices=("simple", "optimized"), default="simple",
                    help="plain in-memory scorer (default) or the streaming vectorized one")
    ap.add_argument("--workers", type=int, default=8, help="optimized only")
    ap.add_argument("--chunk-mb", type=float, default=64,
                    help="optimized only: sequential read chunk (MB)")
    ap.add_argument("--resume", action="store_true")
    return ap


def main():
    run(add_arguments(argparse.ArgumentParser()).parse_args())


if __name__ == "__main__":
    main()
