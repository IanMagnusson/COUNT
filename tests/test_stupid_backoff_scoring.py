#!/usr/bin/env python3
"""Stupid-backoff scoring parity tests.

1. row parity: the optimized kernel `backoff_rows.build_rows` (which wraps
   `rows_for_order`) reproduces the explicit reference
   `tests.support.build_sb_topk` bit for bit on randomized count tables.
2. streaming parity: `stupid_backoff_scoring.run`, with either implementation,
   over range-partitioned local parts reproduces `build_rows` on the whole table.
3. implementation parity: the plain and optimized scorers write byte-identical
   files for the same inputs.
"""
import argparse
import json
import os
import shutil
import tempfile

import numpy as np

from thecount.tables import stupid_backoff_scoring as sbs
from thecount.tables.optimized import backoff_rows as hv
from tests.support import build_sb_topk, prune_histories
from tests.support import recs_from_dict

V = 50          # tiny vocab
ALPHA = 0.4
KMAX = 8


def synth_tables(seed=0, n_orders=3, n_tokens=4000):
    """Ngram count tables from an actual random token stream (sliding
    windows), so suffix-count dominance holds and history pruning is
    suffix-closed like real corpus tables."""
    rng = np.random.default_rng(seed)
    # zipf-ish token stream so some histories are frequent
    probs = 1.0 / np.arange(1, V + 1)
    probs /= probs.sum()
    stream = rng.choice(V, size=n_tokens, p=probs)
    tables = {}
    for n in range(1, n_orders + 1):
        d = {}
        for i in range(len(stream) - n + 1):
            k = tuple(int(x) for x in stream[i:i + n])
            d[k] = d.get(k, 0) + 1
        tables[n] = d
    return tables


def slices_from_tables(tables):
    return {n: recs_from_dict(tables[n], n) for n in tables}


# ---------------------------------------------------------------- row parity
def test_row_parity():
    tables = synth_tables(seed=1)
    slices = slices_from_tables(tables)
    orders = sorted(slices)
    for tau in (1, 4, 16):
        pruned = {k: prune_histories(slices[k], k, tau) for k in orders}
        want = build_sb_topk(pruned, orders, V, alpha=ALPHA, Kmax=KMAX)
        got = hv.build_rows(slices, orders, alpha=ALPHA, Kmax=KMAX,
                            tau=tau)
        for n in orders:
            if n == 1:
                wt, wl = want[1][1], want[1][2]
                gt, gl = got[1]["toks"], got[1]["lps"]
            else:
                assert np.array_equal(want[n][0], got[n]["pref"]), \
                    f"tau={tau} order{n} pref mismatch"
                wt, wl = want[n][1], want[n][2]
                gt, gl = got[n]["toks"], got[n]["lps"]
            fin = np.isfinite(wl)
            assert np.array_equal(fin, np.isfinite(gl)), \
                f"tau={tau} order{n} finite-mask mismatch"
            assert np.array_equal(wt[fin], gt[fin]), \
                f"tau={tau} order{n} token mismatch"
            assert np.allclose(wl[fin], gl[fin], atol=0), \
                f"tau={tau} order{n} value mismatch"
    print("PASS row_parity (vs build_sb_topk, tau in {1,4,16})")


# --------------------------------------------------------- streaming parity
def _run_scored_rows_case(kmax, workers, seed, implementation="optimized"):
    tables = synth_tables(seed=seed, n_orders=4, n_tokens=6000)
    slices = slices_from_tables(tables)
    orders = sorted(slices)
    tau = 4
    want = hv.build_rows(slices, orders, alpha=ALPHA, Kmax=kmax, tau=tau)

    tmp = tempfile.mkdtemp(prefix="sb_scoring_test_")
    try:
        parts_map = {}
        for n in orders:
            rec = 4 * n + 8
            m = np.ascontiguousarray(slices[n], np.uint8).reshape(-1, rec)
            # 3 parts split at 1/3 and 2/3 of the RECORDS — mid-group with
            # high probability, which is the case under test
            cuts = [0, len(m) // 3, 2 * len(m) // 3, len(m)]
            paths = []
            for pi in range(3):
                p = os.path.join(tmp, f"order{n}.range{pi:05d}.bin")
                m[cuts[pi]:cuts[pi + 1]].tofile(p)
                paths.append(p)
            parts_map[str(n)] = paths
        lp = os.path.join(tmp, "parts.json")
        with open(lp, "w") as f:
            json.dump(parts_map, f)
        out = os.path.join(tmp, "scores")
        args = argparse.Namespace(
            parts=lp, out_dir=out,
            min_order=1, max_order=max(orders), tau=tau, alpha=ALPHA,
            kmax=kmax, workers=workers, chunk_mb=0.002, resume=False,
            implementation=implementation)
        # ~2KB chunks: force the carry/held-back-group paths many times
        sbs.run(args)
        for n in orders:
            if n == 1:
                gt = np.fromfile(os.path.join(out, "order1.toks.bin"),
                                 np.uint32).reshape(1, kmax)
                gl = np.fromfile(os.path.join(out, "order1.lps.bin"),
                                 np.float32).reshape(1, kmax)
                fin = np.isfinite(want[1]["lps"])
                assert np.array_equal(np.isfinite(gl), fin)
                assert np.array_equal(gt[fin], want[1]["toks"][fin])
                assert np.array_equal(gl[fin], want[1]["lps"][fin])
                continue
            gp = np.fromfile(os.path.join(out, f"order{n}.pref.bin"),
                             np.uint32).reshape(-1, n - 1)
            gc = np.fromfile(os.path.join(out, f"order{n}.ctot.bin"),
                             np.uint64)
            gt = np.fromfile(os.path.join(out, f"order{n}.toks.bin"),
                             np.uint32).reshape(-1, kmax)
            gl = np.fromfile(os.path.join(out, f"order{n}.lps.bin"),
                             np.float32).reshape(-1, kmax)
            assert np.array_equal(gp, want[n]["pref"]), f"order{n} pref"
            assert np.array_equal(gc, want[n]["ctot"]), f"order{n} ctot"
            fin = np.isfinite(want[n]["lps"])
            assert np.array_equal(np.isfinite(gl), fin), f"order{n} mask"
            assert np.array_equal(gt[fin], want[n]["toks"][fin]), \
                f"order{n} toks"
            assert np.array_equal(gl[fin], want[n]["lps"][fin]), \
                f"order{n} lps"
        meta = json.load(open(os.path.join(out, "scores_meta.json")))
        assert meta["kmax"] == kmax
        for n in orders[1:]:
            assert meta["orders"][str(n)]["records"] == \
                len(slices[n]) // (4 * n + 8), f"order{n} record count"
            assert meta["orders"][str(n)]["missing_parent"] == 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_scored_rows_streaming():
    """stupid_backoff_scoring on range-partitioned local parts — with splits
    deliberately placed INSIDE history groups — must reproduce
    hv.build_rows on the whole table, including the tau filter and the
    cross-part group ownership protocol."""
    _run_scored_rows_case(kmax=KMAX, workers=2, seed=11)
    _run_scored_rows_case(kmax=KMAX, workers=1, seed=11, implementation="simple")
    print("PASS scored_rows_streaming (3-way mid-group splits, 2 workers, "
          "tau=4, all orders bit-identical to build_rows)")


def test_scored_rows_streaming_wide_kmax_512():
    """The wide-K export shape: identical parity at kmax=512 (rows padded
    far past the number of real continuations)."""
    _run_scored_rows_case(kmax=512, workers=2, seed=12)
    print("PASS scored_rows_streaming_wide_kmax_512")


def _write_parts(tmp, slices, orders, n_parts):
    parts_map = {}
    for n in orders:
        rec = 4 * n + 8
        m = np.ascontiguousarray(slices[n], np.uint8).reshape(-1, rec)
        cuts = [len(m) * i // n_parts for i in range(n_parts + 1)]
        paths = []
        for pi in range(n_parts):
            p = os.path.join(tmp, f"order{n}.range{pi:05d}.bin")
            m[cuts[pi]:cuts[pi + 1]].tofile(p)
            paths.append(p)
        parts_map[str(n)] = paths
    lp = os.path.join(tmp, "parts.json")
    with open(lp, "w") as f:
        json.dump(parts_map, f)
    return lp


def test_simple_and_optimized_write_identical_files():
    """Both implementations, same inputs: every output file byte-identical and
    the metadata equal apart from timings. Small vocabularies make equal
    counts, and therefore score ties, common."""
    for seed, kmax, tau, n_parts in ((21, 8, 1, 1), (22, 5, 4, 3), (23, 64, 16, 4)):
        tables = synth_tables(seed=seed, n_orders=4, n_tokens=5000)
        slices = slices_from_tables(tables)
        orders = sorted(slices)
        tmp = tempfile.mkdtemp(prefix="sb_scoring_test_")
        try:
            lp = _write_parts(tmp, slices, orders, n_parts)
            outs = {}
            for impl in ("simple", "optimized"):
                outs[impl] = os.path.join(tmp, impl)
                sbs.run(argparse.Namespace(
                    parts=lp, out_dir=outs[impl], min_order=1, max_order=max(orders),
                    tau=tau, alpha=ALPHA, kmax=kmax, workers=2, chunk_mb=0.002,
                    resume=False, implementation=impl))
            names = sorted(f for f in os.listdir(outs["simple"]) if f.endswith(".bin"))
            assert names == sorted(f for f in os.listdir(outs["optimized"]) if f.endswith(".bin"))
            assert names, "no output files"
            for name in names:
                a = open(os.path.join(outs["simple"], name), "rb").read()
                b = open(os.path.join(outs["optimized"], name), "rb").read()
                assert a == b, f"{name} differs (seed {seed}, kmax {kmax}, tau {tau})"
            metas = []
            for impl in ("simple", "optimized"):
                meta = json.load(open(os.path.join(outs[impl], "scores_meta.json")))
                for order_meta in meta["orders"].values():
                    order_meta.pop("seconds")
                metas.append(meta)
            assert metas[0] == metas[1]
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


def main():
    test_row_parity()
    test_simple_and_optimized_write_identical_files()
    test_scored_rows_streaming()
    test_scored_rows_streaming_wide_kmax_512()


if __name__ == "__main__":
    main()
