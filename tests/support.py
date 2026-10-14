"""Shared test scaffolding: synthetic count records and reference builders."""

from __future__ import annotations

import tempfile

import numpy as np


def ram_temp_dir() -> "tempfile.TemporaryDirectory[str]":
    """A temporary directory in RAM (/dev/shm), where a real job stages a table."""

    return tempfile.TemporaryDirectory(dir="/dev/shm")


def recs_from_dict(d, n):
    """d: {tuple(n tokens): count} -> sorted (m,4n+8) uint8 records."""
    keys = np.array(sorted(d), dtype=np.uint32).reshape(-1, n)
    key_b = 4 * n
    raw = np.ascontiguousarray(keys).view(np.uint8).reshape(-1, key_b)
    order_idx = np.argsort(raw.view(f"S{key_b}").reshape(-1), kind="stable")
    raw = raw[order_idx]
    keys = raw.view(np.uint32).reshape(-1, n)
    counts = np.array([d[tuple(int(x) for x in keys[i])]
                       for i in range(len(keys))], dtype=np.uint64)
    rec = np.empty(len(keys), dtype=np.dtype([("g", "u1", (key_b,)),
                                              ("c", "<u8")]))
    rec["g"] = raw
    rec["c"] = counts
    return rec.view(np.uint8).reshape(-1)


def build_sb_topk(slices, orders, V, alpha=0.4, Kmax=64):
    """Production-order build: apply stupid backoff first, then truncate.
    Per history, the full stupid-backoff distribution is its direct
    continuations at c/c(h) merged with alpha times its backoff parent's
    distribution (recursively) — so strong backed-off candidates can enter
    the stored top-K, matching the deployed stupid-backoff table build. Returns
    {n: (pref, toks, lps_log10, Kmax, V)} with slots SB-rank-ordered, ready
    for the standard index scoring (stored value or residual). Bottom-up so
    each order's lists reuse the parent order's."""
    idx = {}
    prev_pref = None      # (P_{n-1}, n-2) uint32 histories of previous order
    prev_toks = prev_lps = None
    for n in orders:
        rec_w = 4 * n + 8
        m = np.ascontiguousarray(slices[n], np.uint8).reshape(-1, rec_w)
        keys = np.ascontiguousarray(m[:, :4 * n]).view(np.uint32) \
            .reshape(-1, n)
        counts = np.ascontiguousarray(m[:, 4 * n:]).view("<u8") \
            .reshape(-1).astype(np.float64)
        if n == 1:
            N = counts.sum()
            order = np.argsort(counts, kind="stable")[::-1][:Kmax]
            toks = np.zeros((1, Kmax), np.uint32)
            lps = np.full((1, Kmax), -np.inf, np.float32)
            mm = len(order)
            toks[0, :mm] = keys[order, 0]
            lps[0, :mm] = np.log10(counts[order] / N).astype(np.float32)
            pref = np.zeros((1, 0), np.uint32)
            idx[1] = (pref, toks, lps, Kmax, V)
            prev_pref, prev_toks, prev_lps = pref, toks, lps
            continue
        hist = keys[:, :n - 1]
        cont = keys[:, n - 1]
        hb = 4 * (n - 1)
        hkey = np.ascontiguousarray(hist).view(f"S{hb}").reshape(-1)
        neq = hkey[1:] != hkey[:-1]
        starts = np.concatenate(([0], np.flatnonzero(neq) + 1))
        ends = np.concatenate((starts[1:], [len(m)]))
        P_ = len(starts)
        pref = np.ascontiguousarray(hist[starts])
        # locate each history's backoff parent (drop the OLDEST token) in
        # the previous order's pref list
        if n == 2:
            prow = np.zeros(P_, np.int64)          # single unigram row
            pok = np.ones(P_, bool)
        else:
            par = np.ascontiguousarray(pref[:, 1:])
            pk = par.view(f"S{4 * (n - 2)}").reshape(-1)
            ppk = np.ascontiguousarray(prev_pref).view(
                f"S{4 * (n - 2)}").reshape(-1)
            prow = np.searchsorted(ppk, pk)
            pok = prow < len(ppk)
            pok[pok] = ppk[prow[pok]] == pk[pok]
        toks = np.zeros((P_, Kmax), np.uint32)
        lps = np.full((P_, Kmax), -np.inf, np.float32)
        la = np.log10(alpha)
        for i in range(P_):
            a, b = starts[i], ends[i]
            ch = counts[a:b].sum()
            dt = cont[a:b]
            dl = np.log10(counts[a:b] / ch)
            if pok[i]:
                bt = prev_toks[prow[i]]
                bl = prev_lps[prow[i]].astype(np.float64) + la
                fin = np.isfinite(bl)
                # backoff candidates not already direct
                extra = ~np.isin(bt[fin], dt)
                ct_ = np.concatenate([dt, bt[fin][extra]])
                cl_ = np.concatenate([dl, bl[fin][extra]])
            else:
                ct_, cl_ = dt, dl
            sel = np.argsort(cl_, kind="stable")[::-1][:Kmax]
            mm = len(sel)
            toks[i, :mm] = ct_[sel]
            lps[i, :mm] = cl_[sel].astype(np.float32)
        idx[n] = (pref, toks, lps, Kmax, V)
        prev_pref, prev_toks, prev_lps = pref, toks, lps
        print(f"[sbtopk] order{n}: {P_:,} histories", flush=True)
    return idx


def prune_histories(records, k, tau):
    """Drop records whose HISTORY (first k-1 tokens) has total count < tau —
    history-level pruning for the DEPLOYED index (keep only histories seen
    >= tau times, which is what sets the unique-key count). Records are
    byte-lex sorted so each history is a contiguous run. k==1 (single empty
    history) is a no-op."""
    if tau <= 1 or k == 1:
        return records
    rec_w = 4 * k + 8
    m = np.ascontiguousarray(records, np.uint8).reshape(-1, rec_w)
    if len(m) == 0:
        return records
    hb = 4 * (k - 1)
    hist = np.ascontiguousarray(m[:, :hb]).view(f"S{hb}").reshape(-1)
    counts = np.ascontiguousarray(m[:, 4 * k:]).view("<u8").reshape(-1)
    neq = hist[1:] != hist[:-1]
    starts = np.concatenate(([0], np.flatnonzero(neq) + 1))
    ends = np.concatenate((starts[1:], [len(m)]))
    hsum = np.add.reduceat(counts.astype(np.uint64), starts)
    keep = np.repeat(hsum >= tau, ends - starts)
    return np.ascontiguousarray(m[keep]).reshape(-1)
