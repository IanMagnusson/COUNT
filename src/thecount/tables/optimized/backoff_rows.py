#!/usr/bin/env python3
"""Vectorized stupid-backoff row kernel.

`rows_for_order` merges each history's direct continuation counts with
alpha times its backoff parent's row (stupid backoff, log10 scores) and keeps
the top ``Kmax`` slots per history. `tests/test_stupid_backoff_scoring.py` checks it bit for
bit against the explicit reference implementation in `tests/support.py`.
"""
import math

import numpy as np

# ------------------------------------------ vectorized SB production rows
def group_histories(records, n):
    """Byte-lex sorted order-n records -> per-history groups.
    Returns (pref uint32[P, n-1], starts, ends, ctot uint64[P],
    cont uint32[R], counts uint64[R]) where R = record rows."""
    rec_w = 4 * n + 8
    m = np.ascontiguousarray(records, np.uint8).reshape(-1, rec_w)
    keys = np.ascontiguousarray(m[:, :4 * n]).view(np.uint32).reshape(-1, n)
    counts = np.ascontiguousarray(m[:, 4 * n:]).view("<u8").reshape(-1)
    hist = keys[:, :n - 1]
    cont = keys[:, n - 1]
    hb = 4 * (n - 1)
    hkey = np.ascontiguousarray(hist).view(f"S{hb}").reshape(-1)
    neq = hkey[1:] != hkey[:-1]
    starts = np.concatenate(([0], np.flatnonzero(neq) + 1))
    ends = np.concatenate((starts[1:], [len(m)]))
    pref = np.ascontiguousarray(hist[starts])
    ctot = np.add.reduceat(counts, starts).astype(np.uint64) \
        if len(m) else np.zeros(0, np.uint64)
    return pref, starts, ends, ctot, cont, counts


def merge_rank_rows(hid, tok, val, prio, P, Kmax):
    """The per-history SB merge, vectorized. Entries are (history row id,
    candidate token, log10 SB score, tie priority). Direct entries carry
    lower prio than parent entries of the same history. Semantics mirror
    build_sb_topk exactly:
      1. duplicate (hid, tok) keeps the DIRECT entry (lowest prio);
      2. per history, order by score desc, ties by prio DESC (the reversed
         stable argsort of build_sb_topk), keep the first Kmax.
    Returns (toks uint32[P, Kmax], lps float32[P, Kmax]) padded with 0 /
    -inf."""
    toks = np.zeros((P, Kmax), np.uint32)
    lps = np.full((P, Kmax), -np.inf, np.float32)
    if len(hid) == 0:
        return toks, lps
    # -- direct-precedence dedupe: first (lowest prio) entry per (hid, tok)
    order = np.lexsort((prio, tok, hid))
    hid, tok, val, prio = hid[order], tok[order], val[order], prio[order]
    first = np.ones(len(hid), bool)
    first[1:] = (hid[1:] != hid[:-1]) | (tok[1:] != tok[:-1])
    hid, tok, val, prio = hid[first], tok[first], val[first], prio[first]
    # -- rank: per hid by val desc, tie prio desc
    order = np.lexsort((-prio, -val, hid))
    hid, tok, val = hid[order], tok[order], val[order]
    newh = np.ones(len(hid), bool)
    newh[1:] = hid[1:] != hid[:-1]
    run_start = np.maximum.accumulate(np.where(newh, np.arange(len(hid)),
                                               0))
    rank = np.arange(len(hid)) - run_start
    keep = rank < Kmax
    toks[hid[keep], rank[keep]] = tok[keep]
    lps[hid[keep], rank[keep]] = val[keep].astype(np.float32)
    return toks, lps


def unigram_row(records, Kmax):
    """Order-1 row: top-Kmax unigrams by count, log10(c/N), tie order
    matching build_sb_topk (count desc, later file position first).
    Returns (toks uint32[1, Kmax], lps float32[1, Kmax], N)."""
    rec_w = 12
    m = np.ascontiguousarray(records, np.uint8).reshape(-1, rec_w)
    t = np.ascontiguousarray(m[:, :4]).view(np.uint32).reshape(-1)
    c = np.ascontiguousarray(m[:, 4:]).view("<u8").reshape(-1) \
        .astype(np.float64)
    N = float(c.sum())
    order = np.argsort(c, kind="stable")[::-1][:Kmax]
    toks = np.zeros((1, Kmax), np.uint32)
    lps = np.full((1, Kmax), -np.inf, np.float32)
    mm = len(order)
    toks[0, :mm] = t[order]
    lps[0, :mm] = np.log10(c[order] / N).astype(np.float32)
    return toks, lps, N


def rows_for_order(records, n, parent, alpha, Kmax, tau=1,
                   chunk_hists=2_000_000):
    """Build SB production rows for every order-n history with total
    count >= tau. records: sorted order-n record array. parent: for
    n == 2 the (toks, lps) unigram row; for n > 2 a dict with sorted
    'pref' (uint32[Pp, n-2]), 'toks', 'lps' of the order-(n-1) rows.
    Returns (pref, ctot, toks, lps, n_missing_parent)."""
    pref, starts, ends, ctot, cont, counts = group_histories(records, n)
    keep = ctot >= np.uint64(tau)
    pref, starts, ends, ctot = pref[keep], starts[keep], ends[keep], \
        ctot[keep]
    P = len(pref)
    la = math.log10(alpha)
    if n == 2:
        ptoks = np.broadcast_to(parent[0][0], (P, parent[0].shape[1]))
        plps = np.broadcast_to(parent[1][0], (P, parent[1].shape[1]))
        pok = np.ones(P, bool)
        n_missing = 0
    else:
        pp = np.ascontiguousarray(parent["pref"]).view(
            f"S{4 * (n - 2)}").reshape(-1)
        pk = np.ascontiguousarray(pref[:, 1:]).view(
            f"S{4 * (n - 2)}").reshape(-1)
        prow = np.searchsorted(pp, pk)
        pok = prow < len(pp)
        pok[pok] = pp[prow[pok]] == pk[pok]
        n_missing = int((~pok).sum())
        prow = np.where(pok, prow, 0)
        ptoks = parent["toks"][prow]
        plps = parent["lps"][prow]
    W = ptoks.shape[1]
    out_toks = np.zeros((P, Kmax), np.uint32)
    out_lps = np.full((P, Kmax), -np.inf, np.float32)
    for a in range(0, P, chunk_hists):
        b = min(P, a + chunk_hists)
        # direct entries: every kept continuation, prio = in-group index
        lens = (ends[a:b] - starts[a:b]).astype(np.int64)
        total = int(lens.sum())
        hid_d = np.repeat(np.arange(b - a), lens)
        off = np.cumsum(lens) - lens              # chunk-local group start
        prio_d = np.arange(total) - np.repeat(off, lens)
        idx = np.repeat(starts[a:b].astype(np.int64), lens) + prio_d
        tok_d = cont[idx]
        val_d = np.log10(counts[idx].astype(np.float64) /
                         ctot[a:b].astype(np.float64)[hid_d])
        # parent entries: every finite parent slot (dedupe keeps direct),
        # prio = direct length + slot index (order-preserving vs the
        # filtered-extras positions of build_sb_topk)
        pt = ptoks[a:b]
        pl = plps[a:b].astype(np.float64) + la
        fin = np.isfinite(pl) & pok[a:b, None]
        hid_p, slot_p = np.nonzero(fin)
        tok_p = pt[hid_p, slot_p].astype(np.int64)
        val_p = pl[hid_p, slot_p]
        prio_p = lens[hid_p] + slot_p
        hid = np.concatenate([hid_d, hid_p])
        tok = np.concatenate([tok_d.astype(np.int64), tok_p])
        val = np.concatenate([val_d, val_p])
        prio = np.concatenate([prio_d, prio_p])
        t_, l_ = merge_rank_rows(hid, tok, val, prio, b - a, Kmax)
        out_toks[a:b] = t_
        out_lps[a:b] = l_
    return pref, ctot, out_toks, out_lps, n_missing


def build_rows(slices, orders, alpha=0.4, Kmax=64, tau=1):
    """Reference in-memory builder: SB production rows for every history
    with total count >= tau, per order, bottom-up. slices: {n: sorted
    record array}. Returns {1: {'toks','lps','N'},
    n: {'pref','ctot','toks','lps'}}. The row set is suffix-closed for
    any tau (a history's total never exceeds its backoff parent's), so
    the recursion never needs a pruned-away parent."""
    out = {}
    parent = None
    for n in orders:
        if n == 1:
            t, l, N = unigram_row(slices[1], Kmax)
            out[1] = {"toks": t, "lps": l, "N": N}
            parent = (t, l)
            continue
        pref, ctot, t, l, miss = rows_for_order(
            slices[n], n, parent, alpha, Kmax, tau)
        out[n] = {"pref": pref, "ctot": ctot, "toks": t, "lps": l,
                  "missing_parent": miss}
        parent = {"pref": pref, "toks": t, "lps": l}
    return out


