#!/usr/bin/env python3
"""Stupid-backoff scoring: one dense top-Kmax row per retained history.

Streams byte-lex-sorted count tables (order by order, bottom-up) and
materializes, for every history whose total continuation count is >= tau, its
stupid-backoff row: direct continuations merged with alpha times the backoff
parent's row, truncated to Kmax slots. One scoring run at the smallest tau serves
every deeper history threshold as a subset filter, because a history's total
never exceeds its backoff parent's, so the tau-surviving set is suffix-closed
at any tau and each survivor's row is identical whichever tau selected it.

Mechanics: count tables are range-partitioned and individually sorted
(order<k>.range<NNNNN>.bin, record = 4k key bytes + 8 count bytes), so
histories are contiguous runs. Workers process one part each: a worker skips
the leading records belonging to the previous part's last history (that group
is owned by the previous worker) and reads forward into the following part(s)
to complete its own last group, so every history group is built exactly once.
Complete-group buffers feed the vectorized merge kernel
(backoff_rows.rows_for_order). Parent rows for order n come from the order
n-1 scored rows, memory-mapped and shared across workers by fork.

Output per order n (all little-endian, row-aligned, globally sorted):
  order<n>.pref.bin  uint32[P, n-1]   history keys
  order<n>.ctot.bin  uint64[P]        history total counts
  order<n>.toks.bin  uint32[P, Kmax]  rank-ordered slot tokens
  order<n>.lps.bin   float32[P, Kmax] slot log10 stupid-backoff scores (-inf pad)
plus order1.toks/lps.bin (single unigram row) and scores_meta.json.
The parts for each order are listed in a JSON file passed as --parts.
"""

import argparse
import json
import multiprocessing as mp
import os
import queue
import threading
import time

import numpy as np

from thecount.tables.optimized import backoff_rows as hv

class Part:
    """Ranged-read handle for one count part file."""

    def __init__(self, path):
        self.path = path
        self._size = os.path.getsize(path)

    def size(self):
        return self._size

    def read(self, offset, length):
        if length <= 0 or offset >= self._size:
            return b""
        with open(self.path, "rb") as f:
            f.seek(offset)
            return f.read(min(length, self._size - offset))


# worker globals installed before each order's Pool is forked
_G = {}
def _hist_view(m, n):
    hb = 4 * (n - 1)
    return np.ascontiguousarray(m[:, :hb]).view(f"S{hb}").reshape(-1)


def part_group_buffers(parts, i, n, chunk_bytes):
    """Yield complete-group record arrays (m, 4n+8) for part i of a
    range-partitioned sorted table. Ownership protocol: groups STARTING
    in part i belong to worker i; the worker completes its last group by
    reading forward into parts i+1, i+2, ... while the history matches."""
    rec = 4 * n + 8
    hb = 4 * (n - 1)
    src = parts[i]
    nb = src.size()
    prev_hist = None
    if i > 0:
        psz = parts[i - 1].size()
        if psz >= rec:
            # numpy S-dtype comparisons treat trailing NULs as padding;
            # strip them so == / searchsorted against the S-view agree
            prev_hist = bytes(parts[i - 1].read(psz - rec, rec)[:hb]) \
                .rstrip(b"\x00")
    q = queue.Queue(maxsize=2)

    def reader():
        pos = 0
        while pos < nb:
            want = min(chunk_bytes, nb - pos)
            q.put(src.read(pos, want))
            pos += want
        q.put(None)

    t = threading.Thread(target=reader, daemon=True)
    t.start()
    carry = b""                       # held-back last group + partial record
    final = None                      # the part's last group, record array
    skipping = prev_hist is not None
    while True:
        b = q.get()
        last_chunk = b is None
        buf = carry if last_chunk else carry + b
        nrec = len(buf) // rec
        data, carry = buf[:nrec * rec], buf[nrec * rec:]
        if last_chunk:
            assert not carry, "trailing partial record"
        m = np.frombuffer(data, np.uint8).reshape(-1, rec).copy()
        if skipping and len(m):
            hk = _hist_view(m, n)
            if hk[0] == prev_hist:
                m = m[np.searchsorted(hk, prev_hist, side="right"):]
            if len(m):
                skipping = False
        if len(m):
            # hold back the final (possibly continuing) group
            hk = _hist_view(m, n)
            first_last = np.searchsorted(hk, hk[-1], side="left")
            if last_chunk:
                final = m[first_last:]
            else:
                carry = m[first_last:].tobytes() + carry
            m = m[:first_last]
        if len(m):
            yield m
        if last_chunk:
            break
    t.join()
    # complete the part's last group from the following parts
    if final is not None and len(final):
        last = bytes(_hist_view(final, n)[-1])
        extra = []
        aligned = max(rec, chunk_bytes // rec * rec)
        for j in range(i + 1, len(parts)):
            sj = parts[j]
            nbj = sj.size()
            pos = 0
            stop = False
            while pos < nbj and not stop:
                b = sj.read(pos, min(aligned, nbj - pos))
                pos += len(b)
                mm = np.frombuffer(b, np.uint8).reshape(-1, rec)
                hj = _hist_view(mm, n)
                endj = np.searchsorted(hj, last, side="right")
                if endj:
                    extra.append(mm[:endj].copy())
                if endj < len(mm):
                    stop = True
            if stop:
                break
        if extra:
            final = np.concatenate([final] + extra, 0)
        yield final


def _work_part(i):
    """Build rows for the groups owned by part i. Returns the tmp path
    prefix its four arrays were saved under, plus counters."""
    g = _G
    n, tau, alpha, Kmax = g["n"], g["tau"], g["alpha"], g["Kmax"]
    parts = [Part(u) for u in g["urls"]]
    nbytes = parts[i].size()
    out_pref, out_ctot, out_toks, out_lps = [], [], [], []
    nrec = 0
    nmiss = 0
    for m in part_group_buffers(parts, i, n, g["chunk_bytes"]):
        nrec += len(m)
        pref, ctot, toks, lps, miss = hv.rows_for_order(
            m.reshape(-1), n, g["parent"], alpha, Kmax, tau)
        nmiss += miss
        if len(pref):
            out_pref.append(pref)
            out_ctot.append(ctot)
            out_toks.append(toks)
            out_lps.append(lps)
    pfx = os.path.join(g["tmp_dir"], f"order{n}.part{i:05d}")
    P = 0
    if out_pref:
        pref = np.concatenate(out_pref, 0)
        P = len(pref)
        np.save(pfx + ".pref.npy", pref)
        np.save(pfx + ".ctot.npy", np.concatenate(out_ctot))
        np.save(pfx + ".toks.npy", np.concatenate(out_toks, 0))
        np.save(pfx + ".lps.npy", np.concatenate(out_lps, 0))
    return i, P, nrec, nmiss, nbytes


def load_parent(out_dir, n, Kmax):
    """Memory-map the order-(n-1) scored rows as the parent row store for
    order n (fork-shared page cache, no copies)."""
    if n == 2:
        toks = np.fromfile(os.path.join(out_dir, "order1.toks.bin"),
                           np.uint32).reshape(1, Kmax)
        lps = np.fromfile(os.path.join(out_dir, "order1.lps.bin"),
                          np.float32).reshape(1, Kmax)
        return (toks, lps)
    p = n - 1
    pref = np.memmap(os.path.join(out_dir, f"order{p}.pref.bin"),
                     np.uint32, "r").reshape(-1, p - 1)
    toks = np.memmap(os.path.join(out_dir, f"order{p}.toks.bin"),
                     np.uint32, "r").reshape(-1, Kmax)
    lps = np.memmap(os.path.join(out_dir, f"order{p}.lps.bin"),
                    np.float32, "r").reshape(-1, Kmax)
    return {"pref": pref, "toks": toks, "lps": lps}


def build_order(urls, n, out_dir, tmp_dir, tau, alpha, Kmax, workers,
                chunk_bytes, meta):
    t0 = time.time()
    if n == 1:
        parts = [Part(u) for u in urls]
        raw = b"".join(p.read(0, p.size()) for p in parts)
        toks, lps, N = hv.unigram_row(np.frombuffer(raw, np.uint8), Kmax)
        toks.tofile(os.path.join(out_dir, "order1.toks.bin"))
        lps.tofile(os.path.join(out_dir, "order1.lps.bin"))
        meta["orders"]["1"] = {"records": len(raw) // 12, "rows": 1,
                               "total_count": N,
                               "seconds": round(time.time() - t0, 1)}
        return
    _G.update(n=n, tau=tau, alpha=alpha, Kmax=Kmax, urls=urls,
              chunk_bytes=chunk_bytes, tmp_dir=tmp_dir,
              parent=load_parent(out_dir, n, Kmax))
    results = []
    gb_done = 0.0
    if workers <= 1:
        for i in range(len(urls)):
            results.append(_work_part(i))
    else:
        with mp.Pool(workers) as pool:
            for r in pool.imap_unordered(_work_part, range(len(urls))):
                results.append(r)
                gb_done += r[4] / 1e9
                if len(results) % 8 == 0:
                    el = time.time() - t0
                    print(f"[sb-scoring] order{n}: {len(results)}/"
                          f"{len(urls)} parts | {gb_done:.0f} GB | "
                          f"{gb_done / max(el, 1e-9):.2f} GB/s | "
                          f"kept {sum(x[1] for x in results):,} | "
                          f"{el:.0f}s", flush=True)
    results.sort()
    P = sum(r[1] for r in results)
    nrec = sum(r[2] for r in results)
    nmiss = sum(r[3] for r in results)
    # concatenate part outputs in part order -> globally sorted scored rows
    files = {}
    for suff, dt in (("pref", np.uint32), ("ctot", np.uint64),
                     ("toks", np.uint32), ("lps", np.float32)):
        files[suff] = open(os.path.join(out_dir, f"order{n}.{suff}.bin"),
                           "wb")
    for i, pcount, _, _, _ in results:
        if pcount == 0:
            continue
        pfx = os.path.join(tmp_dir, f"order{n}.part{i:05d}")
        for suff in ("pref", "ctot", "toks", "lps"):
            a = np.load(pfx + f".{suff}.npy")
            files[suff].write(np.ascontiguousarray(a).tobytes())
            os.remove(pfx + f".{suff}.npy")
    for f in files.values():
        f.close()
    meta["orders"][str(n)] = {
        "records": int(nrec), "rows": int(P),
        "missing_parent": int(nmiss),
        "seconds": round(time.time() - t0, 1)}
    print(f"[sb-scoring] order{n}: {P:,} rows from {nrec:,} records, "
          f"missing_parent={nmiss}, {time.time() - t0:.0f}s", flush=True)
    if nmiss:
        # suffix-closure says this cannot happen on real corpus tables;
        # a nonzero count is table-anomaly evidence, so surface it loudly
        print(f"[sb-scoring] WARNING order{n}: {nmiss} histories had no "
              f"surviving backoff parent (direct-only rows)", flush=True)


def run(args):
    os.makedirs(args.out_dir, exist_ok=True)
    tmp_dir = os.path.join(args.out_dir, "tmp_parts")
    os.makedirs(tmp_dir, exist_ok=True)
    orders = list(range(args.min_order, args.max_order + 1))
    local = json.load(open(args.parts))
    meta_path = os.path.join(args.out_dir, "scores_meta.json")
    meta = {"tau": args.tau, "alpha": args.alpha,
            "kmax": args.kmax, "orders": {}}
    if os.path.exists(meta_path) and args.resume:
        meta = json.load(open(meta_path))
    for n in orders:
        if args.resume and str(n) in meta["orders"]:
            print(f"[sb-scoring] order{n}: already done, skip", flush=True)
            continue
        urls = local[str(n)]
        if not urls:
            raise RuntimeError(f"no parts for order{n}")
        print(f"[sb-scoring] order{n}: {len(urls)} parts", flush=True)
        build_order(urls, n, args.out_dir, tmp_dir, args.tau, args.alpha,
                    args.kmax, args.workers,
                    max(1024, int(args.chunk_mb * 1_000_000)), meta)
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=1)
    print("[sb-scoring] DONE", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", required=True,
                    help="JSON file {order: [count part paths]}")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--min-order", type=int, default=1)
    ap.add_argument("--max-order", type=int, default=7)
    ap.add_argument("--tau", type=int, default=4096,
                    help="min history total count; deeper tiers filter "
                         "these scored rows by ctot")
    ap.add_argument("--alpha", type=float, default=0.4)
    ap.add_argument("--kmax", type=int, default=64,
                    help="stored slots per row (row policies are prefix "
                         "cuts of this)")
    ap.add_argument("--workers", type=int, default=30)
    ap.add_argument("--chunk-mb", type=float, default=64,
                    help="sequential read chunk (MB); fractional values "
                         "allowed so tests can force tiny chunks")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
