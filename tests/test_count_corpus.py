#!/usr/bin/env python3
"""Gate `count_corpus` against a plain dictionary count and against the
canonical record encoder the accepted scored rows use.

Three properties are required:

1. counts equal a brute-force dictionary count over the same shards, per order;
2. emitted parts are byte-identical to `recs_from_dict`, the encoder that
   defines the accepted scored rows' key encoding and byte-lex sort order;
3. concatenating parts in name order yields a globally byte-lex-sorted table,
   which is what the scorer's forward-reading protocol assumes.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tests.support import recs_from_dict
from thecount.counts.optimized import count_corpus as cc

MAX_ORDER = 4
VOCAB = 300
SEP = 297


def _write_shards(tmp_path: Path, shards: list[np.ndarray]) -> Path:
    d = tmp_path / "corpus"
    d.mkdir()
    for i, s in enumerate(shards):
        s.astype("<u4").tofile(d / f"part-{i:03d}-00000.npy")
    return d


def _brute(shards: list[np.ndarray], order: int) -> dict[tuple, int]:
    """Reference count: per-shard windows and the separator rule.

    A window is invalid when the separator appears STRICTLY BEFORE its final
    position, mirroring `sa_extract_direct.py`'s `past_boundary` / `_close`.
    The separator may end an n-gram; it may never sit inside its history.
    """
    out: dict[tuple, int] = {}
    for sh in shards:
        for i in range(len(sh) - order + 1):
            w = [int(x) for x in sh[i:i + order]]
            if SEP in w[:-1]:
                continue
            k = tuple(w)
            out[k] = out.get(k, 0) + 1
    return out


def _make_shards(seed: int = 7) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    probs = 1.0 / np.arange(1, VOCAB + 1)
    probs /= probs.sum()
    shards = []
    for _ in range(3):
        s = rng.choice(VOCAB, size=rng.integers(4000, 6000), p=probs)
        # sprinkle document separators so boundary tokens are exercised
        s[rng.choice(len(s), size=len(s) // 200, replace=False)] = SEP
        shards.append(s.astype("<u4"))
    return shards


def _run(corpus: Path, out: Path, partitions: int, min_count: int = 1,
         groups: int = 16, resume: bool = False,
         workers: int = 1) -> None:
    cc.run(argparse.Namespace(
        corpus_dir=str(corpus), pattern="part-*.npy", out_dir=str(out),
        max_order=MAX_ORDER, min_count=min_count, partitions=partitions,
        groups=groups, resume=resume, workers=workers,
        vocab_size=VOCAB, doc_separator_token_id=SEP))


def _read_concat(out: Path, order: int, partitions: int) -> np.ndarray:
    rec_w = 4 * order + 8
    blobs = [np.fromfile(out / f"order{order}.range{p:05d}.bin", dtype=np.uint8)
             for p in range(partitions)]
    return np.concatenate(blobs).reshape(-1, rec_w) if blobs else \
        np.zeros((0, rec_w), np.uint8)


@pytest.mark.parametrize("partitions", [256, 64, 1])
def test_counts_match_bruteforce_and_encoding(tmp_path, partitions):
    shards = _make_shards()
    corpus = _write_shards(tmp_path, shards)
    out = tmp_path / f"counts{partitions}"
    _run(corpus, out, partitions)

    for order in range(1, MAX_ORDER + 1):
        want = _brute(shards, order)
        got = _read_concat(out, order, partitions)
        key_b = 4 * order

        # (1) counts equal the dictionary count
        assert len(got) == len(want), f"order {order}: distinct count"
        keys = np.ascontiguousarray(got[:, :key_b]).view("<u4").reshape(-1, order)
        counts = np.ascontiguousarray(got[:, key_b:]).view("<u8").reshape(-1)
        as_dict = {tuple(int(x) for x in keys[i]): int(counts[i])
                   for i in range(len(keys))}
        assert as_dict == want, f"order {order}: counts differ"

        # (2) byte-identical to the canonical encoder
        expect = recs_from_dict(want, order).reshape(-1, key_b + 8)
        assert np.array_equal(got, expect), f"order {order}: encoding differs"

        # (3) globally byte-lex sorted across parts
        flat = np.ascontiguousarray(got[:, :key_b]).view(f"S{key_b}").reshape(-1)
        assert np.array_equal(flat, np.sort(flat)), f"order {order}: not sorted"
        assert len(np.unique(flat)) == len(flat), f"order {order}: dup keys"


def test_partitioning_is_invariant(tmp_path):
    """Part count changes the layout, never the concatenated table."""
    shards = _make_shards(seed=11)
    corpus = _write_shards(tmp_path, shards)
    a, b = tmp_path / "a", tmp_path / "b"
    _run(corpus, a, 256)
    _run(corpus, b, 16)
    for order in range(1, MAX_ORDER + 1):
        assert np.array_equal(_read_concat(a, order, 256),
                              _read_concat(b, order, 16)), f"order {order}"


def test_min_count_floor_drops_exactly_the_rare_ngrams(tmp_path):
    shards = _make_shards(seed=13)
    corpus = _write_shards(tmp_path, shards)
    out = tmp_path / "floor2"
    _run(corpus, out, 64, min_count=2)
    for order in range(1, MAX_ORDER + 1):
        want = {k: v for k, v in _brute(shards, order).items() if v >= 2}
        got = _read_concat(out, order, 64)
        key_b = 4 * order
        assert len(got) == len(want), f"order {order}"
        if len(want):
            expect = recs_from_dict(want, order).reshape(-1, key_b + 8)
            assert np.array_equal(got, expect), f"order {order}"


def test_cli_entrypoint(tmp_path):
    shards = _make_shards(seed=17)
    corpus = _write_shards(tmp_path, shards)
    out = tmp_path / "cli"
    src = Path(__file__).resolve().parents[1] / "src"
    r = subprocess.run(
        [sys.executable, "-m", "thecount.counts.count_corpus",
         "--corpus-dir", str(corpus), "--out-dir", str(out),
         "--max-order", "3", "--partitions", "8",
         "--vocab-size", str(VOCAB), "--doc-separator-token-id", str(SEP)],
        capture_output=True, text=True, env={"PYTHONPATH": str(src),
                                             "PATH": "/usr/bin:/bin"})
    assert r.returncode == 0, r.stderr
    assert (out / "count_meta.json").exists()

def test_group_sweep_count_is_invariant(tmp_path):
    """Sweep grouping is a performance knob; it must not move a single byte."""
    shards = _make_shards(seed=23)
    corpus = _write_shards(tmp_path, shards)
    outs = {}
    for groups in (1, 4, 16, 256):
        d = tmp_path / f"g{groups}"
        _run(corpus, d, 256, groups=groups)
        outs[groups] = d
    for order in range(1, MAX_ORDER + 1):
        ref = _read_concat(outs[16], order, 256)
        for groups, d in outs.items():
            assert np.array_equal(_read_concat(d, order, 256), ref), \
                f"order {order}, groups {groups}"


def test_resume_skips_completed_orders(tmp_path):
    shards = _make_shards(seed=29)
    corpus = _write_shards(tmp_path, shards)
    out = tmp_path / "resume"
    _run(corpus, out, 16)
    ref = {n: _read_concat(out, n, 16) for n in range(1, MAX_ORDER + 1)}
    # corrupt one part; a resumed run must NOT rewrite it (order was complete)
    victim = out / "order3.range00000.bin"
    victim.write_bytes(b"")
    _run(corpus, out, 16, resume=True)
    assert victim.stat().st_size == 0, "resume rewrote a completed order"
    # a non-resume run into a populated directory is refused outright
    with pytest.raises(SystemExit, match="already contains a count"):
        _run(corpus, out, 16, resume=False)
    # a fresh directory reproduces the original bytes exactly
    fresh = out.parent / "resume_fresh"
    _run(corpus, fresh, 16)
    for n in range(1, MAX_ORDER + 1):
        assert np.array_equal(_read_concat(fresh, n, 16), ref[n]), f"order {n}"

def test_worker_count_is_invariant(tmp_path):
    """Parallelism is a performance knob; it must not move a single byte."""
    shards = _make_shards(seed=31)
    corpus = _write_shards(tmp_path, shards)
    serial, par = tmp_path / "w1", tmp_path / "w4"
    _run(corpus, serial, 64, workers=1)
    _run(corpus, par, 64, workers=4)
    for order in range(1, MAX_ORDER + 1):
        assert np.array_equal(_read_concat(serial, order, 64),
                              _read_concat(par, order, 64)), f"order {order}"


def test_separator_never_sits_inside_a_history(tmp_path):
    """Document-boundary case: [1, SEP, 2, 3].

    Order 2 must keep (1, SEP) and (2, 3) and must NOT emit (SEP, 2): the
    separator may end an n-gram but never appear inside its history.
    """
    d = tmp_path / "corpus"
    d.mkdir()
    np.array([1, SEP, 2, 3], dtype="<u4").tofile(d / "part-000-00000.npy")
    out = tmp_path / "counts"
    _run(d, out, 1)
    got = _read_concat(out, 2, 1)
    keys = {tuple(int(x) for x in k) for k in
            np.ascontiguousarray(got[:, :8]).view("<u4").reshape(-1, 2)}
    assert (SEP, 2) not in keys, "separator leaked into a history"
    assert keys == {(1, SEP), (2, 3)}, keys
    # At order 3 BOTH windows are invalid -- [1,SEP,2] and [SEP,2,3] each carry
    # the separator strictly before their final position -- so nothing is
    # emitted. The separator would have to be the LAST token to survive.
    keys3 = {tuple(int(x) for x in k) for k in
             np.ascontiguousarray(_read_concat(out, 3, 1)[:, :12])
             .view("<u4").reshape(-1, 3)}
    assert keys3 == set(), keys3


def test_boundary_gap_matches_a_scalar_reference(tmp_path):
    """The chunked gap scan must equal a plain sequential one, across chunks."""
    rng = np.random.default_rng(5)
    toks = rng.choice(VOCAB, size=5000).astype("<u4")
    toks[rng.choice(len(toks), size=200, replace=False)] = SEP
    for chunk in (7, 64, 999, 1 << 20):
        got = cc.boundary_gap(toks, SEP, chunk=chunk)
        want, run = [], 0
        for t in toks:
            run = 0 if int(t) == SEP else min(run + 1, 255)
            want.append(run)
        assert np.array_equal(got, np.array(want, dtype=np.uint8)), chunk


def test_resume_refuses_changed_parameters(tmp_path):
    shards = _make_shards(seed=37)
    corpus = _write_shards(tmp_path, shards)
    out = tmp_path / "params"
    _run(corpus, out, 64, min_count=1)
    with pytest.raises(SystemExit, match="different parameters"):
        _run(corpus, out, 64, min_count=2, resume=True)
    with pytest.raises(SystemExit, match="different parameters"):
        _run(corpus, out, 16, resume=True)


def _read_meta_without_timings(path: Path) -> dict:
    meta = json.loads(path.read_text())
    meta.pop("total_seconds", None)
    for entry in meta["orders"].values():
        entry.pop("seconds", None)
    return meta


@pytest.mark.parametrize("partitions,min_count", [(256, 1), (64, 2), (1, 1)])
def test_simple_and_optimized_write_identical_parts(tmp_path, partitions, min_count):
    """Both implementations, same shards: every part file byte-identical and the
    metadata equal apart from timings."""
    from thecount.counts import count_corpus as plain

    shards = _make_shards(seed=19)
    corpus = _write_shards(tmp_path, shards)
    outs = {}
    for impl in ("simple", "optimized"):
        outs[impl] = tmp_path / impl
        plain.run(argparse.Namespace(
            corpus_dir=str(corpus), pattern="part-*.npy", out_dir=str(outs[impl]),
            max_order=MAX_ORDER, min_count=min_count, partitions=partitions,
            groups=min(16, partitions), resume=False, workers=2,
            vocab_size=VOCAB, doc_separator_token_id=SEP, implementation=impl))
    names = sorted(p.name for p in outs["simple"].glob("order*.range*.bin"))
    assert names == sorted(p.name for p in outs["optimized"].glob("order*.range*.bin"))
    assert len(names) == MAX_ORDER * partitions
    for name in names:
        assert (outs["simple"] / name).read_bytes() == (outs["optimized"] / name).read_bytes(), name
    assert _read_meta_without_timings(outs["simple"] / "count_meta.json") == \
        _read_meta_without_timings(outs["optimized"] / "count_meta.json")
