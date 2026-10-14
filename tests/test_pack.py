from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import struct
import sys
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]

from thecount.tables.optimized.pack import (  # noqa: E402
    ScoredRowsFormatError,
    estimate_table_bytes,
    pack_scored_rows,
    plan_scored_rows,
    select_k_for_budget,
)
from tests.support import ram_temp_dir  # noqa: E402


def _load_production_module():
    """Load the exact reader module without importing olmo_core.data."""
    path = REPO_ROOT / "olmo-core" / "src" / "olmo_core" / "data" / "count_table.py"
    spec = importlib.util.spec_from_file_location("_count_table_reader", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load production reader from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PRODUCTION_MODULE = _load_production_module()
CountTable = PRODUCTION_MODULE.CountTable
CountLookup = PRODUCTION_MODULE.CountLookup


def _lookup_one(reader, context):
    """Longest-match lookup through the reader's per-order search."""
    reader._ensure_open()
    for order in range(min(max(reader.sections_by_order), len(context) + 1), 0, -1):
        row = reader.lookup_order(context, order)
        if row is not None:
            return row
    raise AssertionError("order-1 row is always present")


U32 = np.dtype("<u4")
F32 = np.dtype("<f4")


def _assert_scores_close(actual, expected):
    """Compare decoded 8-bit scores with their float sources: -inf slots must
    match exactly and finite slots within one quantization step."""
    actual = np.asarray(actual, dtype=F32)
    expected = np.asarray(expected, dtype=F32)
    np.testing.assert_array_equal(np.isneginf(actual), np.isneginf(expected))
    finite = np.isfinite(expected)
    np.testing.assert_allclose(actual[finite], expected[finite], atol=0.01)

U64 = np.dtype("<u8")


def _source_byte_order(prefixes: np.ndarray) -> np.ndarray:
    prefixes = np.ascontiguousarray(prefixes, dtype=U32)
    raw = prefixes.view(np.uint8).reshape(len(prefixes), prefixes.shape[1] * 4)
    keys = np.ascontiguousarray(raw).view(f"S{raw.shape[1]}").reshape(-1)
    return np.argsort(keys, kind="stable")


def _write_dense_order(
    root: Path,
    order: int,
    rows: list[tuple[tuple[int, ...], int, list[int], list[float]]],
    *,
    kmax: int,
) -> None:
    prefixes = np.asarray([row[0] for row in rows], dtype=U32)
    totals = np.asarray([row[1] for row in rows], dtype=U64)
    tokens = np.asarray([row[2] for row in rows], dtype=U32).reshape(-1, kmax)
    scores = np.asarray([row[3] for row in rows], dtype=F32).reshape(-1, kmax)
    permutation = _source_byte_order(prefixes)
    prefixes[permutation].tofile(root / f"order{order}.pref.bin")
    totals[permutation].tofile(root / f"order{order}.ctot.bin")
    tokens[permutation].tofile(root / f"order{order}.toks.bin")
    scores[permutation].tofile(root / f"order{order}.lps.bin")


def _write_synthetic_scored_rows(root: Path) -> dict[int, list]:
    kmax = 4
    root.mkdir(parents=True)
    np.asarray([[11, 12, 13, 14]], dtype=U32).tofile(root / "order1.toks.bin")
    np.asarray([[-0.10, -0.20, -0.30, -0.40]], dtype=F32).tofile(
        root / "order1.lps.bin"
    )
    rows2 = [
        ((1,), 30, [101, 102, 103, 104], [-0.11, -0.23, -0.35, -0.47]),
        ((256,), 20, [601, 602, 603, 604], [-0.14, -0.26, -0.38, -0.50]),
        ((2,), 15, [201, 0, 203, 204], [-0.31, -math.inf, -0.43, -0.55]),
        ((3,), 5, [301, 302, 303, 304], [-0.16, -0.28, -0.40, -0.52]),
    ]
    rows3 = [
        ((1, 256), 20, [501, 502, 503, 504], [-0.17, -0.29, -0.41, -0.53]),
        ((9, 2), 5, [521, 522, 523, 524], [-0.18, -0.30, -0.42, -0.54]),
        ((2, 3), 11, [541, 542, 543, 544], [-0.19, -0.62, -0.44, -0.56]),
        ((7, 999), 5, [561, 562, 563, 564], [-0.20, -0.32, -0.45, -0.57]),
    ]
    _write_dense_order(root, 2, rows2, kmax=kmax)
    _write_dense_order(root, 3, rows3, kmax=kmax)
    metadata = {
        "tau": 1,
        "alpha": 0.4,
        "kmax": kmax,
        # Deliberately no ctot_dtype: this matches the original scorer. The
        # packer's documented default must read these files as uint64.
        "orders": {
            "1": {"rows": 1, "records": 4},
            "2": {"rows": len(rows2), "records": len(rows2)},
            "3": {"rows": len(rows3), "records": len(rows3)},
        },
    }
    (root / "scores_meta.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return {2: rows2, 3: rows3}


def _close_reader(reader) -> None:
    # CountTable.close() closes mmap before clearing its own numpy
    # views. Clear those views first so Python's mmap does not report
    # exported pointers during this focused test.
    reader.sections_by_order.clear()
    reader.close()


def _grid8(values: np.ndarray, lo: float) -> np.ndarray:
    out = values.astype(np.float64).copy()
    finite = np.isfinite(out)
    q = np.rint((out[finite] - lo) / (-lo) * 255.0)
    out[finite] = q / 255.0 * (-lo) + lo
    return out.astype(F32)


class PackScoredRowsTest(unittest.TestCase):
    def test_selects_widest_k_within_exact_byte_budget(self) -> None:
        rows = ((1, 1), (2, 10))
        k2_bytes = estimate_table_bytes(rows, K=2, logprob_bytes=1)
        k3_bytes = estimate_table_bytes(rows, K=3, logprob_bytes=1)
        selection = select_k_for_budget(
            rows,
            source_kmax=4,
            budget_bytes=k3_bytes - 1,
            logprob_bytes=1,
        )
        self.assertEqual(selection["K"], 2)
        self.assertEqual(selection["bytes"], k2_bytes)
        self.assertEqual(selection["closest_K"], 3)

    def test_wide_source_kmax_512_packs_and_reads_back(self) -> None:
        # The widest supported source row must pack and read back end to end,
        # and anything wider must be rejected.
        kmax = 512
        with ram_temp_dir() as tmp:
            root = Path(tmp)
            source = root / "source"
            source.mkdir(parents=True)
            tokens1 = np.arange(1, kmax + 1, dtype=U32).reshape(1, kmax)
            scores1 = (-0.001 * np.arange(1, kmax + 1, dtype=np.float64) - 0.01).astype(
                F32
            )
            tokens1.tofile(source / "order1.toks.bin")
            scores1.reshape(1, kmax).tofile(source / "order1.lps.bin")
            rows2 = [
                (
                    (prefix,),
                    total,
                    list(range(prefix * 600, prefix * 600 + kmax)),
                    [-0.002 * slot - 0.02 for slot in range(1, kmax + 1)],
                )
                for prefix, total in ((5, 100), (2, 40), (7, 9))
            ]
            _write_dense_order(source, 2, rows2, kmax=kmax)
            metadata = {
                "tau": 1,
                "alpha": 0.4,
                "kmax": kmax,
                "orders": {
                    "1": {"rows": 1, "records": kmax},
                    "2": {"rows": len(rows2), "records": len(rows2)},
                },
            }
            (source / "scores_meta.json").write_text(
                json.dumps(metadata) + "\n", encoding="utf-8"
            )

            plan = plan_scored_rows(
                source,
                max_order=2,
                K=kmax,
                min_history_total=10,
                vocab_size=600 * 8 + kmax,
                sentinel_eos_token_id=1,
                chunk_rows=2,
            )
            self.assertEqual(plan.source_kmax, kmax)
            self.assertEqual(
                [(p.input_rows, p.output_rows) for p in plan.orders],
                [(1, 1), (3, 2)],
            )
            output = root / "artifact" / "forward_index_topk.bin"
            pack_scored_rows(
                plan,
                output_path=output,
                manifest_path=output.parent / "build_manifest.json",
            )
            reader = CountTable(str(output))
            reader._ensure_open()
            try:
                self.assertEqual(reader.K, kmax)
                np.testing.assert_array_equal(
                    reader.sections_by_order[2].prefixes_2d,
                    np.asarray([[2], [5]], dtype=U32),
                )
                np.testing.assert_array_equal(
                    reader.sections_by_order[2].topk_tokens[1],
                    np.arange(5 * 600, 5 * 600 + kmax, dtype=U32),
                )
            finally:
                _close_reader(reader)

            # One wider than the supported maximum must be rejected loudly.
            metadata["kmax"] = kmax + 1
            (source / "scores_meta.json").write_text(
                json.dumps(metadata) + "\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ScoredRowsFormatError, "kmax"):
                plan_scored_rows(
                    source,
                    max_order=2,
                    K=kmax,
                    min_history_total=10,
                    vocab_size=600 * 8 + kmax,
                    sentinel_eos_token_id=1,
                    chunk_rows=2,
                )

    def test_uint64_filter_numeric_resort_reader_and_manifest(self) -> None:
        with ram_temp_dir() as tmp:
            root = Path(tmp)
            source = root / "source"
            _write_synthetic_scored_rows(source)
            output = root / "artifact" / "forward_index_topk.bin"
            manifest_path = output.parent / "build_manifest.json"
            plan = plan_scored_rows(
                source,
                max_order=3,
                K=2,
                min_history_total=10,
                vocab_size=1000,
                sentinel_eos_token_id=999,
                # Exercise the original uint64 default explicitly.
                ctot_dtype="uint64",
                chunk_rows=2,
            )
            self.assertEqual(
                [(p.input_rows, p.output_rows) for p in plan.orders],
                [(1, 1), (4, 3), (4, 2)],
            )
            manifest = pack_scored_rows(
                plan,
                output_path=output,
                manifest_path=manifest_path,
            )

            reader = CountTable(str(output))
            reader._ensure_open()
            try:
                self.assertEqual(reader.K, 2)
                self.assertEqual(reader.vocab_size, 1000)
                np.testing.assert_array_equal(
                    reader.sections_by_order[2].prefixes_2d,
                    np.asarray([[1], [2], [256]], dtype=U32),
                )
                np.testing.assert_array_equal(
                    reader.sections_by_order[2].topk_tokens,
                    np.asarray([[101, 102], [201, 0], [601, 602]], dtype=U32),
                )
                _assert_scores_close(
                    reader.sections_by_order[2].logprobs_at(np.arange(3)),
                    [[-0.11, -0.23], [-0.31, -math.inf], [-0.14, -0.26]],
                )
                np.testing.assert_array_equal(
                    reader.sections_by_order[3].prefixes_2d,
                    np.asarray([[1, 256], [2, 3]], dtype=U32),
                )
                np.testing.assert_array_equal(
                    reader.sections_by_order[3].topk_tokens,
                    np.asarray([[501, 502], [541, 542]], dtype=U32),
                )

                tokens, scores = _lookup_one(reader, [42, 1, 256])
                np.testing.assert_array_equal(tokens, [501, 502])
                _assert_scores_close(scores, [-0.17, -0.29])

                # (9, 2) was removed by the history-total threshold, so the
                # reader falls back to the retained order-2 prefix (2,).
                tokens, scores = _lookup_one(reader, [9, 2])
                np.testing.assert_array_equal(tokens, [201, 0])
                _assert_scores_close(scores, [-0.31, -math.inf])

                # Neither (7, 999) nor (999,) exists, so this reaches unigrams.
                tokens, scores = _lookup_one(reader, [7, 999])
                np.testing.assert_array_equal(tokens, [11, 12])
                _assert_scores_close(scores, [-0.10, -0.20])
            finally:
                del tokens, scores
                _close_reader(reader)

            written = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(written, manifest)
            self.assertEqual(written["format_version"], 2)
            self.assertEqual(written["config"]["ctot_dtype"], "uint64")
            self.assertEqual(written["config"]["score_family"], "stupid_backoff")
            self.assertEqual(written["config"]["stupid_backoff_alpha"], 0.4)
            self.assertEqual(
                written["config"]["source_sentinel_mapping"]["target_eos_token_id"],
                999,
            )
            self.assertEqual(written["orders"]["2"]["input_rows"], 4)
            self.assertEqual(written["orders"]["2"]["output_rows"], 3)
            self.assertEqual(written["orders"]["2"]["filtered_rows"], 1)
            expected_masses = np.power(
                10.0,
                np.asarray(
                    [[-0.11, -0.23], [-0.31, -math.inf], [-0.14, -0.26]],
                    dtype=F32,
                ).astype(np.float64),
            ).sum(axis=1)
            diagnostics = written["orders"]["2"]["retained_row_mass"]
            self.assertEqual(diagnostics["row_count"], 3)
            self.assertEqual(diagnostics["quantile_method"], "exact all rows")
            self.assertAlmostEqual(
                diagnostics["mean"], float(expected_masses.mean()), delta=0.01
            )
            self.assertAlmostEqual(
                diagnostics["min"], float(expected_masses.min()), delta=0.01
            )
            self.assertAlmostEqual(
                diagnostics["max"], float(expected_masses.max()), delta=0.01
            )
            self.assertEqual(
                written["source"]["manifest_sha256"],
                hashlib.sha256((source / "scores_meta.json").read_bytes()).hexdigest(),
            )
            self.assertEqual(
                written["output"]["sha256"], hashlib.sha256(output.read_bytes()).hexdigest()
            )
            self.assertEqual(written["output"]["bytes"], output.stat().st_size)

    def test_emit_orders_subset_leaves_empty_sections_and_reader_skips_them(self) -> None:
        with ram_temp_dir() as tmp:
            root = Path(tmp)
            source = root / "source"
            _write_synthetic_scored_rows(source)
            output = root / "artifact" / "forward_index_topk.bin"
            plan = plan_scored_rows(
                source,
                max_order=3,
                K=2,
                min_history_total=10,
                vocab_size=1000,
                sentinel_eos_token_id=999,
                chunk_rows=2,
                emit_orders=frozenset({1, 3}),
            )
            # Order 2 is validated and counted in the inputs but emits no rows.
            self.assertEqual(
                [(p.input_rows, p.output_rows) for p in plan.orders],
                [(1, 1), (4, 0), (4, 2)],
            )
            manifest = pack_scored_rows(
                plan,
                output_path=output,
                manifest_path=output.parent / "build_manifest.json",
            )
            self.assertEqual(manifest["config"]["emitted_orders"], [1, 3])
            self.assertEqual(manifest["orders"]["2"]["output_rows"], 0)

            reader = CountTable(str(output))
            reader._ensure_open()
            try:
                # The empty section exists in the header with zero rows.
                self.assertEqual(
                    reader.sections_by_order[2].prefixes_2d.shape[0], 0
                )
                # An order-3 context whose history survives resolves normally.
                # Copy the returned rows: they are views into the reader's
                # mmap, and holding them past close() trips a BufferError.
                tokens = np.array(_lookup_one(reader, [42, 2, 3])[0])
                np.testing.assert_array_equal(tokens, [541, 542])
                # A context that would have matched an order-2 row in the full
                # table now falls through the empty section to order 1: (9, 2)
                # has no order-3 row above threshold and no order-2 section.
                tokens = np.array(_lookup_one(reader, [9, 2])[0])
                np.testing.assert_array_equal(tokens, [11, 12])
            finally:
                _close_reader(reader)

            # The trainer reads through CountLookup.lookup_batch, a different
            # code path from lookup_one above; it clamps a searchsorted result
            # to n_pref - 1, which is -1 for an empty section, so the batch path
            # needs its own coverage.
            source = CountLookup(str(output.parent), K=2, N_max=3)
            try:
                ids, _ = source.lookup_batch([[42, 2, 3], [9, 2], [7]])
                np.testing.assert_array_equal(np.array(ids[0, 0]), [541, 542])
                # Falls through the empty order-2 section to the unigram row.
                np.testing.assert_array_equal(np.array(ids[1, 0]), [11, 12])
            finally:
                close = getattr(source, "close", None)
                if close is not None:
                    close()

            # emit_orders without order 1 is rejected: it is the backoff floor.
            with self.assertRaises(ValueError):
                plan_scored_rows(
                    source,
                    max_order=3,
                    K=2,
                    min_history_total=10,
                    vocab_size=1000,
                    sentinel_eos_token_id=999,
                    chunk_rows=2,
                    emit_orders=frozenset({3}),
                )

    def test_rejects_source_declaring_a_key_transform(self) -> None:
        with ram_temp_dir() as tmp:
            source = Path(tmp) / "source"
            _write_synthetic_scored_rows(source)
            meta_path = source / "scores_meta.json"
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            metadata["key_transform"] = {"type": "canonical_history_v1"}
            meta_path.write_text(json.dumps(metadata) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ScoredRowsFormatError, "key_transform"):
                plan_scored_rows(
                    source,
                    max_order=3,
                    K=2,
                    min_history_total=10,
                    vocab_size=1000,
                    sentinel_eos_token_id=999,
                    chunk_rows=2,
                )

    def test_logprob_bytes_one_writes_physical_uint8_and_loader_reads_it(self) -> None:
        with ram_temp_dir() as tmp:
            root = Path(tmp)
            source = root / "source"
            _write_synthetic_scored_rows(source)
            # Exercise the finite-mask contract: token ID 0 can be a real,
            # finite continuation as well as the token value used in a padded
            # slot. The packed mask, not the token ID, distinguishes them.
            prefixes = np.memmap(
                source / "order2.pref.bin", dtype=U32, mode="r", shape=(4, 1)
            )
            tokens = np.memmap(
                source / "order2.toks.bin", dtype=U32, mode="r+", shape=(4, 4)
            )
            finite_zero_row = int(np.flatnonzero(prefixes[:, 0] == 1)[0])
            tokens[finite_zero_row, 1] = 0
            tokens.flush()
            del prefixes, tokens
            output = root / "rounded" / "forward_index_topk.bin"
            manifest_path = output.parent / "build_manifest.json"
            plan = plan_scored_rows(
                source,
                max_order=3,
                K=2,
                min_history_total=10,
                vocab_size=1000,
                sentinel_eos_token_id=999,
                logprob_bytes=1,
                chunk_rows=2,
            )
            manifest = pack_scored_rows(
                plan,
                output_path=output,
                manifest_path=manifest_path,
            )

            reader = CountTable(str(output))
            reader._ensure_open()
            try:
                order2_source = np.asarray(
                    [[-0.11, -0.23], [-0.31, -math.inf], [-0.14, -0.26]],
                    dtype=F32,
                )
                # -0.31 is the minimum finite retained/cropped order-2 score.
                order2_low = float(order2_source[np.isfinite(order2_source)].min())
                section = reader.sections_by_order[2]
                self.assertEqual(section.topk_score_codes.dtype, np.uint8)
                np.testing.assert_array_equal(
                    section.logprobs_at(np.arange(3)),
                    _grid8(order2_source, order2_low),
                )
                self.assertTrue(
                    np.isneginf(section.logprobs_at(1)[1])
                )
                finite_tokens, finite_scores = _lookup_one(reader, [1])
                np.testing.assert_array_equal(finite_tokens, [101, 0])
                self.assertTrue(np.isfinite(finite_scores[1]))
            finally:
                del finite_tokens, finite_scores, section
                _close_reader(reader)

            source_loader = CountLookup(output.parent, K=2, N_max=3)
            ids, natural_log_scores = source_loader.lookup_batch([[1, 256], [2], [99]])
            ids, natural_log_scores = ids[:, 0], natural_log_scores[:, 0]
            try:
                np.testing.assert_array_equal(
                    ids,
                    np.asarray([[501, 502], [201, 0], [11, 12]], dtype=np.int32),
                )
                expected_log10 = np.asarray(
                    [
                        _grid8(
                            np.asarray([[-0.17, -0.29]], F32),
                            float(np.asarray([-0.19, -0.62], F32).min()),
                        )[0],
                        _grid8(
                            np.asarray([[-0.31, -math.inf]], F32),
                            order2_low,
                        )[0],
                        _grid8(
                            np.asarray([[-0.10, -0.20]], F32),
                            float(np.asarray([-0.10, -0.20], F32).min()),
                        )[0],
                    ],
                    dtype=F32,
                )
                np.testing.assert_array_equal(
                    natural_log_scores,
                    expected_log10 * np.float32(np.log(10.0)),
                )
            finally:
                del ids, natural_log_scores
                assert source_loader._index is not None
                _close_reader(source_loader._index)
                source_loader._index = None

            self.assertEqual(manifest["config"]["logprob_bytes"], 1)
            self.assertEqual(manifest["format_version"], 2)
            self.assertEqual(
                manifest["config"]["logprob_encoding"],
                "linear_uint8_per_order_with_rowwise_little_endian_finite_mask",
            )
            self.assertIn(
                "one uint8 linear-quantization code",
                manifest["config"]["logprob_storage_note"],
            )
            self.assertAlmostEqual(
                manifest["orders"]["2"]["logprob_quantization_min"], -0.31, places=6
            )
            self.assertEqual(
                manifest["orders"]["2"]["logprob_quantization_max"], 0.0
            )
            order2_bytes = manifest["orders"]["2"]["output_bytes"]
            self.assertEqual(order2_bytes["logprobs"], 3 * 2)
            self.assertEqual(order2_bytes["finite_mask"], 3)
            total_rows = sum(order.output_rows for order in plan.orders)
            score_slots = total_rows * plan.K
            finite_mask_bytes = total_rows * ((plan.K + 7) // 8)
            self.assertEqual(
                plan.projected_output_bytes - estimate_table_bytes(
                    [(order.order, order.output_rows) for order in plan.orders], K=2, logprob_bytes=1
                ),
                0,
            )
            del score_slots, finite_mask_bytes
            self.assertEqual(output.stat().st_size, plan.projected_output_bytes)
            order_rows = [
                (order.order, order.output_rows) for order in plan.orders
            ]
            self.assertEqual(
                estimate_table_bytes(order_rows, K=2, logprob_bytes=1),
                plan.projected_output_bytes,
            )

            with output.open("rb") as handle:
                magic, version, K, n_orders, vocab_size = struct.unpack(
                    "<4sIIII", handle.read(20)
                )
                self.assertEqual((magic, version, K, n_orders, vocab_size), (
                    b"COUN",
                    2,
                    2,
                    3,
                    1000,
                ))
                handle.seek(64 + 64 + 40)
                encoding, score_min, score_max, finite_mask_offset = struct.unpack(
                    "<IffQ", handle.read(20)
                )
                self.assertEqual(encoding, 1)
                self.assertAlmostEqual(score_min, order2_low, places=6)
                self.assertEqual(score_max, 0.0)
                self.assertGreater(finite_mask_offset, 0)

    def test_malformed_dense_file_size_fails_before_output(self) -> None:
        with ram_temp_dir() as tmp:
            source = Path(tmp) / "source"
            _write_synthetic_scored_rows(source)
            tokens_path = source / "order2.toks.bin"
            with tokens_path.open("r+b") as handle:
                handle.truncate(tokens_path.stat().st_size - 4)
            with self.assertRaisesRegex(
                ScoredRowsFormatError, r"order 2 tokens has .* expected exactly"
            ):
                plan_scored_rows(
                    source,
                    max_order=3,
                    K=2,
                    min_history_total=10,
                    vocab_size=1000,
                    sentinel_eos_token_id=999,
                )

    def test_raw_boundary_candidates_map_to_eos_and_padding_stays_empty(self) -> None:
        with ram_temp_dir() as tmp:
            root = Path(tmp)
            source = root / "source"
            _write_synthetic_scored_rows(source)
            prefixes = np.memmap(
                source / "order2.pref.bin", dtype=U32, mode="r", shape=(4, 1)
            )
            tokens = np.memmap(
                source / "order2.toks.bin", dtype=U32, mode="r+", shape=(4, 4)
            )
            scores = np.memmap(
                source / "order2.lps.bin", dtype=F32, mode="r+", shape=(4, 4)
            )
            doc_row = int(np.flatnonzero(prefixes[:, 0] == 1)[0])
            eof_row = int(np.flatnonzero(prefixes[:, 0] == 256)[0])
            tokens[doc_row] = [0xFFFFFFFF, 101, 0, 104]
            scores[doc_row] = [-0.11, -0.23, -math.inf, -0.47]
            tokens[eof_row, 0] = 0xFFFFFFFE
            tokens.flush()
            scores.flush()
            del prefixes, tokens, scores

            output = root / "artifact" / "forward_index_topk.bin"
            manifest_path = output.parent / "build_manifest.json"
            plan = plan_scored_rows(
                source,
                max_order=2,
                K=4,
                min_history_total=10,
                vocab_size=1000,
                sentinel_eos_token_id=999,
                chunk_rows=1,
            )
            manifest = pack_scored_rows(
                plan,
                output_path=output,
                manifest_path=manifest_path,
            )

            reader = CountTable(str(output))
            reader._ensure_open()
            try:
                doc_tokens, doc_scores = _lookup_one(reader, [1])
                np.testing.assert_array_equal(doc_tokens, [999, 101, 0, 104])
                _assert_scores_close(doc_scores, [-0.11, -0.23, -math.inf, -0.47])
                eof_tokens, eof_scores = _lookup_one(reader, [256])
                np.testing.assert_array_equal(eof_tokens, [999, 602, 603, 604])
                _assert_scores_close(eof_scores, [-0.14, -0.26, -0.38, -0.50])
            finally:
                del doc_tokens, doc_scores, eof_tokens, eof_scores
                _close_reader(reader)

            counts = manifest["orders"]["2"][
                "source_sentinel_candidate_slots_remapped"
            ]
            self.assertEqual(counts[str(0xFFFFFFFF)], 1)
            self.assertEqual(counts[str(0xFFFFFFFE)], 1)
            self.assertEqual(counts["total"], 2)

    def test_default_and_invalid_sentinel_eos_configuration(self) -> None:
        with ram_temp_dir() as tmp:
            source = Path(tmp) / "source"
            _write_synthetic_scored_rows(source)
            plan = plan_scored_rows(
                source,
                max_order=2,
                K=2,
                min_history_total=10,
                vocab_size=100278,
            )
            self.assertEqual(plan.sentinel_eos_token_id, 100257)
            for invalid in (-1, 1000, True, 1.5):
                with self.subTest(invalid=invalid), self.assertRaisesRegex(
                    ScoredRowsFormatError,
                    r"sentinel_eos_token_id must be an integer model token",
                ):
                    plan_scored_rows(
                        source,
                        max_order=2,
                        K=2,
                        min_history_total=10,
                        vocab_size=1000,
                        sentinel_eos_token_id=invalid,
                    )

    def test_other_oov_candidates_and_raw_sentinel_prefixes_still_fail(self) -> None:
        for case in ("candidate", "prefix"):
            with self.subTest(case=case), ram_temp_dir() as tmp:
                root = Path(tmp)
                source = root / "source"
                _write_synthetic_scored_rows(source)
                prefixes = np.memmap(
                    source / "order2.pref.bin", dtype=U32, mode="r+", shape=(4, 1)
                )
                row = int(np.flatnonzero(prefixes[:, 0] == 1)[0])
                if case == "candidate":
                    tokens = np.memmap(
                        source / "order2.toks.bin",
                        dtype=U32,
                        mode="r+",
                        shape=(4, 4),
                    )
                    tokens[row, 0] = 0xFFFFFFFD
                    tokens.flush()
                    del tokens
                    expected = r"token id 4294967293 outside vocab_size"
                else:
                    prefixes[row, 0] = 0xFFFFFFFF
                    prefixes.flush()
                    expected = r"prefix token id 4294967295 outside vocab_size"
                del prefixes

                plan = plan_scored_rows(
                    source,
                    max_order=2,
                    K=2,
                    min_history_total=10,
                    vocab_size=1000,
                    sentinel_eos_token_id=999,
                    chunk_rows=2,
                )
                with self.assertRaisesRegex(ScoredRowsFormatError, expected):
                    pack_scored_rows(
                        plan,
                        output_path=root / "artifact" / "forward_index_topk.bin",
                        manifest_path=root / "artifact" / "build_manifest.json",
                    )

    def test_failed_build_preserves_pid_qualified_partial(self) -> None:
        with ram_temp_dir() as tmp:
            root = Path(tmp)
            source = root / "source"
            _write_synthetic_scored_rows(source)
            tokens = np.memmap(
                source / "order2.toks.bin",
                dtype=U32,
                mode="r+",
                shape=(4, 4),
            )
            tokens[0, 0] = 0xFFFFFFFD
            tokens.flush()
            del tokens

            plan = plan_scored_rows(
                source,
                max_order=2,
                K=2,
                min_history_total=10,
                vocab_size=1000,
                sentinel_eos_token_id=999,
                chunk_rows=2,
            )
            artifact_dir = root / "artifact"
            output = artifact_dir / "forward_index_topk.bin"
            manifest_path = artifact_dir / "build_manifest.json"
            with self.assertRaisesRegex(
                ScoredRowsFormatError, r"token id 4294967293 outside vocab_size"
            ):
                pack_scored_rows(
                    plan,
                    output_path=output,
                    manifest_path=manifest_path,
                )

            self.assertFalse(output.exists())
            self.assertFalse(manifest_path.exists())
            partials = list(artifact_dir.glob(".forward_index_topk.bin.*.partial"))
            self.assertEqual(len(partials), 1)
            self.assertEqual(partials[0].stat().st_size, plan.projected_output_bytes)


if __name__ == "__main__":
    unittest.main()


class SimpleAndOptimizedPackersAgree(unittest.TestCase):
    """The plain packer and the optimized one write byte-identical tables and
    manifests (apart from the timestamp and output path) for the same inputs."""

    @staticmethod
    def _manifest_without_volatile(path: Path) -> dict:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest.pop("generated_at_utc")
        manifest["output"].pop("path")
        return manifest

    def _assert_agree(self, source: Path, root: Path, **config) -> None:
        import argparse as _argparse

        from thecount.tables import pack as plain

        root = Path(root)
        eos = config.pop("sentinel_eos_token_id", 999)
        outputs = {}
        for impl in ("simple", "optimized"):
            out_dir = root / f"out-{impl}-{len(list(root.iterdir()))}"
            plain.run(_argparse.Namespace(
                input_dir=Path(source), output_dir=out_dir, output_file=None, manifest_path=None,
                sentinel_eos_token_id=eos, ctot_dtype="uint64", chunk_rows=config.get("chunk_rows", 2),
                orders=config.get("orders", ""), implementation=impl, dry_run=False, force=False,
                max_order=config["max_order"], K=config["K"], min_history_total=config["min_history_total"],
                vocab_size=config["vocab_size"]))
            table = (out_dir / "forward_index_topk.bin").read_bytes()
            manifest = self._manifest_without_volatile(out_dir / "build_manifest.json")
            outputs[impl] = (table, manifest)
        self.assertEqual(outputs["simple"][0], outputs["optimized"][0], "table bytes differ")
        self.assertEqual(outputs["simple"][1], outputs["optimized"][1], "manifests differ")

    def test_synthetic_rows_across_configurations(self) -> None:
        with ram_temp_dir() as root:
            root = Path(root)
            source = root / "source"
            _write_synthetic_scored_rows(source)
            # Inject raw document-boundary candidates so the sentinel remap runs.
            tokens = np.fromfile(source / "order2.toks.bin", dtype=U32).reshape(-1, 4)
            tokens[0, 1] = 0xFFFFFFFF
            tokens[1, 2] = 0xFFFFFFFE
            tokens.tofile(source / "order2.toks.bin")
            for config in (
                dict(max_order=3, K=2, min_history_total=10, vocab_size=1000),
                dict(max_order=3, K=4, min_history_total=1, vocab_size=1000),
                dict(max_order=3, K=3, min_history_total=5, vocab_size=1000, orders="3,1"),
                dict(max_order=2, K=1, min_history_total=20, vocab_size=1000, chunk_rows=1),
            ):
                self._assert_agree(source, root, **config)

    def test_scorer_output_end_to_end(self) -> None:
        import argparse as _argparse

        from tests.test_stupid_backoff_scoring import _write_parts, slices_from_tables, synth_tables
        from thecount.tables import stupid_backoff_scoring as sbs

        with ram_temp_dir() as root:
            root = Path(root)
            tables = synth_tables(seed=31, n_orders=4, n_tokens=5000)
            slices = slices_from_tables(tables)
            parts = _write_parts(root, slices, sorted(slices), 2)
            scores = root / "scores"
            sbs.run(_argparse.Namespace(parts=parts, out_dir=str(scores), min_order=1, max_order=4,
                                        tau=2, alpha=0.4, kmax=8, workers=1, chunk_mb=64,
                                        resume=False, implementation="simple"))
            for config in (
                dict(max_order=4, K=8, min_history_total=2, vocab_size=64, sentinel_eos_token_id=1),
                dict(max_order=4, K=3, min_history_total=6, vocab_size=64, sentinel_eos_token_id=1),
                dict(max_order=3, K=5, min_history_total=2, vocab_size=64, sentinel_eos_token_id=1, orders="2,1"),
            ):
                self._assert_agree(scores, root, **config)
