"""Check COUNT's stored scores, selected orders, and causal input contexts."""

import struct

import numpy as np
import pytest

from olmo_core.data.count_table import CountLookup


def _write_table(path, *, sections=None):
    """Write a small COUNT table without depending on production packing code."""
    if sections is None:
        sections = [
            (1, [[]], [[10, 11, 0]], [[-1.5, -0.5, -np.inf]]),
            (
                2,
                [[9], [12]],
                [[20, 21, 22], [23, 24, 25]],
                [[-1.0, -0.5, -1.5], [-np.inf, -np.inf, -np.inf]],
            ),
            (3, [[7, 9]], [[30, 31, 0]], [[-0.25, -1.25, -np.inf]]),
            (4, [[5, 7, 9]], [[40, 41, 42]], [[-np.inf, -np.inf, -np.inf]]),
        ]
    width = len(sections[0][2][0])
    header = bytearray(64)
    struct.pack_into("<4sIIII", header, 0, b"COUN", 2, width, len(sections), 64)
    section_headers = bytearray()
    payload = bytearray()
    offset = 64 + 64 * len(sections)
    for order, prefixes, tokens, scores in sections:
        count = len(prefixes)
        prefix_bytes = np.asarray(prefixes, dtype="<u4").reshape(count, order - 1).tobytes()
        token_bytes = np.asarray(tokens, dtype="<u4").reshape(count, width).tobytes()
        values = np.asarray(scores, dtype="<f4").reshape(count, width)
        prefix_offset = offset
        payload.extend(prefix_bytes)
        offset += len(prefix_bytes)
        token_offset = offset
        payload.extend(token_bytes)
        offset += len(token_bytes)
        score_offset = offset
        finite = np.isfinite(values)
        # Calibration is fixed at [-2, 0]; separate mask bits represent
        # absent candidates, including when their physical code is zero.
        codes = np.rint((np.where(finite, values, -2.0) + 2.0) * 127.5)
        score_bytes = codes.astype(np.uint8).tobytes()
        payload.extend(score_bytes)
        offset += len(score_bytes)
        section_header = bytearray(64)
        struct.pack_into(
            "<IIQQQQ",
            section_header,
            0,
            order,
            0,
            count,
            prefix_offset,
            token_offset,
            score_offset,
        )
        struct.pack_into("<IffQ", section_header, 40, 1, -2.0, 0.0, offset)
        mask_bytes = np.packbits(finite, axis=1, bitorder="little").tobytes()
        payload.extend(mask_bytes)
        offset += len(mask_bytes)
        section_headers.extend(section_header)
    path.write_bytes(header + section_headers + payload)


def test_selects_highest_matching_orders_and_preserves_stored_scores(tmp_path):
    _write_table(tmp_path / "forward_index_topk.bin")
    source = CountLookup(tmp_path, K=3, N_max=4, num_orders=3)
    ids, scores = source.lookup_batch([[5, 7, 9], [8, 9], [6, 12]])

    # The matching order-4 row and the order-2 row for history [12] have no
    # finite scores, so neither consumes a selected-order slot.
    np.testing.assert_array_equal(ids[:, :, 0], [[30, 20, 10], [20, 10, 0], [10, 0, 0]])
    assert ids.shape == scores.shape == (3, 3, 3)
    assert ids.dtype == np.int32
    assert scores.dtype == np.float32
    # Codes 223, 128, and 64 under the fixed [-2, 0] calibration.
    expected = np.asarray([-2 + 2 * code / 255 for code in (223, 128, 64)], dtype=np.float32)
    np.testing.assert_array_equal(scores[0, :, 0], expected * np.float32(np.log(10.0)))
    np.testing.assert_array_equal(scores[1, :2], scores[0, 1:])
    assert np.isneginf(scores[1, 2]).all()
    assert np.isneginf(scores[2, 1:]).all()
    assert np.isneginf(scores[0, 0, 2])


def test_requested_k_selects_scores_from_unsorted_rows(tmp_path):
    _write_table(tmp_path / "forward_index_topk.bin")
    source = CountLookup(tmp_path, K=1, N_max=2, num_orders=2)
    ids, scores = source.lookup_batch([[5, 7, 9]])
    np.testing.assert_array_equal(ids, [[[21], [11]]])
    stored = np.float32(-2 + 2 * 191 / 255)
    np.testing.assert_allclose(scores, stored * np.float32(np.log(10.0)), rtol=1e-7)


def test_missing_and_rowless_orders_leave_zero_slots(tmp_path):
    _write_table(
        tmp_path / "forward_index_topk.bin",
        sections=[
            (1, [[]], [[10, 0]], [[-0.5, -np.inf]]),
            (2, [[9]], [[20, 0]], [[-0.25, -np.inf]]),
            (4, [], [], []),
        ],
    )
    ids, scores = CountLookup(tmp_path, K=2, N_max=4, num_orders=4).lookup_batch([[5, 7, 9]])
    np.testing.assert_array_equal(ids[0], [[20, 0], [10, 0], [0, 0], [0, 0]])
    assert np.isneginf(scores[0, 2:]).all()


def test_all_absent_rows_produce_only_zero_slots(tmp_path):
    _write_table(
        tmp_path / "forward_index_topk.bin",
        sections=[(1, [[]], [[10, 11]], [[-np.inf, -np.inf]])],
    )
    ids, scores = CountLookup(tmp_path, K=1, N_max=1).lookup_batch([[9]])
    np.testing.assert_array_equal(ids, [[[0]]])
    assert np.isneginf(scores).all()


def test_unigram_limit_does_not_use_the_supplied_history(tmp_path):
    _write_table(tmp_path / "forward_index_topk.bin")
    source = CountLookup(tmp_path, K=1, N_max=1)
    ids, scores = source.lookup_tokens(np.asarray([[5, 7, 9], [9, 7, 5]]))
    assert ids.shape == scores.shape == (2, 3, 1, 1)
    np.testing.assert_array_equal(ids, np.full((2, 3, 1, 1), 11))
    expected = source.lookup_batch([[], [], [], [], [], []])
    np.testing.assert_array_equal(scores.reshape(6, 1, 1), expected[1])


def test_shared_token_lookup_is_causal_and_keeps_sequences_separate(tmp_path):
    _write_table(tmp_path / "forward_index_topk.bin")
    source = CountLookup(tmp_path, K=3, N_max=3, num_orders=2)
    tokens = np.asarray([[5, 7, 9], [9, 12, 9]], dtype=np.uint32)
    ids, scores = source.lookup_tokens(tokens)
    direct_ids, direct_scores = source.lookup_batch([[5], [5, 7], [7, 9], [9], [9, 12], [12, 9]])
    np.testing.assert_array_equal(ids, direct_ids.reshape(2, 3, 2, 3))
    np.testing.assert_array_equal(scores, direct_scores.reshape(2, 3, 2, 3))
    np.testing.assert_array_equal(ids[:, :, 0, 0], [[10, 10, 30], [20, 10, 20]])
    single_ids, single_scores = source.lookup_tokens(tokens[0])
    np.testing.assert_array_equal(single_ids, ids[0])
    np.testing.assert_array_equal(single_scores, scores[0])
    changed = tokens.copy()
    changed[:, 2] = 12
    changed_ids, changed_scores = source.lookup_tokens(changed)
    np.testing.assert_array_equal(changed_ids[:, :2], ids[:, :2])
    np.testing.assert_array_equal(changed_scores[:, :2], scores[:, :2])


def test_contexts_do_not_reset_at_special_tokens(tmp_path):
    # A special token such as an EOS is still an ordinary keyed-table token.
    _write_table(
        tmp_path / "forward_index_topk.bin",
        sections=[
            (1, [[]], [[10]], [[-1.0]]),
            (3, [[63, 9]], [[20]], [[-0.5]]),
        ],
    )
    source = CountLookup(tmp_path, K=1, N_max=3)
    ids, _ = source.lookup_tokens(np.asarray([63, 9]))
    np.testing.assert_array_equal(ids[:, 0, 0], [10, 20])


def test_uint8_mask_spans_multiple_bytes(tmp_path):
    values = [[-2.0, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -1.0, 0.0]]
    _write_table(
        tmp_path / "forward_index_topk.bin",
        sections=[(1, [[]], [list(range(9))], values)],
    )
    _, scores = CountLookup(tmp_path, K=9, N_max=1).lookup_batch([[]])
    np.testing.assert_array_equal(
        np.isfinite(scores[0, 0]), [[True] + [False] * 6 + [True, True]][0]
    )
    np.testing.assert_allclose(
        scores[0, 0, [0, 7, 8]], np.asarray([-2.0, -2 + 256 / 255, 0.0]) * np.log(10.0), rtol=1e-7
    )


@pytest.mark.parametrize("num_orders", [0, -1, 5, 1.5, True])
def test_num_orders_must_be_an_integer_within_the_order_limit(tmp_path, num_orders):
    with pytest.raises(ValueError, match="num_orders"):
        CountLookup(tmp_path, K=1, N_max=4, num_orders=num_orders)


def test_table_width_and_format_are_checked(tmp_path):
    _write_table(tmp_path / "forward_index_topk.bin")
    with pytest.raises(ValueError, match="K=3"):
        CountLookup(tmp_path, K=4, N_max=2).lookup_batch([[]])
    with pytest.raises(ValueError, match="max order 4"):
        CountLookup(tmp_path, K=3, N_max=5).lookup_batch([[]])
    (tmp_path / "ngram_table_meta.json").write_text("{}")
    with pytest.raises(ValueError, match="canonical-history"):
        CountLookup(tmp_path, K=3, N_max=2)


def test_rejects_incompatible_table_header(tmp_path):
    path = tmp_path / "forward_index_topk.bin"
    _write_table(path)
    data = bytearray(path.read_bytes())
    struct.pack_into("<I", data, 4, 1)
    path.write_bytes(data)
    with pytest.raises(ValueError, match="unsupported COUNT n-gram table format"):
        CountLookup(tmp_path, K=3, N_max=2).lookup_batch([[]])


@pytest.mark.parametrize("shape", [(), (1, 2, 3)])
def test_token_input_requires_one_or_two_dimensions(tmp_path, shape):
    _write_table(tmp_path / "forward_index_topk.bin")
    source = CountLookup(tmp_path, K=3, N_max=2)
    with pytest.raises(ValueError, match="input_ids"):
        source.lookup_tokens(np.zeros(shape))


@pytest.mark.parametrize("shape", [(0,), (2, 0), (0, 3)])
def test_empty_token_inputs_keep_the_requested_shape(tmp_path, shape):
    _write_table(tmp_path / "forward_index_topk.bin")
    ids, scores = CountLookup(tmp_path, K=1, N_max=3, num_orders=2).lookup_tokens(
        np.zeros(shape)
    )
    assert ids.shape == scores.shape == (*shape, 2, 1)


@pytest.mark.parametrize("mix_lambda", [0.0, 1.0])
def test_mixture_endpoints_preserve_single_table_rows(tmp_path, mix_lambda):
    table_a, table_b = tmp_path / "a", tmp_path / "b"
    table_a.mkdir()
    table_b.mkdir()
    _write_table(table_a / "forward_index_topk.bin")
    _write_table(
        table_b / "forward_index_topk.bin",
        sections=[
            (1, [[]], [[14, 15, 0]], [[-1.0, -0.5, -np.inf]]),
            (3, [[7, 9]], [[34, 35, 0]], [[-0.75, -0.25, -np.inf]]),
        ],
    )
    kwargs = dict(K=1, N_max=3, num_orders=3)
    mixed = CountLookup(table_a, mix_table_dir=table_b, mix_lambda=mix_lambda, **kwargs)
    single = CountLookup(table_a if mix_lambda == 0 else table_b, **kwargs)
    actual = mixed.lookup_tokens(np.asarray([7, 9, 12]))
    expected = single.lookup_tokens(np.asarray([7, 9, 12]))
    for actual_field, expected_field in zip(actual, expected):
        np.testing.assert_array_equal(actual_field, expected_field)


def test_mixture_sums_duplicate_tokens_before_selecting_k(tmp_path):
    table_a, table_b = tmp_path / "a", tmp_path / "b"
    table_a.mkdir()
    table_b.mkdir()
    _write_table(
        table_a / "forward_index_topk.bin",
        sections=[(1, [[]], [[4, 8, 4]], [[-1.0, -2.0, -1.0]])],
    )
    _write_table(
        table_b / "forward_index_topk.bin",
        sections=[(1, [[]], [[4, 6, 9]], [[-1.0, -1.0, -2.0]])],
    )
    source = CountLookup(table_a, K=2, N_max=1, mix_table_dir=table_b, mix_lambda=0.25)
    ids, scores = source.lookup_batch([[]])
    np.testing.assert_array_equal(ids, [[[4, 6]]])
    # A stored log score of -1 rounds to code 128 under [-2, 0].
    weight = 10 ** (-2 + 2 * 128 / 255)
    np.testing.assert_allclose(np.exp(scores), [[[1.75 * weight, 0.25 * weight]]], rtol=2e-7)


def test_mixture_aligns_exact_orders_before_selecting_them(tmp_path):
    table_a, table_b = tmp_path / "a", tmp_path / "b"
    table_a.mkdir()
    table_b.mkdir()
    _write_table(
        table_a / "forward_index_topk.bin",
        sections=[
            (1, [[]], [[10]], [[-1.0]]),
            (3, [[7, 9]], [[30]], [[-1.0]]),
        ],
    )
    _write_table(
        table_b / "forward_index_topk.bin",
        sections=[
            (1, [[]], [[10]], [[-1.0]]),
            (2, [[9]], [[20]], [[-1.0]]),
            (3, [], [], []),
        ],
    )
    source = CountLookup(
        table_a, K=1, N_max=3, num_orders=3, mix_table_dir=table_b, mix_lambda=0.25
    )
    ids, scores = source.lookup_batch([[7, 9]])
    np.testing.assert_array_equal(ids, [[[30], [20], [10]]])
    weight = 10 ** (-2 + 2 * 128 / 255)
    np.testing.assert_allclose(
        np.exp(scores), [[[0.75 * weight], [0.25 * weight], [weight]]], rtol=2e-7
    )


@pytest.mark.parametrize("mix_lambda", [-0.1, 1.1, float("nan")])
def test_mixture_weight_must_be_within_zero_and_one(tmp_path, mix_lambda):
    with pytest.raises(ValueError, match="mix_lambda"):
        CountLookup(tmp_path, mix_lambda=mix_lambda)


def _write_random_table(path, rng, *, width, max_order, rows_per_order, vocab):
    """Write a table with sorted random histories and random stored rows."""
    sections = []
    for order in range(1, max_order + 1):
        if order == 1:
            prefixes = [[]]
        else:
            keys = rng.integers(0, vocab, size=(rows_per_order, order - 1), dtype=np.uint32)
            struct_dtype = np.dtype([(f"f{column}", "<u4") for column in range(order - 1)])
            keys = np.unique(keys.view(struct_dtype).reshape(-1))  # sorted, distinct
            prefixes = keys.view(np.uint32).reshape(-1, order - 1).tolist()
        count = len(prefixes)
        tokens = rng.integers(0, vocab, size=(count, width)).tolist()
        scores = rng.uniform(-2.0, 0.0, size=(count, width))
        scores[rng.random((count, width)) < 0.3] = -np.inf
        scores[rng.random(count) < 0.1] = -np.inf  # rows with no finite candidate
        sections.append((order, prefixes, tokens, scores.tolist()))
    _write_table(path, sections=sections)


def _reference_lookup_batch(source, contexts):
    """The per-context lookup the batched search replaced, kept as an oracle."""
    shape = (len(contexts), source.num_orders, source.K)
    token_ids = np.zeros(shape, dtype=np.int32)
    log_scores = np.full(shape, -np.inf, dtype=np.float32)
    max_prefix_length = source.N_max - 1
    for row, context in enumerate(contexts):
        limited_context = context[-max_prefix_length:] if max_prefix_length else ()
        slot = 0
        for order in range(min(source.N_max, len(limited_context) + 1), 0, -1):
            selected = source._lookup_order(limited_context, order)
            if selected is None or not np.isfinite(selected[1]).any():
                continue
            token_ids[row, slot], log_scores[row, slot] = selected
            slot += 1
            if slot == source.num_orders:
                break
    return token_ids, log_scores


@pytest.mark.parametrize("K,N_max,num_orders", [(4, 5, 5), (2, 5, 3), (4, 3, 1), (1, 1, 1)])
def test_batched_lookup_matches_per_context_reference(tmp_path, K, N_max, num_orders):
    rng = np.random.default_rng(20260924)
    _write_random_table(
        tmp_path / "forward_index_topk.bin", rng, width=4, max_order=5, rows_per_order=400, vocab=12
    )
    source = CountLookup(tmp_path, K=K, N_max=N_max, num_orders=num_orders)
    tokens = rng.integers(0, 12, size=(3, 40), dtype=np.uint32)
    ids, scores = source.lookup_tokens(tokens)
    width = N_max - 1
    contexts = [
        sequence[max(0, position + 1 - width) : position + 1] if width else sequence[:0]
        for sequence in tokens
        for position in range(tokens.shape[1])
    ]
    expected_ids, expected_scores = _reference_lookup_batch(source, contexts)
    np.testing.assert_array_equal(ids.reshape(-1, num_orders, K), expected_ids)
    np.testing.assert_array_equal(scores.reshape(-1, num_orders, K), expected_scores)
    assert np.isfinite(scores).any(), "the random table should produce some matches"
    if N_max > 1:
        assert np.isfinite(scores[:, :, 1:]).any() or num_orders == 1
    # Ragged contexts through lookup_batch take the same batched path.
    ragged = [[], [3], [1, 2, 3, 4, 5, 6, 7], list(tokens[0, :N_max])]
    ragged_ids, ragged_scores = source.lookup_batch(ragged)
    expected_ids, expected_scores = _reference_lookup_batch(source, ragged)
    np.testing.assert_array_equal(ragged_ids, expected_ids)
    np.testing.assert_array_equal(ragged_scores, expected_scores)


def test_interpolated_tables_still_use_the_per_context_path(tmp_path):
    rng = np.random.default_rng(7)
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    _write_random_table(first / "forward_index_topk.bin", rng, width=3, max_order=3, rows_per_order=50, vocab=6)
    _write_random_table(second / "forward_index_topk.bin", rng, width=3, max_order=3, rows_per_order=50, vocab=6)
    mixed = CountLookup(first, K=3, N_max=3, num_orders=2, mix_table_dir=second, mix_lambda=0.5)
    tokens = rng.integers(0, 6, size=(2, 9), dtype=np.uint32)
    ids, scores = mixed.lookup_tokens(tokens)
    contexts = [seq[max(0, i - 1) : i + 1] for seq in tokens for i in range(9)]
    expected_ids, expected_scores = mixed.lookup_batch(contexts)
    np.testing.assert_array_equal(ids.reshape(-1, 2, 3), expected_ids)
    np.testing.assert_array_equal(scores.reshape(-1, 2, 3), expected_scores)
    # The endpoints of the interpolation reduce to the batched single-table lookup.
    for lam, table in ((0.0, first), (1.0, second)):
        endpoint = CountLookup(first, K=3, N_max=3, num_orders=2, mix_table_dir=second, mix_lambda=lam)
        single = CountLookup(table, K=3, N_max=3, num_orders=2)
        np.testing.assert_array_equal(endpoint.lookup_tokens(tokens)[1], single.lookup_tokens(tokens)[1])
