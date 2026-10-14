"""Compare history shifts and row selection with LongCat's released implementation."""

import ast
import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import pytest
import torch
from torch import nn

from olmo_core.nn.transformer import LearnedNgramEmbedding


@pytest.fixture(scope="module")
def released_embedding_class():
    path = Path(__file__).parent / "references/longcat_flash_lite/modeling_longcat_ngram.py.txt"
    source = path.read_bytes()
    assert hashlib.sha256(source).hexdigest() == (
        "d75c6bfc20a1c8045b3958eb2e2167d3e80bb110e8475d64390abe8eb160968f"
    )
    # Execute the unchanged released class, without importing its unrelated
    # Transformers backbone. No methods or formulas are replaced by test code.
    tree = ast.parse(source, filename=str(path))
    definition = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "NgramEmbedding"
    )
    namespace = {
        "torch": torch,
        "nn": nn,
        "Dict": Dict,
        "List": List,
        "Optional": Optional,
        "Tuple": Tuple,
    }
    exec(compile(ast.Module(body=[definition], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["NgramEmbedding"]


def _compare_with_release(released_class, tokens, *, vocab_size, eos_token_id, order, tables):
    d_model = 2 * (order - 1) * tables
    config = SimpleNamespace(
        vocab_size=vocab_size,
        hidden_size=d_model,
        ngram_vocab_size_ratio=32 / vocab_size,
        emb_split_num=tables,
        emb_neighbor_num=order,
        eos_token_id=eos_token_id,
        pad_token_id=None,
    )
    reference = released_class(config, nn.Embedding(vocab_size, d_model))
    sizes = tuple(table.num_embeddings for table in reference.embedders)
    actual = LearnedNgramEmbedding(
        d_model=d_model,
        vocab_size=vocab_size,
        max_order=order,
        tables_per_order=tables,
        table_sizes=sizes,
        eos_token_id=eos_token_id,
    )
    input_ids = torch.tensor(tokens, dtype=torch.long)
    shifted = actual._shifted_token_ids(input_ids)
    torch.testing.assert_close(shifted[0], input_ids, rtol=0, atol=0)
    for lag in range(1, order):
        expected = reference._shift_right_ignore_eos(input_ids, lag, eos_token_id=eos_token_id)
        torch.testing.assert_close(shifted[lag], expected, rtol=0, atol=0)

    # Observe the actual lookup arguments passed by the released forward method.
    # Its shifts, power calculation, table ordering, and final modulus all run.
    captured = []
    hooks = [
        table.register_forward_pre_hook(lambda _module, args: captured.append(args[0].clone()))
        for table in reference.embedders
    ]
    try:
        with torch.no_grad():
            reference(input_ids)
    finally:
        for hook in hooks:
            hook.remove()
    assert len(captured) == len(sizes)
    # The release uses separate embeddings; our implementation concatenates them.
    # Derive offsets from the reference sizes, independently of our stored offsets.
    expected_rows = torch.stack(
        [rows + sum(sizes[:index]) for index, rows in enumerate(captured)], dim=-1
    )
    torch.testing.assert_close(actual.ngram_rows(input_ids), expected_rows, rtol=0, atol=0)


@pytest.mark.parametrize("order,tables", [(2, 1), (3, 2), (4, 4), (8, 2)])
@pytest.mark.parametrize(
    "tokens",
    [
        pytest.param([[1, 2, 16, 3, 4, 5]], id="two-documents"),
        pytest.param([[16, 1, 2, 3, 4, 16]], id="leading-and-trailing-eos"),
        pytest.param([[1, 16, 16, 2, 3, 16]], id="consecutive-eos"),
        pytest.param([[16, 16, 16, 16]], id="only-eos"),
        pytest.param([[0, 1, 2, 0, 3, 4, 5, 6, 7]], id="no-eos-and-real-token-zero"),
        pytest.param([[7]], id="one-token"),
        pytest.param([[16]], id="one-eos"),
        pytest.param(
            [[1, 2, 16, 3, 4, 5], [1, 16, 2, 3, 16, 4]],
            id="different-boundaries-per-batch-row",
        ),
    ],
)
def test_longcat_histories_and_rows_match_release(
    released_embedding_class, tokens, order, tables
):
    _compare_with_release(
        released_embedding_class,
        tokens,
        vocab_size=17,
        eos_token_id=16,  # Deliberately differs from the released helper's default.
        order=order,
        tables=tables,
    )


@pytest.mark.parametrize("order", [4, 8])
def test_longcat_large_token_ids_match_release(released_embedding_class, order):
    _compare_with_release(
        released_embedding_class,
        [[100350, 99999, 70000, 20000, 8, 19, 34, 55, 100351, 12, 13]],
        vocab_size=100352,
        eos_token_id=100351,
        order=order,
        tables=2,
    )
