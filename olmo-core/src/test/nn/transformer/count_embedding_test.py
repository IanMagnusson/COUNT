"""Check the COUNT and learned n-gram equations against independent scalar formulas."""

import math

import pytest
import torch

from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.attention import AttentionConfig
from olmo_core.nn.feed_forward import FeedForwardConfig
from olmo_core.nn.layer_norm import LayerNormConfig
from olmo_core.nn.lm_head import LMHeadConfig
from olmo_core.nn.transformer import (
    CountNgramEmbedding,
    LearnedNgramEmbedding,
    TransformerBlockConfig,
    TransformerConfig,
)
from olmo_core.testing import GPU_MARKS

DEVICES = ["cpu", pytest.param("cuda", marks=GPU_MARKS)]


def _amplify(value, gain, bias):
    centered = value - value.mean(dim=-1, keepdim=True)
    normalized = centered / (centered.square().mean(dim=-1, keepdim=True) + 1e-5).sqrt()
    return (normalized * gain + bias) * math.sqrt(value.shape[-1])


def _compare_gradients(actual, expected, parameters, reference_parameters):
    objective = torch.randn_like(actual)
    actual_gradients = torch.autograd.grad((actual * objective).sum(), parameters)
    expected_gradients = torch.autograd.grad((expected * objective).sum(), reference_parameters)
    torch.testing.assert_close(actual, expected)
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(actual_gradient, expected_gradient)
    return actual_gradients


@pytest.mark.parametrize("device", DEVICES)
def test_count_equation_and_tied_head_gradients(device):
    torch.manual_seed(13)
    count = CountNgramEmbedding(d_model=4, num_orders=3, dtype=torch.float64, init_device=device)
    base = torch.randn(1, 2, 4, device=device, dtype=torch.float64, requires_grad=True)
    head = torch.randn(7, 4, device=device, dtype=torch.float64, requires_grad=True)
    ids = torch.tensor([[[[1, 1], [3, 2], [0, 0]], [[4, 5], [1, 2], [0, 0]]]], device=device)
    scores = torch.randn(1, 2, 3, 2, device=device, dtype=torch.float64, requires_grad=True)
    with torch.no_grad():
        scores[:, :, 2, :] = -torch.inf
        count.output_norm.weight.uniform_(0.5, 1.5)
        count.output_norm.bias.uniform_(-0.2, 0.2)
    parameters = [base, head, scores, count.output_norm.weight, count.output_norm.bias]
    refs = [value.detach().clone().requires_grad_() for value in parameters]
    ref_base, ref_head, ref_scores, gain, bias = refs
    positions = []
    for position in range(2):
        contribution = torch.zeros(4, device=device, dtype=torch.float64)
        for order in range(3):
            for candidate in range(2):
                token = ids[0, position, order, candidate]
                contribution = (
                    contribution
                    + ref_scores[0, position, order, candidate].exp() * ref_head[token] / 3
                )
        positions.append(ref_base[0, position] + contribution)
    expected = _amplify(torch.stack(positions).unsqueeze(0), gain, bias)
    gradients = _compare_gradients(count(base, ids, scores, head), expected, parameters, refs)
    assert gradients[1][1].abs().sum() > 0  # Repeated candidates still train the tied head.
    assert torch.count_nonzero(gradients[1][6]) == 0  # Unused head rows receive no COUNT gradient.
    assert set(dict(count.named_parameters())) == {"output_norm.weight", "output_norm.bias"}


@pytest.mark.parametrize("shape", [(1, 2, 3), (1, 2, 1, 3)])
def test_count_rejects_missing_order_slots(shape):
    count = CountNgramEmbedding(d_model=4, num_orders=2)
    with pytest.raises(OLMoConfigurationError, match="num_orders"):
        count(
            torch.zeros(1, 2, 4),
            torch.zeros(shape, dtype=torch.long),
            torch.zeros(shape),
            torch.zeros(7, 4),
        )


def _scalar_rows(tokens, tables_per_order, sizes, vocab_size, eos_token_id):
    rows = []
    for sequence in tokens.tolist():
        document_start = 0
        positions = []
        for position, token in enumerate(sequence):
            offset = 0
            branches = []
            for index, size in enumerate(sizes):
                order = 2 + index // tables_per_order
                # Python integers make this reference independent of machine overflow.
                polynomial = sum(
                    sequence[position - lag] * vocab_size**lag
                    for lag in range(order)
                    if position - lag >= document_start
                )
                branches.append(offset + polynomial % size)
                offset += size
            positions.append(branches)
            if token == eos_token_id:
                document_start = position + 1
        rows.append(positions)
    return torch.tensor(rows, dtype=torch.long, device=tokens.device)


@pytest.mark.parametrize("device", DEVICES)
def test_learned_ngram_equation_and_dense_gradients(device):
    torch.manual_seed(19)
    model = LearnedNgramEmbedding(
        d_model=8,
        vocab_size=17,
        max_order=3,
        tables_per_order=2,
        table_sizes=(5, 7, 11, 13),
        eos_token_id=16,
        dtype=torch.float64,
        init_device=device,
    )
    tokens = torch.tensor([[3, 4, 16, 2, 5]], device=device)
    base = torch.randn(1, 5, 8, device=device, dtype=torch.float64, requires_grad=True)
    parameters = [
        base,
        model.tables.weight,
        model.projection.weight,
        model.output_norm.weight,
        model.output_norm.bias,
    ]
    refs = [value.detach().clone().requires_grad_() for value in parameters]
    ref_base, table, projection, gain, bias = refs
    rows = _scalar_rows(tokens, 2, (5, 7, 11, 13), 17, 16)
    positions = []
    for position in range(5):
        combined = ref_base[0, position]
        for branch in range(4):
            branch_projection = projection[:, branch * 2 : (branch + 1) * 2]
            combined = combined + branch_projection @ table[rows[0, position, branch]]
        positions.append(combined / 5)
    expected = _amplify(torch.stack(positions).unsqueeze(0), gain, bias)
    gradients = _compare_gradients(model(tokens, base), expected, parameters, refs)
    assert model.tables.sparse is False
    assert not gradients[1].is_sparse
    assert gradients[1].abs().sum() > 0 and gradients[2].abs().sum() > 0


def test_learned_ngram_hash_causality_and_document_reset():
    sizes = (101, 103, 107, 109, 113, 127, 131)
    model = LearnedNgramEmbedding(
        d_model=14,
        vocab_size=100352,
        max_order=8,
        tables_per_order=1,
        table_sizes=sizes,
        eos_token_id=100351,
    )
    tokens = torch.tensor([[100350, 99999, 70000, 20000, 8, 19, 34, 55, 100351, 12, 13]])
    expected = _scalar_rows(tokens, 1, sizes, 100352, 100351)
    torch.testing.assert_close(model.ngram_rows(tokens), expected)
    base = torch.randn(1, tokens.shape[1], 14)
    changed = tokens.clone()
    changed[:, 6:8] = 42
    torch.testing.assert_close(model(changed, base)[:, :6], model(tokens, base)[:, :6])
    torch.testing.assert_close(model(changed, base)[:, 9:], model(tokens, base)[:, 9:])


def _transformer():
    norm = LayerNormConfig()
    return TransformerConfig(
        d_model=8,
        vocab_size=17,
        n_layers=1,
        block=TransformerBlockConfig(
            sequence_mixer=AttentionConfig(n_heads=2),
            layer_norm=norm,
            feed_forward=FeedForwardConfig(hidden_size=16),
        ),
        lm_head=LMHeadConfig(layer_norm=norm),
        embed_scale=2.0,
    ).build()


def _learned_ngram():
    return LearnedNgramEmbedding(
        d_model=8,
        vocab_size=17,
        max_order=3,
        tables_per_order=2,
        table_sizes=(5, 7, 11, 13),
        eos_token_id=16,
    )


def test_count_requires_inputs_and_rejects_combined_models():
    model = _transformer()
    model.count_ngram = CountNgramEmbedding(d_model=8, num_orders=2)
    tokens = torch.tensor([[1, 2]])
    with pytest.raises(OLMoConfigurationError, match="requires candidate"):
        model(tokens)
    model.learned_ngram = _learned_ngram()
    with pytest.raises(OLMoConfigurationError, match="cannot be combined"):
        model(tokens)
    del model.count_ngram
    with pytest.raises(OLMoConfigurationError, match="require a count_ngram"):
        model(tokens, count_token_ids=torch.zeros(1, 2, 2, 1, dtype=torch.long))


@pytest.mark.parametrize("kind", ["count", "learned"])
def test_model_applies_ngram_embedding_after_scaling(kind):
    model = _transformer()
    model.init_weights(device=torch.device("cpu"))
    tokens = torch.tensor([[1, 2, 3]])
    base = model.embeddings(tokens) * 2.0
    kwargs = {}
    if kind == "count":
        model.count_ngram = CountNgramEmbedding(d_model=8, num_orders=2)
        ids = torch.tensor([[[[4], [5]], [[5], [6]], [[6], [7]]]])
        scores = torch.full(ids.shape, -0.5)
        kwargs = {"count_token_ids": ids, "count_log_scores": scores}
        expected = model.count_ngram(base, ids, scores, model.lm_head.w_out.weight)
    else:
        model.learned_ngram = _learned_ngram()
        expected = model.learned_ngram(tokens, base)
    captured = []
    hook = model.blocks["0"].register_forward_pre_hook(lambda _, args: captured.append(args[0]))
    try:
        model(tokens, **kwargs)
    finally:
        hook.remove()
    torch.testing.assert_close(captured[0], expected)


@pytest.mark.parametrize("kind", ["count", "learned", "baseline"])
def test_fsdp_keeps_count_head_in_root_unit(monkeypatch, kind):
    import olmo_core.nn.transformer.model as implementation

    model = _transformer()
    if kind == "count":
        model.count_ngram = CountNgramEmbedding(d_model=8, num_orders=2)
    elif kind == "learned":
        model.learned_ngram = _learned_ngram()
    wrapped = []

    def record_shard(module, **kwargs):
        wrapped.append(module)
        module.set_unshard_in_backward = lambda value: None

    monkeypatch.setattr(implementation, "fully_shard", record_shard)
    for block in model.blocks.values():
        monkeypatch.setattr(block, "apply_fsdp", lambda **kwargs: None)
    model.apply_fsdp(prefetch_factor=0)
    assert model in wrapped
    assert (model.lm_head in wrapped) == (kind != "count")
    if kind != "baseline":
        assert getattr(model, f"{kind}_ngram") in wrapped
