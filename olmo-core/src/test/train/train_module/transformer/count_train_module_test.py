"""Exercise both paper methods through the trusted training and checkpoint code."""

import copy
from test.data.count_table_test import _write_table

import pytest
import torch

from olmo_core.data.utils import get_labels
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.attention import AttentionConfig
from olmo_core.nn.feed_forward import FeedForwardConfig
from olmo_core.nn.layer_norm import LayerNormConfig
from olmo_core.nn.lm_head import LMHeadConfig
from olmo_core.nn.transformer import TransformerBlockConfig, TransformerConfig
from olmo_core.optim import AdamWConfig
from olmo_core.train.checkpoint import Checkpointer
from olmo_core.train.train_module.transformer import TransformerTrainModuleConfig


class Metrics:
    global_step = 0
    max_steps = 3

    def __init__(self):
        self.values = {}

    def record_metric(self, name, value, *args, namespace=None, **kwargs):
        self.values[f"{namespace}/{name}" if namespace else name] = value

    def record_ce_loss(self, value, *args, **kwargs):
        self.values["train/CE loss"] = value


def build_module(method, table_dir, **overrides):
    norm = LayerNormConfig(name="rms", bias=False)
    model = TransformerConfig(
        d_model=8,
        vocab_size=32,
        n_layers=1,
        init_seed=17,
        block=TransformerBlockConfig(
            name="default",
            attention=AttentionConfig(n_heads=2, backend="torch", bias=False),
            layer_norm=norm,
            feed_forward=FeedForwardConfig(hidden_size=16, bias=False),
        ),
        lm_head=LMHeadConfig(layer_norm=norm, bias=False),
    ).build()
    options = dict(
        rank_microbatch_size=8,
        max_sequence_length=8,
        optim=AdamWConfig(lr=0.01, foreach=False, fused=False),
        max_grad_norm=1.0,
        count_ngram=method == "count",
        count_table_dir=str(table_dir),
        count_k=2,
        count_max_order=3,
        count_num_orders=2,
        learned_ngram=method == "learned",
        learned_ngram_max_order=3,
        learned_ngram_tables_per_order=2,
        learned_ngram_table_sizes=(11, 13, 15, 17),
        learned_ngram_eos_token_id=31,
    )
    options.update(overrides)
    module = TransformerTrainModuleConfig(**options).build(model, device=torch.device("cpu"))
    module._trainer = Metrics()
    return module


@pytest.fixture
def table_dir(tmp_path):
    _write_table(
        tmp_path / "forward_index_topk.bin",
        sections=[
            (1, [[]], [[1, 2]], [[-0.5, -1.0]]),
            (2, [[2], [3]], [[4, 5], [6, 7]], [[-0.2, -0.8], [-0.4, -0.9]]),
            (3, [[2, 3]], [[8, 9]], [[-0.1, -0.7]]),
        ],
    )
    return tmp_path


def batch():
    return {"input_ids": torch.tensor([[1, 2, 3, 4, 31, 2, 3, 7], [2, 3, 8, 9, 10, 11, 12, 13]])}


def step(module):
    module.trainer.global_step += 1
    module.zero_grads()
    module.train_batch(batch())
    assert torch.isfinite(module.trainer.values["train/CE loss"])
    module.optim_step()


@pytest.mark.parametrize("method", ["count", "learned"])
def test_native_training_checkpoint_and_resume(table_dir, method):
    module = build_module(method, table_dir)
    assert type(module.optim) is torch.optim.AdamW
    assert not hasattr(module, "sparse_optim")
    all_params = {id(p) for p in module.model.parameters()}
    assert all_params == {id(p) for group in module.optim.param_groups for p in group["params"]}
    before = {name: p.detach().clone() for name, p in module.model.named_parameters()}
    step(module)
    step(module)
    assert any(not torch.equal(p, before[name]) for name, p in module.model.named_parameters())
    checkpoint = Checkpointer(work_dir=table_dir / "checkpoint-work")
    path = table_dir / "step2"
    checkpoint.save(path, module, {"global_step": 2})
    resumed = build_module(method, table_dir)
    state = checkpoint.load(path, resumed)
    assert state == {"global_step": 2}
    resumed.trainer.global_step = 2
    torch.testing.assert_close(
        resumed.state_dict_to_save(), module.state_dict_to_save(), rtol=0, atol=0
    )
    step(module)
    step(resumed)
    torch.testing.assert_close(
        resumed.state_dict_to_save(), module.state_dict_to_save(), rtol=0, atol=0
    )
    output = resumed.eval_batch(batch(), labels=get_labels(batch()))
    assert torch.isfinite(output.ce_loss).all()


def test_count_train_and_eval_share_causal_lookup(table_dir):
    module = build_module("count", table_dir)
    train_inputs, _, train_kwargs = module._prepare_batch(batch())
    module.model.eval()
    eval_inputs, _, eval_kwargs = module._prepare_batch(batch())
    torch.testing.assert_close(train_inputs, eval_inputs)
    for name in ("count_token_ids", "count_log_scores"):
        torch.testing.assert_close(train_kwargs[name], eval_kwargs[name])
    changed = batch()
    changed["input_ids"][:, 4:] = 20
    _, _, changed_kwargs = module._prepare_batch(changed)
    for name in ("count_token_ids", "count_log_scores"):
        torch.testing.assert_close(train_kwargs[name][:, :4], changed_kwargs[name][:, :4])


def test_count_table_swap_preserves_learned_state(table_dir, tmp_path):
    original = build_module("count", table_dir)
    alternate_dir = tmp_path / "alternate"
    alternate_dir.mkdir()
    _write_table(
        alternate_dir / "forward_index_topk.bin",
        sections=[
            (1, [[]], [[14, 15]], [[-0.1, -0.2]]),
        ],
    )
    alternate = build_module(
        "count", alternate_dir, count_max_order=1, count_num_orders=1
    )
    alternate.load_state_dict(copy.deepcopy(original.state_dict_to_save(optim=False)))
    for name, value in original.model.state_dict().items():
        torch.testing.assert_close(alternate.model.state_dict()[name], value, rtol=0, atol=0)
    assert not torch.equal(original.eval_batch(batch()), alternate.eval_batch(batch()))


def test_rejects_combined_methods(table_dir):
    with pytest.raises(OLMoConfigurationError, match="separate methods"):
        build_module("count", table_dir, learned_ngram=True)


@pytest.mark.parametrize("method", ["count", "learned"])
def test_model_seed_controls_all_parameters_and_preserves_backbone(table_dir, method):
    torch.manual_seed(1)
    first = build_module(method, table_dir)
    torch.manual_seed(999)
    second = build_module(method, table_dir)
    baseline = build_module("baseline", table_dir)
    torch.testing.assert_close(first.model.state_dict(), second.model.state_dict(), rtol=0, atol=0)
    for name, value in baseline.model.state_dict().items():
        torch.testing.assert_close(first.model.state_dict()[name], value, rtol=0, atol=0)
    assert first.model.num_params == sum(p.numel() for p in first.model.parameters())


def test_sparse_learned_ngram_optimizer_state_is_refused():
    from torch.distributed.checkpoint.metadata import (
        Metadata,
        TensorProperties,
        TensorStorageMetadata,
    )

    from olmo_core.train.train_module.transformer.train_module import (
        SPARSE_LEARNED_NGRAM_STEP_KEY,
        _check_dense_learned_ngram_optimizer_state,
    )

    def metadata(step_shape):
        return Metadata(
            state_dict_metadata={
                SPARSE_LEARNED_NGRAM_STEP_KEY: TensorStorageMetadata(
                    properties=TensorProperties(dtype=torch.float32),
                    size=torch.Size(step_shape),
                    chunks=[],
                )
            }
        )

    _check_dense_learned_ngram_optimizer_state(metadata(()))  # dense: one scalar step
    _check_dense_learned_ngram_optimizer_state(Metadata(state_dict_metadata={}))
    with pytest.raises(RuntimeError, match="sparse row-wise optimizer state"):
        _check_dense_learned_ngram_optimizer_state(metadata((56,)))
