import argparse
import importlib.util
import sys
from pathlib import Path

import pytest

from olmo_core.exceptions import OLMoConfigurationError

SCRIPT = (
    Path(__file__).resolve().parents[2] / "scripts" / "train" / "ladder" / "count_ladder.py"
)


@pytest.fixture(scope="module")
def ladder_module():
    spec = importlib.util.spec_from_file_location("count_ladder", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["count_ladder"] = module
    spec.loader.exec_module(module)
    return module


def _args(**overrides) -> argparse.Namespace:
    base = dict(
        name="count-test",
        cluster="local/h200",
        max_gpus=8,
        chinchilla_multiple=2.0,
        sequence_length=8192,
        rank_mbz=None,
        data_root="/data/root",
        data_source=None,
        seed=1042,
        minimum_world_size=None,
        fixed_global_batch_size=None,
        attn_backend=None,
        count_table_dir=None,
        count_k=32,
        count_max_order=6,
        count_num_orders=6,
        count_mix_table_dir=None,
        count_mix_lambda=0.0,
        learned_ngram_rows_per_table=None,
        ignore_data_fingerprints=False,
        learned_ngram_max_order=3,
        learned_ngram_tables_per_order=4,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def test_parser_exposes_method_flags(ladder_module):
    parser = argparse.ArgumentParser()
    ladder_module.add_additional_args("run", parser)
    args = parser.parse_args(
        ["--data-root", "/d", "--count-table-dir", "/t", "--count-k", "8"]
    )
    assert args.count_table_dir == "/t" and args.count_k == 8
    assert args.learned_ngram_rows_per_table is None
    assert not args.ignore_data_fingerprints
    args = parser.parse_args(["--data-root", "/d", "--ignore-data-fingerprints"])
    assert args.ignore_data_fingerprints


def test_vanilla_ladder_uses_shared_backbone(ladder_module, monkeypatch, tmp_path):
    monkeypatch.setenv("OLMO_ROOT_DIR", str(tmp_path))
    ladder = ladder_module.configure_ladder(_args())
    assert ladder.seed == 1042
    assert ladder.device_type == "NVIDIA H200"
    assert str(ladder.dir).startswith(str(tmp_path))
    assert ladder.model_configurator.model_construction_kwargs == {"sliding_window": None}
    config = ladder.model_configurator.train_module_config(
        sequence_length=8192, rank_microbatch_size=8192, optim_config=None, scheduler=None
    )
    assert not config.count_ngram and not config.learned_ngram
    assert config.z_loss_multiplier == 1e-5 and config.max_grad_norm == 1.0


def test_count_ladder_passes_table_settings(ladder_module, monkeypatch, tmp_path):
    monkeypatch.setenv("OLMO_ROOT_DIR", str(tmp_path))
    ladder = ladder_module.configure_ladder(
        _args(count_table_dir="/table", count_k=8, count_mix_table_dir="/mix", count_mix_lambda=0.25)
    )
    config = ladder.model_configurator.train_module_config(
        sequence_length=8192, rank_microbatch_size=8192, optim_config=None, scheduler=None
    )
    assert config.count_ngram and config.count_table_dir == "/table"
    assert config.count_k == 8 and config.count_max_order == 6 and config.count_num_orders == 6
    assert config.count_mix_table_dir == "/mix" and config.count_mix_lambda == 0.25
    assert not config.learned_ngram


def test_learned_ngram_ladder_sizes_tables(ladder_module, monkeypatch, tmp_path):
    monkeypatch.setenv("OLMO_ROOT_DIR", str(tmp_path))
    ladder = ladder_module.configure_ladder(_args(learned_ngram_rows_per_table=148_267))
    config = ladder.model_configurator.train_module_config(
        sequence_length=8192, rank_microbatch_size=8192, optim_config=None, scheduler=None
    )
    assert config.learned_ngram and not config.count_ngram
    # Every table hashes with the same polynomial, so the sizes must differ: m+1, m+3, ...
    assert config.learned_ngram_table_sizes == tuple(148_267 + 2 * i + 1 for i in range(8))
    assert config.learned_ngram_eos_token_id == ladder.tokenizer.eos_token_id


def test_learned_ngram_sizes_match_released_checkpoints(ladder_module):
    # release_args.json of the released 60M learned n-gram checkpoints
    assert ladder_module.learned_table_sizes(49_563, 8) == (
        49_564, 49_566, 49_568, 49_570, 49_572, 49_574, 49_576, 49_578
    )


def test_ignore_data_fingerprints_reaches_data_loader(ladder_module, monkeypatch, tmp_path):
    monkeypatch.setenv("OLMO_ROOT_DIR", str(tmp_path))
    assert not ladder_module.configure_ladder(_args()).data_loader.ignore_source_fingerprints
    ladder = ladder_module.configure_ladder(_args(ignore_data_fingerprints=True))
    assert ladder.data_loader.ignore_source_fingerprints


def test_methods_are_mutually_exclusive(ladder_module, monkeypatch, tmp_path):
    monkeypatch.setenv("OLMO_ROOT_DIR", str(tmp_path))
    with pytest.raises(OLMoConfigurationError):
        ladder_module.configure_ladder(
            _args(count_table_dir="/table", learned_ngram_rows_per_table=16)
        )
    with pytest.raises(OLMoConfigurationError):
        ladder_module.configure_ladder(_args(count_mix_table_dir="/mix"))


def test_fixed_global_batch_requires_matching_world_size(ladder_module, monkeypatch, tmp_path):
    monkeypatch.setenv("OLMO_ROOT_DIR", str(tmp_path))
    ladder = ladder_module.configure_ladder(
        _args(max_gpus=4, minimum_world_size=4, fixed_global_batch_size=524_288, rank_mbz=8)
    )
    assert ladder.run_configurator.configure_target_batch_size(190_000_000) == 524_288
    assert ladder.model_configurator.configure_minimal_device_mesh_spec(
        size_spec="190M", sequence_length=8192, device_type="NVIDIA H200"
    ).world_size == 4
    with pytest.raises(OLMoConfigurationError):
        ladder_module.configure_ladder(
            _args(max_gpus=8, minimum_world_size=4, fixed_global_batch_size=524_288, rank_mbz=8)
        )
    with pytest.raises(OLMoConfigurationError):
        ladder_module.configure_ladder(_args(fixed_global_batch_size=524_288))


@pytest.mark.parametrize(
    "path, step",
    [
        ("/root/model-ladders/count-190m/190M/step14523", 14523),
        ("/root/model-ladders/count-190m/190M/step14523/", 14523),
        ("s3://bucket/model-ladders/vanilla-60m/60M/step8762", 8762),
        ("/root/checkpoint", None),
        ("/root/step12-extra", None),
    ],
)
def test_eval_checkpoint_step_from_path(path, step):
    from olmo_core.model_ladder.base import checkpoint_step_from_path

    assert checkpoint_step_from_path(path) == step
