"""Model ladder for the paper's three treatments on one shared backbone.

The same full-attention OLMo 3 ladder trains a vanilla backbone, a backbone
with COUNT embedding (a frozen top-K n-gram table whose score-weighted output
head rows are added to the token embedding and amplified), or a backbone with
learned n-gram embeddings (hashed dense tables averaged with the token
embedding and amplified). Which treatment runs is chosen by flags:

* no method flags: vanilla backbone;
* ``--count-table-dir``: COUNT embedding;
* ``--learned-ngram-rows-per-table``: learned n-gram embeddings.

Runs launch locally under ``torchrun`` with ``--cluster local/h200`` and the
environment variables ``OLMO_ROOT_DIR`` (save root) and
``OLMO_EVAL_MIX_BASE_DIR`` (in-loop language-model evaluation data).
``eval-checkpoint`` re-runs the in-loop evaluators on a saved checkpoint, which
is how a COUNT model is evaluated with a swapped or mixed table.
"""

from __future__ import annotations

import argparse
import dataclasses

import olmo_core.io as io
from olmo_core.config import DType
from olmo_core.data import TokenizerConfig
from olmo_core.data.composable import (
    ComposableDataLoaderConfig,
    ConcatAndChunkInstanceSourceConfig,
    InstanceFilterConfig,
    InstanceSourceConfig,
    NumpyDocumentSourceConfig,
)
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.internal.common import get_gpu_type, get_root_dir
from olmo_core.internal.ladder import main
from olmo_core.model_ladder import (
    DeviceMeshSpec,
    ModelLadder,
    Olmo3ModelConfigurator,
    TransformerSize,
    WSDSChinchillaRunConfigurator,
)
from olmo_core.train.train_module import (
    TransformerDataParallelConfig,
    TransformerDataParallelWrappingStrategy,
    TransformerTrainModule,
    TransformerTrainModuleConfig,
)

def learned_table_sizes(rows_per_table: int, num_tables: int) -> tuple[int, ...]:
    """Row counts for table r = 0, ..., T-1 around a target of ``rows_per_table`` (m): m + 2r + 1.

    Every table applies the same polynomial hash and differs only in its modulus, its row
    count, so the tables must have different sizes to hash differently.
    """
    return tuple(rows_per_table + 2 * i + 1 for i in range(num_tables))


# Sub-directories of ``--data-root`` holding the tokenized pretraining sample.
DEFAULT_DATA_SOURCES = [
    "all-dressed-snazzy2-fixed",
    "arxiv",
    "finemath-3plus",
    "s2pdf",
    "stack-edu",
    "wikipedia",
]


@dataclasses.dataclass(kw_only=True, eq=True)
class CountLadderConfigurator(Olmo3ModelConfigurator):
    """The OLMo 3 configurator with an optional COUNT or learned n-gram embedding."""

    minimum_world_size: int | None = None

    count_table_dir: str | None = None
    count_k: int = 32
    count_max_order: int = 6
    count_num_orders: int = 6
    count_mix_table_dir: str | None = None
    count_mix_lambda: float = 0.0

    learned_ngram_rows_per_table: int | None = None
    learned_ngram_max_order: int = 3
    learned_ngram_tables_per_order: int = 4
    learned_ngram_eos_token_id: int | None = None

    def __post_init__(self):
        if self.count_table_dir is not None and self.learned_ngram_rows_per_table is not None:
            raise OLMoConfigurationError(
                "COUNT embedding and learned n-gram embeddings are separate treatments"
            )

    def learned_ngram_sizes(self) -> tuple[int, ...] | None:
        if self.learned_ngram_rows_per_table is None:
            return None
        num_tables = (self.learned_ngram_max_order - 1) * self.learned_ngram_tables_per_order
        return learned_table_sizes(self.learned_ngram_rows_per_table, num_tables)

    def configure_minimal_device_mesh_spec(
        self, *, size_spec: str, sequence_length: int, device_type: str
    ) -> DeviceMeshSpec:
        if self.minimum_world_size is not None:
            return DeviceMeshSpec(
                world_size=self.minimum_world_size, dp_world_size=self.minimum_world_size
            )
        return super().configure_minimal_device_mesh_spec(
            size_spec=size_spec, sequence_length=sequence_length, device_type=device_type
        )

    def train_module_config(
        self, *, sequence_length, rank_microbatch_size, optim_config, scheduler
    ) -> TransformerTrainModuleConfig:
        learned_sizes = self.learned_ngram_sizes()
        return TransformerTrainModuleConfig(
            rank_microbatch_size=rank_microbatch_size,
            max_sequence_length=sequence_length,
            optim=optim_config,
            compile_model=True,
            dp_config=TransformerDataParallelConfig(
                name=DataParallelType.fsdp,
                param_dtype=DType.bfloat16,
                reduce_dtype=DType.float32,
                wrapping_strategy=TransformerDataParallelWrappingStrategy.full,
            ),
            z_loss_multiplier=1e-5,
            max_grad_norm=1.0,
            scheduler=scheduler,
            count_ngram=self.count_table_dir is not None,
            count_table_dir=self.count_table_dir,
            count_k=self.count_k,
            count_max_order=self.count_max_order,
            count_num_orders=self.count_num_orders,
            count_mix_table_dir=self.count_mix_table_dir,
            count_mix_lambda=self.count_mix_lambda,
            learned_ngram=learned_sizes is not None,
            learned_ngram_max_order=self.learned_ngram_max_order,
            learned_ngram_tables_per_order=self.learned_ngram_tables_per_order,
            learned_ngram_table_sizes=learned_sizes,
            learned_ngram_eos_token_id=self.learned_ngram_eos_token_id,
        )

    def build_train_module(
        self,
        *,
        size_spec,
        sequence_length,
        rank_microbatch_size,
        model_config,
        optim_config,
        scheduler,
        device_type,
    ) -> TransformerTrainModule:
        assert sequence_length in {2048, 4096, 8192}
        TransformerSize(size_spec)
        config = self.train_module_config(
            sequence_length=sequence_length,
            rank_microbatch_size=rank_microbatch_size,
            optim_config=optim_config,
            scheduler=scheduler,
        )
        train_module = config.build(model_config.build(init_device="meta"))
        assert isinstance(train_module, TransformerTrainModule)
        return train_module


@dataclasses.dataclass(kw_only=True)
class FixedGlobalBatchRunConfigurator(WSDSChinchillaRunConfigurator):
    """The WSD-S schedule with an explicit global batch size in tokens."""

    fixed_global_batch_size: int

    def __post_init__(self):
        super().__post_init__()
        if self.fixed_global_batch_size <= 0:
            raise OLMoConfigurationError("--fixed-global-batch-size must be positive")

    def configure_target_batch_size(self, num_params: int) -> int:
        del num_params
        return self.fixed_global_batch_size


def configure_ladder(args: argparse.Namespace) -> ModelLadder:
    tokenizer = TokenizerConfig.dolma2()
    model_construction_kwargs: dict = {"sliding_window": None}
    if args.attn_backend is not None:
        model_construction_kwargs["attn_backend"] = args.attn_backend

    if (args.minimum_world_size is None) != (args.fixed_global_batch_size is None):
        raise OLMoConfigurationError(
            "--minimum-world-size and --fixed-global-batch-size must be set together"
        )
    if args.minimum_world_size is not None:
        if args.minimum_world_size <= 0:
            raise OLMoConfigurationError("--minimum-world-size must be positive")
        if args.max_gpus != args.minimum_world_size:
            raise OLMoConfigurationError(
                "--max-gpus must equal --minimum-world-size so the world size cannot expand "
                "and silently change the global batch"
            )
        if args.rank_mbz is None:
            raise OLMoConfigurationError("--rank-mbz is required with a fixed global batch")
        if args.fixed_global_batch_size % (
            args.rank_mbz * args.sequence_length * args.minimum_world_size
        ):
            raise OLMoConfigurationError(
                "--fixed-global-batch-size must be divisible by "
                "rank-mbz * sequence-length * minimum-world-size"
            )

    if args.count_table_dir is not None and args.learned_ngram_rows_per_table is not None:
        raise OLMoConfigurationError(
            "--count-table-dir and --learned-ngram-rows-per-table are separate treatments"
        )
    if args.count_mix_table_dir is not None and args.count_table_dir is None:
        raise OLMoConfigurationError("--count-mix-table-dir requires --count-table-dir")

    sources = args.data_source or DEFAULT_DATA_SOURCES
    root = args.data_root.rstrip("/")
    instance_sources: list[InstanceSourceConfig] = [
        ConcatAndChunkInstanceSourceConfig(
            sources=[
                NumpyDocumentSourceConfig(
                    source_paths=[f"{root}/{name}/**/*.npy" for name in sources],
                    tokenizer=tokenizer,
                )
            ],
            sequence_length=args.sequence_length,
        )
    ]

    run_configurator: WSDSChinchillaRunConfigurator
    if args.fixed_global_batch_size is None:
        run_configurator = WSDSChinchillaRunConfigurator(
            chinchilla_multiple=args.chinchilla_multiple
        )
    else:
        run_configurator = FixedGlobalBatchRunConfigurator(
            chinchilla_multiple=args.chinchilla_multiple,
            fixed_global_batch_size=args.fixed_global_batch_size,
        )

    ladder_kwargs = {}
    if args.seed is not None:
        ladder_kwargs["seed"] = args.seed

    return ModelLadder(
        name=args.name,
        dir=str(io.join_path(get_root_dir(args.cluster), "model-ladders", args.name)),
        sizes=[size for size in TransformerSize if size.approx_num_params <= 1e9],
        max_devices=args.max_gpus,
        device_type=get_gpu_type(args.cluster),
        model_configurator=CountLadderConfigurator(
            model_construction_kwargs=model_construction_kwargs,
            rank_microbatch_size=(
                None if args.rank_mbz is None else args.rank_mbz * args.sequence_length
            ),
            minimum_world_size=args.minimum_world_size,
            count_table_dir=args.count_table_dir,
            count_k=args.count_k,
            count_max_order=args.count_max_order,
            count_num_orders=args.count_num_orders,
            count_mix_table_dir=args.count_mix_table_dir,
            count_mix_lambda=args.count_mix_lambda,
            learned_ngram_rows_per_table=args.learned_ngram_rows_per_table,
            learned_ngram_max_order=args.learned_ngram_max_order,
            learned_ngram_tables_per_order=args.learned_ngram_tables_per_order,
            learned_ngram_eos_token_id=tokenizer.eos_token_id,
        ),
        run_configurator=run_configurator,
        sequence_length=args.sequence_length,
        tokenizer=tokenizer,
        instance_sources=instance_sources,
        data_loader=ComposableDataLoaderConfig(
            num_workers=8,
            instance_filter_config=InstanceFilterConfig(),
            ignore_source_fingerprints=args.ignore_data_fingerprints,
        ),
        **ladder_kwargs,
    )


def add_additional_args(cmd: str, parser: argparse.ArgumentParser) -> None:
    del cmd
    data = parser.add_argument_group("data")
    data.add_argument(
        "--data-root",
        type=str,
        required=True,
        help="Root holding one sub-directory of tokenized .npy shards per source.",
    )
    data.add_argument(
        "--data-source",
        action="append",
        default=None,
        help=f"Source sub-directory under --data-root; repeatable. Default: {DEFAULT_DATA_SOURCES}",
    )
    data.add_argument(
        "--ignore-data-fingerprints",
        action="store_true",
        help=(
            "Resume from a checkpoint whose data sources had different fingerprints, e.g. a "
            "released checkpoint whose data now lives at your own paths. Exact only if the "
            "tokenized data itself is the same."
        ),
    )

    schedule = parser.add_argument_group("schedule")
    schedule.add_argument("--seed", type=int, default=None, help="Ladder seed (default 42).")
    schedule.add_argument(
        "--minimum-world-size",
        type=int,
        default=None,
        help="Data-parallel world size; must equal --max-gpus and pair with --fixed-global-batch-size.",
    )
    schedule.add_argument(
        "--fixed-global-batch-size",
        type=int,
        default=None,
        help="Global batch size in tokens (paper: 524,288 at 190M; 1,572,864 at 1B).",
    )
    schedule.add_argument(
        "--attn-backend",
        choices=("torch", "flash_2", "flash_3", "flash_4", "te"),
        default=None,
        help="Attention backend; the paper's runs used flash_3 on H200 GPUs.",
    )

    count = parser.add_argument_group("COUNT embedding")
    count.add_argument(
        "--count-table-dir",
        type=str,
        default=None,
        help="Directory holding the packed top-K table (forward_index_topk.bin).",
    )
    count.add_argument(
        "--count-k", type=int, default=32, help="Candidates used per order (at most the table's K)."
    )
    count.add_argument(
        "--count-max-order", type=int, default=6, help="Longest history order to look up."
    )
    count.add_argument(
        "--count-num-orders",
        type=int,
        default=6,
        help="Number of longest matching orders averaged into the COUNT embedding.",
    )
    count.add_argument(
        "--count-mix-table-dir",
        type=str,
        default=None,
        help="Second table whose scores are interpolated with the first at lookup time.",
    )
    count.add_argument(
        "--count-mix-lambda",
        type=float,
        default=0.0,
        help="Weight of --count-mix-table-dir in [0, 1]; the first table carries 1 - lambda.",
    )

    learned = parser.add_argument_group("learned n-gram embeddings")
    learned.add_argument(
        "--learned-ngram-rows-per-table",
        type=int,
        default=None,
        help=(
            "Target rows per hashed table, m; table r gets m + 2r + 1 rows so the tables hash "
            "differently (paper 1:1 setting at 190M: 148,267; a released checkpoint's m is on "
            "its model card)."
        ),
    )
    learned.add_argument("--learned-ngram-max-order", type=int, default=3)
    learned.add_argument("--learned-ngram-tables-per-order", type=int, default=4)


if __name__ == "__main__":
    main(configure_ladder, add_additional_args=add_additional_args)
