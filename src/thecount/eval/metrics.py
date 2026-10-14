"""Canonical metric selection for COUNT endpoint comparisons."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from typing import Any

LM_SOURCES = (
    "c4_en",
    "dolma_books",
    "dolma_common-crawl",
    "dolma_pes2o",
    "dolma_reddit",
    "dolma_stack",
    "dolma_wiki",
    "ice",
    "m2d2_s2orc",
    "pile",
    "wikitext_103",
)
LM_KEYS = tuple(f"eval/lm/{source}-validation/CE loss" for source in LM_SOURCES)


def endpoint_key_sets(metrics: Mapping[str, Any]) -> dict[str, tuple[str, ...]]:
    """Select the exact keys in the fixed 11-LM, 74-BPB, 41-task contract.

    Accuracy selects length-normalized accuracy when a task provides both
    variants, otherwise ordinary accuracy. Version-two downstream keys are
    excluded throughout to retain the published comparison definition.
    """
    missing_lm = [key for key in LM_KEYS if key not in metrics]
    if missing_lm:
        raise ValueError(f"missing LM validation keys: {missing_lm}")
    bpb_keys = sorted(
        key
        for key in metrics
        if re.match(r"^eval/downstream/.*\(BPB\)$", key) and "v2" not in key
    )
    selected: dict[str, tuple[str, str]] = {}
    for key in metrics:
        if "v2" in key or not key.startswith("eval/downstream/"):
            continue
        match = re.match(
            r"^eval/downstream/(.*) \((length-normalized accuracy|accuracy)\)$", key
        )
        if not match:
            continue
        task, kind = match.groups()
        previous = selected.get(task)
        if previous is None or kind == "length-normalized accuracy":
            selected[task] = (kind, key)
    if len(bpb_keys) != 74:
        raise ValueError(
            f"expected exactly 74 downstream BPB keys, got {len(bpb_keys)}"
        )
    if len(selected) != 41:
        raise ValueError(
            f"expected exactly 41 downstream accuracy tasks, got {len(selected)}"
        )
    return {
        "lm_macro_ce": LM_KEYS,
        "downstream_macro_bpb": tuple(bpb_keys),
        "downstream_macro_accuracy": tuple(
            key for _, key in sorted(selected.values(), key=lambda item: item[1])
        ),
    }


def endpoint_summary(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Compute fail-closed endpoint macros from :func:`endpoint_key_sets`."""

    keys = endpoint_key_sets(metrics)
    lm = [float(metrics[key]) for key in keys["lm_macro_ce"]]
    bpb = [float(metrics[key]) for key in keys["downstream_macro_bpb"]]
    accuracy = [
        float(metrics[key]) for key in keys["downstream_macro_accuracy"]
    ]
    if not all(math.isfinite(value) for value in (*lm, *bpb, *accuracy)):
        raise ValueError("endpoint macro inputs must all be finite")
    return {
        "lm_macro_ce": sum(lm) / len(lm),
        "ds_macro_bpb": sum(bpb) / len(bpb),
        "ds_macro_acc": sum(accuracy) / len(accuracy),
        "n_bpb": len(keys["downstream_macro_bpb"]),
        "n_acc": len(keys["downstream_macro_accuracy"]),
    }


macros = endpoint_summary
