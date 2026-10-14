"""Print endpoint macro metrics for saved in-loop metrics files.

For each metrics_step*.json given, prints one line with the run-averaged
endpoint macros from thecount.eval.metrics.endpoint_summary (the
figure convention): LM macro CE (mean cross-entropy over the LM validation
sources), downstream macro BPB (mean bits-per-byte over downstream tasks),
downstream macro accuracy, plus the single c4_en validation CE for
continuity with earlier record tables.

Usage:
  python -m thecount.eval.endpoint_summary <metrics_step.json> [...]

Output columns (tab-separated):
  path  lm_macro_ce  ds_macro_bpb  ds_macro_acc  c4_en_ce
"""
import json
import sys

from thecount.eval.metrics import endpoint_summary

C4_KEY = "eval/lm/c4_en-validation/CE loss"


def main(paths: list[str]) -> None:
    for p in paths:
        with open(p) as f:
            metrics = json.load(f)
        s = endpoint_summary(metrics)
        c4 = float(metrics[C4_KEY]) if C4_KEY in metrics else float("nan")
        print(
            f"{p}\t{s['lm_macro_ce']:.5f}\t{s['ds_macro_bpb']:.5f}"
            f"\t{s['ds_macro_acc']:.5f}\t{c4:.5f}"
        )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1:])
