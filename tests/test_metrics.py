"""Focused tests for the published endpoint macro selection."""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from thecount.eval import endpoint_summary as cli
from thecount.eval.metrics import LM_KEYS, endpoint_summary


def complete_metrics() -> dict[str, float]:
    metrics = {key: 1.0 for key in LM_KEYS}
    metrics.update({f"eval/downstream/task-{i} (BPB)": 2.0 for i in range(74)})
    metrics.update({f"eval/downstream/task-{i} (accuracy)": 0.1 for i in range(41)})
    metrics["eval/downstream/task-0 (length-normalized accuracy)"] = 0.9
    metrics["eval/downstream/v2-ignored (BPB)"] = 99.0
    return metrics


class EndpointSummaryTest(unittest.TestCase):
    def test_exact_contract_and_length_normalized_preference(self) -> None:
        summary = endpoint_summary(complete_metrics())
        self.assertEqual(summary["n_bpb"], 74)
        self.assertEqual(summary["n_acc"], 41)
        self.assertEqual(summary["lm_macro_ce"], 1.0)
        self.assertEqual(summary["ds_macro_bpb"], 2.0)
        self.assertAlmostEqual(summary["ds_macro_acc"], (0.9 + 40 * 0.1) / 41)

    def test_rejects_incomplete_downstream_payload(self) -> None:
        metrics = complete_metrics()
        del metrics["eval/downstream/task-73 (BPB)"]
        with self.assertRaisesRegex(ValueError, "exactly 74"):
            endpoint_summary(metrics)

    def test_cli_prints_one_tab_separated_line_per_file(self) -> None:
        metrics = complete_metrics()
        metrics[cli.C4_KEY] = 1.5
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "metrics_step1.json"
            path.write_text(json.dumps(metrics))
            out = io.StringIO()
            with redirect_stdout(out):
                cli.main([str(path)])
            fields = out.getvalue().rstrip("\n").split("\t")
            self.assertEqual(fields[0], str(path))
        self.assertEqual(fields[1:], ["1.04545", "2.00000", "0.11951", "1.50000"])

        metrics = complete_metrics()
        del metrics["eval/downstream/task-40 (accuracy)"]
        with self.assertRaisesRegex(ValueError, "exactly 41"):
            endpoint_summary(metrics)

    def test_rejects_nonfinite_values(self) -> None:
        metrics = complete_metrics()
        metrics[LM_KEYS[0]] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            endpoint_summary(metrics)


if __name__ == "__main__":
    unittest.main()
