import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from coupon_lab.agent import revise_candidate, revise_candidate_deepseek
from coupon_lab.cli import main, run_experiment
from coupon_lab.evaluate import Budget


class RunTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        frame = pd.DataFrame({
            "id": range(100), "arm": [i % 2 for i in range(100)],
            "x": [i / 100 for i in range(100)],
            "active": [i % 2 for i in range(100)],
            "margin": [5.0 if i % 2 else 0.0 for i in range(100)],
            "cost": [1.0 if i % 2 else 0.0 for i in range(100)],
        })
        frame.to_csv(self.root / "data.csv", index=False)
        self.manifest = {
            "dataset": "data.csv", "unit_id": "id",
            "treatment": {"column": "arm", "control": 0, "treated": 1, "probability": 0.5},
            "features": ["x"], "feature_timing": "pre_treatment",
            "outcomes": {"active": "active", "gross_margin": "margin", "coupon_cost": "cost"},
            "margin_includes_coupon_cost": False,
        }
        self.path = self.root / "manifest.json"
        self.save_manifest()

    def save_manifest(self):
        self.path.write_text(json.dumps(self.manifest))

    def test_full_capabilities_and_reproducible_run(self):
        report1 = run_experiment(self.path, Budget("cost", 0.25), seed=11, output=self.root / "runs")
        report2 = run_experiment(self.path, Budget("cost", 0.25), seed=11, output=self.root / "runs")
        self.assertEqual(report1, report2)
        self.assertEqual(report1["prediction_sha256"], report2["prediction_sha256"])
        self.assertIn("active", report1["policies"])
        self.assertIn("net_margin", report1["policies"])
        self.assertIn("random", report1["policies"])
        self.assertIn("cost", report1["policies"]["active"])
        self.assertTrue((self.root / "runs" / report1["run_id"] / "report.md").exists())
        self.assertEqual((self.root / "runs" / report1["run_id"] / "candidate.py").read_bytes(),
                         Path(__file__).resolve().parents[1].joinpath("src/coupon_lab/candidate.py").read_bytes())

    def test_feature_gap_note_is_copied_into_report(self):
        gap = self.root / "feature_gaps.md"
        gap.write_text("- 建议字段：发券前 7 天登录天数；来源：行为日志；验证：下一版数据重新训练。")
        report = run_experiment(self.path, Budget("count", 0.2), seed=11,
                                output=self.root / "runs", feature_gaps=gap)
        artifact = self.root / "runs" / report["run_id"]
        self.assertIn("发券前 7 天登录天数", (artifact / "report.md").read_text())
        self.assertEqual((artifact / "feature_gaps.md").read_text(), gap.read_text())

    def test_candidate_receives_only_holdout_features(self):
        observed = {}

        def candidate(train, target, **kwargs):
            observed["columns"] = list(target)
            return pd.DataFrame({"active_uplift": [0.1] * len(target),
                                 "gross_margin_uplift": [0.5] * len(target),
                                 "expected_cost": [0.1] * len(target)})

        with patch("coupon_lab.cli._candidate_function", return_value=candidate):
            run_experiment(self.path, Budget("count", 0.2), seed=11, output=self.root / "runs")
        self.assertEqual(observed["columns"], ["x"])

    def test_conversion_only_reports_no_profit(self):
        self.manifest["outcomes"] = {"conversion": "active"}
        self.manifest.pop("margin_includes_coupon_cost")
        self.save_manifest()
        report = run_experiment(self.path, Budget("count", 0.2), seed=11, output=self.root / "runs")
        self.assertEqual(set(report["policies"]), {"conversion", "random"})
        self.assertIn("net_margin", report["unavailable_objectives"])
        self.assertNotIn("cost", report["policies"]["conversion"])

    def test_fixed_send_cost_reports_assumed_net_only(self):
        self.manifest["outcomes"].pop("coupon_cost")
        self.manifest["fixed_send_cost"] = 1.0
        self.save_manifest()
        report = run_experiment(self.path, Budget("cost", 0.25), seed=11, output=self.root / "runs")
        entry = report["policies"]["net_margin"]
        self.assertNotIn("cost", entry)
        self.assertIn("assumed_cost", entry)
        self.assertEqual(entry["net_label"], "假设增量净收益")
        self.assertAlmostEqual(entry["net"]["mean"],
                               entry["effects"]["gross_margin"]["mean"] - entry["assumed_cost"])

    def test_agent_revision_only_copies_candidate(self):
        candidate = self.root / "candidate.py"
        candidate.write_text("def fit_predict(*args, **kwargs):\n    return None\n")
        report = self.root / "report.md"
        report.write_text("Improve uplift estimate")

        def fake_codex(args, **kwargs):
            scratch = Path(args[args.index("-C") + 1])
            with (scratch / "candidate.py").open("a") as handle:
                handle.write("\n# revised\n")
            (scratch / "unwanted.py").write_text("bad")
            (scratch / "feature_gaps.md").write_text("- 建议字段：历史活跃度")
            return subprocess.CompletedProcess(args, 0, "", "")

        with patch("coupon_lab.agent.subprocess.run", side_effect=fake_codex):
            revise_candidate(candidate, report, feature_gaps_path=self.root / "feature_gaps.md",
                             codex_bin="fake-codex")
        self.assertIn("# revised", candidate.read_text())
        self.assertIn("历史活跃度", (self.root / "feature_gaps.md").read_text())
        self.assertFalse((self.root / "unwanted.py").exists())

    def test_agent_revision_can_be_reevaluated_without_dataset_changes(self):
        candidate = self.root / "candidate.py"
        source = Path(__file__).resolve().parents[1] / "src/coupon_lab/candidate.py"
        candidate.write_bytes(source.read_bytes())
        first = run_experiment(self.path, Budget("count", 0.2), seed=11,
                               output=self.root / "runs", candidate_path=candidate)

        def fake_codex(args, **kwargs):
            scratch = Path(args[args.index("-C") + 1])
            with (scratch / "candidate.py").open("a") as handle:
                handle.write("\n# next candidate revision\n")
            (scratch / "feature_gaps.md").write_text("建议字段：发券前登录天数")
            return subprocess.CompletedProcess(args, 0, "", "")

        notes = self.root / "feature_gaps.md"
        with patch("coupon_lab.agent.subprocess.run", side_effect=fake_codex):
            revise_candidate(candidate, self.root / "runs" / first["run_id"] / "report.md",
                             feature_gaps_path=notes, codex_bin="fake-codex")
        second = run_experiment(self.path, Budget("count", 0.2), seed=11,
                                output=self.root / "runs", candidate_path=candidate,
                                feature_gaps=notes)
        self.assertNotEqual(first["run_id"], second["run_id"])
        self.assertEqual(first["dataset_sha256"], second["dataset_sha256"])
        self.assertIn("发券前登录天数", (self.root / "runs" / second["run_id"] / "report.md").read_text())
        self.assertEqual(first["prediction_sha256"], second["prediction_sha256"])

    def test_deepseek_revision_parses_json_and_feature_notes(self):
        candidate = self.root / "candidate.py"
        candidate.write_text("def fit_predict(*args, **kwargs):\n    return None\n")
        report = self.root / "report.md"
        report.write_text("Improve uplift estimate")
        payload = {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps({
            "candidate_py": "```python\ndef fit_predict(*args, **kwargs):\n    return 1\n```",
            "feature_gaps_md": "建议字段：历史活跃度",
        })}}]}

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *_):
                return False

            def read(self):
                return json.dumps(payload).encode()

        with patch("coupon_lab.agent.urllib.request.urlopen", return_value=FakeResponse()) as call:
            revise_candidate_deepseek(candidate, report, api_key="test-key",
                                      feature_gaps_path=self.root / "feature_gaps.md")
        self.assertIn("return 1", candidate.read_text())
        self.assertIn("历史活跃度", (self.root / "feature_gaps.md").read_text())
        request = call.call_args.args[0]
        body = json.loads(request.data)
        self.assertEqual(body["model"], "deepseek-flash")
        self.assertEqual(body["thinking"], {"type": "disabled"})

    def test_deepseek_repairs_invalid_python_once(self):
        candidate = self.root / "candidate.py"
        candidate.write_text("def fit_predict(*args, **kwargs):\n    return None\n")
        report = self.root / "report.md"
        report.write_text("Improve uplift estimate")

        class FakeResponse:
            def __init__(self, code):
                self.code = code

            def __enter__(self):
                return self

            def __exit__(self, *_):
                return False

            def read(self):
                return json.dumps({"choices": [{"finish_reason": "stop", "message": {
                    "content": json.dumps({"candidate_py": self.code})}}]}).encode()

        responses = [FakeResponse("def fit_predict(:"),
                     FakeResponse("def fit_predict(*args, **kwargs):\n    return 1\n")]
        with patch("coupon_lab.agent.urllib.request.urlopen", side_effect=responses) as call:
            revise_candidate_deepseek(candidate, report, api_key="test-key")
        self.assertEqual(call.call_count, 2)
        self.assertIn("return 1", candidate.read_text())

    def test_agent_command_rejects_code_only_change_and_restores_candidate(self):
        candidate = self.root / "candidate.py"
        source = Path(__file__).resolve().parents[1] / "src/coupon_lab/candidate.py"
        candidate.write_bytes(source.read_bytes())
        original = candidate.read_bytes()

        def fake_revision(path, *_args, **_kwargs):
            path.write_bytes(path.read_bytes() + b"\n# no prediction change\n")

        args = ["coupon-lab", "agent", str(self.path), "--budget-kind", "count",
                "--budget", "0.2", "--candidate", str(candidate), "--output",
                str(self.root / "runs"), "--agent-provider", "deepseek"]
        with patch.object(sys, "argv", args), patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-key"}), \
                patch("coupon_lab.cli.revise_candidate_deepseek", side_effect=fake_revision):
            with self.assertRaisesRegex(ValueError, "predictions unchanged"):
                main()
            self.assertNotIn("DEEPSEEK_API_KEY", os.environ)
        self.assertEqual(candidate.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
