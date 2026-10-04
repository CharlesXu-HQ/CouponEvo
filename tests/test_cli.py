import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from couponevo.agent import revise_candidate, revise_candidate_deepseek
from couponevo.cli import main, run_experiment
from couponevo.evaluate import Budget


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
            "treatment": {"column": "arm", "control": 0, "treated": 1, "probability": 0.5,
                          "probability_source": "protocol", "probability_reference": "experiment-plan-v1"},
            "features": ["x"], "feature_timing": "pre_treatment",
            "outcomes": {"active": "active", "gross_margin": "margin", "coupon_cost": "cost"},
            "margin_includes_coupon_cost": False,
        }
        self.path = self.root / "manifest.json"
        self.save_manifest()

    def save_manifest(self):
        self.path.write_text(json.dumps(self.manifest))

    def test_agent_requires_isolated_execution(self):
        args = ["couponevo", "search", str(self.path), "--budget-kind", "count",
                "--budget", "0.2", "--objective", "active", "--search-id", "isolated",
                "--agent-provider", "deepseek"]
        with patch.object(sys, "argv", args), self.assertRaises(SystemExit) as error:
            main()
        self.assertEqual(error.exception.code, 2)

    def test_candidate_receives_only_declared_training_columns(self):
        seen = {}

        class FakeSandbox:
            def __init__(self, *_args, **_kwargs):
                pass

            def predict(self, _candidate, train, target, _args, _seed, _device):
                seen["train"] = train
                seen["target"] = target
                result = pd.DataFrame({"active_uplift": np.ones(len(target)),
                                       "gross_margin_uplift": np.ones(len(target)),
                                       "expected_cost": np.ones(len(target))})
                result.attrs["model_device"] = "cpu"
                return result

            def choose(self, *_args):
                return None

        with patch("couponevo.cli.DockerSandbox", FakeSandbox):
            run_experiment(self.path, Budget("count", 0.2), seed=7,
                           output=self.root / "runs", sandbox_image="test-image")
        self.assertEqual(list(seen["train"]), ["x", "__treatment", "active", "margin", "cost"])
        self.assertEqual(list(seen["target"]), ["x"])
        self.assertEqual(seen["train"].index.tolist(), list(range(len(seen["train"]))))
        self.assertEqual(seen["target"].index.tolist(), list(range(len(seen["target"]))))

    def test_cuda_sandbox_requires_measured_gpu_allocation(self):
        class FakeSandbox:
            def __init__(self, *_args, **_kwargs):
                pass

            def predict(self, _candidate, _train, target, _args, _seed, _device):
                result = pd.DataFrame({"active_uplift": np.ones(len(target)),
                                       "gross_margin_uplift": np.ones(len(target)),
                                       "expected_cost": np.ones(len(target))})
                result.attrs["model_device"] = "cuda"
                result.attrs["cuda_peak_bytes"] = 0
                return result

        with patch("couponevo.cli.DockerSandbox", FakeSandbox):
            with self.assertRaisesRegex(ValueError, "allocate a CUDA tensor"):
                run_experiment(self.path, Budget("count", 0.2), seed=7,
                               output=self.root / "runs", sandbox_image="test-image", device="cuda")

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
                         Path(__file__).resolve().parents[1].joinpath("src/couponevo/candidate.py").read_bytes())
        self.assertEqual(report1["data_validation"]["feature_timing"], "declared_only")
        self.assertEqual(set(report1["ranking_diagnostics"]), {"active", "gross_margin"})
        self.assertTrue(np.isfinite(report1["ranking_diagnostics"]["active"]["qini"]))
        split_file = self.root / "runs" / report1["run_id"] / "split_manifest.json"
        split = json.loads(split_file.read_text())
        self.assertEqual(sorted(split["train"] + split["validation"] + split["test"]), list(range(100)))

    def test_strict_run_rejects_unverifiable_feature_timing(self):
        with self.assertRaisesRegex(ValueError, "assignment_time_column"):
            run_experiment(self.path, Budget("count", 0.2), seed=11,
                           output=self.root / "runs", strict_data=True)

    def test_feature_gap_note_is_copied_into_report(self):
        gap = self.root / "feature_gaps.md"
        gap.write_text("- 建议字段：发券前 7 天登录天数；来源：行为日志；验证：下一版数据重新训练。")
        report = run_experiment(self.path, Budget("count", 0.2), seed=11,
                                output=self.root / "runs", feature_gaps=gap)
        artifact = self.root / "runs" / report["run_id"]
        self.assertIn("发券前 7 天登录天数", (artifact / "report.md").read_text())
        self.assertEqual((artifact / "feature_gaps.md").read_text(), gap.read_text())

    def test_same_candidate_has_zero_paired_difference_on_final_holdout(self):
        candidate = Path(__file__).resolve().parents[1] / "src/couponevo/candidate.py"
        report = run_experiment(self.path, Budget("count", 0.2), seed=11,
                                output=self.root / "runs", final=True,
                                compare_candidate_path=candidate)
        self.assertEqual(report["holdout"], "test")
        for metrics in report["paired_vs_baseline"].values():
            for estimate in metrics.values():
                self.assertEqual((estimate["mean"], estimate["lower"], estimate["upper"]), (0, 0, 0))
        self.assertTrue((self.root / "runs" / report["run_id"] / "baseline_candidate.py").exists())

    def test_candidate_can_change_budget_policy_without_changing_model(self):
        baseline = Path(__file__).resolve().parents[1] / "src/couponevo/candidate.py"
        candidate = self.root / "policy_candidate.py"
        candidate.write_text(baseline.read_text() + "\n"
                             "def choose_policy(scores, costs, budget_kind, budget_value):\n"
                             "    chosen = np.zeros(len(scores), dtype=bool)\n"
                             "    chosen[0] = True\n"
                             "    return chosen\n")
        report = run_experiment(self.path, Budget("count", 0.2), seed=11,
                                output=self.root / "runs", candidate_path=candidate,
                                compare_candidate_path=baseline)
        self.assertEqual(report["policies"]["active"]["selected_count"], 1)
        self.assertNotEqual(report["policy_sha256"]["active"],
                            report["baseline_policy_sha256"]["active"])

    def test_stochastic_candidate_policy_repeats_with_same_seed(self):
        baseline = Path(__file__).resolve().parents[1] / "src/couponevo/candidate.py"
        candidate = self.root / "policy_candidate.py"
        candidate.write_text(baseline.read_text() + "\n"
                             "def choose_policy(scores, costs, budget_kind, budget_value):\n"
                             "    chosen = np.zeros(len(scores), dtype=bool)\n"
                             "    chosen[np.random.permutation(len(scores))[:3]] = True\n"
                             "    return chosen\n")
        first = run_experiment(self.path, Budget("count", 0.2), seed=11,
                               output=self.root / "runs", candidate_path=candidate)
        second = run_experiment(self.path, Budget("count", 0.2), seed=11,
                                output=self.root / "runs", candidate_path=candidate)
        self.assertEqual(first["policy_sha256"], second["policy_sha256"])

    def test_run_id_changes_if_candidate_decisions_change(self):
        baseline = Path(__file__).resolve().parents[1] / "src/couponevo/candidate.py"
        candidate = self.root / "policy_candidate.py"
        candidate.write_text(baseline.read_text() + "\nimport os\n"
                             "def choose_policy(scores, costs, budget_kind, budget_value):\n"
                             "    chosen = np.zeros(len(scores), dtype=bool)\n"
                             "    chosen[int(os.environ['POLICY_INDEX'])] = True\n"
                             "    return chosen\n")
        with patch.dict(os.environ, {"POLICY_INDEX": "0"}):
            first = run_experiment(self.path, Budget("count", 0.2), seed=11,
                                   output=self.root / "runs", candidate_path=candidate)
        with patch.dict(os.environ, {"POLICY_INDEX": "1"}):
            second = run_experiment(self.path, Budget("count", 0.2), seed=11,
                                    output=self.root / "runs", candidate_path=candidate)
        self.assertNotEqual(first["run_id"], second["run_id"])

    def test_candidate_policy_cannot_exceed_budget(self):
        baseline = Path(__file__).resolve().parents[1] / "src/couponevo/candidate.py"
        candidate = self.root / "policy_candidate.py"
        candidate.write_text(baseline.read_text() + "\n"
                             "def choose_policy(scores, costs, budget_kind, budget_value):\n"
                             "    return np.ones(len(scores), dtype=bool)\n")
        with self.assertRaisesRegex(ValueError, "budget"):
            run_experiment(self.path, Budget("count", 0.2), seed=11,
                           output=self.root / "runs", candidate_path=candidate)

    def test_candidate_policy_rejects_invalid_predicted_cost(self):
        baseline = Path(__file__).resolve().parents[1] / "src/couponevo/candidate.py"
        candidate = self.root / "policy_candidate.py"
        candidate.write_text(baseline.read_text().replace(
            'result["expected_cost"] = np.maximum(0, cost_model.predict(x_target))',
            'result["expected_cost"] = -np.ones(len(target))') + "\n"
            "def choose_policy(scores, costs, budget_kind, budget_value):\n"
            "    return np.zeros(len(scores), dtype=bool)\n")
        with self.assertRaisesRegex(ValueError, "expected_cost"):
            run_experiment(self.path, Budget("cost", 0.25), seed=11,
                           output=self.root / "runs", candidate_path=candidate)

    def test_report_includes_paired_interval_against_random_policy(self):
        report = run_experiment(self.path, Budget("count", 0.2), seed=11,
                                output=self.root / "runs")
        paired = report["paired_vs_random"]["active"]["active"]
        self.assertAlmostEqual(paired["mean"], report["policies"]["active"]["delta_vs_random"]["active"])
        self.assertLessEqual(paired["lower"], paired["mean"])
        self.assertGreaterEqual(paired["upper"], paired["mean"])
        self.assertIn("net", report["paired_vs_random"]["net_margin"])

    def test_final_report_can_bootstrap_paired_intervals(self):
        baseline = Path(__file__).resolve().parents[1] / "src/couponevo/candidate.py"
        report = run_experiment(self.path, Budget("count", 0.2), seed=11,
                                output=self.root / "runs", final=True,
                                compare_candidate_path=baseline, bootstrap_reps=100)
        paired = report["paired_vs_baseline_bootstrap"]["active"]["active"]
        self.assertEqual((paired["mean"], paired["lower"], paired["upper"]), (0, 0, 0))
        self.assertIn("active", report["paired_vs_random_bootstrap"])

    def test_stochastic_candidate_is_repeatable_for_one_run_id(self):
        def candidate(train, target, **kwargs):
            values = torch.rand(len(target)).numpy()
            return pd.DataFrame({"active_uplift": values,
                                 "gross_margin_uplift": values,
                                 "expected_cost": values})

        with patch("couponevo.cli._candidate_function", return_value=candidate):
            first = run_experiment(self.path, Budget("count", 0.2), seed=11, output=self.root / "runs")
            second = run_experiment(self.path, Budget("count", 0.2), seed=11, output=self.root / "runs")
        self.assertEqual(first["prediction_sha256"], second["prediction_sha256"])

    def test_candidate_receives_only_holdout_features(self):
        observed = {}

        def candidate(train, target, **kwargs):
            observed["columns"] = list(target)
            return pd.DataFrame({"active_uplift": [0.1] * len(target),
                                 "gross_margin_uplift": [0.5] * len(target),
                                 "expected_cost": [0.1] * len(target)})

        with patch("couponevo.cli._candidate_function", return_value=candidate):
            run_experiment(self.path, Budget("count", 0.2), seed=11, output=self.root / "runs")
        self.assertEqual(observed["columns"], ["x"])

    def test_cuda_run_rejects_candidate_without_cuda_confirmation(self):
        def cpu_candidate(train, target, **kwargs):
            return pd.DataFrame({"active_uplift": [0.1] * len(target),
                                 "gross_margin_uplift": [0.5] * len(target),
                                 "expected_cost": [0.1] * len(target)})

        with patch("couponevo.cli._candidate_function", return_value=cpu_candidate):
            with self.assertRaisesRegex(ValueError, "CUDA"):
                run_experiment(self.path, Budget("count", 0.2), seed=11,
                               output=self.root / "runs", device="cuda")

    def test_conversion_only_reports_no_profit(self):
        self.manifest["outcomes"] = {"conversion": "active"}
        self.manifest.pop("margin_includes_coupon_cost")
        self.save_manifest()
        report = run_experiment(self.path, Budget("count", 0.2), seed=11, output=self.root / "runs")
        self.assertEqual(set(report["policies"]), {"conversion", "random"})
        self.assertIn("net_margin", report["unavailable_objectives"])
        self.assertNotIn("cost", report["policies"]["conversion"])

    def test_revenue_only_with_observed_cost_has_net_policy_and_paired_net(self):
        self.manifest["outcomes"] = {"revenue": "margin", "coupon_cost": "cost"}
        self.manifest.pop("margin_includes_coupon_cost")
        self.save_manifest()
        candidate = Path(__file__).resolve().parents[1] / "src/couponevo/candidate.py"
        report = run_experiment(self.path, Budget("count", 0.2), seed=11,
                                output=self.root / "runs", compare_candidate_path=candidate)
        self.assertIn("net_revenue", report["policies"])
        self.assertIn("net", report["paired_vs_baseline"]["net_revenue"])

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
        self.assertTrue(any("每位选中用户 1.0" in note and "全体评估用户平均" in note
                            for note in report["notes"]))

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

        with patch("couponevo.agent.subprocess.run", side_effect=fake_codex):
            revise_candidate(candidate, report, feature_gaps_path=self.root / "feature_gaps.md",
                             codex_bin="fake-codex")
        self.assertIn("# revised", candidate.read_text())
        self.assertIn("历史活跃度", (self.root / "feature_gaps.md").read_text())
        self.assertFalse((self.root / "unwanted.py").exists())

    def test_agent_revision_can_be_reevaluated_without_dataset_changes(self):
        candidate = self.root / "candidate.py"
        source = Path(__file__).resolve().parents[1] / "src/couponevo/candidate.py"
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
        with patch("couponevo.agent.subprocess.run", side_effect=fake_codex):
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

        with patch("urllib.request.urlopen", return_value=FakeResponse()) as call:
            revise_candidate_deepseek(candidate, report, api_key="test-key",
                                      feature_gaps_path=self.root / "feature_gaps.md")
        self.assertIn("return 1", candidate.read_text())
        self.assertIn("历史活跃度", (self.root / "feature_gaps.md").read_text())
        request = call.call_args.args[0]
        body = json.loads(request.data)
        self.assertEqual(body["model"], "deepseek-flash")
        self.assertEqual(body["thinking"], {"type": "enabled"})
        self.assertEqual(body["reasoning_effort"], "high")

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
        with patch("urllib.request.urlopen", side_effect=responses) as call:
            revise_candidate_deepseek(candidate, report, api_key="test-key")
        self.assertEqual(call.call_count, 2)
        self.assertIn("return 1", candidate.read_text())

    def test_agent_accepts_torch_transformed_outcome_candidate(self):
        candidate = self.root / "candidate.py"
        source = Path(__file__).resolve().parents[1] / "src/couponevo/candidate.py"
        candidate.write_bytes(source.read_bytes())
        report = self.root / "report.md"
        report.write_text("Report")
        before = candidate.read_text()
        old = ("        learner = TLearner(models=TorchRidgeRegressor(device=device))\n"
               "        learner.fit(y, arm, X=x_train)\n"
               "        result[f\"{name}_uplift\"] = learner.effect(x_target)")
        new = ("        propensity = arm.mean()\n"
               "        pseudo = y * (arm / propensity - (1 - arm) / (1 - propensity))\n"
               "        model = TorchRidgeRegressor(device=device).fit(x_train, pseudo)\n"
               "        result[f\"{name}_uplift\"] = model.predict(x_target)")
        self.assertIn(old, before)
        revised = (before.replace("from econml.metalearners import TLearner\n", "")
                  .replace("from sklearn.base import BaseEstimator, RegressorMixin\n", "")
                  .replace("class TorchRidgeRegressor(RegressorMixin, BaseEstimator):",
                           "class TorchRidgeRegressor:")
                  .replace(old, new)
                  .replace('"econml.TLearner"', '"torch.transformed_outcome"'))

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *_):
                return False

            def read(self):
                return json.dumps({"choices": [{"finish_reason": "stop", "message": {
                    "content": json.dumps({"candidate_py": revised})}}]}).encode()

        with patch("urllib.request.urlopen", return_value=FakeResponse()):
            revise_candidate_deepseek(candidate, report, api_key="test-key")
        self.assertIn("torch.transformed_outcome", candidate.read_text())
        result = run_experiment(self.path, Budget("count", 0.2), seed=11,
                                output=self.root / "runs", candidate_path=candidate)
        self.assertEqual(result["model_device"], "cpu")

    def test_agent_command_rejects_code_only_change_and_restores_candidate(self):
        candidate = self.root / "candidate.py"
        source = Path(__file__).resolve().parents[1] / "src/couponevo/candidate.py"
        candidate.write_bytes(source.read_bytes())
        original = candidate.read_bytes()

        def fake_revision(path, *_args, **_kwargs):
            path.write_bytes(path.read_bytes() + b"\n# no prediction change\n")

        args = ["couponevo", "agent", str(self.path), "--budget-kind", "count",
                "--budget", "0.2", "--candidate", str(candidate), "--output",
                str(self.root / "runs"), "--agent-provider", "deepseek", "--unsafe-local-execution"]
        with patch.object(sys, "argv", args), patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-key"}), \
                patch("couponevo.cli.revise_candidate_deepseek", side_effect=fake_revision):
            with self.assertRaisesRegex(ValueError, "predictions and policies unchanged"):
                main()
            self.assertNotIn("DEEPSEEK_API_KEY", os.environ)
        self.assertEqual(candidate.read_bytes(), original)

    def test_agent_command_accepts_policy_only_revision(self):
        candidate = self.root / "candidate.py"
        source = Path(__file__).resolve().parents[1] / "src/couponevo/candidate.py"
        candidate.write_bytes(source.read_bytes())

        def fake_revision(path, *_args, **_kwargs):
            path.write_text(path.read_text() + "\n"
                            "def choose_policy(scores, costs, budget_kind, budget_value):\n"
                            "    chosen = np.zeros(len(scores), dtype=bool)\n"
                            "    chosen[0] = True\n"
                            "    return chosen\n")

        args = ["couponevo", "agent", str(self.path), "--budget-kind", "count",
                "--budget", "0.2", "--candidate", str(candidate), "--output",
                str(self.root / "runs"), "--agent-provider", "deepseek", "--unsafe-local-execution"]
        with patch.object(sys, "argv", args), patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-key"}), \
                patch("couponevo.cli.revise_candidate_deepseek", side_effect=fake_revision), \
                patch("couponevo.cli.analyze_reports_deepseek") as analyze:
            main()
        prior, revised = analyze.call_args.args
        self.assertEqual(prior["prediction_sha256"], revised["prediction_sha256"])
        self.assertNotEqual(prior["policy_sha256"]["active"], revised["policy_sha256"]["active"])

    def test_deepseek_agent_analyzes_revised_report(self):
        candidate = self.root / "candidate.py"
        source = Path(__file__).resolve().parents[1] / "src/couponevo/candidate.py"
        candidate.write_bytes(source.read_bytes())

        def fake_revision(path, *_args, **_kwargs):
            path.write_text(path.read_text().replace(
                'result[f"{name}_uplift"] = learner.effect(x_target)',
                'result[f"{name}_uplift"] = learner.effect(x_target) + 0.1'))

        args = ["couponevo", "agent", str(self.path), "--budget-kind", "count",
                "--budget", "0.2", "--candidate", str(candidate), "--output",
                str(self.root / "runs"), "--agent-provider", "deepseek", "--unsafe-local-execution"]
        with patch.object(sys, "argv", args), patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-key"}), \
                patch("couponevo.cli.revise_candidate_deepseek", side_effect=fake_revision), \
                patch("couponevo.cli.analyze_reports_deepseek") as analyze:
            main()
        self.assertEqual(analyze.call_count, 1)
        prior, revised = analyze.call_args.args
        self.assertNotEqual(prior["prediction_sha256"], revised["prediction_sha256"])
        self.assertEqual(analyze.call_args.kwargs["provider"].api_key, "test-key")
        self.assertEqual(analyze.call_args.kwargs["output_dir"], self.root / "runs" / revised["run_id"])

    def test_provider_config_routes_model_and_secret_to_both_agent_steps(self):
        candidate = self.root / "candidate.py"
        source = Path(__file__).resolve().parents[1] / "src/couponevo/candidate.py"
        candidate.write_bytes(source.read_bytes())
        config = self.root / "agent.json"
        config.write_text(json.dumps({
            "provider_url": "https://proxy.example/v1", "model": "flash-proxy",
            "api_key_env": "TEST_AGENT_KEY", "thinking": "enabled",
            "iteration_effort": "high", "review_effort": "max",
        }))

        def fake_revision(path, *_args, **_kwargs):
            path.write_text(path.read_text().replace(
                'result[f"{name}_uplift"] = learner.effect(x_target)',
                'result[f"{name}_uplift"] = learner.effect(x_target) + 0.1'))

        args = ["couponevo", "agent", str(self.path), "--budget-kind", "count",
                "--budget", "0.2", "--candidate", str(candidate), "--output",
                str(self.root / "runs"), "--agent-config", str(config), "--unsafe-local-execution"]
        with patch.object(sys, "argv", args), patch.dict(os.environ, {"TEST_AGENT_KEY": "test-key"}), \
                patch("couponevo.cli.revise_candidate_deepseek", side_effect=fake_revision) as revise, \
                patch("couponevo.cli.analyze_reports_deepseek") as analyze:
            main()
            self.assertNotIn("TEST_AGENT_KEY", os.environ)
        revision_provider = revise.call_args.kwargs["provider"]
        analysis_provider = analyze.call_args.kwargs["provider"]
        self.assertIs(revision_provider, analysis_provider)
        self.assertEqual(revision_provider.url, "https://proxy.example/v1")
        self.assertEqual(revision_provider.model, "flash-proxy")
        self.assertEqual(revision_provider.api_key, "test-key")
        self.assertEqual(revision_provider.iteration_effort, "high")
        self.assertEqual(revision_provider.review_effort, "max")
        revised = analyze.call_args.args[1]
        artifact = self.root / "runs" / revised["run_id"]
        self.assertEqual(revised["agent"]["provider"], "api")
        self.assertNotIn("test-key", (artifact / "report.json").read_text())

    def test_direct_deepseek_url_enables_thinking(self):
        candidate = self.root / "candidate.py"
        source = Path(__file__).resolve().parents[1] / "src/couponevo/candidate.py"
        candidate.write_bytes(source.read_bytes())

        def fake_revision(path, *_args, **_kwargs):
            path.write_text(path.read_text().replace(
                'result[f"{name}_uplift"] = learner.effect(x_target)',
                'result[f"{name}_uplift"] = learner.effect(x_target) + 0.1'))

        args = ["couponevo", "agent", str(self.path), "--budget-kind", "count",
                "--budget", "0.2", "--candidate", str(candidate), "--output",
                str(self.root / "runs"), "--agent-provider-url", "https://api.deepseek.com",
                "--agent-model", "deepseek-flash", "--unsafe-local-execution"]
        with patch.object(sys, "argv", args), patch.dict(os.environ, {"AGENT_API_KEY": "test-key"}), \
                patch("couponevo.cli.revise_candidate_deepseek", side_effect=fake_revision) as revise, \
                patch("couponevo.cli.analyze_reports_deepseek"):
            main()
        provider = revise.call_args.kwargs["provider"]
        self.assertEqual(provider.thinking, "enabled")
        self.assertEqual(provider.iteration_effort, "high")
        self.assertEqual(provider.review_effort, "max")

    def test_search_command_runs_agent_steps_and_resumes(self):
        candidate = Path(__file__).resolve().parents[1] / "src/couponevo/candidate.py"
        calls = []

        def propose(_provider, context):
            calls.append(context)
            source = context["available"]["seed"]["candidate_py"]
            return {"operator": "draft", "parent_ids": [], "hypothesis": f"search idea {len(calls)}",
                    "candidate_py": source + f"\n# search idea {len(calls)}\n"}

        args = ["couponevo", "search", str(self.path), "--budget-kind", "count", "--budget", "0.2",
                "--candidate", str(candidate), "--objective", "active", "--max-steps", "1",
                "--search-id", "test-search", "--output", str(self.root / "runs"),
                "--agent-provider", "deepseek", "--unsafe-local-execution"]
        with patch.object(sys, "argv", args), patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-key"}), \
                patch("couponevo.cli.propose_search_candidate", side_effect=propose), \
                patch("couponevo.cli.reflect_search_step", return_value={
                    "verdict": "inconclusive", "evidence": "validation only",
                    "lesson": "Try another policy", "next_direction": "Continue"}), \
                patch("couponevo.cli.analyze_reports_deepseek", return_value={"high": {}}):
            main()
            self.assertNotIn("DEEPSEEK_API_KEY", os.environ)
        journal_path = self.root / "runs/test-search/journal.json"
        first = json.loads(journal_path.read_text())
        self.assertEqual(len(first["steps"]), 1)
        self.assertEqual(first["steps"][0]["report"]["holdout"], "validation")
        self.assertEqual(first["agent"]["model"], "deepseek-flash")
        self.assertEqual(first["steps"][0]["reflection"]["lesson"], "Try another policy")
        self.assertEqual(first["budget_exhausted_decision"],
                         {"after_step": 1, "reason": "no_terminal_action"})
        self.assertEqual(len(calls), 2)
        self.assertTrue(calls[1]["experiment_budget_exhausted"])
        self.assertNotIn("test-key", journal_path.read_text())

        resumed_args = args.copy()
        resumed_args[resumed_args.index("--max-steps") + 1] = "2"
        resumed_args.append("--resume")
        with patch.object(sys, "argv", resumed_args), patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-key"}), \
                patch("couponevo.cli.propose_search_candidate", side_effect=propose), \
                patch("couponevo.cli.reflect_search_step", return_value={
                    "verdict": "inconclusive", "evidence": "validation only",
                    "lesson": "Try another policy", "next_direction": "Continue"}), \
                patch("couponevo.cli.analyze_reports_deepseek", return_value={"high": {}}):
            main()
        resumed = json.loads(journal_path.read_text())
        self.assertEqual(len(resumed["steps"]), 2)
        self.assertEqual(resumed["budget_exhausted_decision"],
                         {"after_step": 2, "reason": "no_terminal_action"})
        self.assertEqual(len(calls), 4)
        self.assertFalse(calls[2].get("experiment_budget_exhausted", False))
        self.assertIn("analysis", calls[2]["history"][-1])
        self.assertIn("reflection", calls[2]["history"][-1])
        self.assertTrue(calls[3]["experiment_budget_exhausted"])

    def test_search_rejects_final_holdout(self):
        args = ["couponevo", "search", str(self.path), "--budget-kind", "count", "--budget", "0.2",
                "--objective", "active", "--final", "--agent-provider", "deepseek"]
        with patch.object(sys, "argv", args), self.assertRaises(SystemExit):
            main()

    def test_finalize_command_evaluates_frozen_search(self):
        candidate = Path(__file__).resolve().parents[1] / "src/couponevo/candidate.py"
        args = ["couponevo", "search", str(self.path), "--budget-kind", "count", "--budget", "0.2",
                "--candidate", str(candidate), "--objective", "active", "--max-steps", "1",
                "--search-id", "finalize-trial", "--output", str(self.root / "runs"),
                "--agent-provider", "deepseek", "--unsafe-local-execution"]
        proposal = {"operator": "draft", "parent_ids": [], "hypothesis": "candidate",
                    "candidate_py": candidate.read_text() + "\n# experiment\n"}
        with patch.object(sys, "argv", args), patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-key"}), \
                patch("couponevo.cli.propose_search_candidate", return_value=proposal), \
                patch("couponevo.cli.reflect_search_step", return_value={
                    "verdict": "inconclusive", "evidence": "validation only",
                    "lesson": "No clear gain", "next_direction": "Stop"}), \
                patch("couponevo.cli.analyze_reports_deepseek", return_value={"high": {}}):
            main()
        final_args = ["couponevo", "finalize", str(self.path), "--budget-kind", "count", "--budget", "0.2",
                      "--candidate", str(candidate), "--objective", "active", "--search-id", "finalize-trial",
                      "--output", str(self.root / "runs"), "--bootstrap-reps", "100"]
        with patch.object(sys, "argv", final_args):
            main()
        journal = json.loads((self.root / "runs/finalize-trial/journal.json").read_text())
        self.assertEqual(journal["final"]["report"]["holdout"], "test")


if __name__ == "__main__":
    unittest.main()
