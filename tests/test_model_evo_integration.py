"""The optional external research harness at CouponEvo's search boundary."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from couponevo.data import load_dataset, split_dataset
from couponevo.evaluate import Budget
from couponevo.harness import build_harness_context, validate_research
from couponevo.search import _task, run_search


@unittest.skipUnless(importlib.util.find_spec("model_evo_harness"),
                     "install the optional model-evo-harness package to run integration tests")
class ModelEvoIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.csv = self.root / "data.csv"
        pd.DataFrame({"id": range(100), "arm": [i % 2 for i in range(100)],
                      "x": [i / 100 for i in range(100)],
                      "y": [i % 3 == 0 for i in range(100)],
                      "y2": [i % 5 == 0 for i in range(100)]}).to_csv(self.csv, index=False)
        self.manifest = self.root / "manifest.json"
        self.spec = {"dataset": "data.csv", "unit_id": "id", "features": ["x"],
                     "feature_timing": "pre_treatment", "outcomes": {"active": "y"},
                     "treatment": {"column": "arm", "control": 0, "treated": 1,
                                   "probability": 0.5, "probability_source": "protocol",
                                   "probability_reference": "test"}}
        self.manifest.write_text(json.dumps(self.spec))
        self.seed = Path(__file__).resolve().parents[1] / "src/couponevo/candidate.py"
        self.mode = Path("model-evo")
        self.kwargs = dict(manifest_path=self.manifest, budget=Budget("count", 0.2), seed=7,
                           output=self.root / "runs", initial_candidate=self.seed,
                           objective="active", search_id="external-test", harness_path=self.mode)
        self.research = {
            "direction": "Test a new policy threshold", "mechanism": "Adjust the threshold over existing predictions",
            "why_now": "The validation policy has room to improve", "data_rationale": "No new raw input is needed",
            "input_features": [], "comparison": "Compare with the same fixed-budget baseline",
            "falsification": "No policy value gain under the frozen validation split",
            "alternatives": [{"direction": "Retune the predictor", "mechanism": "Change model depth",
                              "reason": "Does not isolate the policy rule"}],
        }
        self.proposal = {"action": "experiment", "operator": "draft", "parent_ids": [],
                         "hypothesis": "A threshold change improves policy value",
                         "expected_result": "Validation active policy value increases",
                         "candidate_py": self.seed.read_text(), "research": self.research}
        self.report = {"run_id": "unit", "holdout": "validation",
                       "policies": {"active": {"effects": {"active": {"mean": 0.0}}}}}

    def context(self):
        return build_harness_context(self.mode, self.manifest, 7, False, objective="active")

    def test_snapshot_exposes_only_proven_capabilities_and_training_statistics(self):
        before = self.context()
        snapshot = before["task_snapshot"]
        self.assertEqual(before["source"], "ModelEvoHarness")
        self.assertEqual(snapshot["stage"], "policy")
        self.assertEqual(snapshot["fields"], ["x"])
        self.assertEqual(snapshot["objective"], {"name": "active", "direction": "max"})
        self.assertEqual(set(snapshot["capabilities"]),
                         {"tabular_features", "observed_outcome_labels",
                          "assignment_or_exposure_propensity"})
        self.assertEqual(before["dataset_profile"]["training_rows"], 60)
        by_id = {entry["family_id"]: entry for entry in before["applicability"]}
        self.assertEqual(by_id["feature_interactions"]["status"], "ready")
        by_method = {entry["method_id"]: entry for entry in before["method_applicability"]}
        self.assertEqual(by_method["fm"]["status"], "ready")
        self.assertEqual(by_method["din"]["status"], "needs_data")
        self.assertEqual(by_id["sequence_ranking"]["status"], "needs_data")
        self.assertIn("families", before["catalog"])

        split = split_dataset(load_dataset(self.manifest), 7)
        frame = pd.read_csv(self.csv)
        frame.loc[[*split.validation.index, *split.test.index], "x"] = float("nan")
        frame.to_csv(self.csv, index=False)
        after = self.context()
        self.assertEqual(before["dataset_profile"], after["dataset_profile"])

        self.spec["outcomes"]["visit"] = "y2"
        self.manifest.write_text(json.dumps(self.spec))
        self.assertIn("multiple_outcomes", self.context()["task_snapshot"]["capabilities"])

    def test_external_validator_blocks_unavailable_family_but_accepts_novel_direction(self):
        context = self.context()
        blocked = {**self.proposal, "research": {**self.research, "family_id": "sequence_ranking"}}
        with self.assertRaisesRegex(ValueError, "needs_data"):
            validate_research(blocked, context)
        canonical = validate_research(self.proposal, context)
        self.assertEqual(canonical["direction"], self.research["direction"])
        self.assertEqual(canonical["input_fields"], [])
        self.assertEqual(canonical["expected_result"], self.proposal["expected_result"])
        self.assertNotIn("input_features", canonical)
        self.assertNotIn("family_id", canonical)
        method = {**self.proposal, "research": {**self.research, "method_id": "fm"}}
        self.assertEqual(validate_research(method, context)["method_id"], "fm")
        unavailable = {**self.proposal, "research": {**self.research, "method_id": "din"}}
        with self.assertRaisesRegex(ValueError, "needs_data"):
            validate_research(unavailable, context)

    def test_package_digests_isolate_resume_and_record_canonical_research(self):
        task = _task(self.manifest, self.kwargs["budget"], 7, self.seed, "active", "cpu", False,
                     harness_path=self.mode)
        self.assertEqual(task["harness"]["source"], "ModelEvoHarness")
        self.assertEqual(len(task["harness"]["catalog_digest"]), 64)
        self.assertEqual(len(task["harness"]["implementation_digest"]), 64)
        with patch("model_evo_harness.implementation_digest", return_value="changed"):
            changed = _task(self.manifest, self.kwargs["budget"], 7, self.seed, "active", "cpu", False,
                            harness_path=self.mode)
            self.assertNotEqual(task, changed)
        with patch("couponevo.harness._model_evo_revision", return_value="new-commit"):
            changed = _task(self.manifest, self.kwargs["budget"], 7, self.seed, "active", "cpu", False,
                            harness_path=self.mode)
            self.assertNotEqual(task, changed)

        with patch("couponevo.search._evaluate", return_value=self.report):
            journal = run_search(**self.kwargs, max_steps=1, proposer=lambda _: self.proposal)
            with patch("model_evo_harness.catalog_digest", return_value="changed"):
                with self.assertRaisesRegex(ValueError, "changed"):
                    run_search(**self.kwargs, max_steps=2, proposer=lambda _: self.proposal, resume=True)
        self.assertEqual(journal["steps"][0]["research"]["input_fields"], [])
        self.assertEqual(journal["steps"][0]["research"]["expected_result"],
                         self.proposal["expected_result"])

    def test_cli_model_evo_flag_reaches_real_search_loop(self):
        from couponevo.cli import main

        args = ["couponevo", "search", str(self.manifest), "--budget-kind", "count",
                "--budget", "0.2", "--objective", "active", "--search-id", "cli-external",
                "--seed", "7", "--max-steps", "1", "--output", str(self.root / "runs"),
                "--harness", "model-evo", "--agent-provider", "deepseek", "--unsafe-local-execution"]
        reflection = {"verdict": "inconclusive", "evidence": "No clear gain",
                      "lesson": "Try another mechanism", "next_direction": "Feature interaction"}
        with patch("sys.argv", args), patch.dict("os.environ", {"DEEPSEEK_API_KEY": "test-key"}), \
                patch("couponevo.search._evaluate", return_value=self.report), \
                patch("couponevo.cli.propose_search_candidate", side_effect=[
                    self.proposal, {"action": "stop", "reason": "Budget exhausted"}]) as propose, \
                patch("couponevo.cli.reflect_search_step", return_value=reflection), \
                patch("couponevo.cli.analyze_reports_deepseek", return_value={"high": {}}):
            main()
        context = propose.call_args_list[0].args[1]
        self.assertEqual(context["harness"]["source"], "ModelEvoHarness")
        journal = json.loads((self.root / "runs/cli-external/journal.json").read_text())
        self.assertEqual(journal["steps"][0]["research"]["input_fields"], [])
        self.assertNotIn("test-key", json.dumps(journal))


if __name__ == "__main__":
    unittest.main()
