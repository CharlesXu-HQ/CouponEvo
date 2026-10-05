"""The optional external research harness at CouponEvo's search boundary."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

import couponevo.harness as harness_module
from couponevo.data import load_dataset, split_dataset
from couponevo.evaluate import Budget
from couponevo.harness import build_harness_context, refresh_model_evo, validate_research
from couponevo.search import _evaluate, _task, _validate_proposal, run_search


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

    def test_refresh_requires_clean_checkout_and_local_package(self):
        with patch("couponevo.harness.subprocess.run",
                   return_value=SimpleNamespace(stdout=" M README.md\n")) as git:
            with self.assertRaisesRegex(RuntimeError, "uncommitted changes"):
                refresh_model_evo(self.mode)
        self.assertEqual(len(git.call_args_list), 1)
        self.assertEqual(git.call_args.args[0][-2:], ["status", "--porcelain"])

        revision = harness_module._model_evo_revision(harness_module._model_evo_package())

        def clean_git(command, **_):
            return SimpleNamespace(stdout=revision + "\n" if command[-2:] == ["rev-parse", "HEAD"]
                                   else "")

        with patch("couponevo.harness.subprocess.run",
                   side_effect=clean_git) as git, \
                patch("couponevo.harness._model_evo_package",
                      return_value=SimpleNamespace(__file__="/tmp/other/model_evo_harness/__init__.py")):
            with self.assertRaisesRegex(RuntimeError, "submodule checkout"):
                refresh_model_evo(self.mode)
        self.assertTrue(any("update" in call.args[0] and "--remote" in call.args[0]
                            for call in git.call_args_list))

    def test_refresh_rejects_loaded_old_code_again_on_retry(self):
        package = harness_module._model_evo_package()
        old_commit = harness_module._model_evo_revision(package)
        self.assertIsNotNone(old_commit)
        current = old_commit

        def git_run(command, **_):
            nonlocal current
            if "--remote" in command:
                current = "different-commit"
            if command[-2:] == ["rev-parse", "HEAD"]:
                return SimpleNamespace(stdout=current + "\n")
            return SimpleNamespace(stdout="")

        with patch("couponevo.harness.subprocess.run", side_effect=git_run) as git:
            with self.assertRaisesRegex(RuntimeError, "changed while loaded"):
                refresh_model_evo(self.mode)
            updates_after_first = sum("update" in call.args[0] for call in git.call_args_list)
            self.assertEqual(updates_after_first, 1)
            with self.assertRaisesRegex(RuntimeError, "changed while loaded"):
                refresh_model_evo(self.mode)
            self.assertEqual(sum("update" in call.args[0] for call in git.call_args_list),
                             updates_after_first)

    def test_new_search_refreshes_once_but_resume_and_existing_root_do_not(self):
        with patch("couponevo.search.refresh_model_evo") as refresh, \
                patch("couponevo.search._evaluate", return_value=self.report):
            run_search(**self.kwargs, max_steps=1,
                       proposer=lambda _: {"action": "stop", "reason": "done"})
            refresh.assert_called_once_with(self.mode)
            with self.assertRaisesRegex(ValueError, "already exists"):
                run_search(**self.kwargs, max_steps=1,
                           proposer=lambda _: {"action": "stop", "reason": "done"})
            refresh.assert_called_once_with(self.mode)

        self.kwargs["search_id"] = "resumable"
        with patch("couponevo.search.refresh_model_evo") as refresh, \
                patch("couponevo.search._evaluate", return_value=self.report):
            run_search(**self.kwargs, max_steps=1, proposer=lambda _: self.proposal)
            run_search(**self.kwargs, max_steps=2, proposer=lambda _: self.proposal,
                       resume=True)
            refresh.assert_called_once_with(self.mode)

    def test_snapshot_exposes_only_proven_capabilities_and_training_statistics(self):
        before = self.context()
        snapshot = before["task_snapshot"]
        self.assertEqual(before["source"], "ModelEvoHarness")
        self.assertEqual(snapshot["framework"], "pytorch")
        self.assertEqual(snapshot["stage"], "policy")
        self.assertEqual(snapshot["fields"], ["x"])
        self.assertEqual(snapshot["objective"], {"name": "active", "direction": "max"})
        self.assertEqual(set(snapshot["capabilities"]),
                         {"tabular_features", "observed_outcome_labels",
                          "assignment_or_exposure_propensity", "decision_rule_adapter"})
        self.assertEqual(snapshot["evaluation_protocol"]["metric"], "active")
        self.assertIn("seed=7", snapshot["evaluation_protocol"]["split"])
        self.assertEqual(before["dataset_profile"]["training_rows"], 60)
        by_id = {entry["family_id"]: entry for entry in before["applicability"]}
        self.assertEqual(by_id["feature_interactions"]["status"], "ready")
        by_method = {entry["method_id"]: entry for entry in before["method_applicability"]}
        self.assertEqual(by_method["fm"]["status"], "ready")
        self.assertIn("fm", before["model_api"]["pytorch"])
        self.assertIn("cardinalities", before["model_api"]["pytorch"]["fm"]["constructor"])
        self.assertEqual(by_method["din"]["status"], "needs_data")

        self.assertEqual(by_method["dcn_v2"]["status"], "needs_data")
        self.assertIn("typed_feature_schema", by_method["dcn_v2"]["missing_capabilities"])
        self.assertEqual(by_id["sequence_ranking"]["status"], "needs_data")
        self.assertEqual(by_id["decision_mapping"]["status"], "ready")
        by_check = {entry["check_id"]: entry for entry in before["decision_applicability"]}
        self.assertEqual(by_check["prediction_to_decision"]["status"], "ready")
        self.assertEqual(by_check["negative_sampling"]["status"], "other_stage")
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

    def test_candidate_import_allowlist_uses_declared_pytorch_modules(self):
        harness = {"source": "ModelEvoHarness", "task_snapshot": {"fields": ["x"]},
                   "catalog": {"model_implementations": [
                       {"framework": "pytorch",
                        "symbol": "model_evo_harness.models.pytorch.interactions:FM"},
                       {"framework": "tensorflow",
                        "symbol": "model_evo_harness.models.tensorflow.interactions:FM"}]}}
        proposal = {"operator": "draft", "parent_ids": [], "hypothesis": "Test a reference FM",
                    "expected_result": "Validation value rises"}
        for module, symbol in (("model_evo_harness.models.pytorch.interactions", "FM"),
                               ("model_evo_harness.models.pytorch.training", "focal_loss")):
            source = f"from {module} import {symbol}\ndef fit_predict(*args): pass\n"
            with self.subTest(module=module):
                self.assertEqual(_validate_proposal({**proposal, "candidate_py": source}, {}, harness)[
                    "candidate_py"], source)
        imported = "import model_evo_harness.models.pytorch.interactions as reference\n" \
                   "def fit_predict(*args): pass\n"
        self.assertEqual(_validate_proposal({**proposal, "candidate_py": imported}, {}, harness)[
            "candidate_py"], imported)
        forbidden = ("model_evo_harness.provider", "model_evo_harness.models.tensorflow.interactions",
                     "model_evo_harness.models.pytorch.undeclared", "model_evo_harness.models.pytorch",
                     ".model_evo_harness.models.pytorch.interactions")
        for module in forbidden:
            source = f"from {module} import FM\ndef fit_predict(*args): pass\n"
            with self.subTest(module=module), self.assertRaisesRegex(ValueError, "unsupported module"):
                _validate_proposal({**proposal, "candidate_py": source}, {}, harness)
        plain = {**proposal, "candidate_py":
                 "from model_evo_harness.models.pytorch.interactions import FM\n"
                 "def fit_predict(*args): pass\n"}
        with self.assertRaisesRegex(ValueError, "unsupported module"):
            _validate_proposal(plain, {})

    def test_evaluator_forwards_model_evo_mode_to_isolated_run(self):
        output = self.root / "isolated-runs"
        report = output / "case-001" / "report.json"
        report.parent.mkdir(parents=True)
        report.write_text(json.dumps(self.report))
        with patch("couponevo.search.subprocess.run",
                   return_value=SimpleNamespace(returncode=0, stdout=str(report.with_suffix(".md")),
                                                stderr="")) as run:
            self.assertEqual(_evaluate(self.manifest, Budget("count", 0.2), 7, output, self.seed,
                                       "cpu", False, 30, sandbox_image="test-image",
                                       harness_path=self.mode), self.report)
        command = run.call_args.args[0]
        self.assertEqual(command[command.index("--harness") + 1], "model-evo")

    def test_explicit_domain_requirements_are_validated_before_snapshot(self):
        requirement = {"id": "recency", "source": "coupon operations owner",
                       "fields": ["prior_coupon_use"], "as_of": "before assignment"}
        self.spec["domain_requirements"] = [requirement]
        self.manifest.write_text(json.dumps(self.spec))
        self.assertEqual(self.context()["task_snapshot"]["domain_requirements"], [requirement])
        self.spec["domain_requirements"] = [{**requirement, "as_of": ""}]
        self.manifest.write_text(json.dumps(self.spec))
        with self.assertRaisesRegex(ValueError, "domain_requirements"):
            self.context()

    def test_ready_family_knowledge_and_training_applicability_reach_agent_context(self):
        package = harness_module._model_evo_package()
        with patch.object(package, "load_guide", create=True,
                          side_effect=lambda family_id: {"family_id": family_id,
                                                          "content": "Local guidance"}), \
                patch.object(package, "training_applicability", create=True,
                             return_value=[{"pattern_id": "hard_mining", "status": "ready"}]):
            context = self.context()
        self.assertEqual(context["knowledge"]["feature_interactions"]["content"],
                         "Local guidance")
        self.assertNotIn("sequence_ranking", context["knowledge"])
        self.assertEqual(context["training_applicability"][0]["pattern_id"], "hard_mining")

    def test_model_evo_data_request_needs_two_evaluated_distinct_mechanisms_or_domain_requirement(self):
        common = {"name": "prior_coupon_use", "definition": "Prior coupon redemption count",
                  "source": "coupon event log", "as_of": "before assignment",
                  "evidence": "Repeated error in coupon response", "validation_plan": "Check cutoff and coverage"}
        experiments = [{"id": "step-001", "status": "evaluated", "research": {"mechanism": "FM cross"}},
                       {"id": "step-002", "status": "evaluated", "research": {"mechanism": "weighted loss"}}]
        request = {**common, "basis": "experimental_evidence", "trial_ids": ["step-001", "step-002"]}
        snapshot = self.context()["task_snapshot"]
        validate_model_evo_data_request = harness_module.validate_model_evo_data_request
        self.assertEqual(validate_model_evo_data_request(request, snapshot, experiments), request)
        for trials in (experiments[:1], [{**experiments[0]},
                                       {**experiments[1], "status": "failed"}],
                       [{**experiments[0]}, {**experiments[1],
                                             "research": {"mechanism": "FM cross"}}]):
            with self.subTest(trials=trials), self.assertRaisesRegex(ValueError, "trial_ids"):
                validate_model_evo_data_request(request, snapshot, trials)
        requirement = {"id": "recency", "source": "coupon operations owner",
                       "fields": ["prior_coupon_use"], "as_of": "before assignment"}
        self.spec["domain_requirements"] = [requirement]
        self.manifest.write_text(json.dumps(self.spec))
        direct = {**common, "basis": "domain_requirement", "requirement_id": "recency"}
        self.assertEqual(validate_model_evo_data_request(direct,
                         self.context()["task_snapshot"], []), direct)
        with self.assertRaisesRegex(ValueError, "requirement_id"):
            validate_model_evo_data_request({**direct, "requirement_id": "invented"},
                                             self.context()["task_snapshot"], [])

    def test_experiment_feature_gap_note_requires_a_validated_data_request(self):
        attempts = []

        def propose(context):
            attempts.append(context)
            if len(attempts) == 1:
                return {**self.proposal, "feature_gaps_md": "Add prior coupon use"}
            self.assertIn("feature_request", context["proposal_error"])
            return self.proposal

        with patch("couponevo.search.refresh_model_evo"), \
                patch("couponevo.search._evaluate", return_value=self.report):
            journal = run_search(**self.kwargs, max_steps=1, proposer=propose)
        self.assertEqual(len(attempts), 2)
        self.assertNotIn("feature_request", journal["steps"][0])
        self.assertFalse((self.root / "runs/external-test/steps/step-001/feature_gaps.md").exists())

    def test_domain_requirement_can_support_experiment_feature_gap_note(self):
        requirement = {"id": "recency", "source": "coupon operations owner",
                       "fields": ["prior_coupon_use"], "as_of": "before assignment"}
        self.spec["domain_requirements"] = [requirement]
        self.manifest.write_text(json.dumps(self.spec))
        request = {"name": "prior_coupon_use", "definition": "Prior coupon redemption count",
                   "source": "coupon event log", "as_of": "before assignment",
                   "evidence": "Explicit business requirement",
                   "validation_plan": "Check cutoff and coverage",
                   "basis": "domain_requirement", "requirement_id": "recency"}
        proposal = {**self.proposal, "feature_gaps_md": "Build prior_coupon_use before assignment",
                    "feature_request": request}
        with patch("couponevo.search.refresh_model_evo"), \
                patch("couponevo.search._evaluate", return_value=self.report):
            journal = run_search(**self.kwargs, max_steps=1, proposer=lambda _: proposal)
        self.assertEqual(journal["steps"][0]["feature_request"], request)
        self.assertTrue((self.root / "runs/external-test/steps/step-001/feature_gaps.md").exists())

    def test_reflection_persists_both_experiences_with_validation_policy_observation(self):
        self.report = {"holdout": "validation", "run_id": "unit",
                       "policies": {"active": {"effects": {"active": {"mean": 0.03,
                                                                          "lower": 0.01,
                                                                          "upper": 0.05}}}}}
        seen = []

        def reflect(observation):
            seen.extend(observation["business_observations"])
            self.assertEqual(observation["harness_source"], "ModelEvoHarness")
            return {"verdict": "inconclusive", "evidence": "Validation interval is exploratory",
                    "lesson": "Policy score alone is insufficient", "next_direction": "Test a loss change",
                    "technical_experience": {"lesson": "Interaction did not isolate loss effects",
                                             "evidence": "Validation score 0.03",
                                             "uncertainty": "One split only", "next_test": "Compare weighted loss"},
                    "business_experience": {"status": "observed",
                                            "observation_id": seen[0]["id"],
                                            "insight": "The active policy estimate is positive",
                                            "limitations": "Validation only; no subgroup inference"}}

        with patch("couponevo.search.refresh_model_evo"), \
                patch("couponevo.search._evaluate", return_value=self.report):
            journal = run_search(**self.kwargs, max_steps=1,
                                 proposer=lambda _: self.proposal, reflector=reflect)
        self.assertEqual(seen[0]["holdout"], "validation")
        self.assertEqual(seen[0]["metric"], "active")
        self.assertEqual(seen[0]["mean"], 0.03)
        self.assertEqual(journal["steps"][0]["business_observations"], seen)
        self.assertEqual(journal["steps"][0]["reflection"]["business_experience"]["observation_id"],
                         seen[0]["id"])
        from couponevo.experience import load_experience
        lessons = load_experience(self.root / "runs", journal["task"])
        self.assertEqual(lessons[0]["technical_experience"]["next_test"], "Compare weighted loss")
        self.assertEqual(lessons[0]["business_experience"]["status"], "observed")
        self.assertNotIn("final", json.dumps(lessons))

    def test_business_observations_use_policy_cost_and_exclude_final_holdout(self):
        report = {"holdout": "validation", "policies": {"net_revenue": {
            "effects": {"revenue": {"mean": 0.07}}, "cost": {"mean": 0.02},
            "net": {"mean": 0.05}}}}
        observations = harness_module.validation_business_observations(report)
        self.assertEqual({item["metric"] for item in observations},
                         {"revenue", "coupon_cost", "net"})
        self.assertEqual(harness_module.validation_business_observations(
            {**report, "holdout": "test"}), [])

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
            with self.assertRaisesRegex(RuntimeError, "changed while loaded"):
                _task(self.manifest, self.kwargs["budget"], 7, self.seed, "active", "cpu", False,
                      harness_path=self.mode)

        with patch("couponevo.search.refresh_model_evo"), \
                patch("couponevo.search._evaluate", return_value=self.report):
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
                      "lesson": "Try another mechanism", "next_direction": "Feature interaction",
                      "technical_experience": {"lesson": "Current candidate did not establish a gain",
                                               "evidence": "No clear gain", "uncertainty": "Validation only",
                                               "next_test": "Try feature interaction"},
                      "business_experience": {"status": "not_observable",
                                              "reason": "No validation policy observation was supplied"}}
        with patch("sys.argv", args), patch.dict("os.environ", {"DEEPSEEK_API_KEY": "test-key"}), \
                patch("couponevo.search._evaluate", return_value=self.report), \
                patch("couponevo.search.refresh_model_evo"), \
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
