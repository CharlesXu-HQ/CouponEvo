"""Component lineage survives CouponEvo's proposal, review, and memory boundaries."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from couponevo.experience import load_experience
from couponevo.harness import research_history, validate_model_evo_reflection, validate_research
from couponevo.search import _context


def design(backbone="MLP", scope="initialize", parent=None):
    return {"estimator": "T-learner", "backbone": backbone, "change_scope": scope,
            "parent_trial_id": parent, "rationale": "Test cross terms within the backbone",
            "data_fit": "Existing numeric inputs support learned interactions",
            "comparison_plan": "Freeze all other training settings",
            "components": [{"id": "cross", "mechanism": "Learned feature crossing",
                            "code_sections": ["CrossBlock"], "input_fields": ["x"],
                            "required_capabilities": ["tabular_features"]}],
            "inheritance": []}


class ModelCompositionTests(unittest.TestCase):
    def test_history_keeps_model_design_and_component_reflection(self):
        recorded = design()
        step = {"id": "a", "status": "evaluated", "research": {
            "direction": "cross", "mechanism": "cross", "model_design": recorded},
            "reflection": {"technical_experience": {"component_assessments": [{
                "component_id": "cross", "outcome": "inconclusive"}]}}}
        history = research_history([step])
        self.assertEqual(history[0]["model_design"], recorded)
        self.assertEqual(history[0]["reflection"], step["reflection"])

    def test_research_bridge_passes_current_history_as_composition_sources(self):
        steps = [{"id": "a", "status": "evaluated", "research": {"model_design": design()}}]
        sources = [{"id": "a", "research": steps[0]["research"]}]
        package = SimpleNamespace(composition_sources=Mock(return_value=sources),
                                  validate_research=Mock(return_value={"model_design": design()}))
        harness = {"source": "ModelEvoHarness", "task_snapshot": {}, "catalog": {}}
        proposal = {"research": {"model_design": design(), "prediction_semantics": "direct_cate"}}
        with patch("couponevo.harness._model_evo_package", return_value=package):
            result = validate_research(proposal, harness, steps=steps, evidence=[])
        self.assertEqual(result["model_design"], design())
        self.assertEqual(package.validate_research.call_args.kwargs["sources"], sources)
        self.assertEqual(package.composition_sources.call_args.args[0][0]["id"], "a")

    def test_declared_lineage_parent_must_match_the_actual_primary_code_parent(self):
        current = design(scope="local", parent="a")
        package = SimpleNamespace(validate_research=Mock(return_value={"model_design": current}))
        harness = {"source": "ModelEvoHarness", "task_snapshot": {}, "catalog": {}}
        proposal = {"operator": "improve", "parent_ids": ["different-parent"],
                    "research": {"model_design": current}}
        with patch("couponevo.harness._model_evo_package", return_value=package):
            with self.assertRaisesRegex(ValueError, "primary.*parent"):
                validate_research(proposal, harness)

    def test_context_adds_at_most_three_latest_eligible_backbone_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "seed.py").write_text("seed source")
            steps = []
            for index in range(9):
                identifier = f"step-{index}"
                (root / f"{identifier}.py").write_text(f"source {identifier}")
                steps.append({"id": identifier, "candidate": f"{identifier}.py",
                              "status": "evaluated", "score": index, "eligibility": "eligible",
                              "research": {"direction": "cross", "mechanism": "cross",
                                           "model_design": design(f"network-{index}")}})
            steps[5]["eligibility"] = "blocked_implementation"
            journal = {"task": {"dataset": "d", "manifest": "m", "objective": "active", "budget": {}},
                       "baseline": {"id": "seed", "candidate": "seed.py", "status": "evaluated", "score": 0},
                       "steps": steps, "best_id": "step-8"}
            context = _context(journal, root)
        self.assertEqual(set(context["available"]), {"seed", "step-8", "step-7", "step-6", "step-4", "step-3"})
        self.assertEqual(len(context["history"]), 10)
        self.assertEqual(context["available"]["step-4"]["candidate_py"], "source step-4")

    def test_context_groups_stable_ids_independently_of_model_descriptions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "seed.py").write_text("seed source")
            steps = []
            for index, backbone_id in enumerate(("a", "a", "b", "c", "main", "main", "main", "main")):
                identifier = f"step-{index}"
                description = "same prose, different networks" if index in (2, 3) else f"local description {index}"
                recorded = {**design(description), "estimator_id": "t-learner", "backbone_id": backbone_id}
                (root / f"{identifier}.py").write_text(f"source {identifier}")
                steps.append({"id": identifier, "candidate": f"{identifier}.py", "status": "evaluated",
                              "score": index, "eligibility": "eligible", "research": {
                                  "direction": "cross", "mechanism": "cross", "model_design": recorded}})
            journal = {"task": {"dataset": "d", "manifest": "m", "objective": "active", "budget": {}},
                       "baseline": {"id": "seed", "candidate": "seed.py", "status": "evaluated", "score": 0},
                       "steps": steps, "best_id": "step-7"}
            original = json.dumps(journal, sort_keys=True)
            context = _context(journal, root)
            self.assertEqual(json.dumps(journal, sort_keys=True), original)
        self.assertEqual(set(context["available"]), {"seed", "step-7", "step-6", "step-3", "step-2", "step-1"})
        self.assertEqual(context["available"]["step-3"]["model_identity"],
                         {"estimator_id": "t-learner", "backbone_id": "c"})

    def test_recorded_gpu_local_ablation_keeps_identity_despite_changed_description(self):
        path = Path(__file__).resolve().parents[1] / "docs/research/compositional-harness-gpu-validation-2026-10-05.json"
        attempt = json.loads(path.read_text())["attempts"][1]
        source = attempt["steps"][0]
        response = next(item for item in attempt["provider_final_responses"] if item["file"] == "call-009.json")
        proposal = json.loads(response["content"])
        snapshot = {"fields": [f"V{index}" for index in range(1, 8)], "capabilities": [
            "tabular_features", "observed_outcome_labels", "assignment_or_exposure_propensity", "decision_rule_adapter"],
            "framework": "pytorch", "stage": "policy", "model_design_required": True}
        before = json.dumps(source, sort_keys=True)
        harness = {"source": "ModelEvoHarness", "task_snapshot": snapshot, "catalog": {}}
        self.assertNotEqual(proposal["research"]["model_design"]["backbone"], source["research"]["model_design"]["backbone"])
        from couponevo.research_evidence import search_evidence
        facts = search_evidence({"baseline": attempt["baseline"], "steps": [source], "harness": {
            "dataset_profile": {"training_rows": attempt["splits"][0]["counts"]["train"]}}})
        checked = validate_research(proposal, harness, steps=[source], evidence=facts)
        from couponevo.harness import model_design_identity
        self.assertEqual(model_design_identity(checked["model_design"]), model_design_identity(source["research"]["model_design"]))
        self.assertEqual(checked["model_design"]["change_scope"], "local")
        self.assertEqual([item["decision"] for item in checked["model_design"]["inheritance"]],
                         ["retain", "adapt", "retain", "retain", "retain"])
        self.assertEqual(json.dumps(source, sort_keys=True), before)

    def test_reflection_bridge_identifies_current_design_and_trial_status(self):
        core = Mock(side_effect=lambda reflection, *_args, **_kwargs: reflection)
        reflection = {"technical_experience": {key: "evidence" for key in
                      ("lesson", "evidence", "uncertainty", "next_test")},
                      "business_experience": {"status": "not_observable", "reason": "No semantic labels"}}
        observation = {"trial_id": "b", "status": "failed", "research": {"model_design": design()},
                       "task_snapshot": {}, "trial_history": []}
        with patch("couponevo.harness._model_evo_package", return_value=SimpleNamespace(validate_reflection=core)):
            validate_model_evo_reflection(reflection, observation, evidence=[])
        evaluation = core.call_args.args[1]
        self.assertEqual(evaluation["research"], observation["research"])
        self.assertEqual(evaluation["trial_id"], "b")
        self.assertEqual(evaluation["trial_status"], "failed")

    def test_experience_preserves_component_assessments_and_invalidates_false_claims(self):
        task = {"dataset": "v1", "manifest": "m1", "objective": "active", "budget": {},
                "seed": 7, "framework": "e1", "agent_workflow": "a1", "strict_data": False,
                "harness": {"source": "ModelEvoHarness"}}
        component = {"component_id": "cross", "outcome": "promising", "evidence": "Joint validation gain",
                     "compatibility_limits": "numeric tabular only", "next_test": "Remove cross block",
                     "attribution": "unverified"}
        reflection = {"verdict": "inconclusive", "technical_experience": {
            "lesson": "Joint candidate promising", "component_assessments": [component]},
            "business_experience": {"status": "not_observable", "reason": "No semantics"}}
        step = {"id": "a", "status": "evaluated", "operator": "improve", "score": 0.02,
                "hypothesis": "Crosses may help", "research": {"model_design": design()}, "reflection": reflection}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "completed"
            path.mkdir()
            journal = {"task": task, "baseline": {"score": 0.01}, "stop": {"reason": "budget"}, "steps": [step]}
            (path / "journal.json").write_text(json.dumps(journal))
            lesson = load_experience(Path(directory), task)[0]
            self.assertEqual(lesson["technical_experience"]["component_assessments"], [component])
            step["eligibility"] = "blocked_implementation"
            (path / "journal.json").write_text(json.dumps(journal))
            invalid = load_experience(Path(directory), task)[0]
        assessment = invalid["technical_experience"]["component_assessments"][0]
        self.assertEqual(assessment["component_id"], "cross")
        self.assertEqual(assessment["outcome"], "invalid")
        self.assertEqual(assessment["attribution"], "unverified")


if __name__ == "__main__":
    unittest.main()
