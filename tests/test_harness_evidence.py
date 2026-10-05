"""Evidence-bearing CouponEvo requests must reach the external harness intact."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from couponevo.harness import (validate_model_evo_audit_recommendations,
                               validate_model_evo_data_request,
                               validate_model_evo_reflection, validate_research)


FACTS = [{"id": "task.profile", "status": "observed", "scope": "task",
          "source": "training profile", "statement": "Seven numeric input columns"}]
SNAPSHOT = {"fields": ["V2"], "domain_requirements": []}


class HarnessEvidenceBridgeTests(unittest.TestCase):
    def test_data_request_maps_coupon_fields_and_trial_research_to_core(self):
        request = {"name": "recent_spend", "definition": "Need recent spend before coupon",
                   "source": "order log", "as_of": "before assignment",
                   "evidence": "Two trial-specific residuals", "validation_plan": "Rebuild dataset",
                   "basis": "experimental_evidence", "trial_ids": ["step-001", "step-002"],
                   "alternatives_considered": "Checked crossed and pooled models",
                   "evidence_ids": ["step-001.gap", "step-002.gap"]}
        steps = [{"id": "step-001", "status": "evaluated",
                  "research": {"mechanism": "field cross"}},
                 {"id": "step-002", "status": "evaluated",
                  "research": {"mechanism": "weighted loss"}}]
        core = Mock(return_value={})
        with patch("couponevo.harness._model_evo_package",
                   return_value=SimpleNamespace(validate_data_request=core)):
            result = validate_model_evo_data_request(request, SNAPSHOT, steps,
                                                      evidence=FACTS)
        mapped, snapshot, converted = core.call_args.args
        self.assertEqual(mapped["fields"], ["recent_spend"])
        self.assertEqual(mapped["reason"], request["definition"])
        self.assertEqual(mapped["evidence_ids"], request["evidence_ids"])
        self.assertEqual(mapped["alternatives_considered"], request["alternatives_considered"])
        self.assertEqual(snapshot, SNAPSHOT)
        self.assertEqual(converted[0]["proposal"]["research"]["mechanism"], "field cross")
        self.assertEqual(converted[1]["proposal"]["research"]["mechanism"], "weighted loss")
        self.assertEqual(core.call_args.kwargs, {"evidence": FACTS})
        self.assertEqual(result, request)

    def test_data_request_rejects_core_evidence_failure(self):
        request = {"name": "recent_spend", "definition": "Recent spend",
                   "source": "order log", "as_of": "before assignment",
                   "evidence": "Declared timing only", "validation_plan": "Audit",
                   "basis": "experimental_evidence", "trial_ids": ["a", "b"],
                   "alternatives_considered": "Two models", "evidence_ids": ["task.profile"]}
        trials = [{"id": "a", "status": "evaluated", "research": {"mechanism": "A"}},
                  {"id": "b", "status": "evaluated", "research": {"mechanism": "B"}}]
        core = Mock(side_effect=ValueError("trial-specific measured gap evidence"))
        with patch("couponevo.harness._model_evo_package",
                   return_value=SimpleNamespace(validate_data_request=core)):
            with self.assertRaisesRegex(ValueError, "trial-specific measured gap"):
                validate_model_evo_data_request(request, SNAPSHOT, trials, evidence=FACTS)

    def test_reflection_passes_host_checks_and_business_observations_to_core(self):
        reflection = {"verdict": "inconclusive",
                      "technical_experience": {"lesson": "Joint change", "evidence": "receipt",
                                               "uncertainty": "one split", "next_test": "ablate"},
                      "business_experience": {"status": "observed", "observation_id": "policy.net",
                                              "insight": "positive estimate", "limitations": "validation"}}
        observation = {"status": "evaluated", "eligibility": "pending_reflection",
                       "task_snapshot": SNAPSHOT,
                       "trial_history": [{"id": "step-001", "status": "evaluated",
                                          "mechanism": "expert gate"}],
                       "business_observations": [{"id": "policy.net"}],
                       "implementation_check": {"status": "verified"},
                       "change_audit": {"status": "verified", "changed_factors": ["loss", "model"]}}
        core = Mock(side_effect=lambda value, *_args, **_kwargs: value)
        with patch("couponevo.harness._model_evo_package",
                   return_value=SimpleNamespace(validate_reflection=core)):
            self.assertIs(validate_model_evo_reflection(reflection, observation,
                                                        evidence=FACTS), reflection)
        sent_reflection, evaluation, snapshot, steps = core.call_args.args
        self.assertIs(sent_reflection, reflection)
        self.assertEqual(evaluation["business_observations"], [{"id": "policy.net"}])
        self.assertEqual(evaluation["implementation_check"], {"status": "verified"})
        self.assertEqual(evaluation["change_audit"]["changed_factors"], ["loss", "model"])
        self.assertEqual(snapshot, SNAPSHOT)
        self.assertEqual(steps[0]["proposal"]["research"]["mechanism"], "expert gate")
        self.assertEqual(core.call_args.kwargs, {"evidence": FACTS})

    def test_blocked_implementation_requires_invalid_verdict(self):
        reflection = {"verdict": "inconclusive", "technical_experience": {
            "lesson": "wrong score", "evidence": "receipt", "uncertainty": "known",
            "next_test": "fix score"}, "business_experience": {"status": "not_observable",
                                                            "reason": "invalid candidate"}}
        observation = {"status": "evaluated", "eligibility": "blocked_implementation",
                       "task_snapshot": SNAPSHOT, "trial_history": [],
                       "business_observations": []}
        with self.assertRaisesRegex(ValueError, "invalid verdict"):
            validate_model_evo_reflection(reflection, observation, evidence=FACTS)

    def test_missing_implementation_and_audit_checks_reach_core_as_objects(self):
        reflection = {"verdict": "invalid", "technical_experience": {
            "lesson": "Unverified model", "evidence": "No runtime check", "uncertainty": "unknown",
            "next_test": "Run probe"}, "business_experience": {"status": "not_observable",
                                                               "reason": "No valid estimate"}}
        observation = {"status": "failed", "task_snapshot": SNAPSHOT, "trial_history": [],
                       "business_observations": [], "implementation_check": None,
                       "change_audit": None}
        core = Mock(return_value=reflection)
        with patch("couponevo.harness._model_evo_package",
                   return_value=SimpleNamespace(validate_reflection=core)):
            validate_model_evo_reflection(reflection, observation, evidence=FACTS)
        self.assertEqual(core.call_args.args[1]["implementation_check"], {})
        self.assertEqual(core.call_args.args[1]["change_audit"], {})

    def test_research_requires_prediction_semantics_and_preserves_it(self):
        research = {"direction": "test experts", "mechanism": "gate", "why_now": "residual",
                    "data_rationale": "current inputs", "comparison": "same loss",
                    "expected_result": "higher value", "falsification": "no gain",
                    "input_fields": ["V2"], "alternatives": [{"direction": "cross",
                    "mechanism": "FM", "reason": "cheaper"}],
                    "evidence_ids": ["task.profile"], "change_factors": ["model"],
                    "prediction_semantics": "direct_cate"}
        proposal = {"expected_result": "higher value", "research": research}
        core = Mock(return_value={key: value for key, value in research.items()
                                  if key != "prediction_semantics"})
        harness = {"source": "ModelEvoHarness", "task_snapshot": SNAPSHOT, "catalog": {}}
        with patch("couponevo.harness._model_evo_package",
                   return_value=SimpleNamespace(validate_research=core)):
            result = validate_research(proposal, harness, evidence=FACTS)
            with self.assertRaisesRegex(ValueError, "prediction_semantics"):
                validate_research({"research": {key: value for key, value in research.items()
                                                if key != "prediction_semantics"}},
                                  harness, evidence=FACTS)
        self.assertEqual(result["prediction_semantics"], "direct_cate")
        self.assertEqual(core.call_args_list[0].kwargs, {"evidence": FACTS})

    def test_audit_recommendations_forward_facts(self):
        items = [{"issue": "Check assignment time", "evidence_ids": ["task.profile"],
                  "validation_plan": "Audit event log"}]
        core = Mock(return_value=items)
        with patch("couponevo.harness._model_evo_package",
                   return_value=SimpleNamespace(validate_audit_recommendations=core)):
            self.assertEqual(validate_model_evo_audit_recommendations(items,
                                                                      evidence=FACTS), items)
        core.assert_called_once_with(items, evidence=FACTS)

    def test_legacy_calls_without_host_facts_keep_the_existing_contract(self):
        request = {"name": "recent_spend", "definition": "Recent spend",
                   "source": "order log", "as_of": "before assignment",
                   "evidence": "Host rule", "validation_plan": "Audit",
                   "basis": "domain_requirement", "requirement_id": "spend_rule"}
        snapshot = {"fields": ["V2"], "domain_requirements": [{
            "id": "spend_rule", "source": "order log", "as_of": "before assignment",
            "fields": ["recent_spend"]}]}
        with patch("couponevo.harness._model_evo_package",
                   side_effect=AssertionError("unexpected core call")):
            self.assertEqual(validate_model_evo_data_request(request, snapshot, []), request)
            self.assertEqual(validate_model_evo_audit_recommendations([]), [])


if __name__ == "__main__":
    unittest.main()
