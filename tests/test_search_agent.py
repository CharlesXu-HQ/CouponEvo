import json
import unittest
from unittest.mock import patch

from couponevo.agent import (diagnose_search_state, propose_search_candidate,
                             reflect_search_step)
from couponevo.provider import ApiProvider, IncompleteResponseError


class FakeResponse:
    def __init__(self, answer, finish_reason="stop"):
        self.answer = answer
        self.finish_reason = finish_reason

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def read(self):
        return json.dumps({"choices": [{"finish_reason": self.finish_reason, "message": {
            "content": json.dumps(self.answer)}}]}).encode()


class SearchAgentTests(unittest.TestCase):
    def setUp(self):
        self.provider = ApiProvider("https://api.deepseek.com", "deepseek-flash", "test-key")
        self.context = {"objective": "active", "budget": {"kind": "count", "value": 0.2},
                        "best_id": "seed", "history": [{"id": "seed", "score": 0.1}],
                        "available": {"seed": {"candidate_py": "source", "report": {"score": 0.1}}}}
        self.answer = {"action": "experiment", "operator": "improve", "parent_ids": ["seed"],
                       "hypothesis": "try a smaller model", "expected_result": "active rises",
                       "candidate_py": "new source"}

    def test_proposal_uses_high_and_receives_experiment_history(self):
        with patch("urllib.request.urlopen", return_value=FakeResponse(self.answer)) as call:
            result = propose_search_candidate(self.provider, self.context)
        self.assertEqual(result, self.answer)
        body = json.loads(call.call_args.args[0].data)
        self.assertEqual(body["reasoning_effort"], "high")
        self.assertIn("seed", body["messages"][1]["content"])
        self.assertIn("source", body["messages"][1]["content"])
        self.assertIn("candidate-side", body["messages"][0]["content"])
        self.assertNotIn("TLearner", body["messages"][0]["content"])
        self.assertIn("experiment_budget_exhausted", body["messages"][0]["content"])
        self.assertIn("Unverified feature timing", body["messages"][0]["content"])

    def test_model_evo_reads_real_source_before_writing_candidate(self):
        from importlib import import_module
        package = import_module("model_evo_harness")
        self.context["harness"] = {"source": "ModelEvoHarness",
                                   "catalog": package.load_catalog(),
                                   "task_snapshot": {"framework": "pytorch", "fields": ["x"]}}
        read = {"action": "read_reference", "framework": "pytorch",
                "method_ids": ["fm"], "include_training": True}
        with patch("couponevo.harness._model_evo_package", return_value=package), \
             patch("urllib.request.urlopen", side_effect=[FakeResponse(read),
                                                           FakeResponse(self.answer)]) as call:
            result = propose_search_candidate(self.provider, self.context)
        self.assertEqual(call.call_count, 2)
        second = json.loads(call.call_args.args[0].data)["messages"][1]["content"]
        self.assertIn("def focal_loss", second)
        self.assertIn("class FM", second)
        self.assertIn("models/pytorch/architectures.py", result["reference_reads"])

    def test_model_evo_contract_retry_preserves_source_reads(self):
        from importlib import import_module
        package = import_module("model_evo_harness")
        self.context["harness"] = {"source": "ModelEvoHarness",
                                   "catalog": package.load_catalog(),
                                   "task_snapshot": {"framework": "pytorch", "fields": ["x"]}}
        read = {"action": "read_reference", "framework": "pytorch", "method_ids": ["afm"]}
        with patch("couponevo.harness._model_evo_package", return_value=package), \
             patch("urllib.request.urlopen", side_effect=[FakeResponse(read),
                     FakeResponse({"action": "experiment"}), FakeResponse(self.answer)]) as call:
            result = propose_search_candidate(self.provider, self.context)
        self.assertEqual(call.call_count, 3)
        self.assertIn("models/pytorch/interactions.py", result["reference_reads"])
        self.assertIn("class AFM", json.loads(call.call_args.args[0].data)["messages"][1]["content"])

    def test_harness_reaches_provider_with_open_research_contract(self):
        self.context["harness"] = {"plugin": {"name": "custom-research", "directions": []},
                                   "dataset_profile": {"features": {"x": {"dtype": "float64"}}}}
        with patch("urllib.request.urlopen", return_value=FakeResponse(self.answer)) as call:
            propose_search_candidate(self.provider, self.context)
        messages = json.loads(call.call_args.args[0].data)["messages"]
        self.assertIn("custom-research", messages[1]["content"])
        self.assertIn("not an allowed-model list", messages[0]["content"])
        self.assertIn("falsification", messages[0]["content"])
        self.assertIn("input_features", messages[0]["content"])

    def test_invalid_shape_is_retried_once(self):
        with patch("urllib.request.urlopen", side_effect=[FakeResponse({"operator": "other"}),
                                                          FakeResponse(self.answer)]) as call:
            result = propose_search_candidate(self.provider, self.context)
        self.assertEqual(call.call_count, 2)
        self.assertEqual(result["operator"], "improve")

    def test_non_string_action_is_retried(self):
        with patch("urllib.request.urlopen", side_effect=[FakeResponse({"action": []}),
                                                          FakeResponse(self.answer)]) as call:
            self.assertEqual(propose_search_candidate(self.provider, self.context), self.answer)
        self.assertEqual(call.call_count, 2)

    def test_agent_can_choose_diagnosis_or_stop_without_candidate_code(self):
        for answer in ({"action": "diagnose", "question": "Why did net value fall?"},
                       {"action": "stop", "reason": "No useful next hypothesis"}):
            with self.subTest(answer=answer), patch("urllib.request.urlopen",
                                                    return_value=FakeResponse(answer)):
                self.assertEqual(propose_search_candidate(self.provider, self.context), answer)

    def test_agent_can_request_specific_pre_treatment_data(self):
        answer = {"action": "request_data", "reason": "Current rows lack purchase history",
                  "feature_request": {
                      "name": "prior_purchase_count",
                      "definition": "Purchases in the 30 days before assignment",
                      "source": "Order event log", "as_of": "Before assignment time",
                      "evidence": "Recent cold-start errors correlate with missing history",
                      "validation_plan": "Join by user and verify cutoff and coverage"}}
        with patch("urllib.request.urlopen", return_value=FakeResponse(answer)) as call:
            self.assertEqual(propose_search_candidate(self.provider, self.context), answer)
        instruction = json.loads(call.call_args.args[0].data)["messages"][0]["content"]
        for field in answer["feature_request"]:
            self.assertIn(field, instruction)

    def test_invalid_data_request_is_retried(self):
        incomplete = {"action": "request_data", "reason": "Missing history",
                      "feature_request": {"name": "prior_purchase_count"}}
        complete = {"action": "stop", "reason": "No testable hypothesis remains"}
        with patch("urllib.request.urlopen", side_effect=[FakeResponse(incomplete),
                                                          FakeResponse(complete)]) as call:
            self.assertEqual(propose_search_candidate(self.provider, self.context), complete)
        self.assertEqual(call.call_count, 2)

    def test_model_evo_rejects_experimental_feature_request_without_completed_trials(self):
        self.context["harness"] = {"source": "ModelEvoHarness",
                                   "task_snapshot": {"fields": ["x"], "domain_requirements": []}}
        request = {"action": "request_data", "reason": "Need coupon recency",
                   "feature_request": {"name": "prior_coupon_use", "definition": "Past use count",
                                       "source": "coupon log", "as_of": "before assignment",
                                       "evidence": "One weak model", "validation_plan": "Check cutoff",
                                       "basis": "experimental_evidence", "trial_ids": ["step-001"]}}
        corrected = {"action": "stop", "reason": "More model-side tests are needed"}
        with patch("urllib.request.urlopen", side_effect=[FakeResponse(request),
                                                          FakeResponse(corrected)]) as call:
            self.assertEqual(propose_search_candidate(self.provider, self.context), corrected)
        self.assertEqual(call.call_count, 2)

    def test_model_evo_reflection_rejects_unlisted_business_observation(self):
        observation = {"harness_source": "ModelEvoHarness", "status": "evaluated",
                       "business_observations": [{"id": "policy:active:effect:active",
                                                  "holdout": "validation", "metric": "active",
                                                  "mean": 0.03}]}
        base = {"verdict": "inconclusive", "evidence": "Validation is exploratory",
                "lesson": "Test further", "next_direction": "Compare losses",
                "technical_experience": {"lesson": "Interaction is uncertain",
                                         "evidence": "Validation interval overlaps zero",
                                         "uncertainty": "One split", "next_test": "Test weighting"}}
        invented = {**base, "business_experience": {"status": "observed",
                                                  "observation_id": "cohort:high_spend",
                                                  "insight": "High spenders prefer coupons",
                                                  "limitations": "Validation only"}}
        corrected = {**base, "business_experience": {"status": "not_observable",
                                                   "reason": "No supported segment statistic"}}
        with patch("urllib.request.urlopen", side_effect=[FakeResponse(invented),
                                                          FakeResponse(corrected)]) as call:
            self.assertEqual(reflect_search_step(self.provider, observation), corrected)
        self.assertEqual(call.call_count, 2)

    def test_experiment_action_may_be_omitted(self):
        answer = {key: value for key, value in self.answer.items() if key != "action"}
        with patch("urllib.request.urlopen", return_value=FakeResponse(answer)):
            self.assertEqual(propose_search_candidate(self.provider, self.context), answer)

    def test_reflection_compares_hypothesis_with_validation_result(self):
        observation = {"hypothesis": "A smaller model improves active",
                       "expected_result": "active rises", "report": {"holdout": "validation"},
                       "score": 0.03, "baseline_score": 0.02}
        answer = {"verdict": "inconclusive", "evidence": "Paired difference is uncertain",
                  "lesson": "Size alone did not establish a gain", "next_direction": "Test allocation"}
        with patch("urllib.request.urlopen", return_value=FakeResponse(answer)) as call:
            self.assertEqual(reflect_search_step(self.provider, observation), answer)
        body = json.loads(call.call_args.args[0].data)
        self.assertEqual(body["reasoning_effort"], "high")
        self.assertIn("A smaller model", body["messages"][1]["content"])
        self.assertIn("validation", body["messages"][1]["content"])
        self.assertIn("inconclusive", body["messages"][0]["content"].lower())

    def test_reflection_length_retry_preserves_json_and_schema_repair_budget(self):
        class BadResponse(FakeResponse):
            def read(self):
                return json.dumps({"choices": [{"finish_reason": "stop", "message": {
                    "content": '{"verdict":'}}]}).encode()

        answer = {"verdict": "invalid", "evidence": "Candidate exceeded its budget",
                  "lesson": "Repair the budget", "next_direction": "Debug allocation"}
        observation = {"status": "failed", "error": "budget exceeded"}
        for invalid in (BadResponse(answer), FakeResponse({**answer, "verdict": "consistent"})):
            with self.subTest(invalid=type(invalid).__name__), patch("urllib.request.urlopen", side_effect=[
                    FakeResponse(answer, "length"), invalid, FakeResponse(answer)]) as call:
                self.assertEqual(reflect_search_step(self.provider, observation), answer)
            payloads = [json.loads(item.args[0].data) for item in call.call_args_list]
            self.assertEqual([item["max_tokens"] for item in payloads], [10000, 20000, 20000])
            self.assertEqual([item.kwargs["timeout"] for item in call.call_args_list], [180, 300, 300])
            self.assertEqual([item["reasoning_effort"] for item in payloads], ["high"] * 3)
            self.assertIn("Invalid reflection:", payloads[-1]["messages"][-1]["content"])

    def test_reflection_consecutive_length_stops_after_three_attempts(self):
        with patch("urllib.request.urlopen", side_effect=[FakeResponse({}, "length") for _ in range(3)]) as call:
            with self.assertRaises(IncompleteResponseError) as caught:
                reflect_search_step(self.provider, {"status": "evaluated"})
        self.assertEqual(caught.exception.reason, "length")
        self.assertEqual([json.loads(item.args[0].data)["max_tokens"]
                          for item in call.call_args_list], [10000, 20000, 32768])

    def test_reflection_non_length_incomplete_response_does_not_retry(self):
        with patch("urllib.request.urlopen", return_value=FakeResponse({}, "content_filter")) as call:
            with self.assertRaises(IncompleteResponseError) as caught:
                reflect_search_step(self.provider, {"status": "evaluated"})
        self.assertEqual(caught.exception.reason, "content_filter")
        self.assertEqual(call.call_count, 1)

    def test_reflection_timeout_does_not_retry(self):
        with patch("urllib.request.urlopen", side_effect=TimeoutError("provider timed out")) as call:
            with self.assertRaisesRegex(TimeoutError, "provider timed out"):
                reflect_search_step(self.provider, {"status": "evaluated"})
        self.assertEqual(call.call_count, 1)

    def test_diagnosis_uses_only_search_context_and_question(self):
        answer = {"finding": "Cost rose", "evidence": "net delta below zero",
                  "next_direction": "Try a lower-cost policy"}
        with patch("urllib.request.urlopen", return_value=FakeResponse(answer)) as call:
            self.assertEqual(diagnose_search_state(self.provider, self.context,
                                                   "Why did net fall?"), answer)
        body = json.loads(call.call_args.args[0].data)
        self.assertIn("Why did net fall?", body["messages"][1]["content"])
        self.assertIn("seed", body["messages"][1]["content"])

    def test_failed_step_reflection_rejects_a_success_verdict(self):
        wrong = {"verdict": "consistent", "evidence": "No metrics", "lesson": "Good",
                 "next_direction": "Continue"}
        corrected = {**wrong, "verdict": "invalid", "lesson": "Repair the budget"}
        with patch("urllib.request.urlopen", side_effect=[FakeResponse(wrong),
                                                          FakeResponse(corrected)]) as call:
            result = reflect_search_step(self.provider, {
                "hypothesis": "Try a broad policy", "status": "failed",
                "error": "budget exceeded", "report": None})
        self.assertEqual(result["verdict"], "invalid")
        self.assertEqual(call.call_count, 2)


if __name__ == "__main__":
    unittest.main()
