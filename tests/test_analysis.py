import json
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

from couponevo.analysis import analyze_reports_deepseek
from couponevo.provider import ApiProvider


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


class AnalysisTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.candidate = self.root / "candidate.py"
        self.candidate.write_text("def fit_predict(*args, **kwargs):\n    return None\n")
        self.report = {
            "run_id": "baseline", "features": ["pre_sessions"],
            "policies": {
                "active": {"effects": {"active": {"mean": 0.2}},
                           "net": {"mean": 0.5, "lower": 0.1, "upper": 0.9},
                           "cost": {"mean": 0.1}},
                "net_margin": {"effects": {"active": {"mean": 0.25}},
                               "net": {"mean": 0.8, "lower": 0.2, "upper": 1.4},
                               "cost": {"mean": 0.1}},
                "random": {"effects": {"active": {"mean": 0.05}},
                           "net": {"mean": 0.1, "lower": 0.0, "upper": 0.2},
                           "cost": {"mean": 0.1}},
            },
        }
        self.high = {"summary": "Positive uplift with clear cost signal",
                     "uplift_anomaly": {"flag": False, "evidence": ""},
                     "feature_leakage": {"flag": False, "confirmed": False, "evidence": ""},
                     "cost_tradeoff_unclear": {"flag": False, "evidence": ""},
                     "recommendation": "Keep for further validation"}

    def test_clear_report_uses_high_once(self):
        with patch("urllib.request.urlopen", return_value=FakeResponse(self.high)) as call:
            result = analyze_reports_deepseek(self.report, self.report,
                                              candidate_path=self.candidate,
                                              output_dir=self.root, api_key="test-key")
        self.assertEqual(call.call_count, 1)
        self.assertEqual(json.loads(call.call_args.args[0].data)["reasoning_effort"], "high")
        self.assertEqual(result["review_reasons"], [])
        self.assertIsNone(result["max"])
        self.assertIn("Positive uplift", (self.root / "analysis.md").read_text())
        self.assertEqual(json.loads((self.root / "analysis.json").read_text())["model"], "deepseek-flash")

    def test_experiment_analysis_requires_implementation_check_and_retries(self):
        revised = {**self.report, "experiment_context": {
            "hypothesis": "Mixed encoding improves uplift", "parent_candidate_py": "parent code",
            "candidate_diff": "+ model change"}}
        valid = {**self.high, "implementation_check": {
            "status": "verified", "evidence": "Observed features match the proposed encoding",
            "changed_factors": ["network"]}}
        with patch("urllib.request.urlopen", side_effect=[FakeResponse(self.high), FakeResponse(valid)]) as call:
            result = analyze_reports_deepseek(self.report, revised, candidate_path=self.candidate,
                                              output_dir=self.root, api_key="test-key")
        self.assertEqual(call.call_count, 2)
        self.assertEqual(result['high']['implementation_check']['status'], 'verified')

    def test_prediction_semantics_error_triggers_max_review(self):
        revised = {**self.report, "experiment_context": {"hypothesis": "Probability uplift"}}
        answer = {**self.high, "implementation_check": {
            "status": "contradicted", "evidence": "BCE logit difference returned as probability difference",
            "changed_factors": ["network", "loss"]}}
        with patch("urllib.request.urlopen", return_value=FakeResponse(answer)) as call:
            result = analyze_reports_deepseek(self.report, revised, candidate_path=self.candidate,
                                              output_dir=self.root, api_key="test-key")
        self.assertEqual(call.call_count, 2)
        self.assertIn('implementation_check', result['review_reasons'])
        self.assertEqual(result['max']['implementation_check']['status'], 'contradicted')

    def test_high_analysis_repairs_malformed_json_once(self):
        class BadResponse(FakeResponse):
            def read(self):
                return json.dumps({"choices": [{"finish_reason": "stop", "message": {
                    "content": '{"summary":'}}]}).encode()

        with patch("urllib.request.urlopen",
                   side_effect=[BadResponse(self.high), FakeResponse(self.high)]) as call:
            result = analyze_reports_deepseek(self.report, self.report,
                                              candidate_path=self.candidate,
                                              output_dir=self.root, api_key="test-key")
        self.assertEqual(call.call_count, 2)
        self.assertEqual(result["high"]["summary"], self.high["summary"])
        self.assertIsNone(result["max"])

    def test_analysis_repairs_original_content_and_respects_configured_budget(self):
        class BadResponse(FakeResponse):
            def read(self):
                return json.dumps({"choices": [{"finish_reason": "stop", "message": {
                    "content": '{"summary":'}}]}).encode()

        provider = ApiProvider("https://api.deepseek.com", "deepseek-flash", "test-key",
                               request_timeout_seconds=240, token_budgets={"analysis": 22000})
        with patch("urllib.request.urlopen", side_effect=[BadResponse(self.high),
                                                          FakeResponse(self.high)]) as call:
            analyze_reports_deepseek(self.report, self.report, candidate_path=self.candidate,
                                     output_dir=self.root, provider=provider)
        payloads = [json.loads(item.args[0].data) for item in call.call_args_list]
        self.assertEqual(payloads[-1]["messages"][-2], {"role": "assistant", "content": '{"summary":'})
        self.assertEqual([item["max_tokens"] for item in payloads], [22000, 22000])
        self.assertEqual([item.kwargs["timeout"] for item in call.call_args_list], [240, 240])

    def test_review_uses_its_stage_budget_independently_of_effort(self):
        revised = json.loads(json.dumps(self.report))
        revised["policies"]["random"]["effects"]["active"]["mean"] = 0.06
        provider = ApiProvider("https://api.deepseek.com", "deepseek-flash", "test-key",
                               iteration_effort="max", review_effort="high",
                               token_budgets={"analysis": 21000, "review": 26000})
        with patch("urllib.request.urlopen", return_value=FakeResponse(self.high)) as call:
            analyze_reports_deepseek(self.report, revised, candidate_path=self.candidate,
                                     output_dir=self.root, provider=provider)
        self.assertEqual([json.loads(item.args[0].data)["max_tokens"]
                          for item in call.call_args_list], [21000, 26000])

    def test_analysis_transient_timeout_retries_once(self):
        with patch("urllib.request.urlopen", side_effect=[TimeoutError("transient"),
                                                          FakeResponse(self.high)]) as call:
            result = analyze_reports_deepseek(self.report, self.report, candidate_path=self.candidate,
                                              output_dir=self.root, api_key="test-key")
        self.assertEqual(call.call_count, 2)
        self.assertEqual(result["high"], self.high)

    def test_analysis_persistent_timeout_stops_after_two_attempts(self):
        with patch("urllib.request.urlopen", side_effect=TimeoutError("transient")) as call:
            with self.assertRaises(TimeoutError):
                analyze_reports_deepseek(self.report, self.report, candidate_path=self.candidate,
                                         output_dir=self.root, api_key="test-key")
        self.assertEqual(call.call_count, 2)

    def test_high_length_retries_with_double_tokens_and_same_effort(self):
        with patch("urllib.request.urlopen", side_effect=[
                FakeResponse(self.high, "length"), FakeResponse(self.high)]) as call:
            result = analyze_reports_deepseek(self.report, self.report,
                                              candidate_path=self.candidate,
                                              output_dir=self.root, api_key="test-key")
        payloads = [json.loads(item.args[0].data) for item in call.call_args_list]
        self.assertEqual([item["max_tokens"] for item in payloads], [10000, 20000])
        self.assertEqual([item.kwargs["timeout"] for item in call.call_args_list], [180, 300])
        self.assertEqual([item["reasoning_effort"] for item in payloads], ["high", "high"])
        self.assertEqual(result["high"]["summary"], self.high["summary"])

    def test_max_length_retries_with_double_tokens_and_same_effort(self):
        revised = json.loads(json.dumps(self.report))
        revised["policies"]["random"]["effects"]["active"]["mean"] = 0.06
        with patch("urllib.request.urlopen", side_effect=[
                FakeResponse(self.high), FakeResponse(self.high, "length"),
                FakeResponse(self.high)]) as call:
            result = analyze_reports_deepseek(self.report, revised,
                                              candidate_path=self.candidate,
                                              output_dir=self.root, api_key="test-key")
        payloads = [json.loads(item.args[0].data) for item in call.call_args_list]
        self.assertEqual([item["max_tokens"] for item in payloads], [10000, 16000, 32000])
        self.assertEqual([item.kwargs["timeout"] for item in call.call_args_list], [180, 180, 300])
        self.assertEqual([item["reasoning_effort"] for item in payloads], ["high", "max", "max"])
        self.assertIsNotNone(result["max"])

    def test_length_retry_keeps_one_attempt_for_json_repair(self):
        class BadResponse(FakeResponse):
            def read(self):
                return json.dumps({"choices": [{"finish_reason": "stop", "message": {
                    "content": '{"summary":'}}]}).encode()

        with patch("urllib.request.urlopen", side_effect=[
                FakeResponse(self.high, "length"), BadResponse(self.high), FakeResponse(self.high)]) as call:
            result = analyze_reports_deepseek(self.report, self.report,
                                              candidate_path=self.candidate,
                                              output_dir=self.root, api_key="test-key")
        self.assertEqual(call.call_count, 3)
        self.assertEqual(result["high"]["summary"], self.high["summary"])
        self.assertEqual([json.loads(item.args[0].data)["max_tokens"]
                          for item in call.call_args_list], [10000, 20000, 20000])

    def test_consecutive_length_stops_after_three_attempts(self):
        with patch("urllib.request.urlopen", side_effect=[
                FakeResponse(self.high, "length"), FakeResponse(self.high, "length"),
                FakeResponse(self.high, "length")]) as call:
            with self.assertRaises(RuntimeError) as captured:
                analyze_reports_deepseek(self.report, self.report,
                                         candidate_path=self.candidate,
                                         output_dir=self.root, api_key="test-key")
        self.assertEqual(getattr(captured.exception, "reason", None), "length")
        self.assertEqual([json.loads(item.args[0].data)["max_tokens"]
                          for item in call.call_args_list], [10000, 20000, 32768])

    def test_non_length_incomplete_response_does_not_retry(self):
        with patch("urllib.request.urlopen", return_value=FakeResponse(self.high, "content_filter")) as call:
            with self.assertRaises(RuntimeError) as captured:
                analyze_reports_deepseek(self.report, self.report,
                                         candidate_path=self.candidate,
                                         output_dir=self.root, api_key="test-key")
        self.assertEqual(getattr(captured.exception, "reason", None), "content_filter")
        self.assertEqual(call.call_count, 1)

    def test_http_error_does_not_retry(self):
        error = urllib.error.HTTPError("https://example.test", 429, "Too Many Requests", {}, None)
        with patch("urllib.request.urlopen", side_effect=error) as call:
            with self.assertRaisesRegex(RuntimeError, "HTTP 429"):
                analyze_reports_deepseek(self.report, self.report,
                                         candidate_path=self.candidate,
                                         output_dir=self.root, api_key="test-key")
        self.assertEqual(call.call_count, 1)

    def test_cost_uncertainty_triggers_max_review(self):
        revised = json.loads(json.dumps(self.report))
        revised["policies"]["net_margin"]["net"]["lower"] = -0.2
        responses = [FakeResponse(self.high), FakeResponse({**self.high, "summary": "Cost benefit remains uncertain"})]
        with patch("urllib.request.urlopen", side_effect=responses) as call:
            result = analyze_reports_deepseek(self.report, revised,
                                              candidate_path=self.candidate,
                                              output_dir=self.root, api_key="test-key")
        self.assertEqual(call.call_count, 2)
        self.assertEqual(json.loads(call.call_args.args[0].data)["reasoning_effort"], "max")
        self.assertIn("cost_tradeoff_unclear", result["review_reasons"])
        self.assertIn("Cost benefit remains uncertain", (self.root / "analysis.md").read_text())

    def test_uplift_below_random_triggers_max_review(self):
        revised = json.loads(json.dumps(self.report))
        revised["policies"]["active"]["effects"]["active"]["mean"] = -0.1
        with patch("urllib.request.urlopen",
                   side_effect=[FakeResponse(self.high), FakeResponse(self.high)]) as call:
            result = analyze_reports_deepseek(self.report, revised,
                                              candidate_path=self.candidate,
                                              output_dir=self.root, api_key="test-key")
        self.assertEqual(call.call_count, 2)
        self.assertIn("uplift_anomaly", result["review_reasons"])

    def test_random_baseline_drift_triggers_max_review(self):
        revised = json.loads(json.dumps(self.report))
        revised["policies"]["random"]["effects"]["active"]["mean"] = 0.06
        with patch("urllib.request.urlopen",
                   side_effect=[FakeResponse(self.high), FakeResponse(self.high)]) as call:
            result = analyze_reports_deepseek(self.report, revised,
                                              candidate_path=self.candidate,
                                              output_dir=self.root, api_key="test-key")
        self.assertEqual(call.call_count, 2)
        self.assertIn("uplift_anomaly", result["review_reasons"])

    def test_suspected_leakage_triggers_max_review(self):
        high = json.loads(json.dumps(self.high))
        high["feature_leakage"] = {"flag": True, "confirmed": False,
                                   "evidence": "feature timing unverified"}
        with patch("urllib.request.urlopen",
                   side_effect=[FakeResponse(high), FakeResponse(high)]) as call:
            result = analyze_reports_deepseek(self.report, self.report,
                                              candidate_path=self.candidate,
                                              output_dir=self.root, api_key="test-key")
        self.assertEqual(call.call_count, 2)
        self.assertIn("feature_leakage", result["review_reasons"])
        self.assertFalse(result["max"]["feature_leakage"]["confirmed"])
        self.assertIn("feature_leakage：风险待核验", (self.root / "analysis.md").read_text())

    def test_leakage_confirmation_requires_consistent_schema(self):
        invalid = json.loads(json.dumps(self.high))
        invalid["feature_leakage"] = {"flag": False, "confirmed": True,
                                      "evidence": "post-treatment feature"}
        with patch("urllib.request.urlopen",
                   side_effect=[FakeResponse(invalid), FakeResponse(self.high)]) as call:
            result = analyze_reports_deepseek(self.report, self.report,
                                              candidate_path=self.candidate,
                                              output_dir=self.root, api_key="test-key")
        self.assertEqual(call.call_count, 2)
        self.assertFalse(result["high"]["feature_leakage"]["confirmed"])


if __name__ == "__main__":
    unittest.main()
