import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from coupon_lab.analysis import analyze_reports_deepseek


class FakeResponse:
    def __init__(self, answer):
        self.answer = answer

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def read(self):
        return json.dumps({"choices": [{"finish_reason": "stop", "message": {
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
                     "feature_leakage": {"flag": False, "evidence": ""},
                     "cost_tradeoff_unclear": {"flag": False, "evidence": ""},
                     "recommendation": "Keep for further validation"}

    def test_clear_report_uses_high_once(self):
        with patch("coupon_lab.agent.urllib.request.urlopen", return_value=FakeResponse(self.high)) as call:
            result = analyze_reports_deepseek(self.report, self.report,
                                              candidate_path=self.candidate,
                                              output_dir=self.root, api_key="test-key")
        self.assertEqual(call.call_count, 1)
        self.assertEqual(json.loads(call.call_args.args[0].data)["reasoning_effort"], "high")
        self.assertEqual(result["review_reasons"], [])
        self.assertIsNone(result["max"])
        self.assertIn("Positive uplift", (self.root / "analysis.md").read_text())
        self.assertEqual(json.loads((self.root / "analysis.json").read_text())["model"], "deepseek-flash")

    def test_cost_uncertainty_triggers_max_review(self):
        revised = json.loads(json.dumps(self.report))
        revised["policies"]["net_margin"]["net"]["lower"] = -0.2
        responses = [FakeResponse(self.high), FakeResponse({**self.high, "summary": "Cost benefit remains uncertain"})]
        with patch("coupon_lab.agent.urllib.request.urlopen", side_effect=responses) as call:
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
        with patch("coupon_lab.agent.urllib.request.urlopen",
                   side_effect=[FakeResponse(self.high), FakeResponse(self.high)]) as call:
            result = analyze_reports_deepseek(self.report, revised,
                                              candidate_path=self.candidate,
                                              output_dir=self.root, api_key="test-key")
        self.assertEqual(call.call_count, 2)
        self.assertIn("uplift_anomaly", result["review_reasons"])

    def test_random_baseline_drift_triggers_max_review(self):
        revised = json.loads(json.dumps(self.report))
        revised["policies"]["random"]["effects"]["active"]["mean"] = 0.06
        with patch("coupon_lab.agent.urllib.request.urlopen",
                   side_effect=[FakeResponse(self.high), FakeResponse(self.high)]) as call:
            result = analyze_reports_deepseek(self.report, revised,
                                              candidate_path=self.candidate,
                                              output_dir=self.root, api_key="test-key")
        self.assertEqual(call.call_count, 2)
        self.assertIn("uplift_anomaly", result["review_reasons"])

    def test_suspected_leakage_triggers_max_review(self):
        high = json.loads(json.dumps(self.high))
        high["feature_leakage"] = {"flag": True, "evidence": "feature timing unverified"}
        with patch("coupon_lab.agent.urllib.request.urlopen",
                   side_effect=[FakeResponse(high), FakeResponse(self.high)]) as call:
            result = analyze_reports_deepseek(self.report, self.report,
                                              candidate_path=self.candidate,
                                              output_dir=self.root, api_key="test-key")
        self.assertEqual(call.call_count, 2)
        self.assertIn("feature_leakage", result["review_reasons"])


if __name__ == "__main__":
    unittest.main()
