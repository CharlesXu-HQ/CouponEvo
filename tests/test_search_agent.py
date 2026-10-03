import json
import unittest
from unittest.mock import patch

from coupon_lab.agent import propose_search_candidate
from coupon_lab.provider import ApiProvider


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


class SearchAgentTests(unittest.TestCase):
    def setUp(self):
        self.provider = ApiProvider("https://api.deepseek.com", "deepseek-flash", "test-key")
        self.context = {"objective": "active", "budget": {"kind": "count", "value": 0.2},
                        "best_id": "seed", "history": [{"id": "seed", "score": 0.1}],
                        "available": {"seed": {"candidate_py": "source", "report": {"score": 0.1}}}}
        self.answer = {"operator": "improve", "parent_ids": ["seed"],
                       "hypothesis": "try a smaller model", "candidate_py": "new source"}

    def test_proposal_uses_high_and_receives_experiment_history(self):
        with patch("urllib.request.urlopen", return_value=FakeResponse(self.answer)) as call:
            result = propose_search_candidate(self.provider, self.context)
        self.assertEqual(result, self.answer)
        body = json.loads(call.call_args.args[0].data)
        self.assertEqual(body["reasoning_effort"], "high")
        self.assertIn("seed", body["messages"][1]["content"])
        self.assertIn("source", body["messages"][1]["content"])

    def test_invalid_shape_is_retried_once(self):
        with patch("urllib.request.urlopen", side_effect=[FakeResponse({"operator": "other"}),
                                                          FakeResponse(self.answer)]) as call:
            result = propose_search_candidate(self.provider, self.context)
        self.assertEqual(call.call_count, 2)
        self.assertEqual(result["operator"], "improve")


if __name__ == "__main__":
    unittest.main()
