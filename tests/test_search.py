import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from coupon_lab.evaluate import Budget
from coupon_lab.search import run_search


class SearchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        pd.DataFrame({
            "id": range(100), "arm": [i % 2 for i in range(100)],
            "x": [i / 100 for i in range(100)],
            "active": [i % 2 for i in range(100)],
        }).to_csv(self.root / "data.csv", index=False)
        self.manifest = self.root / "manifest.json"
        self.manifest.write_text(json.dumps({
            "dataset": "data.csv", "unit_id": "id", "features": ["x"],
            "feature_timing": "pre_treatment", "outcomes": {"active": "active"},
            "treatment": {"column": "arm", "control": 0, "treated": 1,
                          "probability": 0.5, "probability_source": "protocol",
                          "probability_reference": "test-protocol"},
        }))
        self.seed = Path(__file__).resolve().parents[1] / "src/coupon_lab/candidate.py"
        self.kwargs = dict(manifest_path=self.manifest, budget=Budget("count", 0.2),
                           seed=7, output=self.root / "runs", initial_candidate=self.seed,
                           objective="active", search_id="agent-search", device="cpu")

    def test_agent_runs_multiple_steps_and_resume_keeps_history(self):
        contexts = []

        def propose(context):
            contexts.append(context)
            source = context["available"]["seed"]["candidate_py"]
            index = len(contexts)
            return {"operator": "improve", "parent_ids": ["seed"],
                    "hypothesis": f"policy choice {index}",
                    "candidate_py": source + "\ndef choose_policy(scores, costs, budget_kind, budget_value):\n"
                    + ("    return np.zeros(len(scores), dtype=bool)\n" if index == 1 else
                       "    result = np.zeros(len(scores), dtype=bool)\n"
                       "    result[:int(len(scores) * budget_value)] = True\n"
                       "    return result\n")}

        first = run_search(**self.kwargs, max_steps=2, proposer=propose)
        self.assertEqual(len(first["steps"]), 2)
        self.assertEqual([step["status"] for step in first["steps"]], ["evaluated", "evaluated"])
        self.assertEqual(len(contexts), 2)
        self.assertEqual(len(contexts[1]["history"]), 2)
        self.assertTrue((self.root / "runs/agent-search/steps/step-001/candidate.py").exists())
        self.assertEqual(first["baseline"]["report"]["holdout"], "validation")
        scores = {"seed": first["baseline"]["score"]}
        scores.update({step["id"]: step["score"] for step in first["steps"]})
        self.assertEqual(first["best_id"], max(scores, key=scores.get))
        self.assertTrue(all(step["report"]["holdout"] == "validation" for step in first["steps"]))

        second = run_search(**self.kwargs, max_steps=3, proposer=propose, resume=True)
        self.assertEqual(len(second["steps"]), 3)
        self.assertEqual(len(contexts), 3)
        self.assertEqual(second["baseline"], first["baseline"])
        self.assertEqual(json.loads((self.root / "runs/agent-search/journal.json").read_text()), second)

    def test_failure_is_saved_and_agent_can_debug_it(self):
        seen = []

        def propose(context):
            seen.append(context)
            if len(seen) == 1:
                return {"operator": "improve", "parent_ids": ["seed"],
                        "hypothesis": "try broad policy",
                        "candidate_py": context["available"]["seed"]["candidate_py"] +
                        "\ndef choose_policy(scores, costs, budget_kind, budget_value):\n"
                        "    return np.ones(len(scores), dtype=bool)\n"}
            self.assertEqual(context["history"][-1]["status"], "failed")
            return {"operator": "debug", "parent_ids": ["step-001"],
                    "hypothesis": "stay within budget",
                    "candidate_py": context["available"]["step-001"]["candidate_py"].replace(
                        "np.ones(len(scores), dtype=bool)", "np.zeros(len(scores), dtype=bool)")}

        result = run_search(**self.kwargs, max_steps=2, proposer=propose)
        self.assertEqual([step["status"] for step in result["steps"]], ["failed", "evaluated"])
        self.assertIn("budget", result["steps"][0]["error"])
        self.assertEqual(result["steps"][1]["parent_ids"], ["step-001"])

    def test_resume_rejects_changed_dataset(self):
        source = self.seed.read_text()
        propose = lambda context: {"operator": "draft", "parent_ids": [],
                                   "hypothesis": "independent candidate", "candidate_py": source + "\n"}
        run_search(**self.kwargs, max_steps=1, proposer=propose)
        data = self.root / "data.csv"
        data.write_text(data.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "changed"):
            run_search(**self.kwargs, max_steps=2, proposer=propose, resume=True)

    def test_resume_rejects_modified_candidate_snapshot(self):
        source = self.seed.read_text()
        proposal = lambda context: {"operator": "draft", "parent_ids": [],
                                    "hypothesis": "independent candidate", "candidate_py": source + "\n"}
        run_search(**self.kwargs, max_steps=1, proposer=proposal)
        candidate = self.root / "runs/agent-search/steps/step-001/candidate.py"
        candidate.write_text(candidate.read_text() + "\n# altered after validation\n")
        with self.assertRaisesRegex(ValueError, "candidate snapshot changed"):
            run_search(**self.kwargs, max_steps=2, proposer=proposal, resume=True)

    def test_report_analysis_failure_does_not_erase_evaluation(self):
        source = self.seed.read_text()
        proposal = lambda context: {"operator": "draft", "parent_ids": [],
                                    "hypothesis": "independent candidate", "candidate_py": source + "\n"}

        def broken_analysis(*_):
            raise RuntimeError("analysis endpoint unavailable")

        result = run_search(**self.kwargs, max_steps=1, proposer=proposal, analyzer=broken_analysis)
        self.assertEqual(result["steps"][0]["status"], "evaluated")
        self.assertIn("analysis endpoint unavailable", result["steps"][0]["analysis_error"])

    def test_resume_finishes_pending_candidate_without_new_agent_call(self):
        source = self.seed.read_text()
        proposal = lambda context: {"operator": "draft", "parent_ids": [],
                                    "hypothesis": "independent candidate", "candidate_py": source + "\n"}
        run_search(**self.kwargs, max_steps=1, proposer=proposal)
        path = self.root / "runs/agent-search/journal.json"
        journal = json.loads(path.read_text())
        journal["steps"][0].pop("report")
        journal["steps"][0].pop("score")
        journal["steps"][0]["status"] = "pending"
        journal["best_id"] = "seed"
        path.write_text(json.dumps(journal))

        def unexpected(_):
            self.fail("Agent must not propose again for a pending step")

        resumed = run_search(**self.kwargs, max_steps=1, proposer=unexpected, resume=True)
        self.assertEqual(resumed["steps"][0]["status"], "evaluated")

    def test_invalid_parent_selection_gets_one_agent_retry(self):
        source = self.seed.read_text()
        contexts = []

        def propose(context):
            contexts.append(context)
            return {"operator": "improve", "parent_ids": ["missing" if len(contexts) == 1 else "seed"],
                    "hypothesis": "repair parent selection", "candidate_py": source + "\n# revised\n"}

        result = run_search(**self.kwargs, max_steps=1, proposer=propose)
        self.assertEqual(len(contexts), 2)
        self.assertIn("proposal_error", contexts[1])
        self.assertEqual(result["steps"][0]["status"], "evaluated")

    def test_optional_feature_gap_accepts_null(self):
        source = self.seed.read_text()
        proposal = lambda context: {"operator": "draft", "parent_ids": [],
                                    "hypothesis": "new candidate", "candidate_py": source + "\n",
                                    "feature_gaps_md": None}
        result = run_search(**self.kwargs, max_steps=1, proposer=proposal)
        self.assertEqual(result["steps"][0]["status"], "evaluated")

    def test_dataset_changed_during_proposal_is_not_evaluated(self):
        source = self.seed.read_text()

        def propose(_):
            data = self.root / "data.csv"
            data.write_text(data.read_text() + "\n")
            return {"operator": "draft", "parent_ids": [],
                    "hypothesis": "new candidate", "candidate_py": source + "\n# draft\n"}

        with self.assertRaisesRegex(RuntimeError, "changed"):
            run_search(**self.kwargs, max_steps=1, proposer=propose)


if __name__ == "__main__":
    unittest.main()
