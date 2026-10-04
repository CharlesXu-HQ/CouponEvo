import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from couponevo.evaluate import Budget
from couponevo.search import run_search


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
        self.seed = Path(__file__).resolve().parents[1] / "src/couponevo/candidate.py"
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
        self.assertEqual(contexts[0]["dataset_sha256"], first["task"]["dataset"])
        self.assertEqual(contexts[0]["manifest_sha256"], first["task"]["manifest"])
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

    def test_reflection_is_saved_and_informs_the_next_experiment(self):
        contexts = []

        def propose(context):
            contexts.append(context)
            if len(contexts) == 2:
                self.assertEqual(context["history"][-1]["reflection"]["lesson"],
                                 "The first policy selected nobody")
            return {"action": "experiment", "operator": "draft", "parent_ids": [],
                    "hypothesis": "A different allocation may improve activation",
                    "expected_result": "active policy value increases versus seed",
                    "candidate_py": self.seed.read_text() + f"\n# experiment {len(contexts)}\n"}

        def reflect(observation):
            self.assertEqual(observation["expected_result"],
                             "active policy value increases versus seed")
            self.assertEqual(observation["report"]["holdout"], "validation")
            return {"verdict": "inconclusive", "evidence": "No clear paired gain",
                    "lesson": "The first policy selected nobody", "next_direction": "Try another allocation"}

        result = run_search(**self.kwargs, max_steps=2, proposer=propose, reflector=reflect)
        self.assertEqual(len(contexts), 2)
        self.assertTrue(all("reflection" in step for step in result["steps"]))
        self.assertEqual(result["steps"][0]["expected_result"],
                         "active policy value increases versus seed")

    def test_agent_can_diagnose_then_stop_without_using_an_experiment_step(self):
        contexts = []

        def propose(context):
            contexts.append(context)
            if len(contexts) == 1:
                return {"action": "diagnose", "question": "Why is the baseline weak?"}
            self.assertEqual(context["diagnoses"][0]["finding"], "Few users were selected")
            return {"action": "stop", "reason": "Current evidence does not justify GPU work"}

        result = run_search(**self.kwargs, max_steps=2, proposer=propose,
                            diagnoser=lambda context, question: {
                                "finding": "Few users were selected", "evidence": question,
                                "next_direction": "Stop"})
        self.assertEqual(result["steps"], [])
        self.assertEqual(len(result["diagnoses"]), 1)
        self.assertEqual(result["stop"]["reason"], "Current evidence does not justify GPU work")
        with self.assertRaisesRegex(ValueError, "stopped"):
            run_search(**self.kwargs, max_steps=3, proposer=propose, resume=True)

    def test_failed_reflection_is_retried_on_resume_before_new_proposal(self):
        source = self.seed.read_text()
        proposal = lambda context: {"action": "experiment", "operator": "draft", "parent_ids": [],
                                    "hypothesis": "Test a new policy", "expected_result": "active rises",
                                    "candidate_py": source + "\n# change\n"}
        with self.assertRaisesRegex(RuntimeError, "reflection unavailable"):
            run_search(**self.kwargs, max_steps=1, proposer=proposal,
                       reflector=lambda observation: (_ for _ in ()).throw(
                           RuntimeError("reflection unavailable")))
        saved = json.loads((self.root / "runs/agent-search/journal.json").read_text())
        self.assertEqual(saved["steps"][0]["status"], "evaluated")
        self.assertNotIn("reflection", saved["steps"][0])

        def unexpected(_):
            self.fail("Resume must reflect the evaluated step, not propose again")

        resumed = run_search(**self.kwargs, max_steps=1, proposer=unexpected, resume=True,
                             reflector=lambda observation: {
                                 "verdict": "inconclusive", "evidence": "No clear gain",
                                 "lesson": "Try a different policy", "next_direction": "Stop"})
        self.assertEqual(resumed["steps"][0]["reflection"]["lesson"],
                         "Try a different policy")

    def test_leakage_review_prevents_champion_promotion(self):
        from unittest.mock import patch

        proposal = lambda context: {"action": "experiment", "operator": "draft", "parent_ids": [],
                                    "hypothesis": "Try a new feature", "expected_result": "active rises",
                                    "candidate_py": self.seed.read_text() + "\n# feature trial\n"}
        analysis = {"high": {"feature_leakage": {"flag": True, "confirmed": True}},
                    "max": {"feature_leakage": {"flag": True, "confirmed": True}}}
        with patch("couponevo.search._score", side_effect=[0.0, 1.0]):
            result = run_search(**self.kwargs, max_steps=1, proposer=proposal,
                                analyzer=lambda *_: analysis)
        self.assertEqual(result["best_id"], "seed")
        self.assertEqual(result["steps"][0]["eligibility"], "blocked_feature_leakage")

    def test_unverified_feature_timing_does_not_block_promotion(self):
        from unittest.mock import patch

        proposal = lambda context: {"action": "experiment", "operator": "draft", "parent_ids": [],
                                    "hypothesis": "Try a new feature", "expected_result": "active rises",
                                    "candidate_py": self.seed.read_text() + "\n# feature trial\n"}
        analysis = {"high": {"feature_leakage": {"flag": True, "confirmed": True}},
                    "max": {"feature_leakage": {"flag": True, "confirmed": False,
                                                "evidence": "feature timing declared only"}}}
        with patch("couponevo.search._score", side_effect=[0.0, 1.0]):
            result = run_search(**self.kwargs, max_steps=1, proposer=proposal,
                                analyzer=lambda *_: analysis)
        self.assertEqual(result["best_id"], "step-001")
        self.assertEqual(result["steps"][0]["eligibility"], "eligible")
        self.assertTrue(result["steps"][0]["analysis"]["max"]["feature_leakage"]["flag"])

    def test_latest_result_is_visible_even_when_it_is_not_a_top_candidate(self):
        from unittest.mock import patch

        contexts = []

        def propose(context):
            contexts.append(context)
            return {"operator": "draft", "parent_ids": [], "hypothesis": "new trial",
                    "candidate_py": self.seed.read_text() + f"\n# trial {len(contexts)}\n"}

        report = {"holdout": "validation"}
        with patch("couponevo.search._evaluate", return_value=report), \
                patch("couponevo.search._score", side_effect=[10, 3, 2, 1, 0]):
            run_search(**self.kwargs, max_steps=4, proposer=propose)
        self.assertIn("step-003", contexts[3]["available"])
        self.assertEqual(contexts[3]["available"]["step-003"]["report"], report)

    def test_analysis_failure_cannot_promote_unreviewed_candidate(self):
        from unittest.mock import patch

        proposal = lambda context: {"operator": "draft", "parent_ids": [],
                                    "hypothesis": "new trial",
                                    "candidate_py": self.seed.read_text() + "\n# trial\n"}
        with patch("couponevo.search._score", side_effect=[0.0, 1.0]):
            result = run_search(**self.kwargs, max_steps=1, proposer=proposal,
                                analyzer=lambda *_: (_ for _ in ()).throw(
                                    RuntimeError("review unavailable")))
        self.assertEqual(result["best_id"], "seed")
        self.assertEqual(result["steps"][0]["eligibility"], "blocked_analysis_error")

    def test_old_journal_without_diagnoses_can_resume_and_diagnose(self):
        source = self.seed.read_text()
        run_search(**self.kwargs, max_steps=1, proposer=lambda context: {
            "operator": "draft", "parent_ids": [], "hypothesis": "first trial",
            "candidate_py": source + "\n# first trial\n"})
        path = self.root / "runs/agent-search/journal.json"
        journal = json.loads(path.read_text())
        del journal["diagnoses"]
        path.write_text(json.dumps(journal))
        calls = []

        def propose(context):
            calls.append(context)
            return ({"action": "diagnose", "question": "Why did the first trial fail?"}
                    if len(calls) == 1 else
                    {"action": "stop", "reason": "No further experiment"})

        result = run_search(**self.kwargs, max_steps=2, proposer=propose, resume=True,
                            diagnoser=lambda *_: {"finding": "No gain", "evidence": "Validation",
                                                  "next_direction": "Stop"})
        self.assertEqual(result["diagnoses"][0]["finding"], "No gain")

    def test_invalid_reflection_blocks_candidate_before_champion_selection(self):
        from unittest.mock import patch

        proposal = lambda context: {"operator": "draft", "parent_ids": [],
                                    "hypothesis": "new trial", "expected_result": "active rises",
                                    "candidate_py": self.seed.read_text() + "\n# trial\n"}
        with patch("couponevo.search._score", side_effect=[0.0, 1.0]):
            result = run_search(**self.kwargs, max_steps=1, proposer=proposal,
                                reflector=lambda observation: {
                                    "verdict": "invalid", "evidence": "Post-treatment feature suspected",
                                    "lesson": "Do not use that feature", "next_direction": "Try another feature"})
        self.assertEqual(result["best_id"], "seed")
        self.assertEqual(result["steps"][0]["eligibility"], "blocked_reflection")


if __name__ == "__main__":
    unittest.main()
