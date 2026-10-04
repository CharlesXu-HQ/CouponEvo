import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from couponevo.evaluate import Budget
from couponevo.search import _validate_proposal, run_search


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
            if context.get("experiment_budget_exhausted"):
                return {"action": "stop", "reason": "Experiment budget reached"}
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
        self.assertEqual(len(contexts), 3)
        self.assertTrue(contexts[-1]["experiment_budget_exhausted"])
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

        def after_reflection(context):
            self.assertEqual(context["history"][-1]["reflection"]["lesson"],
                             "Try a different policy")
            self.assertTrue(context["experiment_budget_exhausted"])
            return {"action": "stop", "reason": "Experiment budget reached"}

        resumed = run_search(**self.kwargs, max_steps=1, proposer=after_reflection, resume=True,
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

    def test_non_tlearner_candidate_is_evaluated(self):
        source = self.seed.read_text().replace("from econml.metalearners import TLearner\n", "")
        old = ("        learner = TLearner(models=TorchRidgeRegressor(device=device))\n"
               "        learner.fit(y, arm, X=x_train)\n"
               "        result[f\"{name}_uplift\"] = learner.effect(x_target)\n")
        new = ("        control = TorchRidgeRegressor(device=device).fit(x_train[arm == 0], y[arm == 0])\n"
               "        treated = TorchRidgeRegressor(device=device).fit(x_train[arm == 1], y[arm == 1])\n"
               "        result[f\"{name}_uplift\"] = treated.predict(x_target) - control.predict(x_target)\n")
        self.assertIn(old, source)
        source = source.replace(old, new).replace('"econml.TLearner"', '"two-arm torch ridge"')
        result = run_search(**self.kwargs, max_steps=1, proposer=lambda _: {
            "operator": "draft", "parent_ids": [], "hypothesis": "Fit arm models directly",
            "approach": "Two independent Torch ridge models",
            "candidate_py": source})
        self.assertEqual(result["steps"][0]["status"], "evaluated")
        self.assertEqual(result["steps"][0]["approach"], "Two independent Torch ridge models")

    def test_request_data_persists_without_a_candidate_evaluation(self):
        request = {"action": "request_data", "reason": "The response depends on a missing prior-use signal",
                   "feature_request": {"name": "prior_coupon_use", "definition": "Count of coupons used before assignment",
                                       "source": "coupon event log", "as_of": "Before treatment assignment",
                                       "evidence": "Validation errors cluster among repeat recipients",
                                       "validation_plan": "Check event timestamps and missingness, then rerun search on a new dataset"}}
        report = {"policies": {"active": {"effects": {"active": {"mean": 0.0}}}}}
        with patch("couponevo.search._evaluate", return_value=report) as evaluate:
            result = run_search(**self.kwargs, max_steps=2, proposer=lambda _: request)
        self.assertEqual(evaluate.call_count, 1)
        self.assertEqual(result["steps"], [])
        saved = json.loads((self.root / "runs/agent-search/feature_request.json").read_text())
        self.assertEqual(saved, result["data_request"])
        self.assertEqual(saved["feature_request"], request["feature_request"])
        self.assertEqual(saved["dataset_sha256"], result["task"]["dataset"])
        self.assertEqual(saved["after_step"], 0)
        self.assertEqual(json.loads((self.root / "runs/agent-search/journal.json").read_text()), result)
        with self.assertRaisesRegex(ValueError, "data request"):
            run_search(**self.kwargs, max_steps=3, proposer=lambda _: self.fail("unexpected proposal"), resume=True)

    def test_request_data_requires_actionable_fields(self):
        valid = {"action": "request_data", "reason": "Cannot test without the field",
                 "feature_request": {"name": "history", "definition": "Prior use count",
                                     "source": "event log", "as_of": "Before assignment",
                                     "evidence": "Validation slice error", "validation_plan": "Audit timing and coverage"}}
        invalid = [{**valid, "reason": " "}, {**valid, "feature_request": "history"},
                   {**valid, "feature_request": {**valid["feature_request"], "as_of": " "}},
                   {**valid, "feature_request": {key: value for key, value in
                                                 valid["feature_request"].items() if key != "validation_plan"}}]
        for proposal in invalid:
            with self.subTest(proposal=proposal), self.assertRaisesRegex(ValueError, "request_data"):
                _validate_proposal(proposal, {})

    def test_last_reflection_can_request_data_at_experiment_limit(self):
        contexts = []
        request = {"action": "request_data", "reason": "The completed experiment points to a missing field",
                   "feature_request": {"name": "prior_coupon_use", "definition": "Earlier coupon count",
                                       "source": "event log", "as_of": "Before assignment",
                                       "evidence": "Last validation result", "validation_plan": "Audit timing"}}

        def propose(context):
            contexts.append(context)
            if len(contexts) == 1:
                return {"operator": "draft", "parent_ids": [], "hypothesis": "Try a revision",
                        "candidate_py": self.seed.read_text() + "\n# revision\n"}
            self.assertTrue(context["experiment_budget_exhausted"])
            return request

        report = {"policies": {"active": {"effects": {"active": {"mean": 0.0}}}}}
        with patch("couponevo.search._evaluate", return_value=report) as evaluate:
            result = run_search(**self.kwargs, max_steps=1, proposer=propose,
                                reflector=lambda _: {"verdict": "inconclusive", "evidence": "No gain",
                                                     "lesson": "Need prior-use data", "next_direction": "Request field"})
        self.assertEqual(len(contexts), 2)
        self.assertEqual(evaluate.call_count, 2)
        self.assertEqual(result["data_request"]["after_step"], 1)

    def test_experiment_proposed_after_limit_is_not_run_or_requested_again(self):
        calls = []

        def propose(context):
            calls.append(context)
            return {"operator": "draft", "parent_ids": [], "hypothesis": "Try a revision",
                    "candidate_py": self.seed.read_text() + "\n# revision\n"}

        report = {"policies": {"active": {"effects": {"active": {"mean": 0.0}}}}}
        with patch("couponevo.search._evaluate", return_value=report) as evaluate:
            result = run_search(**self.kwargs, max_steps=1, proposer=propose,
                                reflector=lambda _: {"verdict": "inconclusive", "evidence": "No gain",
                                                     "lesson": "Try another mechanism", "next_direction": "Experiment"})
            resumed = run_search(**self.kwargs, max_steps=1, proposer=lambda _: self.fail("unexpected proposal"),
                                 reflector=lambda _: self.fail("unexpected reflection"), resume=True)
        self.assertEqual(evaluate.call_count, 2)
        self.assertEqual(len(calls), 2)
        self.assertTrue(calls[-1]["experiment_budget_exhausted"])
        self.assertEqual(result["budget_exhausted_decision"],
                         {"after_step": 1, "reason": "no_terminal_action"})
        self.assertEqual(resumed, result)

    def test_invalid_terminal_proposal_is_retried(self):
        calls = []

        def propose(context):
            calls.append(context)
            if len(calls) == 1:
                return {"operator": "draft", "parent_ids": [], "hypothesis": "Try a revision",
                        "candidate_py": self.seed.read_text() + "\n# revision\n"}
            if len(calls) == 2:
                return {"action": "unsupported"}
            self.assertIn("proposal_error", context)
            self.assertTrue(context["experiment_budget_exhausted"])
            return {"action": "stop", "reason": "Experiment budget reached"}

        report = {"policies": {"active": {"effects": {"active": {"mean": 0.0}}}}}
        with patch("couponevo.search._evaluate", return_value=report) as evaluate:
            result = run_search(**self.kwargs, max_steps=1, proposer=propose,
                                reflector=lambda _: {"verdict": "inconclusive", "evidence": "No gain",
                                                     "lesson": "Need more evidence", "next_direction": "Stop"})
        self.assertEqual(len(calls), 3)
        self.assertEqual(evaluate.call_count, 2)
        self.assertEqual(result["stop"]["reason"], "Experiment budget reached")
        self.assertNotIn("budget_exhausted_decision", result)

    def test_larger_budget_allows_a_new_terminal_data_request(self):
        calls = []
        request = {"action": "request_data", "reason": "The second result needs prior-use data",
                   "feature_request": {"name": "history", "definition": "Prior use count",
                                       "source": "event log", "as_of": "Before assignment",
                                       "evidence": "Second validation result", "validation_plan": "Audit timing"}}

        def propose(context):
            calls.append(context)
            if len(calls) == 4:
                self.assertTrue(context["experiment_budget_exhausted"])
                return request
            return {"operator": "draft", "parent_ids": [], "hypothesis": "Try a revision",
                    "candidate_py": self.seed.read_text() + f"\n# revision {len(calls)}\n"}

        report = {"policies": {"active": {"effects": {"active": {"mean": 0.0}}}}}
        def reflect(_):
            return {"verdict": "inconclusive", "evidence": "No gain",
                    "lesson": "Need more evidence", "next_direction": "Continue"}
        with patch("couponevo.search._evaluate", return_value=report) as evaluate:
            first = run_search(**self.kwargs, max_steps=1, proposer=propose, reflector=reflect)
            self.assertEqual(first["budget_exhausted_decision"]["after_step"], 1)
            second = run_search(**self.kwargs, max_steps=2, proposer=propose,
                                reflector=reflect, resume=True)
        self.assertEqual(evaluate.call_count, 3)
        self.assertEqual(len(second["steps"]), 2)
        self.assertEqual(second["data_request"]["after_step"], 2)


if __name__ == "__main__":
    unittest.main()
