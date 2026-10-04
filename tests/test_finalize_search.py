import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from couponevo.evaluate import Budget
from couponevo.search import finalize_search, run_search


class FinalizeSearchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        pd.DataFrame({"id": range(100), "arm": [i % 2 for i in range(100)],
                      "x": [i / 100 for i in range(100)],
                      "active": [i % 2 for i in range(100)]}).to_csv(root / "data.csv", index=False)
        self.manifest = root / "manifest.json"
        self.manifest.write_text(json.dumps({"dataset": "data.csv", "unit_id": "id",
                                            "features": ["x"], "feature_timing": "pre_treatment",
                                            "outcomes": {"active": "active"},
                                            "treatment": {"column": "arm", "control": 0,
                                                          "treated": 1, "probability": 0.5,
                                                          "probability_source": "protocol",
                                                          "probability_reference": "test"}}))
        self.seed = Path(__file__).resolve().parents[1] / "src/couponevo/candidate.py"
        self.kwargs = dict(manifest_path=self.manifest, budget=Budget("count", 0.2),
                           seed=7, output=root / "runs", initial_candidate=self.seed,
                           objective="active", search_id="trial", device="cpu")

    def test_finalize_uses_test_once_and_seals_search(self):
        source = self.seed.read_text()
        propose = lambda _: {"operator": "draft", "parent_ids": [],
                             "hypothesis": "new candidate", "candidate_py": source + "\n# draft\n"}
        run_search(**self.kwargs, max_steps=1, proposer=propose)
        first = finalize_search(**self.kwargs, bootstrap_reps=100)
        self.assertEqual(first["holdout"], "test")
        self.assertIn("active", first["paired_vs_baseline_bootstrap"])
        second = finalize_search(**self.kwargs, bootstrap_reps=100)
        self.assertEqual(first, second)
        journal = json.loads((self.kwargs["output"] / "trial/journal.json").read_text())
        self.assertEqual(journal["final"]["report"]["run_id"], first["run_id"])
        with self.assertRaisesRegex(ValueError, "finalized"):
            run_search(**self.kwargs, max_steps=2, proposer=propose, resume=True)

    def test_finalize_rejects_modified_candidate_snapshot(self):
        source = self.seed.read_text()
        propose = lambda _: {"operator": "draft", "parent_ids": [],
                             "hypothesis": "new candidate", "candidate_py": source + "\n# draft\n"}
        run_search(**self.kwargs, max_steps=1, proposer=propose)
        candidate = self.kwargs["output"] / "trial/steps/step-001/candidate.py"
        candidate.write_text(candidate.read_text() + "\n# altered before final test\n")
        with self.assertRaisesRegex(ValueError, "candidate snapshot changed"):
            finalize_search(**self.kwargs, bootstrap_reps=100)

    def test_finalize_waits_for_new_dataset_after_data_request(self):
        request = {"action": "request_data", "reason": "Missing pre-treatment history",
                   "feature_request": {"name": "history", "definition": "Prior use count",
                                       "source": "event log", "as_of": "Before assignment",
                                       "evidence": "Validation failure", "validation_plan": "Audit timing"}}
        run_search(**self.kwargs, max_steps=1, proposer=lambda _: request)
        with self.assertRaisesRegex(ValueError, "data request"):
            finalize_search(**self.kwargs, bootstrap_reps=100)


if __name__ == "__main__":
    unittest.main()
