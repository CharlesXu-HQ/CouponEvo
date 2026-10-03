import json
import tempfile
import unittest
from pathlib import Path

from couponevo.experience import load_experience


class ExperienceTests(unittest.TestCase):
    def test_lessons_require_same_dataset_version_and_evaluation_task(self):
        task = {"dataset": "dataset-v1", "manifest": "manifest-v1", "objective": "conversion",
                "budget": {"kind": "count", "value": 0.2}, "seed": 42,
                "framework": "evaluator-v1", "strict_data": False}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            variants = (("matching", {}, True), ("changed-data", {"dataset": "dataset-v2"}, True),
                        ("changed-schema", {"manifest": "manifest-v2"}, True),
                        ("changed-objective", {"objective": "revenue"}, True),
                        ("unfinished", {}, False))
            for name, changes, finalized in variants:
                path = root / name
                path.mkdir()
                journal = {"task": {**task, **changes},
                           "baseline": {"score": 0.01},
                           "steps": [{"status": "evaluated", "operator": "improve",
                                      "hypothesis": name, "score": 0.02}],
                           "final": {"report": {"private_test_metric": "NEVER_SEND_TO_AGENT"}}}
                if not finalized:
                    del journal["final"]
                (path / "journal.json").write_text(json.dumps(journal))
            lessons = load_experience(root, task)
            self.assertEqual([item["hypothesis"] for item in lessons], ["matching"])
            self.assertEqual(lessons[0]["validation_delta"], 0.01)
            self.assertNotIn("NEVER_SEND_TO_AGENT", json.dumps(lessons))


if __name__ == "__main__":
    unittest.main()
