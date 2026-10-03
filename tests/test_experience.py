import json
import tempfile
import unittest
from pathlib import Path

from coupon_lab.experience import load_experience


class ExperienceTests(unittest.TestCase):
    def test_only_other_finalized_tasks_validation_lessons_are_retrieved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, dataset, finalized in (("other", "other-dataset", True),
                                             ("same", "current-dataset", True),
                                             ("unfinished", "third-dataset", False)):
                path = root / name
                path.mkdir()
                journal = {"task": {"dataset": dataset, "objective": "conversion"},
                           "baseline": {"score": 0.01},
                           "steps": [{"status": "evaluated", "operator": "improve",
                                      "hypothesis": name, "score": 0.02}],
                           "final": {"report": {"private_test_metric": "NEVER_SEND_TO_AGENT"}}}
                if not finalized:
                    del journal["final"]
                (path / "journal.json").write_text(json.dumps(journal))
            lessons = load_experience(root, "current-dataset")
            self.assertEqual([item["hypothesis"] for item in lessons], ["other"])
            self.assertEqual(lessons[0]["validation_delta"], 0.01)
            self.assertNotIn("NEVER_SEND_TO_AGENT", json.dumps(lessons))


if __name__ == "__main__":
    unittest.main()
