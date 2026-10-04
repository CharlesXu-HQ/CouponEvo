import json
import tempfile
import unittest
from pathlib import Path

from couponevo.experience import load_experience


class ExperienceTests(unittest.TestCase):
    def test_lessons_require_same_dataset_version_and_evaluation_task(self):
        task = {"dataset": "dataset-v1", "manifest": "manifest-v1", "objective": "conversion",
                "budget": {"kind": "count", "value": 0.2}, "seed": 42,
                "framework": "evaluator-v1", "agent_workflow": "reviewer-v1",
                "strict_data": False}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            variants = (("matching", {}, True), ("changed-data", {"dataset": "dataset-v2"}, True),
                        ("changed-schema", {"manifest": "manifest-v2"}, True),
                        ("changed-objective", {"objective": "revenue"}, True),
                        ("changed-workflow", {"agent_workflow": "reviewer-v2"}, True),
                        ("legacy-workflow", {}, True),
                        ("unfinished", {}, False))
            for name, changes, finalized in variants:
                path = root / name
                path.mkdir()
                journal = {"task": {**task, **changes},
                           "baseline": {"score": 0.01},
                           "steps": [{"status": "evaluated", "operator": "improve",
                                      "hypothesis": name, "expected_result": "conversion rises",
                                      "score": 0.02,
                                      "reflection": {"verdict": "inconclusive",
                                                     "evidence": "No clear paired gain",
                                                     "lesson": "Try a smaller policy"}}],
                           "final": {"report": {"private_test_metric": "NEVER_SEND_TO_AGENT"}}}
                if name == "legacy-workflow":
                    del journal["task"]["agent_workflow"]
                if not finalized:
                    del journal["final"]
                (path / "journal.json").write_text(json.dumps(journal))
            lessons = load_experience(root, task)
            self.assertEqual([item["hypothesis"] for item in lessons], ["matching"])
            self.assertEqual(lessons[0]["validation_delta"], 0.01)
            self.assertEqual(lessons[0]["lesson"], "Try a smaller policy")
            self.assertEqual(lessons[0]["expected_result"], "conversion rises")
            self.assertEqual(lessons[0]["evidence"], "No clear paired gain")
            self.assertNotIn("NEVER_SEND_TO_AGENT", json.dumps(lessons))

    def test_failed_experiments_can_supply_dataset_bound_lessons(self):
        task = {"dataset": "v1", "manifest": "m1", "objective": "active",
                "budget": {"kind": "count", "value": 0.2}, "seed": 7,
                "framework": "e1", "agent_workflow": "a1", "strict_data": False}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "finished"
            path.mkdir()
            (path / "journal.json").write_text(json.dumps({
                "task": task, "baseline": {"score": 0.01},
                "steps": [{"status": "failed", "operator": "improve",
                           "hypothesis": "Larger policy", "error": "budget exceeded",
                           "reflection": {"verdict": "invalid",
                                          "lesson": "Check the count budget before GPU training"}}],
                "final": {"report": {"secret_test_value": 999}},
            }))
            lessons = load_experience(Path(directory), task)
        self.assertEqual(lessons[0]["lesson"], "Check the count budget before GPU training")
        self.assertEqual(lessons[0]["validation_delta"], None)
        self.assertNotIn("secret_test_value", json.dumps(lessons))


if __name__ == "__main__":
    unittest.main()
