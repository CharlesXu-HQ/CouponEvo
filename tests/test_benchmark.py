import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from promolift.benchmark import run_benchmark
from promolift.tasks import load_task


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.candidate = self.root / "candidate.py"
        self.candidate.write_text("def fit_predict(*args, **kwargs): pass\n")
        self.config = self.root / "agent.json"
        self.config.write_text("{}")

    def task(self, name, objective="active"):
        folder = self.root / name
        folder.mkdir()
        source = folder / "data.csv"
        pd.DataFrame({"id": range(100), "arm": [i % 2 for i in range(100)],
                      "x": [i / 100 for i in range(100)],
                      "active": [i % 2 for i in range(100)]}).to_csv(source, index=False)
        manifest = folder / "manifest.json"
        manifest.write_text(json.dumps({"dataset": "data.csv", "unit_id": "id",
                                        "features": ["x"], "feature_timing": "pre_treatment",
                                        "outcomes": {"active": "active"},
                                        "treatment": {"column": "arm", "control": 0,
                                                      "treated": 1, "probability": 0.5,
                                                      "probability_source": "protocol",
                                                      "probability_reference": "test"}}))
        spec = folder / "task.json"
        spec.write_text(json.dumps({"name": name, "manifest": "manifest.json",
                                    "candidate": "../candidate.py",
                                    "budget": {"kind": "count", "value": 0.2},
                                    "objective": objective, "seed": 7, "max_steps": 1,
                                    "dataset_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                                    "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                                    "candidate_sha256": hashlib.sha256(self.candidate.read_bytes()).hexdigest()}))
        return spec

    def test_task_rejects_wrong_dataset_or_objective(self):
        path = self.task("first")
        self.assertEqual(load_task(path).objective, "active")
        config = json.loads(path.read_text())
        config["dataset_sha256"] = "0" * 64
        path.write_text(json.dumps(config))
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            load_task(path)
        config["dataset_sha256"] = hashlib.sha256((path.parent / "data.csv").read_bytes()).hexdigest()
        config["objective"] = "gross_margin"
        path.write_text(json.dumps(config))
        with self.assertRaisesRegex(ValueError, "unavailable"):
            load_task(path)

    def test_benchmark_uses_equal_steps_and_paired_test(self):
        first, second = self.task("first"), self.task("second")
        # Distinct public tasks must have distinct frozen source bytes.
        with (second.parent / "data.csv").open("a") as output:
            output.write("\n")
        second_config = json.loads(second.read_text())
        second_config["dataset_sha256"] = hashlib.sha256((second.parent / "data.csv").read_bytes()).hexdigest()
        second.write_text(json.dumps(second_config))
        tasks = [load_task(first), load_task(second)]
        calls = []

        def fake_invoke(command):
            calls.append(command)
            kind = command[command.index("promolift.cli") + 1]
            directory = Path(command[command.index("--output") + 1])
            if kind == "search":
                root = directory / command[command.index("--search-id") + 1]
                (root / "steps/seed").mkdir(parents=True)
                (root / "steps/seed/candidate.py").write_bytes(self.candidate.read_bytes())
                journal = {"baseline": {"id": "seed", "score": 1.0,
                                        "candidate": "steps/seed/candidate.py"},
                           "steps": [], "best_id": "seed",
                           "experience": [{"validation_delta": 0.1}] if "--experience-dir" in command else []}
                (root / "journal.json").write_text(json.dumps(journal))
            elif kind == "finalize":
                root = directory / command[command.index("--search-id") + 1]
                journal = json.loads((root / "journal.json").read_text())
                journal["final"] = {"report": {"run_id": "final-1"}}
                (root / "journal.json").write_text(json.dumps(journal))
            else:
                report = directory / "comparison" / "report.json"
                report.parent.mkdir(parents=True)
                report.write_text(json.dumps({"paired_vs_baseline_bootstrap": {
                    "active": {"active": {"mean": 0.01, "lower": -0.01, "upper": 0.03}}}}))

        with patch("promolift.benchmark._invoke", side_effect=fake_invoke):
            result = run_benchmark(tasks, output=self.root / "bench", agent_config=self.config,
                                   sandbox_image="sandbox:test", bootstrap_reps=20)
        searches = [cmd for cmd in calls if cmd[cmd.index("promolift.cli") + 1] == "search"]
        comparisons = [cmd for cmd in calls if cmd[cmd.index("promolift.cli") + 1] == "run"]
        self.assertEqual(len(searches), 4)
        self.assertEqual(len(comparisons), 2)
        self.assertTrue(all(cmd[cmd.index("--max-steps") + 1] == "1" for cmd in searches))
        self.assertTrue(all("--sandbox-image" in cmd for cmd in calls))
        self.assertTrue(all("--final" not in cmd for cmd in searches))
        self.assertTrue(all(command[command.index("--experience-dir") + 1] ==
                            str((self.root / "bench/plain").resolve())
                            for command in searches if "--experience-dir" in command))
        self.assertEqual(result["results"]["experience"]["first"]["paired_vs_plain"]["mean"], 0.01)

    def test_benchmark_accepts_one_dataset(self):
        task = load_task(self.task("only"))
        with patch("promolift.benchmark._invoke", side_effect=RuntimeError("submitted")):
            with self.assertRaisesRegex(RuntimeError, "submitted"):
                run_benchmark([task], output=self.root / "bench", agent_config=self.config,
                              sandbox_image="sandbox:test")


if __name__ == "__main__":
    unittest.main()
