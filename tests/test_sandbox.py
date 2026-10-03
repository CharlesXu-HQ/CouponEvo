import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from couponevo.sandbox import DockerSandbox


class SandboxTests(unittest.TestCase):
    def test_candidate_receives_train_and_holdout_features_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate = root / "candidate.py"
            candidate.write_text("def fit_predict(*args, **kwargs): pass\n")
            train = pd.DataFrame({"x": [1.0], "__treatment": [1], "active": [1]})
            target = pd.DataFrame({"x": [2.0]})
            seen = {}

            def fake_docker(command, **_):
                seen["command"] = command
                mounts = [command[i + 1] for i, item in enumerate(command) if item == "--mount"]
                input_mount = next(item for item in mounts if "dst=/input" in item)
                output_mount = next(item for item in mounts if "dst=/output" in item)
                input_dir = Path(input_mount.split("src=", 1)[1].split(",dst=", 1)[0])
                output_dir = Path(output_mount.split("src=", 1)[1].split(",dst=", 1)[0])
                with (input_dir / "target.json").open() as source:
                    visible = pd.read_json(source, orient="table")
                self.assertEqual(list(visible), ["x"])
                self.assertFalse((input_dir / "data.csv").exists())
                (output_dir / "predictions.json").write_text(
                    pd.DataFrame({"active_uplift": [0.1]}).to_json(orient="table"))
                (output_dir / "response.json").write_text(json.dumps({"model_device": "cuda",
                                                                     "cuda_peak_bytes": 1024}))
                return subprocess.CompletedProcess(command, 0, "", "")

            with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "private-key"}), \
                    patch("couponevo.sandbox.subprocess.run", side_effect=fake_docker):
                result = DockerSandbox("test-image").predict(candidate, train, target,
                                                               {"features": ["x"]}, 42, "cuda")
            command = seen["command"]
            self.assertEqual(result.attrs["model_device"], "cuda")
            self.assertEqual(result.attrs["cuda_peak_bytes"], 1024)
            self.assertIn("--gpus", command)
            self.assertEqual(command[command.index("--network") + 1], "none")
            self.assertIn("--read-only", command)
            self.assertNotIn("private-key", " ".join(command))
            self.assertFalse(any("data.csv" in item for item in command))


if __name__ == "__main__":
    unittest.main()
