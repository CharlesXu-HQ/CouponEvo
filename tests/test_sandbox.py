import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from couponevo.sandbox import DockerSandbox


class SandboxTests(unittest.TestCase):
    def test_model_evo_exposes_only_pytorch_reference_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            references = root / "pytorch"
            references.mkdir()
            (references / "__init__.py").write_text("")
            (references / "reference.py").write_text("class ReferenceModel: pass\n")
            candidate = root / "candidate.py"
            candidate.write_text("from model_evo_harness.models.pytorch.reference import ReferenceModel\n"
                                 "def fit_predict(*args, **kwargs): pass\n")
            real_run = subprocess.run
            seen = {}

            def fake_docker(command, **_):
                seen["command"] = command
                mounts = [command[i + 1] for i, item in enumerate(command) if item == "--mount"]
                reference_mount = next(item for item in mounts if "dst=/opt/model-evo-src" in item)
                self.assertTrue(reference_mount.endswith(",readonly"))
                staged = Path(reference_mount.split("src=", 1)[1].split(",dst=", 1)[0])
                self.assertFalse((staged / "model_evo_harness/provider.py").exists())
                self.assertFalse((staged / "model_evo_harness/models/tensorflow").exists())
                imported = real_run([sys.executable, "-c",
                                     "from model_evo_harness.models.pytorch.reference import ReferenceModel; "
                                     "print(ReferenceModel.__name__)"],
                                    env={**os.environ, "PYTHONPATH": str(staged)},
                                    capture_output=True, text=True, check=True)
                self.assertEqual(imported.stdout.strip(), "ReferenceModel")
                output_mount = next(item for item in mounts if "dst=/output" in item)
                output_dir = Path(output_mount.split("src=", 1)[1].split(",dst=", 1)[0])
                (output_dir / "response.json").write_text(json.dumps({"model_device": "cpu"}))
                (output_dir / "predictions.json").write_text(
                    pd.DataFrame({"active_uplift": [0.1]}).to_json(orient="table"))
                return subprocess.CompletedProcess(command, 0, "", "")

            with patch("couponevo.sandbox._model_evo_models_dir", return_value=references), \
                    patch.dict(os.environ, {"DEEPSEEK_API_KEY": "private-key"}), \
                    patch("couponevo.sandbox.subprocess.run", side_effect=fake_docker):
                DockerSandbox("test-image", model_evo=True).predict(
                    candidate, pd.DataFrame({"x": [1.0], "__treatment": [1], "active": [1]}),
                    pd.DataFrame({"x": [2.0]}), {"features": ["x"]}, 42, "cpu")
            command = seen["command"]
            self.assertIn("PYTHONPATH=/opt/coupon-src:/opt/model-evo-src", command)
            self.assertEqual(command[command.index("--network") + 1], "none")
            self.assertNotIn("private-key", " ".join(command))

    def test_model_evo_requires_a_populated_reference_checkout(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch("couponevo.sandbox._model_evo_models_dir", return_value=Path(directory)):
            with self.assertRaisesRegex(ValueError, "PyTorch reference models"):
                DockerSandbox("test-image", model_evo=True)

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
