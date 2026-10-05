"""Run generated candidates with access only to their declared input files."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


def _model_evo_models_dir() -> Path:
    return (Path(__file__).resolve().parents[2] / "third_party/model-evo-harness/src/"
            "model_evo_harness/models/pytorch")


class DockerSandbox:
    def __init__(self, image: str, *, timeout_seconds: int = 3600, model_evo: bool = False):
        if not image or timeout_seconds < 1:
            raise ValueError("sandbox image and positive timeout are required")
        self.image = image
        self.timeout_seconds = timeout_seconds
        self.source = Path(__file__).resolve().parents[1]
        self.venv = Path(os.environ.get("COUPONEVO_SANDBOX_VENV", os.sys.prefix)).resolve()
        if not (self.venv / "bin/python").exists():
            raise ValueError("sandbox requires a Linux Python 3.12 virtual environment")
        self.reference_models = _model_evo_models_dir() if model_evo else None
        if self.reference_models is not None and (not (self.reference_models / "__init__.py").is_file() or
                                                  not any(path.name != "__init__.py"
                                                          for path in self.reference_models.glob("*.py"))):
            raise ValueError("ModelEvoHarness PyTorch reference models are missing from the submodule checkout")

    def _run(self, candidate: Path, job: dict, frames: dict[str, pd.DataFrame] | None = None) -> tuple[dict, str]:
        with tempfile.TemporaryDirectory(prefix="couponevo-sandbox-") as scratch:
            root = Path(scratch)
            input_dir, output_dir = root / "input", root / "output"
            input_dir.mkdir()
            output_dir.mkdir()
            shutil.copyfile(candidate, input_dir / "candidate.py")
            (input_dir / "job.json").write_text(json.dumps(job))
            for name, frame in (frames or {}).items():
                (input_dir / f"{name}.json").write_text(frame.to_json(orient="table", double_precision=15))
            cidfile = root / "container.id"
            reference_mount = []
            python_path = "/opt/coupon-src"
            if self.reference_models is not None:
                staged = root / "model-evo-src/model_evo_harness/models/pytorch"
                staged.mkdir(parents=True)
                (staged.parent / "__init__.py").write_text("")
                (staged.parent.parent / "__init__.py").write_text("")
                for source in self.reference_models.glob("*.py"):
                    if source.is_symlink():
                        raise ValueError("ModelEvoHarness reference model files must not be symlinks")
                    shutil.copy2(source, staged / source.name)
                reference_mount = ["--mount", f"type=bind,src={root / 'model-evo-src'},"
                                   "dst=/opt/model-evo-src,readonly"]
                python_path += ":/opt/model-evo-src"
            command = ["docker", "run", "--rm", "--cidfile", str(cidfile),
                       "--network", "none", "--read-only", "--cap-drop", "ALL",
                       "--security-opt", "no-new-privileges", "--pids-limit", "256",
                       "--memory", "16g", "--cpus", "8", "--user", f"{os.getuid()}:{os.getgid()}",
                       "--tmpfs", "/tmp:rw,nosuid,size=1g",
                       "--mount", f"type=bind,src={input_dir},dst=/input,readonly",
                       "--mount", f"type=bind,src={output_dir},dst=/output",
                       "--mount", f"type=bind,src={self.source},dst=/opt/coupon-src,readonly",
                       *reference_mount,
                       "--mount", f"type=bind,src={self.venv},dst=/opt/venv,readonly",
                       "--env", f"PYTHONPATH={python_path}", "--env", "PYTHONDONTWRITEBYTECODE=1",
                       self.image, "/opt/venv/bin/python", "-m", "couponevo.sandbox_worker",
                       "/input", "/output"]
            if job.get("device") == "cuda":
                command[2:2] = ["--gpus", "all"]
            try:
                result = subprocess.run(command, stdin=subprocess.DEVNULL, capture_output=True,
                                        text=True, timeout=self.timeout_seconds)
            except subprocess.TimeoutExpired as error:
                if cidfile.exists():
                    subprocess.run(["docker", "rm", "-f", cidfile.read_text().strip()],
                                   capture_output=True, timeout=30)
                raise RuntimeError("sandbox candidate timed out") from error
            if result.returncode:
                raise RuntimeError(f"sandbox candidate failed: {(result.stderr or result.stdout)[-2000:]}")
            response = json.loads((output_dir / "response.json").read_text())
            predictions = (output_dir / "predictions.json").read_text() if job["operation"] == "predict" else ""
            return response, predictions

    def predict(self, candidate: Path, train: pd.DataFrame, target: pd.DataFrame,
                args: dict, seed: int, device: str) -> pd.DataFrame:
        response, raw = self._run(candidate, {"operation": "predict", "args": args,
                                              "seed": seed, "device": device},
                                  {"train": train, "target": target})
        from io import StringIO
        result = pd.read_json(StringIO(raw), orient="table")
        result.attrs["model_device"] = response.get("model_device")
        result.attrs["cuda_peak_bytes"] = response.get("cuda_peak_bytes", 0)
        return result

    def choose(self, candidate: Path, scores: np.ndarray, costs: np.ndarray | None,
               budget_kind: str, budget_value: float, seed: int, device: str):
        response, _ = self._run(candidate, {"operation": "policy", "scores": scores.tolist(),
                                             "costs": None if costs is None else costs.tolist(),
                                             "budget_kind": budget_kind, "budget_value": budget_value,
                                             "seed": seed, "device": device})
        return response["policy"]
