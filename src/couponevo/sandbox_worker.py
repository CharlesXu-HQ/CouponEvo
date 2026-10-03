"""Candidate-only process. The caller mounts training rows and holdout features, never labels."""

from __future__ import annotations

import importlib.util
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def _candidate(path: Path):
    spec = importlib.util.spec_from_file_location("sandbox_candidate", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    input_dir, output_dir = Path(sys.argv[1]), Path(sys.argv[2])
    job = json.loads((input_dir / "job.json").read_text())
    candidate = _candidate(input_dir / "candidate.py")
    seed = job["seed"]
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if job["operation"] == "predict":
        if job["device"] == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is unavailable in the sandbox")
            torch.cuda.reset_peak_memory_stats()
        with (input_dir / "train.json").open() as source:
            train = pd.read_json(source, orient="table")
        with (input_dir / "target.json").open() as source:
            target = pd.read_json(source, orient="table")
        result = candidate.fit_predict(train, target, **job["args"])
        if not isinstance(result, pd.DataFrame):
            raise ValueError("candidate must return a DataFrame")
        (output_dir / "predictions.json").write_text(result.to_json(orient="table", double_precision=15))
        response = {"model_device": result.attrs.get("model_device"),
                    "cuda_peak_bytes": (torch.cuda.max_memory_allocated()
                                        if job["device"] == "cuda" else 0)}
    elif job["operation"] == "policy":
        chooser = getattr(candidate, "choose_policy", None)
        if chooser is None:
            response = {"policy": None}
        else:
            scores = np.asarray(job["scores"], dtype=float)
            costs = None if job["costs"] is None else np.asarray(job["costs"], dtype=float)
            policy = chooser(scores, costs, job["budget_kind"], job["budget_value"])
            response = {"policy": np.asarray(policy).tolist()}
    else:
        raise ValueError("unknown sandbox operation")
    (output_dir / "response.json").write_text(json.dumps(response))


if __name__ == "__main__":
    main()
