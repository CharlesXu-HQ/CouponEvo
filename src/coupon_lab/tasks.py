"""Immutable inputs for one public randomized uplift benchmark task."""

from __future__ import annotations

import json
import re
import hashlib
from dataclasses import dataclass
from pathlib import Path

from .data import load_dataset
from .evaluate import Budget


@dataclass(frozen=True)
class Task:
    name: str
    manifest: Path
    candidate: Path
    budget: Budget
    objective: str
    seed: int
    max_steps: int
    dataset_sha256: str
    manifest_sha256: str
    candidate_sha256: str


def load_task(path: Path) -> Task:
    path = Path(path).resolve()
    config = json.loads(path.read_text())
    if set(config) != {"name", "manifest", "candidate", "budget", "objective", "seed",
                       "max_steps", "dataset_sha256", "manifest_sha256", "candidate_sha256"}:
        raise ValueError("task needs name, manifest, candidate, budget, objective, seed, max_steps, and input SHA-256 values")
    name = config["name"]
    if not isinstance(name, str) or not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", name):
        raise ValueError("task name must be a safe lowercase directory name")
    manifest = (path.parent / config["manifest"]).resolve()
    candidate = (path.parent / config["candidate"]).resolve()
    if hashlib.sha256(manifest.read_bytes()).hexdigest() != config["manifest_sha256"]:
        raise ValueError(f"task manifest SHA-256 changed: {name}")
    dataset = load_dataset(manifest)
    if dataset.source_sha256 != config["dataset_sha256"]:
        raise ValueError(f"task dataset SHA-256 changed: {name}")
    if not candidate.is_file():
        raise ValueError(f"task candidate does not exist: {name}")
    if hashlib.sha256(candidate.read_bytes()).hexdigest() != config["candidate_sha256"]:
        raise ValueError(f"task candidate SHA-256 changed: {name}")
    budget = Budget(config["budget"]["kind"], config["budget"]["value"])
    if budget.kind == "cost" and "coupon_cost" not in dataset.outcomes and dataset.manifest.get("fixed_send_cost") is None:
        raise ValueError("cost budget needs coupon_cost or fixed_send_cost")
    objective = config["objective"]
    available = set(dataset.outcomes) - {"coupon_cost"}
    if "revenue" in available and ("coupon_cost" in dataset.outcomes or dataset.manifest.get("fixed_send_cost") is not None):
        available.add("net_revenue")
    if "gross_margin" in available and ("coupon_cost" in dataset.outcomes or dataset.manifest.get("fixed_send_cost") is not None):
        available.add("net_margin")
    if objective not in available:
        raise ValueError(f"task objective is unavailable: {objective}")
    seed, max_steps = config["seed"], config["max_steps"]
    if not isinstance(seed, int) or not isinstance(max_steps, int) or max_steps < 1:
        raise ValueError("task seed must be an integer and max_steps must be positive")
    return Task(name, manifest, candidate, budget, objective, seed, max_steps,
                dataset.source_sha256, config["manifest_sha256"], config["candidate_sha256"])


def verify_task(task: Task) -> None:
    if (hashlib.sha256(task.manifest.read_bytes()).hexdigest() != task.manifest_sha256 or
            hashlib.sha256(task.candidate.read_bytes()).hexdigest() != task.candidate_sha256 or
            load_dataset(task.manifest).source_sha256 != task.dataset_sha256):
        raise ValueError(f"benchmark task inputs changed: {task.name}")
