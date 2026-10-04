"""Data-only research plugins for the existing Agent experiment loop."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .data import load_dataset, split_dataset


def load_harness(path: Path) -> dict:
    spec = json.loads(Path(path).read_text())
    if not isinstance(spec, dict) or spec.get("schema_version") != 1:
        raise ValueError("harness needs schema_version=1")
    for key in ("name", "guidance"):
        if not isinstance(spec.get(key), str) or not spec[key].strip():
            raise ValueError(f"harness needs {key}")
    directions = spec.get("directions")
    if not isinstance(directions, list) or not directions:
        raise ValueError("harness needs nonempty directions")
    for direction in directions:
        if not isinstance(direction, dict) or any(
                not isinstance(direction.get(key), str) or not direction[key].strip()
                for key in ("name", "question", "requires", "experiment", "pitfalls")):
            raise ValueError("harness direction needs name, question, requires, experiment, pitfalls")
    return spec


def harness_digest(path: Path | None) -> str | None:
    if path is None:
        return None
    payload = json.dumps(load_harness(path), sort_keys=True, ensure_ascii=False).encode()
    return hashlib.sha256(payload).hexdigest()


def build_harness_context(path: Path, manifest: Path, seed: int, strict: bool) -> dict:
    dataset = load_dataset(manifest, strict=strict)
    train = split_dataset(dataset, seed).train
    features = {name: {"dtype": str(train[name].dtype),
                       "unique_count": int(train[name].nunique(dropna=True)),
                       "missing_fraction": float(train[name].isna().mean())}
                for name in dataset.features}
    return {"plugin": load_harness(path), "dataset_profile": {
        "statistics_partition": "train", "training_rows": len(train), "features": features,
        "outcomes": dataset.outcomes,
        "cost": ("observed" if "coupon_cost" in dataset.outcomes else
                 "assumed_fixed_send" if dataset.manifest.get("fixed_send_cost") is not None else "unavailable"),
        "feature_timing": dataset.validation["feature_timing"],
        "assignment_probability": dataset.propensity,
        "contract": "One row per assignment unit; binary treatment; prediction receives declared features only.",
        "semantics": "Dtypes and cardinalities do not establish business meaning, sequence order, or availability before treatment."
    }}


def validate_research(proposal: dict, harness: dict) -> dict:
    """Require a testable design, without enumerating allowed model families."""
    research = proposal.get("research")
    fields = ("direction", "mechanism", "why_now", "data_rationale", "comparison", "falsification")
    if not isinstance(research, dict) or any(
            not isinstance(research.get(key), str) or not research[key].strip() for key in fields):
        raise ValueError("harness experiment needs research: " + ", ".join(fields))
    inputs = research.get("input_features")
    available = harness["dataset_profile"]["features"]
    if not isinstance(inputs, list) or any(
            not isinstance(name, str) or name not in available for name in inputs):
        raise ValueError("research.input_features must name existing manifest features; request missing data instead")
    alternatives = research.get("alternatives")
    if not isinstance(alternatives, list) or not alternatives or any(
            not isinstance(item, dict) or any(not isinstance(item.get(key), str) or not item[key].strip()
                                             for key in ("direction", "mechanism", "reason"))
            for item in alternatives):
        raise ValueError("research.alternatives needs at least one direction, mechanism, and reason")
    return {**{key: research[key].strip() for key in fields}, "input_features": inputs,
            "alternatives": [{key: item[key].strip() for key in ("direction", "mechanism", "reason")}
                             for item in alternatives]}


def research_history(steps: list[dict]) -> list[dict]:
    """Explicitly distinguish recorded trials from untested alternatives."""
    return [{"id": step["id"], "direction": step["research"]["direction"],
             "mechanism": step["research"]["mechanism"], "status": step["status"],
             "eligibility": step.get("eligibility"), "validation_score": step.get("score"),
             "reflection": step.get("reflection")}
            for step in steps if "research" in step]
