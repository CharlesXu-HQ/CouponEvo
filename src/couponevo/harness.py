"""Data-only research plugins for the existing Agent experiment loop."""

from __future__ import annotations

import hashlib
import importlib
import json
import subprocess
import sys
from pathlib import Path

from .data import load_dataset, split_dataset


_model_evo_loaded: tuple[object, str | None] | None = None


def _model_evo_package():
    global _model_evo_loaded
    if "model_evo_harness" in sys.modules and _model_evo_loaded is None:
        raise RuntimeError("model-evo package was preloaded without a recorded revision; "
                           "start a new Python process")
    try:
        package = importlib.import_module("model_evo_harness")
    except ModuleNotFoundError as error:
        if error.name != "model_evo_harness":
            raise
        raise RuntimeError("--harness model-evo requires the optional model-evo-harness package; "
                           "install it from the submodule checkout (see docs/model-evo-harness.md)") from error
    revision = _model_evo_revision(package)
    if _model_evo_loaded is None:
        _model_evo_loaded = (package, revision)
    elif _model_evo_loaded != (package, revision):
        raise RuntimeError("model-evo submodule changed while loaded; start a new Python process")
    return package


def _is_model_evo(path: Path | None) -> bool:
    return path is not None and Path(path) == Path("model-evo")


def _model_evo_revision(package) -> str | None:
    checkout = Path(__file__).resolve().parents[2] / "third_party/model-evo-harness"
    if not Path(package.__file__).resolve().is_relative_to(checkout.resolve()):
        return None
    return subprocess.run(["git", "-C", str(checkout), "rev-parse", "HEAD"], check=True,
                          capture_output=True, text=True).stdout.strip()


def refresh_model_evo(path: Path | None) -> None:
    """Use remote main for a new search, before importing the external package."""
    if not _is_model_evo(path):
        return
    root = Path(__file__).resolve().parents[2]
    checkout = root / "third_party/model-evo-harness"
    loaded = sys.modules.get("model_evo_harness")
    if loaded is not None and (_model_evo_loaded is None or _model_evo_loaded[0] is not loaded):
        raise RuntimeError("model-evo package was preloaded without a recorded revision; "
                           "start a new Python process")
    if loaded is None and _model_evo_loaded is not None:
        raise RuntimeError("model-evo package was unloaded; start a new Python process")
    if loaded is not None and not Path(loaded.__file__).resolve().is_relative_to(checkout.resolve()):
        raise RuntimeError("model-evo package must be imported from the submodule checkout")

    def git(*args: str) -> str:
        return subprocess.run(["git", "-C", str(checkout), *args], check=True,
                              capture_output=True, text=True).stdout.strip()

    if (checkout / ".git").exists():
        if git("status", "--porcelain"):
            raise RuntimeError("model-evo submodule has uncommitted changes")
        if loaded is not None and git("rev-parse", "HEAD") != _model_evo_loaded[1]:
            raise RuntimeError("model-evo submodule changed while loaded; start a new Python process")
    try:
        subprocess.run(["git", "-C", str(root), "submodule", "update", "--init", "--remote",
                        "third_party/model-evo-harness"], check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as error:
        raise RuntimeError("could not update model-evo submodule from remote main") from error
    if not (checkout / ".git").exists():
        raise RuntimeError("model-evo submodule checkout is missing after update")
    if git("status", "--porcelain"):
        raise RuntimeError("model-evo submodule has uncommitted changes")
    if loaded is not None and git("rev-parse", "HEAD") != _model_evo_loaded[1]:
        raise RuntimeError("model-evo submodule changed while loaded; start a new Python process")
    package = _model_evo_package()
    if not Path(package.__file__).resolve().is_relative_to(checkout.resolve()):
        raise RuntimeError("model-evo package must be installed from the submodule checkout")


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


def harness_identity(path: Path | None) -> str | dict | None:
    if not _is_model_evo(path):
        return harness_digest(path)
    package = _model_evo_package()
    catalog = package.load_catalog()
    return {"source": "ModelEvoHarness", "catalog_digest": package.catalog_digest(catalog),
            "implementation_digest": package.implementation_digest(),
            "source_commit": _model_evo_revision(package)}


def build_harness_context(path: Path, manifest: Path, seed: int, strict: bool,
                          objective: str | None = None) -> dict:
    dataset = load_dataset(manifest, strict=strict)
    train = split_dataset(dataset, seed).train
    features = {name: {"dtype": str(train[name].dtype),
                       "unique_count": int(train[name].nunique(dropna=True)),
                       "missing_fraction": float(train[name].isna().mean())}
                for name in dataset.features}
    profile = {
        "statistics_partition": "train", "training_rows": len(train), "features": features,
        "outcomes": dataset.outcomes,
        "cost": ("observed" if "coupon_cost" in dataset.outcomes else
                 "assumed_fixed_send" if dataset.manifest.get("fixed_send_cost") is not None else "unavailable"),
        "feature_timing": dataset.validation["feature_timing"],
        "assignment_probability": dataset.propensity,
        "contract": "One row per assignment unit; binary treatment; prediction receives declared features only.",
        "semantics": "Dtypes and cardinalities do not establish business meaning, sequence order, or availability before treatment."
    }
    if not _is_model_evo(path):
        return {"plugin": load_harness(path), "dataset_profile": profile}
    if not objective:
        raise ValueError("model-evo harness needs a fixed objective")
    package = _model_evo_package()
    catalog = package.load_catalog()
    capabilities = ["tabular_features", "observed_outcome_labels",
                    "assignment_or_exposure_propensity", "decision_rule_adapter"]
    if len(set(dataset.outcomes) - {"coupon_cost"}) > 1:
        capabilities.append("multiple_outcomes")
    snapshot = {
        "task_id": "couponevo-policy",
        "dataset_digest": hashlib.sha256((dataset.source_sha256 +
                                           hashlib.sha256(Path(manifest).read_bytes()).hexdigest() +
                                           str(seed)).encode()).hexdigest(),
        "stage": "policy", "fields": list(dataset.features), "capabilities": capabilities,
        "objective": {"name": objective, "direction": "max"},
        "evaluation_protocol": {
            "unit": "manifest unit_id" if dataset.has_unit_id else "source row",
            "split": f"treatment-stratified unit 60/20/20; seed={seed}",
            "metric": objective,
            "candidate_universe": "send coupon or do not send under the fixed budget",
            "label_provenance": "randomized treatment assignment and observed outcomes",
            "feature_timing": dataset.validation["feature_timing"],
        },
    }
    return {"source": "ModelEvoHarness", "catalog": catalog,
            "applicability": package.applicability(snapshot, catalog),
            "method_applicability": package.method_applicability(snapshot, catalog),
            "decision_applicability": package.decision_applicability(snapshot, catalog),
            "source_commit": _model_evo_revision(package),
            "task_snapshot": snapshot, "dataset_profile": profile}


def validate_research(proposal: dict, harness: dict) -> dict:
    """Require a testable design, without enumerating allowed model families."""
    if harness.get("source") == "ModelEvoHarness":
        research = proposal.get("research")
        bridged = dict(research) if isinstance(research, dict) else {}
        bridged.setdefault("input_fields", bridged.get("input_features"))
        bridged.setdefault("expected_result", proposal.get("expected_result"))
        return _model_evo_package().validate_research(
            {**proposal, "research": bridged}, harness["task_snapshot"], harness["catalog"])
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
