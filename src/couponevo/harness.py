"""Data-only research plugins for the existing Agent experiment loop."""

from __future__ import annotations

import hashlib
import importlib
import json
import math
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


def model_design_identity(design: dict) -> dict:
    """Resolve the core's stable identity without modifying a recorded design."""
    return _model_evo_package().model_design_identity(design)


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
        "stage": "policy", "framework": "pytorch", "fields": list(dataset.features),
        "model_design_required": True,
        "horizontal_expansion_required": True,
        "capabilities": capabilities,
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
    if "feature_groups" in dataset.manifest:
        snapshot["feature_groups"] = package.validate_feature_groups(
            dataset.manifest["feature_groups"], snapshot["fields"])
    requirements = dataset.manifest.get("domain_requirements", [])
    if not isinstance(requirements, list) or any(
            not isinstance(item, dict) or
            any(not isinstance(item.get(key), str) or not item[key].strip()
                for key in ("id", "source", "as_of")) or
            not isinstance(item.get("fields"), list) or not item["fields"] or
            any(not isinstance(field, str) or not field.strip() for field in item["fields"])
            for item in requirements) or len({item["id"] for item in requirements}) != len(requirements):
        raise ValueError("domain_requirements need unique id, source, fields, and as_of")
    snapshot["domain_requirements"] = requirements
    applicable = package.applicability(snapshot, catalog)
    methods = package.method_applicability(snapshot, catalog)
    context = {"source": "ModelEvoHarness", "catalog": catalog,
            "applicability": applicable,
            "method_applicability": methods,
            "decision_applicability": package.decision_applicability(snapshot, catalog),
            "source_commit": _model_evo_revision(package),
            "composition_instructions": package.COMPOSITION_INSTRUCTIONS,
            "task_snapshot": snapshot, "dataset_profile": profile}
    training_applicability = getattr(package, "training_applicability", None)
    if callable(training_applicability):
        context["training_applicability"] = training_applicability(snapshot, catalog)
    model_api = getattr(package, "model_api", None)
    if callable(model_api):
        context["model_api"] = {"pytorch": model_api(
            catalog, framework="pytorch",
            method_ids={item["method_id"] for item in methods if item["status"] == "ready"})}
    load_guide = getattr(package, "load_guide", None)
    if callable(load_guide):
        context["knowledge"] = {item["family_id"]: load_guide(item["family_id"])
                                for item in applicable if item["status"] == "ready"}
    common_knowledge = getattr(package, "common_knowledge", None)
    if callable(common_knowledge):
        context.setdefault("knowledge", {}).update(common_knowledge())
    return context


def _model_evo_steps(steps: list[dict]) -> list[dict]:
    """Expose CouponEvo's recorded research through the core trial shape."""
    converted = []
    for step in steps:
        item = dict(step)
        proposal = dict(item.get("proposal") or {})
        if "research" not in proposal:
            research = item.get("research")
            if research is None and isinstance(item.get("mechanism"), str):
                research = {"mechanism": item["mechanism"]}
            if research is not None:
                proposal["research"] = research
        item["proposal"] = proposal
        converted.append(item)
    return converted


def validate_model_evo_data_request(request: dict, snapshot: dict,
                                    steps: list[dict], *,
                                    evidence: list[dict] | None = None) -> dict:
    fields = ("name", "definition", "source", "as_of", "evidence", "validation_plan")
    if not isinstance(request, dict) or any(
            not isinstance(request.get(key), str) or not request[key].strip() for key in fields):
        raise ValueError("feature_request needs name, definition, source, as_of, evidence, and validation_plan")
    result = {key: request[key].strip() for key in fields}
    if result["name"] in snapshot["fields"]:
        raise ValueError("feature_request.name must be absent from the frozen dataset")
    basis = request.get("basis")
    if basis == "experimental_evidence":
        ids = request.get("trial_ids")
        if not isinstance(ids, list) or len(ids) < 2 or any(
                not isinstance(trial_id, str) or not trial_id.strip() for trial_id in ids) or \
                len(set(ids)) != len(ids):
            raise ValueError("experimental_evidence needs at least two distinct trial_ids")
        evaluated = {step["id"]: step for step in steps
                     if step.get("status") == "evaluated" and isinstance(step.get("id"), str)}
        mechanisms = [evaluated.get(trial_id, {}).get("research", {}).get("mechanism")
                      for trial_id in ids]
        if any(not isinstance(mechanism, str) or not mechanism.strip()
               for mechanism in mechanisms) or len({mechanism.strip().casefold()
                                                     for mechanism in mechanisms}) < 2:
            raise ValueError("trial_ids must cite evaluated trials with distinct mechanisms")
        result.update(basis=basis, trial_ids=ids)
    elif basis == "domain_requirement":
        requirement_id = request.get("requirement_id")
        requirements = {item["id"]: item for item in snapshot.get("domain_requirements", [])}
        if (not isinstance(requirement_id, str) or requirement_id not in requirements or
                result["name"] not in requirements[requirement_id]["fields"] or
                result["source"] != requirements[requirement_id]["source"] or
                result["as_of"] != requirements[requirement_id]["as_of"]):
            raise ValueError("requirement_id must match a domain requirement for the requested field, source, and as_of")
        result.update(basis=basis, requirement_id=requirement_id)
    else:
        raise ValueError("feature_request.basis must be experimental_evidence or domain_requirement")
    for key in ("alternatives_considered", "evidence_ids"):
        if key in request:
            result[key] = request[key]
    if evidence is not None:
        core_request = {key: result[key] for key in
                        ("source", "as_of", "evidence", "validation_plan", "basis")}
        core_request.update(fields=[result["name"]], reason=result["definition"])
        for key in ("trial_ids", "requirement_id", "alternatives_considered", "evidence_ids"):
            if key in result:
                core_request[key] = result[key]
        _model_evo_package().validate_data_request(
            core_request, snapshot, _model_evo_steps(steps), evidence=evidence)
    return result


def validate_model_evo_audit_recommendations(items: object, *,
                                              evidence: list[dict] | None = None) -> list[dict]:
    """Keep audit suggestions separate from blocking feature requests."""
    if evidence is None:
        if items == []:
            return []
        raise ValueError("audit recommendations need host evidence")
    return _model_evo_package().validate_audit_recommendations(items, evidence=evidence)


def validation_business_observations(report: dict | None) -> list[dict]:
    """Expose only evaluator-computed policy aggregates from validation."""
    if not isinstance(report, dict) or report.get("holdout") != "validation":
        return []
    observations = []

    def add(identifier: str, policy: str, metric: str, kind: str, estimate: dict):
        if not isinstance(estimate, dict) or not isinstance(estimate.get("mean"), (int, float)) or \
                not math.isfinite(estimate["mean"]):
            return
        item = {"id": identifier, "holdout": "validation", "policy": policy,
                "metric": metric, "kind": kind, "mean": float(estimate["mean"])}
        for key in ("lower", "upper", "se"):
            value = estimate.get(key)
            if isinstance(value, (int, float)) and math.isfinite(value):
                item[key] = float(value)
        observations.append(item)

    for policy, metrics in (report.get("policies") or {}).items():
        for outcome, estimate in (metrics.get("effects") or {}).items():
            add(f"policy:{policy}:effect:{outcome}", policy, outcome, "policy_effect", estimate)
        if "net" in metrics:
            add(f"policy:{policy}:net", policy, "net", "policy_net", metrics["net"])
        if "cost" in metrics:
            add(f"policy:{policy}:cost", policy, "coupon_cost", "policy_cost",
                metrics["cost"])
    for comparison in ("paired_vs_baseline", "paired_vs_random"):
        for policy, metrics in (report.get(comparison) or {}).items():
            for metric, estimate in metrics.items():
                add(f"{comparison}:{policy}:{metric}", policy, metric, comparison, estimate)
    return observations


def validate_model_evo_reflection(reflection: dict, observation: dict, *,
                                   evidence: list[dict] | None = None) -> dict:
    technical = reflection.get("technical_experience")
    if not isinstance(technical, dict) or any(
            not isinstance(technical.get(key), str) or not technical[key].strip()
            for key in ("lesson", "evidence", "uncertainty", "next_test")):
        raise ValueError("technical_experience needs lesson, evidence, uncertainty, and next_test")
    if (observation.get("eligibility") == "blocked_implementation" and
            reflection.get("verdict") != "invalid"):
        raise ValueError("blocked implementation needs an invalid verdict")
    business = reflection.get("business_experience")
    if not isinstance(business, dict):
        raise ValueError("business_experience must be an object")
    if business.get("status") == "observed":
        valid_ids = {item["id"] for item in observation.get("business_observations", [])}
        if (observation.get("status") != "evaluated" or
                observation.get("eligibility") == "blocked_feature_leakage" or
                business.get("observation_id") not in valid_ids or
                any(not isinstance(business.get(key), str) or not business[key].strip()
                    for key in ("insight", "limitations"))):
            raise ValueError("business_experience.observation_id must cite a validation policy observation")
    elif business.get("status") == "not_observable":
        if not isinstance(business.get("reason"), str) or not business["reason"].strip():
            raise ValueError("business_experience.not_observable needs reason")
    else:
        raise ValueError("business_experience.status must be observed or not_observable")
    if evidence is not None:
        evaluation = {"business_observations": observation.get("business_observations", []),
                      "research": observation.get("research") or {},
                      "trial_id": observation.get("trial_id"),
                      "trial_status": observation.get("status"),
                      "implementation_check": (observation.get("implementation_check")
                                               if isinstance(observation.get("implementation_check"), dict)
                                               else {}),
                      "change_audit": (observation.get("change_audit")
                                       if isinstance(observation.get("change_audit"), dict)
                                       else {})}
        return _model_evo_package().validate_reflection(
            reflection, evaluation, observation["task_snapshot"],
            _model_evo_steps(observation.get("trial_history", [])), evidence=evidence)
    return reflection


def validate_research(proposal: dict, harness: dict, *, steps: list[dict] | None = None,
                      evidence: list[dict] | None = None) -> dict:
    """Require a testable design, without enumerating allowed model families."""
    if harness.get("source") == "ModelEvoHarness":
        research = proposal.get("research")
        bridged = dict(research) if isinstance(research, dict) else {}
        bridged.setdefault("input_fields", bridged.get("input_features"))
        bridged.setdefault("expected_result", proposal.get("expected_result"))
        semantics = bridged.get("prediction_semantics")
        if (semantics is not None or evidence is not None) and semantics not in (
                "probability_difference", "direct_cate", "ranking_score"):
            raise ValueError("research.prediction_semantics needs probability_difference, "
                             "direct_cate, or ranking_score")
        package = _model_evo_package()
        args = ({**proposal, "research": bridged}, harness["task_snapshot"], harness["catalog"])
        kwargs = {"evidence": evidence} if evidence is not None else {}
        if steps is not None:
            kwargs["sources"] = package.composition_sources(_model_evo_steps(steps))
        checked = package.validate_research(*args, **kwargs)
        design = checked.get("model_design")
        if design and design["change_scope"] != "initialize":
            parents = proposal.get("parent_ids") or []
            if not parents or design["parent_trial_id"] != parents[0]:
                raise ValueError("model_design.parent_trial_id must match the primary code parent")
        if semantics is not None:
            checked["prediction_semantics"] = semantics
        return checked
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
             "model_design": step["research"].get("model_design"),
             "reflection": step.get("reflection")}
            for step in steps if "research" in step]
