"""Small, host-owned facts for research decisions; no inferred field semantics."""

from __future__ import annotations

import json


def search_evidence(journal: dict) -> list[dict]:
    facts = []
    profile = (journal.get("harness") or {}).get("dataset_profile") or {}
    if profile:
        facts.append({"id": "task.train_profile", "status": "observed", "scope": "task",
                      "source": "training partition profile",
                      "statement": "Raw training dtypes/cardinalities; these do not establish encoding or business meaning.",
                      "value": {key: profile[key] for key in
                                ("training_rows", "features") if key in profile}})
        for key in ("feature_timing", "cost"):
            if key in profile:
                facts.append({"id": f"task.{key}", "status": "declared", "scope": "task",
                              "source": "dataset contract", "statement": "Task metadata, not a trial-specific finding.",
                              "value": profile[key]})
    for step in [journal["baseline"], *journal["steps"]]:
        if step.get("status") != "evaluated":
            continue
        identifier = step["id"]
        report = step.get("report") or {}
        parts = {"score": step.get("score")}
        parts.update({key: report[key] for key in
                      ("policies", "ranking_diagnostics", "paired_vs_baseline", "paired_vs_random")
                      if key in report})
        parts.update({f"runtime.{key}": value for key, value in
                      report.get("runtime_diagnostics", {}).items() if key != "coverage"})
        for key, value in parts.items():
            if value is not None:
                facts.append({"id": f"{identifier}.{key}", "status": "observed", "scope": "trial",
                              "trial_id": identifier, "source": "evaluator/runtime observer",
                              "statement": f"Recorded {key}; validation metrics are exploratory, not mechanism attribution.",
                              "value": value})
    return facts


def observed_change_audit(before: dict, after: dict) -> dict | None:
    """Flag changed operations; execution alone does not prove their role in the model."""
    old, new = (report.get("runtime_diagnostics") or {} for report in (before, after))
    factors = []
    # Ignore source line numbers, row counts and event order. They do not establish a feature change.
    def encodings(runtime):
        return {json.dumps({key: event[key] for key in ("output_width", "output_columns")
                            if key in event}, sort_keys=True)
                for event in runtime.get("preprocessing", {}).get("get_dummies", [])}
    if encodings(old) and encodings(new) and encodings(old) != encodings(new):
        factors.append("preprocessing")
    old_losses = set(old.get("events", {}).get("losses", []))
    new_losses = set(new.get("events", {}).get("losses", []))
    if old_losses and new_losses and old_losses != new_losses:
        factors.append("loss")
    return {"status": "unverified", "changed_factors": factors, "source": "runtime observer",
            "coverage": "partial", "evidence": "Observed operations changed; final input/gradient lineage is not verified."} \
        if len(factors) > 1 else None
