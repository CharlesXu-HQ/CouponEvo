"""Validation-only lessons from completed searches on the same frozen task."""

from __future__ import annotations

import json
from pathlib import Path


_IDENTITY_FIELDS = ("dataset", "manifest", "objective", "budget", "seed", "framework",
                    "agent_workflow", "strict_data")


def load_experience(directory: Path, task: dict, limit: int = 5) -> list[dict]:
    lessons = []
    paths = sorted(Path(directory).glob("*/journal.json"),
                   key=lambda path: (path.stat().st_mtime_ns, str(path)), reverse=True)
    for path in paths:
        try:
            journal = json.loads(path.read_text())
            previous = journal["task"]
            if previous.get("harness") != task.get("harness"):
                continue
            model_evo = isinstance(task.get("harness"), dict) and \
                task["harness"].get("source") == "ModelEvoHarness"
            complete = "final" in journal or (model_evo and any(
                key in journal for key in ("stop", "data_request", "budget_exhausted_decision")))
            if not complete or any(previous[key] != task[key] for key in _IDENTITY_FIELDS):
                continue
            baseline = float(journal["baseline"]["score"])
            for step in reversed(journal["steps"]):
                if step["status"] not in {"evaluated", "failed"}:
                    continue
                reflection = step.get("reflection") or {}
                lesson = {"source_search": path.parent.name, "operator": step["operator"],
                                "approach": (step.get("approach") or "")[:500],
                                "research": step.get("research"),
                                "hypothesis": step["hypothesis"][:500],
                                "expected_result": (step.get("expected_result") or "")[:500],
                                "status": step["status"],
                                "validation_delta": (float(step["score"]) - baseline
                                                     if step["status"] == "evaluated" else None),
                                "verdict": reflection.get("verdict"),
                                "evidence": (reflection.get("evidence") or "")[:500],
                                "lesson": (reflection.get("lesson") or "")[:500],
                                "next_direction": (reflection.get("next_direction") or "")[:500]}
                if model_evo:
                    technical = reflection.get("technical_experience")
                    business = reflection.get("business_experience")
                    if not isinstance(technical, dict) or not isinstance(business, dict):
                        continue
                    lesson["source_trial"] = step["id"]
                    lesson["technical_experience"] = {
                        key: str(technical.get(key, ""))[:500]
                        for key in ("lesson", "evidence", "uncertainty", "next_test", "attribution")}
                    lesson["audit_recommendations"] = reflection.get("audit_recommendations", [])
                    lesson["business_experience"] = {
                        key: str(business[key])[:500]
                        for key in (("status", "observation_id", "insight", "limitations")
                                    if business.get("status") == "observed" else ("status", "reason"))
                        if key in business}
                review = step.get("analysis") or {}
                review = review.get("max") or review.get("high") or {}
                check = review.get("implementation_check") or {}
                contract = (step.get("report") or {}).get("runtime_diagnostics", {}).get(
                    "prediction_contract", {})
                lesson["knowledge_status"] = check.get("status", "unverified")
                if (step.get("eligibility") == "blocked_implementation" or
                        check.get("status") == "contradicted" or contract.get("status") == "inconsistent"):
                    evidence = str(check.get("evidence") or contract.get("evidence") or
                                   "Implementation does not match the declared experiment")[:1000]
                    repair = "Correct the recorded mismatch and verify runtime evidence before reusing this mechanism."
                    technical = {"lesson": "Implementation contradicted; no mechanism conclusion is supported.",
                                 "evidence": evidence, "uncertainty": "The original hypothesis remains untested.",
                                 "next_test": repair, "attribution": "unverified"}
                    lesson.update(knowledge_status="contradicted", validation_delta=None,
                                  verdict="invalid", evidence=evidence, lesson=technical["lesson"],
                                  next_direction=repair, technical_experience=technical)
                    lesson["invalidated_claim"] = {key: lesson.pop(key) for key in
                                                   ("hypothesis", "expected_result", "research", "approach")}
                lessons.append(lesson)
                if len(lessons) >= limit:
                    return lessons
        except (KeyError, ValueError, TypeError, OSError):
            continue
    return lessons
