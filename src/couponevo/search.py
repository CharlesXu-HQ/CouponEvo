"""Resumable Agent search over a frozen offline uplift task."""

from __future__ import annotations

import ast
import difflib
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Callable

from .data import load_dataset
from .evaluate import Budget
from .experience import load_experience
from .research_evidence import observed_change_audit, search_evidence
from .harness import (build_harness_context, harness_identity, refresh_model_evo,
                      research_history, validate_model_evo_data_request,
                      validate_model_evo_audit_recommendations,
                      validate_model_evo_reflection, validate_research,
                      validation_business_observations)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _save(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def _verify_candidates(journal: dict, root: Path) -> None:
    for entry in [journal["baseline"], *journal["steps"]]:
        candidate = root / entry["candidate"]
        if not candidate.is_file() or _sha(candidate.read_bytes()) != entry["candidate_sha256"]:
            raise ValueError(f"candidate snapshot changed: {entry['id']}")


def _score(report: dict, objective: str) -> float:
    if objective not in report["policies"] or objective == "random":
        raise ValueError(f"objective policy is unavailable: {objective}")
    policy = report["policies"][objective]
    return float(policy["net"]["mean"] if objective.startswith("net_") else
                 policy["effects"][objective]["mean"])


def _evaluate(manifest: Path, budget: Budget, seed: int, output: Path, candidate: Path,
              device: str, strict_data: bool, timeout_seconds: int,
              compare_candidate: Path | None = None, feature_gaps: Path | None = None,
              final: bool = False, bootstrap_reps: int = 0,
              sandbox_image: str | None = None, harness_path: Path | None = None) -> dict:
    command = [sys.executable, "-m", "couponevo.cli", "run", str(manifest),
               "--budget-kind", budget.kind, "--budget", str(budget.value),
               "--seed", str(seed), "--output", str(output),
               "--candidate", str(candidate), "--device", device]
    if strict_data:
        command.append("--strict-data")
    if sandbox_image:
        command.extend(["--sandbox-image", sandbox_image, "--timeout-seconds", str(timeout_seconds)])
    if harness_path == Path("model-evo"):
        command.extend(["--harness", "model-evo"])
    if compare_candidate is not None:
        command.extend(["--compare-candidate", str(compare_candidate)])
    if feature_gaps is not None:
        command.extend(["--feature-gaps", str(feature_gaps)])
    if final:
        command.extend(["--final", "--bootstrap-reps", str(bootstrap_reps)])
    try:
        result = subprocess.run(command, stdin=subprocess.DEVNULL, capture_output=True,
                                text=True, timeout=timeout_seconds + (45 if sandbox_image else 0))
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(f"candidate exceeded {timeout_seconds}s execution limit") from error
    if result.returncode:
        raise RuntimeError((result.stderr.strip() or result.stdout.strip())[-2000:])
    lines = result.stdout.strip().splitlines()
    if not lines:
        raise RuntimeError("evaluator did not return a report path")
    report_path = Path(lines[-1]).with_name("report.json")
    if not report_path.is_file() or output not in report_path.parents:
        raise RuntimeError("evaluator returned an invalid report path")
    return json.loads(report_path.read_text())


def _context(journal: dict, root: Path, *, budget_exhausted: bool = False) -> dict:
    history = [{"id": "seed", "status": "evaluated", "score": journal["baseline"]["score"]}]
    history.extend({key: step.get(key) for key in ("id", "status", "operator", "parent_ids",
                                                   "hypothesis", "approach", "research", "expected_result", "score", "error",
                                                   "analysis", "analysis_error", "reflection",
                                                   "business_observations", "feature_request",
                                                   "eligibility") if key in step}
                   for step in journal["steps"])
    successful = sorted((step for step in journal["steps"] if step["status"] == "evaluated"),
                        key=lambda step: step["score"], reverse=True)[:2]
    latest = next((step for step in reversed(journal["steps"])
                   if step["status"] == "evaluated"), None)
    failed = next((step for step in reversed(journal["steps"]) if step["status"] == "failed"), None)
    selected = [journal["baseline"], *successful]
    if latest is not None:
        selected.append(latest)
    if failed is not None:
        selected.append(failed)
    represented = set()
    selected_ids = {entry["id"] for entry in selected}
    extra = 0
    for entry in reversed(journal["steps"]):
        design = (entry.get("research") or {}).get("model_design")
        if (not design or entry["status"] != "evaluated" or
                str(entry.get("eligibility", "")).startswith("blocked")):
            continue
        pair = (design["estimator"], design["backbone"])
        if pair not in represented:
            represented.add(pair)
            if entry["id"] in selected_ids:
                continue
            selected.append(entry)
            extra += 1
            if extra == 3:
                break
    available = {}
    for entry in selected:
        candidate = root / entry["candidate"]
        available[entry["id"]] = {"candidate_py": candidate.read_text(),
                                   "score": entry.get("score"), "status": entry["status"],
                                   "report": entry.get("report"), "error": entry.get("error"),
                                   "analysis": entry.get("analysis"),
                                   "eligibility": entry.get("eligibility"),
                                   "research": entry.get("research"),
                                   "approach": entry.get("approach")}
    context = {"dataset_sha256": journal["task"]["dataset"],
            "manifest_sha256": journal["task"]["manifest"],
            "objective": journal["task"]["objective"], "budget": journal["task"]["budget"],
            "history": history, "available": available, "best_id": journal["best_id"],
            "experience": journal.get("experience", []),
            "evidence": search_evidence(journal),
            "harness": journal.get("harness"),
            "research_history": research_history(journal["steps"]),
            "diagnoses": journal.get("diagnoses", []),
            "diagnostic_available": not any(
                item["after_step"] == len(journal["steps"])
                for item in journal.get("diagnoses", []))}
    proposal_errors = [error for error in journal.get("proposal_errors", [])
                       if error["after_step"] == len(journal["steps"]) and
                       error.get("budget_exhausted", False) == budget_exhausted][-2:]
    if proposal_errors:
        context.update(proposal_errors=proposal_errors,
                       proposal_error=proposal_errors[-1]["error"])
    return context


def _validate_proposal(proposal: dict, available: dict, harness: dict | None = None,
                       steps: list[dict] | None = None, *, evidence: list[dict] | None = None) -> dict:
    if not isinstance(proposal, dict):
        raise ValueError("Agent proposal must be an object")
    action = proposal.get("action", "experiment")
    if not isinstance(action, str):
        raise ValueError("Agent action must be experiment, diagnose, request_data, or stop")
    if action == "request_data":
        reason, request = proposal.get("reason"), proposal.get("feature_request")
        fields = ("name", "definition", "source", "as_of", "evidence", "validation_plan")
        if (not isinstance(reason, str) or not reason.strip() or not isinstance(request, dict) or
                any(not isinstance(request.get(key), str) or not request[key].strip()
                    for key in fields)):
            raise ValueError("Agent request_data needs reason and feature_request with name, "
                             "definition, source, as_of, evidence, and validation_plan")
        if harness and harness.get("source") == "ModelEvoHarness":
            canonical_request = validate_model_evo_data_request(
                request, harness["task_snapshot"], steps or [], evidence=evidence)
        else:
            canonical_request = {key: request[key].strip() for key in fields}
        return {"action": "request_data", "reason": reason.strip(),
                "feature_request": canonical_request}
    if action in {"stop", "diagnose"}:
        field = "reason" if action == "stop" else "question"
        value = proposal.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Agent {action} needs {field}")
        result = {"action": action, field: value.strip()}
        if action == "stop" and harness and harness.get("source") == "ModelEvoHarness":
            result["audit_recommendations"] = validate_model_evo_audit_recommendations(
                proposal.get("audit_recommendations", []), evidence=evidence)
        return result
    if action != "experiment":
        raise ValueError("Agent action must be experiment, diagnose, request_data, or stop")
    operator = proposal.get("operator")
    parents = proposal.get("parent_ids")
    required = {"draft": 0, "improve": 1, "debug": 1, "crossover": 2}
    if operator not in required or not isinstance(parents, list) or len(parents) != required[operator] or \
            len(set(parents)) != len(parents) or any(parent not in available for parent in parents):
        raise ValueError("Agent proposal has invalid operator or parent IDs")
    source = proposal.get("candidate_py")
    hypothesis = proposal.get("hypothesis")
    expected = proposal.get("expected_result", hypothesis)
    approach = proposal.get("approach")
    gaps = proposal.get("feature_gaps_md")
    if gaps is None:
        gaps = ""
    if not isinstance(source, str) or not source.strip() or not isinstance(hypothesis, str) or \
            not hypothesis.strip() or not isinstance(expected, str) or not expected.strip() or \
            not isinstance(gaps, str) or (approach is not None and
                                         (not isinstance(approach, str) or not approach.strip())):
        raise ValueError("Agent proposal needs candidate_py, hypothesis, expected_result, and optional feature_gaps_md")
    tree = ast.parse(source)
    allowed = {"__future__", "numpy", "pandas", "torch", "econml", "sklearn"}
    references = set()
    if harness and harness.get("source") == "ModelEvoHarness":
        references.add("model_evo_harness.models.pytorch.training")
        references.update(symbol.split(":", 1)[0] for item in
                          harness.get("catalog", {}).get("model_implementations", [])
                          if item.get("framework") == "pytorch"
                          for symbol in [item.get("symbol", "")]
                          if symbol.startswith("model_evo_harness.models.pytorch.") and
                          symbol.count(".") == 3)
    unsupported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            unsupported.update(alias.name for alias in node.names
                               if alias.name.split(".")[0] not in allowed and
                               alias.name not in references)
        if isinstance(node, ast.ImportFrom) and (node.level or
                ((node.module or "").split(".")[0] not in allowed and node.module not in references)):
            unsupported.add("." * node.level + (node.module or ""))
    if unsupported:
        guidance = "Allowed import roots: " + ", ".join(sorted(allowed)) + "."
        if references:
            guidance += (" Harness imports are limited to catalog-declared PyTorch model modules "
                         "and model_evo_harness.models.pytorch.training.")
        raise ValueError("Agent candidate imports unsupported modules: " +
                         ", ".join(sorted(unsupported)) + ". " + guidance)
    if not any(isinstance(node, ast.FunctionDef) and node.name == "fit_predict" for node in tree.body):
        raise ValueError("Agent candidate is missing fit_predict")
    if operator != "draft" and source == available[parents[0]]["candidate_py"]:
        raise ValueError("Agent candidate did not change its parent")
    validated = {"action": "experiment", "operator": operator, "parent_ids": parents,
                 "hypothesis": hypothesis.strip(), "expected_result": expected.strip(),
                 "candidate_py": source.rstrip() + "\n", "feature_gaps_md": gaps.strip()}
    if harness and harness.get("source") == "ModelEvoHarness" and gaps.strip():
        validated["feature_request"] = validate_model_evo_data_request(
            proposal.get("feature_request"), harness["task_snapshot"], steps or [], evidence=evidence)
    if approach is not None:
        validated["approach"] = approach.strip()
    if "reference_reads" in proposal:
        validated["reference_reads"] = proposal["reference_reads"]
    return validated


def _task(manifest_path: Path, budget: Budget, seed: int, initial_candidate: Path,
          objective: str, device: str, strict_data: bool,
          sandbox_image: str | None = None,
          experience_dir: Path | None = None, harness_path: Path | None = None) -> dict:
    data = load_dataset(manifest_path, strict=strict_data)
    framework = b"".join(Path(__file__).with_name(name).read_bytes()
                         for name in ("cli.py", "data.py", "evaluate.py", "sandbox.py",
                                      "sandbox_worker.py", "experience.py", "runtime_diagnostics.py"))
    agent_workflow = b"".join(Path(__file__).with_name(name).read_bytes()
                              for name in ("agent.py", "analysis.py", "provider.py", "search.py", "harness.py",
                                           "research_evidence.py"))
    return {"dataset": data.source_sha256, "manifest": _sha(manifest_path.read_bytes()),
            "framework": _sha(framework), "initial_candidate": _sha(initial_candidate.read_bytes()),
            "agent_workflow": _sha(agent_workflow), "harness": harness_identity(harness_path),
            "budget": {"kind": budget.kind, "value": budget.value}, "seed": seed,
            "objective": objective, "device": device, "strict_data": strict_data,
            "sandbox_image": sandbox_image,
            "experience_dir": str(experience_dir.resolve()) if experience_dir else None}


def run_search(manifest_path: Path, budget: Budget, *, seed: int, output: Path,
               initial_candidate: Path, objective: str, max_steps: int, search_id: str,
               proposer: Callable[[dict], dict], device: str = "cpu", strict_data: bool = False,
               resume: bool = False, timeout_seconds: int = 3600,
               analyzer: Callable[[dict, dict, Path, Path], dict] | None = None,
               reflector: Callable[[dict], dict] | None = None,
               diagnoser: Callable[[dict, str], dict] | None = None,
               agent_info: dict | None = None, sandbox_image: str | None = None,
               experience_dir: Path | None = None, harness_path: Path | None = None) -> dict:
    """Run up to max_steps candidate experiments; Agent actions see validation only."""
    if max_steps < 1 or timeout_seconds < 1:
        raise ValueError("max_steps and timeout_seconds must be positive")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", search_id) or search_id in {".", ".."}:
        raise ValueError("search_id must be a safe directory name")
    manifest_path, initial_candidate, output = (Path(path).resolve() for path in
                                                (manifest_path, initial_candidate, output))
    root = output / search_id
    if not resume:
        if root.exists():
            raise ValueError("search directory already exists; use resume")
        refresh_model_evo(harness_path)
    task = _task(manifest_path, budget, seed, initial_candidate, objective, device, strict_data,
                 sandbox_image, experience_dir, harness_path)
    journal_path = root / "journal.json"
    if resume:
        if not journal_path.is_file():
            raise ValueError("search journal does not exist")
        journal = json.loads(journal_path.read_text())
        if journal["task"] != task:
            raise ValueError("search task, dataset, or evaluator changed")
        if journal.get("agent") != agent_info:
            raise ValueError("search Agent configuration changed")
        if "final" in journal:
            raise ValueError("search was finalized")
        if "stop" in journal:
            raise ValueError("search was stopped by the Agent")
        if "data_request" in journal:
            raise ValueError("search has a data request; publish a new dataset and start a new search")
        _verify_candidates(journal, root)
    else:
        if root.exists():
            raise ValueError("search directory already exists; use resume")
        seed_dir = root / "steps" / "seed"
        seed_dir.mkdir(parents=True)
        seed_candidate = seed_dir / "candidate.py"
        seed_candidate.write_bytes(initial_candidate.read_bytes())
        seed_sha = _sha(seed_candidate.read_bytes())
        report = _evaluate(manifest_path, budget, seed, root / "runs", seed_candidate,
                           device, strict_data, timeout_seconds, sandbox_image=sandbox_image,
                           harness_path=harness_path)
        if _sha(seed_candidate.read_bytes()) != seed_sha:
            raise ValueError("candidate snapshot changed: seed")
        score = _score(report, objective)
        journal = {"task": task, "agent": agent_info,
                   "harness": (build_harness_context(harness_path, manifest_path, seed, strict_data,
                                                     objective)
                               if harness_path is not None else None),
                   "experience": (load_experience(experience_dir, task)
                                  if experience_dir else []),
                   "baseline": {"id": "seed", "status": "evaluated",
                                "candidate": "steps/seed/candidate.py", "candidate_sha256": seed_sha,
                                "score": score,
                                "report": report},
                   "steps": [], "diagnoses": [], "best_id": "seed"}
        _save(journal_path, journal)
    while True:
        _verify_candidates(journal, root)
        pending = next((step for step in journal["steps"] if step["status"] == "pending"), None)
        if pending is None:
            unreflected = next((step for step in journal["steps"]
                                if step["status"] in {"evaluated", "failed"} and
                                "reflection" not in step), None) if reflector else None
            if unreflected is not None:
                assert reflector is not None
                observation = {"trial_id": unreflected["id"],
                               "objective": objective, "budget": journal["task"]["budget"],
                               "hypothesis": unreflected["hypothesis"],
                               "approach": unreflected.get("approach"),
                               "research": unreflected.get("research"),
                               "expected_result": unreflected.get("expected_result",
                                                                   unreflected["hypothesis"]),
                               "status": unreflected["status"],
                               "score": unreflected.get("score"),
                               "baseline_score": journal["baseline"]["score"],
                               "report": unreflected.get("report"),
                               "error": unreflected.get("error"),
                               "analysis": unreflected.get("analysis"),
                               "analysis_error": unreflected.get("analysis_error"),
                               "change_audit": unreflected.get("change_audit"),
                               "eligibility": unreflected.get("eligibility")}
                if (journal.get("harness") or {}).get("source") == "ModelEvoHarness":
                    observation["harness_source"] = "ModelEvoHarness"
                    observation["composition_instructions"] = journal["harness"].get("composition_instructions", "")
                    observation["business_observations"] = unreflected.get(
                        "business_observations", [])
                    review = unreflected.get("analysis") or {}
                    review = review.get("max") or review.get("high") or {}
                    observation.update(evidence=search_evidence(journal),
                                       task_snapshot=journal["harness"]["task_snapshot"],
                                       trial_history=[{key: step[key] for key in ("id", "status", "research", "eligibility", "analysis", "reflection")
                                                       if key in step} for step in journal["steps"]],
                                       implementation_check=review.get("implementation_check"))
                try:
                    reflection = reflector(observation)
                    if (not isinstance(reflection, dict) or
                            reflection.get("verdict") not in {"consistent", "inconsistent",
                                                                "inconclusive", "invalid"} or
                            any(not isinstance(reflection.get(key), str) or
                                not reflection[key].strip() for key in
                                ("evidence", "lesson", "next_direction"))):
                        raise ValueError("Agent reflection needs verdict, evidence, lesson, and next_direction")
                    if (observation["status"] == "failed" or observation.get("eligibility") in {
                            "blocked_implementation", "blocked_feature_leakage"}) and reflection["verdict"] != "invalid":
                        raise ValueError("failed or blocked experiments require an invalid reflection")
                    if observation.get("harness_source") == "ModelEvoHarness":
                        reflection = validate_model_evo_reflection(reflection, observation,
                                                                   evidence=observation["evidence"])
                    unreflected["reflection"] = {key: reflection[key] for key in
                                                 ("verdict", "evidence", "lesson", "next_direction")}
                    if observation.get("harness_source") == "ModelEvoHarness":
                        unreflected["reflection"]["technical_experience"] = reflection["technical_experience"]
                        unreflected["reflection"]["business_experience"] = reflection["business_experience"]
                        for key in ("audit_recommendations", "future_feature_suggestions"):
                            if key in reflection:
                                unreflected["reflection"][key] = reflection[key]
                    unreflected.pop("reflection_error", None)
                    if unreflected.get("eligibility") == "pending_reflection":
                        if reflection["verdict"] == "invalid":
                            unreflected["eligibility"] = "blocked_reflection"
                        else:
                            unreflected["eligibility"] = "eligible"
                            best = (journal["baseline"] if journal["best_id"] == "seed" else
                                    next(step for step in journal["steps"]
                                         if step["id"] == journal["best_id"]))
                            if unreflected["score"] > best["score"]:
                                journal["best_id"] = unreflected["id"]
                    _save(journal_path, journal)
                except Exception as error:
                    unreflected["reflection_error"] = str(error)[-2000:]
                    _save(journal_path, journal)
                    raise
                continue
            exhausted = len(journal["steps"]) >= max_steps
            if exhausted and (reflector is None or
                              journal.get("budget_exhausted_decision", {}).get("after_step") == len(journal["steps"])):
                break
            context = _context(journal, root, budget_exhausted=exhausted)
            if exhausted:
                context["experiment_budget_exhausted"] = True
            proposal: dict | None = None
            for attempt in range(2):
                try:
                    raw = proposer(context)
                    proposal = _validate_proposal(raw, context["available"],
                                                  journal.get("harness"), journal["steps"],
                                                  evidence=context["evidence"])
                    if journal.get("harness") and proposal["action"] == "experiment" and not exhausted:
                        proposal["research"] = validate_research(
                            raw, journal["harness"], steps=[journal["baseline"], *journal["steps"]],
                            evidence=context["evidence"])
                    if exhausted and proposal["action"] not in ("stop", "request_data"):
                        proposal = None
                        break
                    if proposal["action"] == "diagnose" and (
                            diagnoser is None or not context["diagnostic_available"]):
                        raise ValueError("diagnosis is unavailable until the next experiment")
                    break
                except (ValueError, SyntaxError) as error:
                    journal.setdefault("proposal_errors", []).append({
                        "after_step": len(journal["steps"]), "attempt": attempt + 1,
                        "budget_exhausted": exhausted, "error": str(error)[-2000:]})
                    _save(journal_path, journal)
                    if attempt:
                        if not exhausted:
                            raise
                        journal["budget_exhausted_decision"] = {
                            "after_step": len(journal["steps"]),
                            "reason": "invalid_terminal_proposal",
                            "last_error": str(error)[-2000:]}
                        proposal = None
                        break
                    context = {**context, **_context(journal, root, budget_exhausted=exhausted)}
            if _task(manifest_path, budget, seed, initial_candidate, objective, device,
                     strict_data, sandbox_image, experience_dir, harness_path) != task:
                raise RuntimeError("search task or dataset changed during proposal")
            if proposal is None:
                if journal.get("budget_exhausted_decision", {}).get("after_step") != len(journal["steps"]):
                    journal["budget_exhausted_decision"] = {
                        "after_step": len(journal["steps"]), "reason": "no_terminal_action"}
                _save(journal_path, journal)
                break
            if proposal["action"] == "stop":
                journal["stop"] = {"reason": proposal["reason"],
                                   "after_step": len(journal["steps"])}
                if "audit_recommendations" in proposal:
                    journal["stop"]["audit_recommendations"] = proposal["audit_recommendations"]
                _save(journal_path, journal)
                break
            if proposal["action"] == "request_data":
                request = {"reason": proposal["reason"],
                           "feature_request": proposal["feature_request"],
                           "after_step": len(journal["steps"]),
                           "dataset_sha256": task["dataset"],
                           "manifest_sha256": task["manifest"]}
                journal["data_request"] = request
                _save(journal_path, journal)
                _save(root / "feature_request.json", request)
                break
            if proposal["action"] == "diagnose":
                assert diagnoser is not None
                diagnosis = diagnoser(context, proposal["question"])
                if (not isinstance(diagnosis, dict) or
                        any(not isinstance(diagnosis.get(key), str) or
                            not diagnosis[key].strip() for key in
                            ("finding", "evidence", "next_direction"))):
                    raise ValueError("Agent diagnosis needs finding, evidence, and next_direction")
                journal.setdefault("diagnoses", []).append({
                    "after_step": len(journal["steps"]), "question": proposal["question"],
                    **{key: diagnosis[key] for key in ("finding", "evidence", "next_direction")}})
                _save(journal_path, journal)
                continue
            step_id = f"step-{len(journal['steps']) + 1:03d}"
            step_dir = root / "steps" / step_id
            step_dir.mkdir()
            (step_dir / "candidate.py").write_text(proposal.pop("candidate_py"))
            candidate_sha = _sha((step_dir / "candidate.py").read_bytes())
            if proposal["feature_gaps_md"]:
                (step_dir / "feature_gaps.md").write_text(proposal["feature_gaps_md"] + "\n")
            pending = {"id": step_id, "status": "pending", "candidate": f"steps/{step_id}/candidate.py",
                       "candidate_sha256": candidate_sha,
                       **{key: proposal[key] for key in
                          ("operator", "parent_ids", "hypothesis", "expected_result")}}
            if "approach" in proposal:
                pending["approach"] = proposal["approach"]
            if "reference_reads" in proposal:
                pending["reference_reads"] = proposal["reference_reads"]
            if "research" in proposal:
                pending["research"] = proposal["research"]
            if "feature_request" in proposal:
                pending["feature_request"] = proposal["feature_request"]
            journal["steps"].append(pending)
            _save(journal_path, journal)
        if _task(manifest_path, budget, seed, initial_candidate, objective, device,
                 strict_data, sandbox_image, experience_dir, harness_path) != task:
            raise RuntimeError("search task or dataset changed before evaluation")
        candidate = root / pending["candidate"]
        parent_id = pending["parent_ids"][0] if pending["parent_ids"] else journal["best_id"]
        parent = (journal["baseline"] if parent_id == "seed" else
                  next(step for step in journal["steps"] if step["id"] == parent_id))
        if parent["status"] != "evaluated":
            parent = (journal["baseline"] if journal["best_id"] == "seed" else
                      next(step for step in journal["steps"] if step["id"] == journal["best_id"]))
        gap_path = candidate.with_name("feature_gaps.md")
        try:
            report = _evaluate(manifest_path, budget, seed, root / "runs", candidate,
                               device, strict_data, timeout_seconds, root / parent["candidate"],
                               gap_path if gap_path.exists() else None,
                               sandbox_image=sandbox_image, harness_path=harness_path)
            _verify_candidates(journal, root)
            pending["score"] = _score(report, objective)
            pending["report"] = report
            audit = observed_change_audit(parent["report"], report)
            if audit:
                pending["change_audit"] = audit
            if (journal.get("harness") or {}).get("source") == "ModelEvoHarness":
                pending["business_observations"] = validation_business_observations(report)
            pending["status"] = "evaluated"
            _save(journal_path, journal)
        except Exception as error:
            pending["status"] = "failed"
            pending["error"] = str(error)[-2000:]
        if pending["status"] == "evaluated" and analyzer is not None:
            prior_source = (root / parent["candidate"]).read_text()
            candidate_source = candidate.read_text()
            experiment_context = {
                "hypothesis": pending["hypothesis"],
                "research": pending.get("research"),
                "evidence": search_evidence(journal),
                "change_audit": pending.get("change_audit"),
                "parent_id": parent["id"],
                "parent_candidate_py": prior_source,
                "parent_research": parent.get("research"),
                "component_sources": {identifier: {key: source[key] for key in
                                      ("candidate_py", "research", "eligibility", "status")}
                                      for identifier, source in _context(journal, root)["available"].items()
                                      if identifier in {item["source_trial_id"] for item in
                                          (pending.get("research") or {}).get("model_design", {}).get("inheritance", [])}},
                "candidate_diff": "".join(difflib.unified_diff(
                    prior_source.splitlines(keepends=True), candidate_source.splitlines(keepends=True),
                    fromfile="parent.py", tofile="candidate.py")),
            }
            for attempt in range(2):
                pending["analysis_attempts"] = attempt + 1
                try:
                    pending["analysis"] = analyzer(parent["report"],
                                                   {**pending["report"], "experiment_context": experiment_context}, candidate,
                                                   root / "runs" / pending["report"]["run_id"])
                    pending.pop("analysis_error", None)
                    break
                except Exception as error:
                    pending["analysis_error"] = str(error)[-2000:]
        if pending["status"] == "evaluated":
            review = pending.get("analysis") or {}
            review = review.get("max") or review.get("high") or {}
            if pending.get("analysis_error"):
                pending["eligibility"] = "blocked_analysis_error"
            elif (pending["report"].get("runtime_diagnostics", {}).get(
                    "prediction_contract", {}).get("status") == "inconsistent" or
                  (pending.get("research", {}).get("prediction_semantics") == "probability_difference" and
                   pending["report"].get("runtime_diagnostics", {}).get(
                       "prediction_contract", {}).get("status") != "consistent") or
                  review.get("implementation_check", {}).get("status") == "contradicted"):
                pending["eligibility"] = "blocked_implementation"
            elif review.get("feature_leakage", {}).get("confirmed") is True:
                pending["eligibility"] = "blocked_feature_leakage"
            elif reflector is not None:
                pending["eligibility"] = "pending_reflection"
            else:
                pending["eligibility"] = "eligible"
                best = (journal["baseline"] if journal["best_id"] == "seed" else
                        next(step for step in journal["steps"] if step["id"] == journal["best_id"]))
                if pending["score"] > best["score"]:
                    journal["best_id"] = pending["id"]
        _save(journal_path, journal)
    return journal


def finalize_search(manifest_path: Path, budget: Budget, *, seed: int, output: Path,
                    initial_candidate: Path, objective: str, search_id: str,
                    device: str = "cpu", strict_data: bool = False,
                    timeout_seconds: int = 3600, bootstrap_reps: int = 2000,
                    sandbox_image: str | None = None,
                    experience_dir: Path | None = None, harness_path: Path | None = None) -> dict:
    """Evaluate the frozen validation champion against the seed once on test."""
    if bootstrap_reps < 2 or timeout_seconds < 1:
        raise ValueError("finalize needs at least two bootstrap resamples and a positive timeout")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", search_id) or search_id in {".", ".."}:
        raise ValueError("search_id must be a safe directory name")
    manifest_path, initial_candidate, output = (Path(path).resolve() for path in
                                                (manifest_path, initial_candidate, output))
    root = output / search_id
    journal_path = root / "journal.json"
    if not journal_path.is_file():
        raise ValueError("search journal does not exist")
    journal = json.loads(journal_path.read_text())
    if experience_dir is None and journal["task"].get("experience_dir"):
        experience_dir = Path(journal["task"]["experience_dir"])
    if journal["task"] != _task(manifest_path, budget, seed, initial_candidate, objective,
                                 device, strict_data, sandbox_image, experience_dir, harness_path):
        raise ValueError("search task, dataset, or evaluator changed")
    _verify_candidates(journal, root)
    if "data_request" in journal:
        raise ValueError("cannot finalize a search with a data request")
    if any(step["status"] == "pending" for step in journal["steps"]):
        raise ValueError("cannot finalize with a pending experiment")
    if "final" in journal:
        return journal["final"]["report"]
    champion = (journal["baseline"] if journal["best_id"] == "seed" else
                next(step for step in journal["steps"] if step["id"] == journal["best_id"]))
    report = _evaluate(manifest_path, budget, seed, root / "final", root / champion["candidate"],
                       device, strict_data, timeout_seconds,
                       compare_candidate=root / journal["baseline"]["candidate"],
                       final=True, bootstrap_reps=bootstrap_reps,
                       sandbox_image=sandbox_image, harness_path=harness_path)
    journal["final"] = {"candidate_id": champion["id"], "report": report}
    _save(journal_path, journal)
    return report
