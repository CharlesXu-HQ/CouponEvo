"""Validation-only lessons from completed searches on the same frozen task."""

from __future__ import annotations

import json
from pathlib import Path


_IDENTITY_FIELDS = ("dataset", "manifest", "objective", "budget", "seed", "framework",
                    "strict_data")


def load_experience(directory: Path, task: dict, limit: int = 5) -> list[dict]:
    lessons = []
    paths = sorted(Path(directory).glob("*/journal.json"),
                   key=lambda path: (path.stat().st_mtime_ns, str(path)), reverse=True)
    for path in paths:
        try:
            journal = json.loads(path.read_text())
            previous = journal["task"]
            if "final" not in journal or any(previous[key] != task[key] for key in _IDENTITY_FIELDS):
                continue
            baseline = float(journal["baseline"]["score"])
            for step in reversed(journal["steps"]):
                if step["status"] not in {"evaluated", "failed"}:
                    continue
                reflection = step.get("reflection") or {}
                lessons.append({"source_search": path.parent.name, "operator": step["operator"],
                                "hypothesis": step["hypothesis"][:500],
                                "expected_result": (step.get("expected_result") or "")[:500],
                                "status": step["status"],
                                "validation_delta": (float(step["score"]) - baseline
                                                     if step["status"] == "evaluated" else None),
                                "verdict": reflection.get("verdict"),
                                "evidence": (reflection.get("evidence") or "")[:500],
                                "lesson": (reflection.get("lesson") or "")[:500]})
                if len(lessons) >= limit:
                    return lessons
        except (KeyError, ValueError, TypeError, OSError):
            continue
    return lessons
