"""Validation-only lessons from completed searches on the same frozen task."""

from __future__ import annotations

import json
from pathlib import Path


_IDENTITY_FIELDS = ("dataset", "manifest", "objective", "budget", "seed", "framework",
                    "strict_data")


def load_experience(directory: Path, task: dict, limit: int = 5) -> list[dict]:
    lessons = []
    for path in sorted(Path(directory).glob("*/journal.json")):
        try:
            journal = json.loads(path.read_text())
            previous = journal["task"]
            if "final" not in journal or any(previous[key] != task[key] for key in _IDENTITY_FIELDS):
                continue
            baseline = float(journal["baseline"]["score"])
            for step in journal["steps"]:
                if step["status"] != "evaluated":
                    continue
                lessons.append({"source_search": path.parent.name, "operator": step["operator"],
                                "hypothesis": step["hypothesis"][:500],
                                "validation_delta": float(step["score"]) - baseline})
        except (KeyError, ValueError, TypeError, OSError):
            continue
    lessons.sort(key=lambda item: item["validation_delta"], reverse=True)
    return lessons[:limit]
