"""Small, validation-only lessons from completed searches on other datasets."""

from __future__ import annotations

import json
from pathlib import Path


def load_experience(directory: Path, dataset_sha: str, limit: int = 5) -> list[dict]:
    lessons = []
    for path in sorted(Path(directory).glob("*/journal.json")):
        try:
            journal = json.loads(path.read_text())
            if "final" not in journal or journal["task"]["dataset"] == dataset_sha:
                continue
            baseline = float(journal["baseline"]["score"])
            for step in journal["steps"]:
                if step["status"] != "evaluated":
                    continue
                lessons.append({"task_objective": journal["task"]["objective"],
                                "operator": step["operator"],
                                "hypothesis": step["hypothesis"][:500],
                                "validation_delta": float(step["score"]) - baseline})
        except (KeyError, ValueError, TypeError, OSError):
            continue
    lessons.sort(key=lambda item: item["validation_delta"], reverse=True)
    return lessons[:limit]
