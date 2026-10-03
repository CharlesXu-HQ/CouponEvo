"""Compare Agent search with and without same-task, dataset-bound lessons."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from .tasks import Task, load_task, verify_task


def _invoke(command: list[str]) -> None:
    result = subprocess.run(command, stdin=subprocess.DEVNULL, text=True,
                            capture_output=True)
    if result.returncode:
        raise RuntimeError((result.stderr or result.stdout)[-3000:])


def _cli(task: Task, command: str, output: Path, search_id: str, image: str,
         device: str) -> list[str]:
    return [sys.executable, "-m", "promolift.cli", command, str(task.manifest),
            "--budget-kind", task.budget.kind, "--budget", str(task.budget.value),
            "--seed", str(task.seed), "--candidate", str(task.candidate),
            "--objective", task.objective, "--search-id", search_id,
            "--output", str(output), "--device", device, "--sandbox-image", image]


def _champion(journal: dict, root: Path) -> Path:
    chosen = (journal["baseline"] if journal["best_id"] == "seed" else
              next(step for step in journal["steps"] if step["id"] == journal["best_id"]))
    return root / chosen["candidate"]


def run_benchmark(tasks: list[Task], *, output: Path, agent_config: Path,
                  sandbox_image: str, device: str = "cuda", bootstrap_reps: int = 2000) -> dict:
    if not tasks:
        raise ValueError("benchmark needs at least one randomized task")
    if len({task.name for task in tasks}) != len(tasks):
        raise ValueError("benchmark task names must be unique")
    if bootstrap_reps < 2:
        raise ValueError("bootstrap_reps must be at least two")
    output = Path(output).resolve()
    agent_config = Path(agent_config).resolve()
    if not agent_config.is_file() or not sandbox_image:
        raise ValueError("agent config and sandbox image are required")
    if output.exists():
        raise ValueError("benchmark output already exists; use a fresh directory")
    plain_dir, memory_dir = output / "plain", output / "experience"
    output.mkdir(parents=True, exist_ok=True)
    results = {}
    for mode, directory in (("plain", plain_dir), ("experience", memory_dir)):
        results[mode] = {}
        for task in tasks:
            verify_task(task)
            started = time.monotonic()
            print(f"{mode}: searching {task.name}", flush=True)
            search = _cli(task, "search", directory, task.name, sandbox_image, device)
            search += ["--agent-config", str(agent_config), "--max-steps", str(task.max_steps)]
            if mode == "experience":
                search += ["--experience-dir", str(plain_dir)]
            _invoke(search)
            journal_path = directory / task.name / "journal.json"
            journal = json.loads(journal_path.read_text())
            finalize = _cli(task, "finalize", directory, task.name, sandbox_image, device)
            finalize += ["--bootstrap-reps", str(bootstrap_reps)]
            if mode == "experience":
                finalize += ["--experience-dir", str(plain_dir)]
            print(f"{mode}: finalizing {task.name}", flush=True)
            _invoke(finalize)
            results[mode][task.name] = {
                "validation_seed": journal["baseline"]["score"],
                "validation_best": (journal["baseline"]["score"] if journal["best_id"] == "seed" else
                                    next(step["score"] for step in journal["steps"]
                                         if step["id"] == journal["best_id"])),
                "best_id": journal["best_id"],
                "evaluated_steps": sum(step["status"] == "evaluated" for step in journal["steps"]),
                "failed_steps": sum(step["status"] == "failed" for step in journal["steps"]),
                "experience_count": len(journal.get("experience", [])),
                "elapsed_seconds": round(time.monotonic() - started, 2),
                "final_report": journal_path.parent / "final" /
                                json.loads(journal_path.read_text())["final"]["report"]["run_id"] / "report.json",
            }
    for task in tasks:
        verify_task(task)
        print(f"paired test of same-task experience: {task.name}", flush=True)
        plain_root = plain_dir / task.name
        memory_root = memory_dir / task.name
        plain = json.loads((plain_root / "journal.json").read_text())
        memory = json.loads((memory_root / "journal.json").read_text())
        compare_output = output / "paired" / task.name
        comparison = [sys.executable, "-m", "promolift.cli", "run", str(task.manifest),
                      "--budget-kind", task.budget.kind, "--budget", str(task.budget.value),
                      "--seed", str(task.seed), "--candidate", str(_champion(memory, memory_root)),
                      "--compare-candidate", str(_champion(plain, plain_root)),
                      "--output", str(compare_output), "--device", device,
                      "--sandbox-image", sandbox_image, "--final", "--bootstrap-reps",
                      str(bootstrap_reps)]
        _invoke(comparison)
        reports = list(compare_output.glob("*/report.json"))
        if len(reports) != 1:
            raise RuntimeError("paired comparison must produce exactly one report")
        report_path = reports[0]
        report = json.loads(report_path.read_text())
        key = "net" if task.objective.startswith("net_") else task.objective
        results["experience"][task.name]["paired_vs_plain"] = report["paired_vs_baseline_bootstrap"][task.objective][key]
        results["experience"][task.name]["paired_report"] = str(report_path)
    summary = {"tasks": [{"name": task.name, "dataset_sha256": task.dataset_sha256,
                          "objective": task.objective, "max_steps": task.max_steps}
                         for task in tasks], "device": device, "sandbox_image": sandbox_image,
               "results": results}
    for mode in results.values():
        for entry in mode.values():
            entry["final_report"] = str(entry["final_report"])
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark dataset-bound Agent experience on frozen RCT tasks")
    parser.add_argument("tasks", nargs="+", type=Path)
    parser.add_argument("--agent-config", type=Path, required=True)
    parser.add_argument("--sandbox-image", required=True)
    parser.add_argument("--output", type=Path, default=Path("runs/benchmark"))
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--bootstrap-reps", type=int, default=2000)
    args = parser.parse_args()
    result = run_benchmark([load_task(path) for path in args.tasks], output=args.output,
                           agent_config=args.agent_config, sandbox_image=args.sandbox_image,
                           device=args.device, bootstrap_reps=args.bootstrap_reps)
    print(args.output.resolve() / "summary.json")


if __name__ == "__main__":
    main()
