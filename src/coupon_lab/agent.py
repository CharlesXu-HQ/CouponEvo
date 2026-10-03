"""Let Codex revise only a temporary candidate file, then copy that file back."""

from __future__ import annotations

import ast
import hashlib
import json
import shutil
import subprocess
import tempfile
import urllib.error
import urllib.request
from pathlib import Path


def revise_candidate(candidate_path: Path, report_path: Path, *, feature_gaps_path: Path | None = None,
                     codex_bin: str = "codex", model: str | None = None) -> str:
    candidate_path, report_path = Path(candidate_path), Path(report_path)
    before = candidate_path.read_bytes()
    with tempfile.TemporaryDirectory(prefix="coupon-lab-agent-") as temp:
        scratch = Path(temp)
        shutil.copy2(candidate_path, scratch / "candidate.py")
        shutil.copy2(report_path, scratch / "report.md")
        prompt = (
            "Improve candidate.py based on report.md for the next offline uplift experiment. "
            "Edit candidate.py and keep the fit_predict signature and return columns. "
            "Use only pandas and numpy. If the evidence suggests missing pre-treatment user features, "
            "write feature_gaps.md with proposed field, source, timing, evidence, leakage risk, and "
            "a future dataset validation plan. Do not change report.md or other files."
        )
        command = [codex_bin, "exec", "--skip-git-repo-check", "-s", "workspace-write",
                   "-C", str(scratch)]
        if model:
            command.extend(["-m", model])
        result = subprocess.run([*command, prompt], stdin=subprocess.DEVNULL,
                                capture_output=True, text=True)
        if result.returncode:
            detail = result.stderr.strip() or result.stdout.strip()
            raise RuntimeError(f"Agent CLI failed: {detail[-3000:]}")
        revised = (scratch / "candidate.py").read_bytes()
        compile(revised, str(candidate_path), "exec")
        if revised == before:
            raise ValueError("Agent did not change candidate.py")
        candidate_path.write_bytes(revised)
        gap_file = scratch / "feature_gaps.md"
        if feature_gaps_path is not None and gap_file.exists():
            Path(feature_gaps_path).write_bytes(gap_file.read_bytes())
    return hashlib.sha256(revised).hexdigest()


def revise_candidate_deepseek(candidate_path: Path, report_path: Path, *, api_key: str,
                              feature_gaps_path: Path | None = None,
                              model: str = "deepseek-flash") -> str:
    """Ask DeepSeek for one complete candidate revision using the frozen report."""
    candidate_path, report_path = Path(candidate_path), Path(report_path)
    before = candidate_path.read_text()
    prompt = (
        "Return one JSON object with candidate_py (complete Python source) and "
        "feature_gaps_md (empty string if none). Make one small, testable change to "
        "the uplift algorithm or pre-treatment feature engineering based on the report. "
        "Preserve fit_predict and its required return columns. Use only pandas and numpy. "
        "Create engineered features before building the feature matrix so predictions "
        "actually change. The target contains only pre-treatment features; never use "
        "target treatment or outcomes. Suggest a missing field only with a concrete "
        "hypothesis, source, pre-treatment timing, leakage risk and validation plan."
    )
    user_message = f"Candidate:\n{before}\n\nReport:\n{report_path.read_text()}"
    for attempt in range(2):
        payload = json.dumps({
            "model": model, "thinking": {"type": "enabled"},
            "reasoning_effort": "high", "response_format": {"type": "json_object"},
            "max_tokens": 16384,
            "messages": [{"role": "system", "content": prompt},
                         {"role": "user", "content": user_message}],
        }).encode()
        request = urllib.request.Request(
            "https://api.deepseek.com/chat/completions", data=payload,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                result = json.load(response)
        except urllib.error.HTTPError as error:
            raise RuntimeError(f"DeepSeek API returned HTTP {error.code}") from None
        choice = result["choices"][0]
        if choice["finish_reason"] != "stop":
            raise RuntimeError(f"DeepSeek response incomplete: {choice['finish_reason']}")
        try:
            answer = json.loads(choice["message"]["content"])
            revised = answer["candidate_py"].strip()
            if revised.startswith("```"):
                revised = revised.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            revised += "\n"
            tree = ast.parse(revised, filename=str(candidate_path))
            allowed = {"__future__", "numpy", "pandas"}
            for node in ast.walk(tree):
                if isinstance(node, ast.Import) and any(alias.name.split(".")[0] not in allowed for alias in node.names):
                    raise ValueError("Agent candidate imports an unsupported module")
                if isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] not in allowed:
                    raise ValueError("Agent candidate imports an unsupported module")
            if not any(isinstance(node, ast.FunctionDef) and node.name == "fit_predict" for node in tree.body):
                raise ValueError("Agent candidate is missing fit_predict")
            compile(tree, str(candidate_path), "exec")
            if revised == before:
                raise ValueError("Agent did not change candidate.py")
            break
        except (KeyError, ValueError, SyntaxError) as error:
            if attempt:
                raise ValueError(f"Agent returned invalid candidate: {error}") from None
            user_message += f"\n\nYour previous response was invalid ({type(error).__name__}: {error}). Return corrected JSON with valid Python source."
    candidate_path.write_text(revised)
    notes = answer.get("feature_gaps_md") or ""
    if feature_gaps_path is not None and notes.strip():
        Path(feature_gaps_path).write_text(notes)
    return hashlib.sha256(revised.encode()).hexdigest()
