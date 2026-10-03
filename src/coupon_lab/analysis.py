"""Interpret frozen experiment reports with a bounded high/max review."""

from __future__ import annotations

import json
from pathlib import Path

from .provider import ApiProvider, request_json


FLAGS = ("uplift_anomaly", "feature_leakage", "cost_tradeoff_unclear")


def _ask(provider: ApiProvider, effort: str, messages: list[dict]) -> dict:
    for attempt in range(2):
        try:
            answer = request_json(provider, effort, messages,
                                  max_tokens=10000 if effort == "high" else 16000)
            if not isinstance(answer.get("summary"), str) or not isinstance(answer.get("recommendation"), str):
                raise ValueError("Agent analysis needs a summary and recommendation")
            for name in FLAGS:
                if not isinstance(answer.get(name), dict) or not isinstance(answer[name].get("flag"), bool):
                    raise ValueError(f"Agent analysis needs a boolean {name} flag")
            return answer
        except ValueError:
            if attempt:
                raise
            messages = [*messages, {"role": "user", "content":
                        "The previous response was invalid. Return exactly one complete JSON object with summary, recommendation, and all three flag/evidence objects."}]


def _review_reasons(prior: dict, report: dict, high: dict) -> list[str]:
    policies = report["policies"]
    random = policies["random"]
    primary = next((name for name in ("active", "visit", "click", "conversion") if name in policies), None)
    anomaly = high["uplift_anomaly"]["flag"] or prior["policies"]["random"] != random
    if primary:
        outcome = policies[primary]["effects"][primary]["mean"]
        anomaly |= outcome < 0 or outcome < random["effects"][primary]["mean"]
    net_policy = next((policies[name] for name in ("net_margin", "net_revenue") if name in policies), None)
    if net_policy and "net" in random:
        anomaly |= net_policy["net"]["mean"] < random["net"]["mean"]

    cost_unclear = high["cost_tradeoff_unclear"]["flag"]
    if any("cost" in entry or "assumed_cost" in entry for entry in policies.values()):
        cost_unclear |= any("net" in entry and entry["net"]["lower"] <= 0 <= entry["net"]["upper"]
                            for name, entry in policies.items() if name != "random")
        if primary and net_policy and "net" in policies[primary]:
            cost_unclear |= (policies[primary]["effects"][primary]["mean"] >
                             net_policy["effects"][primary]["mean"] and
                             net_policy["net"]["mean"] > policies[primary]["net"]["mean"])
    return (["uplift_anomaly"] if anomaly else []) + \
           (["feature_leakage"] if high["feature_leakage"]["flag"] else []) + \
           (["cost_tradeoff_unclear"] if cost_unclear else [])


def analyze_reports_deepseek(prior: dict, revised: dict, *, candidate_path: Path,
                             output_dir: Path, api_key: str | None = None,
                             model: str = "deepseek-flash",
                             provider: ApiProvider | None = None) -> dict:
    """Run high analysis, then max review only when evidence warrants it."""
    provider = provider or ApiProvider("https://api.deepseek.com", model, api_key or "")
    context = json.dumps({"before": prior, "after": revised}, ensure_ascii=False)
    candidate = Path(candidate_path).read_text()
    instruction = (
        "Analyze an offline randomized coupon uplift experiment. Return JSON with "
        "summary, recommendation, and uplift_anomaly, feature_leakage, "
        "cost_tradeoff_unclear objects, each containing flag (boolean) and evidence "
        "(string). Use Chinese. Distinguish point estimates from uncertainty and do "
        "not claim a model improvement from overlapping confidence intervals. "
        "Check treatment timing for leakage and distinguish actual from assumed cost. "
        "Flag suspicion only with concrete evidence; state uncertainty explicitly."
    )
    messages = [{"role": "system", "content": instruction},
                {"role": "user", "content": f"Reports:\n{context}\n\nCandidate code:\n{candidate}"}]
    high = _ask(provider, provider.iteration_effort, messages)
    reasons = _review_reasons(prior, revised, high)
    maximum = None
    if reasons:
        maximum = _ask(provider, provider.review_effort, [
            {"role": "system", "content": instruction + " Independently audit the first analysis; confirm or refute each review trigger with evidence."},
            {"role": "user", "content": f"Reports:\n{context}\n\nCandidate code:\n{candidate}\n\nHigh analysis:\n{json.dumps(high, ensure_ascii=False)}\n\nReview triggers: {', '.join(reasons)}"},
        ])
    result = {"model": provider.model, "provider_url": provider.url,
              "initial_effort": provider.iteration_effort,
              "review_effort": provider.review_effort if maximum else None,
              "run_id": revised["run_id"], "high": high,
              "review_reasons": reasons, "max": maximum}
    output_dir = Path(output_dir)
    lines = ["# Agent 报告分析", "", f"模型：{provider.model}", "", f"## {provider.iteration_effort} 初评", "",
             high["summary"], "", f"建议：{high['recommendation']}", ""]
    for name in FLAGS:
        flag = high[name]
        lines.append(f"- {name}：{'需复核' if flag['flag'] else '未发现'}；{flag.get('evidence', '')}")
    lines.extend(["", f"## {provider.review_effort} 复核", ""])
    if maximum:
        lines.extend([f"触发项：{', '.join(reasons)}", "", maximum["summary"], "",
                      f"建议：{maximum['recommendation']}", ""])
    else:
        lines.extend(["未触发。", ""])
    (output_dir / "analysis.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    (output_dir / "analysis.md").write_text("\n".join(lines))
    return result
