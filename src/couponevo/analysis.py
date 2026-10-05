"""Interpret frozen experiment reports with a bounded high/max review."""

from __future__ import annotations

import json
from pathlib import Path

from .provider import ApiProvider, IncompleteResponseError, request_json


FLAGS = ("uplift_anomaly", "feature_leakage", "cost_tradeoff_unclear")


def _ask(provider: ApiProvider, effort: str, messages: list[dict], *,
         implementation_required: bool = False) -> dict:
    max_tokens = 10000 if effort == "high" else 16000
    for attempt in range(2):
        try:
            answer = request_json(provider, effort, messages,
                                  max_tokens=max_tokens, timeout=300 if max_tokens > 16000 else 180)
            if not isinstance(answer.get("summary"), str) or not isinstance(answer.get("recommendation"), str):
                raise ValueError("Agent analysis needs a summary and recommendation")
            for name in FLAGS:
                if not isinstance(answer.get(name), dict) or not isinstance(answer[name].get("flag"), bool):
                    raise ValueError(f"Agent analysis needs a boolean {name} flag")
            leakage = answer["feature_leakage"]
            if (not isinstance(leakage.get("confirmed"), bool) or
                    (leakage["confirmed"] and
                     (not leakage["flag"] or not isinstance(leakage.get("evidence"), str) or
                      not leakage["evidence"].strip()))):
                raise ValueError("feature_leakage needs confirmed=true only with a flagged, evidenced leak")
            if implementation_required:
                check = answer.get("implementation_check")
                if (not isinstance(check, dict) or
                        check.get("status") not in {"verified", "contradicted", "unverified"} or
                        not isinstance(check.get("evidence"), str) or not check["evidence"].strip() or
                        not isinstance(check.get("changed_factors"), list) or any(
                            not isinstance(factor, str) or not factor.strip()
                            for factor in check["changed_factors"])):
                    raise ValueError("implementation_check needs status, evidence and changed_factors")
            return answer
        except IncompleteResponseError as error:
            if error.reason != "length" or attempt:
                raise
            max_tokens = min(max_tokens * 2, 32768)
        except ValueError as error:
            if attempt:
                raise
            messages = [*messages, {"role": "user", "content":
                        f"The previous response was invalid: {error}. Return one complete JSON object following the analysis contract."}]


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
    implementation_issue = high.get("implementation_check", {}).get("status") == "contradicted"
    implementation_issue |= report.get("runtime_diagnostics", {}).get(
        "prediction_contract", {}).get("status") == "inconsistent"
    return (["implementation_check"] if implementation_issue else []) + \
           (["uplift_anomaly"] if anomaly else []) + \
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
    implementation_required = "experiment_context" in revised
    instruction = (
        "Analyze an offline randomized coupon uplift experiment. Return JSON with "
        "summary, recommendation, and uplift_anomaly, feature_leakage, "
        "cost_tradeoff_unclear objects, each containing flag (boolean) and evidence "
        "(string); feature_leakage must also contain confirmed (boolean). Use Chinese. "
        "feature_leakage.flag marks suspicion or confirmation and triggers independent review. "
        "Set confirmed=true only with concrete evidence that a feature used by the candidate "
        "contains treatment/outcome information or was measured after assignment. "
        "feature_timing='declared_only' means timing is unverified, not proven leakage: "
        "record the concern with flag=true, confirmed=false and explain the missing evidence. "
        "Assess new versus old using paired_vs_baseline intervals; "
        "do not claim a final improvement from validation intervals or separate interval overlap. "
        "Check treatment timing for leakage and distinguish actual from assumed cost. "
        "Policy effects and costs are averaged over all eligible holdout users; "
        "fixed_send_cost is the assumed cost per selected treated user. "
        "Flag suspicion only with concrete evidence; state uncertainty explicitly."
        " A paired policy-effect interval crossing zero is not a diagnosis of CATE model variance, "
        "nor proof of equivalence. Different policies can have identical binary IPW means: at p=0.5 "
        "the mean is twice the selected treated-minus-control positive count divided by N. "
        "Different hashes and a nonzero paired standard error can coexist with that tie; "
        "do not call it an evaluator/version anomaly without an additional contradiction."
    )
    if implementation_required:
        instruction += (
            " Also return implementation_check {status: verified|contradicted|unverified, "
            "evidence: string, changed_factors: [string]}. Independently compare the experiment "
            "hypothesis, parent code, candidate_diff and runtime_diagnostics with the candidate. "
            "Check the premise as well as the implementation: pandas get_dummies with columns=None "
            "does not one-hot numeric DataFrame columns; use observed shapes and dtypes. "
            "Distinguish preprocessing width (possibly excluding a bias column) from model width. "
            "Check whether each claimed change really happened. Distinguish logit contrast from "
            "sigmoid(logit1)-sigmoid(logit0), predicted CATE and a ranking score; inspect every "
            "learned component and the exact return expression. A score gain cannot validate an "
            "incorrect premise or output scale. Contradicted means concrete code/runtime evidence "
            "refutes a claimed mechanism or semantics, and blocks promotion pending repair. "
            "Unverified means evidence is missing, not that the method is invalid. Verified means "
            "implementation agrees with the stated change on the available evidence, not that its "
            "causal explanation or final improvement is proven. List actual joint changes in "
            "encoding, estimator, network, loss, training and policy; do not infer isolation from "
            "the proposal alone. Prioritize correcting a detected implementation error and a "
            "small diagnostic/ablation using current data before speculative complexity or a "
            "terminal request for already-known metadata concerns."
        )
    messages = [{"role": "system", "content": instruction},
                {"role": "user", "content": f"Reports:\n{context}\n\nCandidate code:\n{candidate}"}]
    high = _ask(provider, provider.iteration_effort, messages,
                implementation_required=implementation_required)
    reasons = _review_reasons(prior, revised, high)
    maximum = None
    if reasons:
        maximum = _ask(provider, provider.review_effort, [
            {"role": "system", "content": instruction + " Independently audit the first analysis; confirm or refute each review trigger with evidence."},
            {"role": "user", "content": f"Reports:\n{context}\n\nCandidate code:\n{candidate}\n\nHigh analysis:\n{json.dumps(high, ensure_ascii=False)}\n\nReview triggers: {', '.join(reasons)}"},
        ], implementation_required=implementation_required)
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
        status = "未发现"
        if flag["flag"]:
            status = ("已证实" if flag["confirmed"] else "风险待核验") if name == "feature_leakage" else "需复核"
        lines.append(f"- {name}：{status}；{flag.get('evidence', '')}")
    if implementation_required:
        check = (maximum or high)["implementation_check"]
        lines.extend(["", f"实现核验：{check['status']}；{check['evidence']}",
                      "实际改动项：" + ", ".join(check["changed_factors"])])
    lines.extend(["", f"## {provider.review_effort} 复核", ""])
    if maximum:
        lines.extend([f"触发项：{', '.join(reasons)}", "", maximum["summary"], "",
                      f"建议：{maximum['recommendation']}", ""])
        leakage = maximum["feature_leakage"]
        lines.append("- feature_leakage：" +
                     ("已证实" if leakage["confirmed"] else
                      "风险待核验" if leakage["flag"] else "未发现") +
                     f"；{leakage.get('evidence', '')}")
    else:
        lines.extend(["未触发。", ""])
    (output_dir / "analysis.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    (output_dir / "analysis.md").write_text("\n".join(lines))
    return result
