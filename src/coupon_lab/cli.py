"""Run a repeatable uplift experiment and save an evidence report."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .agent import revise_candidate, revise_candidate_deepseek
from .analysis import analyze_reports_deepseek
from .data import load_dataset, split_dataset
from .evaluate import Budget, estimate_cost, evaluate_policy, select_policy


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _candidate_function(path: Path):
    spec = importlib.util.spec_from_file_location("uplift_candidate", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot import candidate: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.fit_predict


def _random_policy(n: int, expected_cost: np.ndarray | None, budget: Budget, seed: int) -> np.ndarray:
    order = np.random.default_rng(seed).permutation(n)
    policy = np.zeros(n, dtype=bool)
    if budget.kind == "count":
        policy[order[:int(np.floor(n * budget.value))]] = True
    else:
        remaining = n * budget.value
        for i in order:
            if expected_cost[i] <= remaining + 1e-12:
                policy[i] = True
                remaining -= expected_cost[i]
    return policy


def _split_sha(parts) -> str:
    assignments = {name: sorted(map(str, frame["__unit_id"].unique()))
                   for name, frame in (("train", parts.train), ("validation", parts.validation), ("test", parts.test))}
    return _sha(json.dumps(assignments, sort_keys=True).encode())


def _render(report: dict) -> str:
    lines = ["# Coupon Uplift 实验报告", "", f"运行 ID：`{report['run_id']}`", "",
             f"数据 SHA-256：`{report['dataset_sha256']}`", "",
             f"候选代码 SHA-256：`{report['candidate_sha256']}`", "",
             f"预测 SHA-256：`{report['prediction_sha256']}`", "",
             f"干预前特征：{', '.join(report['features'])}", "",
             "结果字段：" + "；".join(f"{name}={column}" for name, column in report["outcome_columns"].items()), "",
             f"切分：{report['holdout']}；预算：{report['budget']['kind']} = {report['budget']['value']}", "",
             "## 策略结果", ""]
    if report.get("agent"):
        agent = report["agent"]
        lines[2:2] = [f"Agent：{agent['provider']} / {agent['model']}；前一轮：`{agent['previous_run_id']}`", ""]
    for name, result in report["policies"].items():
        lines.append(f"### {name}")
        lines.append(f"选中 {result['selected_count']} / {report['holdout_size']} 个单位。")
        for outcome, estimate in result["effects"].items():
            lines.append(f"- {outcome} 增量/单位：{estimate['mean']:.6g}（95% 区间 {estimate['lower']:.6g} 至 {estimate['upper']:.6g}）")
        if "cost" in result:
            lines.append(f"- 成本/单位：{result['cost']['mean']:.6g}（95% 区间 {result['cost']['lower']:.6g} 至 {result['cost']['upper']:.6g}）")
        if "assumed_cost" in result:
            lines.append(f"- 假设成本/单位：{result['assumed_cost']:.6g}")
        if "net" in result:
            lines.append(f"- {result['net_label']}/单位：{result['net']['mean']:.6g}（95% 区间 {result['net']['lower']:.6g} 至 {result['net']['upper']:.6g}）")
        if "delta_vs_random" in result:
            lines.append("- 相对随机发券的点估计差值/单位：" +
                         "；".join(f"{key} {value:.6g}" for key, value in result["delta_vs_random"].items()))
        lines.append("")
    lines.extend(["## 数据与特征缺口", "", report.get("feature_gaps") or "当前轮未提出新字段。Agent 可根据指标、可用特征和干预前时点在下一轮补充建议。", "",
                  "## 口径与限制", "", *[f"- {note}" for note in report["notes"]], ""])
    return "\n".join(lines)


def run_experiment(manifest_path: Path, budget: Budget, *, seed: int, output: Path,
                   candidate_path: Path | None = None, final: bool = False,
                   feature_gaps: Path | None = None, agent_info: dict | None = None) -> dict:
    manifest_path, output = Path(manifest_path), Path(output)
    candidate_path = Path(candidate_path or Path(__file__).with_name("candidate.py"))
    manifest_bytes, candidate_bytes = manifest_path.read_bytes(), candidate_path.read_bytes()
    gap_bytes = Path(feature_gaps).read_bytes() if feature_gaps is not None else None
    data = load_dataset(manifest_path)
    parts = split_dataset(data, seed)
    target = parts.test if final else parts.validation
    outcome_cols = {name: column for name, column in data.outcomes.items() if name != "coupon_cost"}
    cost_col = data.outcomes.get("coupon_cost")
    predictions = _candidate_function(candidate_path)(parts.train, target[data.features].copy(), features=data.features,
                                                        treatment="__treatment", outcomes=outcome_cols, cost=cost_col)
    expected_columns = {f"{name}_uplift" for name in outcome_cols}
    if cost_col:
        expected_columns.add("expected_cost")
    if not isinstance(predictions, pd.DataFrame) or not expected_columns <= set(predictions) or len(predictions) != len(target):
        raise ValueError("candidate returned invalid prediction columns or row count")
    prediction_sha = _sha(predictions[sorted(expected_columns)].to_numpy(dtype="<f8").tobytes())
    if cost_col:
        expected_cost = predictions["expected_cost"].to_numpy(dtype=float)
    elif data.manifest.get("fixed_send_cost") is not None:
        expected_cost = np.full(len(target), data.manifest["fixed_send_cost"], dtype=float)
    else:
        expected_cost = None
    if budget.kind == "cost" and expected_cost is None:
        raise ValueError("cost budget needs observed coupon_cost or fixed_send_cost")

    primary = next((name for name in ("active", "visit", "click", "conversion") if name in outcome_cols), None)
    if primary is None:
        primary = next(iter(outcome_cols))
    score_columns = {primary: predictions[f"{primary}_uplift"].to_numpy(dtype=float)}
    money = "gross_margin" if "gross_margin" in outcome_cols else "revenue" if "revenue" in outcome_cols else None
    if money and money != primary:
        label = ("net_margin" if money == "gross_margin" else "net_revenue") if expected_cost is not None else money
        score = predictions[f"{money}_uplift"].to_numpy(dtype=float)
        if expected_cost is not None and not (money == "gross_margin" and data.manifest.get("margin_includes_coupon_cost")):
            score = score - expected_cost
        score_columns[label] = score
    policies = {name: select_policy(score, expected_cost, budget) for name, score in score_columns.items()}
    policies["random"] = _random_policy(len(target), expected_cost, budget, seed)
    results = {}
    for name, policy in policies.items():
        entry = {"selected_count": int(policy.sum()),
                 "effects": {outcome: evaluate_policy(target, policy, col, data.propensity).to_dict()
                             for outcome, col in outcome_cols.items()}}
        if expected_cost is not None:
            entry["predicted_cost_per_unit"] = float((policy * expected_cost).mean())
        if cost_col:
            entry["cost"] = estimate_cost(target, policy, cost_col, data.propensity).to_dict()
        elif expected_cost is not None:
            assumed = float((policy * expected_cost).mean())
            entry["assumed_cost"] = assumed
        if money and cost_col:
            net_frame = target.copy()
            net_frame["__net"] = net_frame[outcome_cols[money]]
            if money == "revenue" or not data.manifest.get("margin_includes_coupon_cost"):
                net_frame["__net"] -= net_frame[cost_col]
            entry["net"] = evaluate_policy(net_frame, policy, "__net", data.propensity).to_dict()
            entry["net_label"] = "增量净收益" if money == "gross_margin" else "增量收入扣券成本"
        elif money and expected_cost is not None:
            assumed = (0.0 if money == "gross_margin" and data.manifest.get("margin_includes_coupon_cost")
                       else entry["assumed_cost"])
            raw_effect = entry["effects"][money]
            entry["net"] = {key: value - assumed if key != "se" else value
                            for key, value in raw_effect.items()}
            entry["net_label"] = "假设增量净收益" if money == "gross_margin" else "假设增量净收入"
        results[name] = entry
    random_result = results["random"]
    for name, entry in results.items():
        if name == "random":
            continue
        keys = [primary]
        if "net" in entry:
            keys.append("net")
        elif money and money != primary:
            keys.append(money)
        entry["delta_vs_random"] = {
            key: (entry["net"]["mean"] - random_result["net"]["mean"] if key == "net" else
                  entry["effects"][key]["mean"] - random_result["effects"][key]["mean"])
            for key in keys}
    unavailable = []
    if "active" not in outcome_cols:
        unavailable.append("active")
    if "gross_margin" not in outcome_cols or not cost_col:
        unavailable.append("net_margin")
    notes = []
    if not data.has_unit_id:
        notes.append("无用户 ID；按行切分，无法排除同一用户跨切分。")
    if data.manifest["treatment"]["probability"] == "empirical":
        notes.append("分组概率由样本比例估计，依赖简单随机分组与不按组差异抽样的声明。")
    if "active" not in outcome_cols:
        notes.append("当前数据没有符合 App 活跃定义的标签；访问、点击或转化不得称为促活。")
    if money == "revenue" and cost_col:
        notes.append("收入扣券成本不等于利润；缺少商品或服务毛利。")
    if expected_cost is None:
        notes.append("无成本字段；预算按选中人数比例计算。")
    elif not cost_col:
        notes.append("成本来自固定每次发券费用假设，不是实际核销成本。")
    if budget.kind == "cost":
        notes.append("发券选择使用预测成本；报告中的观测成本及区间可能超过预算。")
    identity = {"dataset": data.source_sha256, "manifest": _sha(manifest_bytes),
                "candidate": _sha(candidate_bytes), "split": _split_sha(parts),
                "budget": {"kind": budget.kind, "value": budget.value}, "seed": seed, "final": final,
                "feature_gaps": _sha(gap_bytes) if gap_bytes is not None else None}
    if agent_info is not None:
        identity["agent"] = agent_info
    run_id = _sha(json.dumps(identity, sort_keys=True).encode())[:16]
    report = {"run_id": run_id, "dataset_sha256": data.source_sha256,
              "manifest_sha256": identity["manifest"], "candidate_sha256": identity["candidate"],
              "prediction_sha256": prediction_sha,
              "split_sha256": identity["split"], "budget": identity["budget"], "seed": seed,
              "holdout": "test" if final else "validation", "holdout_size": len(target),
              "features": data.features, "outcome_columns": data.outcomes,
              "available_outcomes": list(outcome_cols), "unavailable_objectives": unavailable,
              "policies": results, "notes": notes,
              "feature_gaps": gap_bytes.decode("utf-8") if gap_bytes is not None else None}
    if agent_info is not None:
        report["agent"] = agent_info
    if (_sha(data.dataset_path.read_bytes()) != data.source_sha256 or
            _sha(manifest_path.read_bytes()) != identity["manifest"] or
            _sha(candidate_path.read_bytes()) != identity["candidate"] or
            (gap_bytes is not None and _sha(Path(feature_gaps).read_bytes()) != identity["feature_gaps"])):
        raise RuntimeError("dataset, manifest, candidate, or feature notes changed during the run")
    directory = output / run_id
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "candidate.py").write_bytes(candidate_bytes)
    if gap_bytes is not None:
        (directory / "feature_gaps.md").write_bytes(gap_bytes)
    (directory / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    (directory / "report.md").write_text(_render(report))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run offline coupon uplift experiments")
    parser.add_argument("command", choices=["run", "agent"])
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--budget-kind", choices=["count", "cost"], required=True)
    parser.add_argument("--budget", type=float, required=True, help="fraction of units, or cost per eligible unit")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("runs"))
    parser.add_argument("--candidate", type=Path, default=Path(__file__).with_name("candidate.py"))
    parser.add_argument("--final", action="store_true", help="evaluate the final test split")
    parser.add_argument("--feature-gaps", type=Path, help="existing human or Agent feature suggestions")
    parser.add_argument("--agent-provider", choices=["codex", "deepseek"], default="codex")
    parser.add_argument("--agent-model", help="Agent model override; DeepSeek defaults to deepseek-flash")
    args = parser.parse_args()
    if args.command == "agent" and args.final:
        parser.error("agent revisions must use validation; run --final separately after selection")
    api_key = os.environ.pop("DEEPSEEK_API_KEY", None) if args.command == "agent" and args.agent_provider == "deepseek" else None
    if args.command == "agent" and args.agent_provider == "deepseek" and not api_key:
        parser.error("DEEPSEEK_API_KEY is required for the DeepSeek agent")
    budget = Budget(args.budget_kind, args.budget)
    original_candidate = args.candidate.read_bytes() if args.command == "agent" else None
    try:
        if args.command == "agent":
            prior = run_experiment(args.manifest, budget, seed=args.seed, output=args.output,
                                   candidate_path=args.candidate)
            args.feature_gaps = args.output / "agent-feature-gaps.md"
            args.feature_gaps.unlink(missing_ok=True)
            prior_report = args.output / prior["run_id"] / "report.md"
            model = args.agent_model or ("deepseek-flash" if args.agent_provider == "deepseek" else "default")
            if args.agent_provider == "deepseek":
                revise_candidate_deepseek(args.candidate, prior_report, api_key=api_key,
                                          feature_gaps_path=args.feature_gaps, model=model)
            else:
                revise_candidate(args.candidate, prior_report,
                                 feature_gaps_path=args.feature_gaps, model=args.agent_model)
            if not args.feature_gaps.exists():
                args.feature_gaps = None
            agent_info = {"provider": args.agent_provider, "model": model,
                          "previous_run_id": prior["run_id"]}
        else:
            agent_info = None
        report = run_experiment(args.manifest, budget, seed=args.seed, output=args.output,
                                candidate_path=args.candidate, final=args.final,
                                feature_gaps=args.feature_gaps, agent_info=agent_info)
        if args.command == "agent" and prior["prediction_sha256"] == report["prediction_sha256"]:
            raise ValueError("Agent revision left predictions unchanged")
        if args.command == "agent" and args.agent_provider == "deepseek":
            analyze_reports_deepseek(prior, report, candidate_path=args.candidate,
                                     output_dir=args.output / report["run_id"],
                                     api_key=api_key, model=model)
            api_key = None
    except Exception:
        if original_candidate is not None:
            args.candidate.write_bytes(original_candidate)
        raise
    print(args.output / report["run_id"] / "report.md")


if __name__ == "__main__":
    main()
