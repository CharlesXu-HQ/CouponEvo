"""Run a repeatable uplift experiment and save an evidence report."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import random
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
import pandas as pd
import torch

from .agent import propose_search_candidate, revise_candidate, revise_candidate_deepseek
from .analysis import analyze_reports_deepseek
from .data import load_dataset, split_dataset
from .evaluate import Budget, bootstrap_policy_difference, compare_policies, estimate_cost, evaluate_policy, ranking_diagnostic, select_policy
from .provider import ApiProvider
from .sandbox import DockerSandbox
from .search import finalize_search, run_search


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _candidate_module(path: Path):
    spec = importlib.util.spec_from_file_location("uplift_candidate", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot import candidate: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _candidate_function(path: Path):
    return _candidate_module(path).fit_predict


def _predict(path: Path, train: pd.DataFrame, target: pd.DataFrame,
             args: dict, seed: int, device: str) -> pd.DataFrame:
    devices = [torch.cuda.current_device()] if device == "cuda" and torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        return _candidate_function(path)(train.copy(deep=True), target.copy(deep=True), **args)


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


def _candidate_policies(predictions: pd.DataFrame, data, budget: Budget,
                        primary: str, money: str | None, candidate_path: Path,
                        seed: int, sandbox: DockerSandbox | None = None,
                        device: str = "cpu") -> tuple[dict[str, np.ndarray], np.ndarray | None]:
    cost_col = data.outcomes.get("coupon_cost")
    if cost_col:
        expected_cost = predictions["expected_cost"].to_numpy(dtype=float)
    elif data.manifest.get("fixed_send_cost") is not None:
        expected_cost = np.full(len(predictions), data.manifest["fixed_send_cost"], dtype=float)
    else:
        expected_cost = None
    if budget.kind == "cost" and expected_cost is None:
        raise ValueError("cost budget needs observed coupon_cost or fixed_send_cost")
    if expected_cost is not None and (not np.isfinite(expected_cost).all() or (expected_cost < 0).any()):
        raise ValueError("candidate expected_cost must be finite and nonnegative")
    scores = {primary: predictions[f"{primary}_uplift"].to_numpy(dtype=float)}
    if money:
        label = ("net_margin" if money == "gross_margin" else "net_revenue") if expected_cost is not None else money
        score = predictions[f"{money}_uplift"].to_numpy(dtype=float)
        if expected_cost is not None and not (money == "gross_margin" and data.manifest.get("margin_includes_coupon_cost")):
            score = score - expected_cost
        scores[label] = score
    chooser = getattr(_candidate_module(candidate_path), "choose_policy", None) if sandbox is None else None
    policies = {}
    for name, score in scores.items():
        if sandbox is not None:
            raw = sandbox.choose(candidate_path, score, expected_cost, budget.kind, budget.value,
                                 seed, device)
            if raw is None:
                raw = select_policy(score, expected_cost, budget)
        elif chooser:
            numpy_state, python_state = np.random.get_state(), random.getstate()
            devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
            try:
                np.random.seed(seed)
                random.seed(seed)
                with torch.random.fork_rng(devices=devices):
                    torch.manual_seed(seed)
                    raw = chooser(score.copy(), None if expected_cost is None else expected_cost.copy(),
                                  budget.kind, budget.value)
            finally:
                np.random.set_state(numpy_state)
                random.setstate(python_state)
        else:
            raw = select_policy(score, expected_cost, budget)
        policy = np.asarray(raw)
        if policy.dtype != bool or policy.shape != (len(score),):
            raise ValueError("candidate policy must be a boolean vector for every holdout unit")
        if budget.kind == "count" and policy.sum() > int(np.floor(len(policy) * budget.value)):
            raise ValueError("candidate policy exceeds count budget")
        if budget.kind == "cost" and float((policy * expected_cost).sum()) > len(policy) * budget.value + 1e-9:
            raise ValueError("candidate policy exceeds predicted cost budget")
        policies[name] = policy
    return policies, expected_cost


def _policy_shas(policies: dict[str, np.ndarray]) -> dict[str, str]:
    return {name: _sha(np.asarray(policy, dtype=np.uint8).tobytes()) for name, policy in policies.items()}


def _paired_metrics(target: pd.DataFrame, new: np.ndarray, old: np.ndarray, data,
                    outcome_cols: dict[str, str], primary: str, money: str | None,
                    expected_cost: np.ndarray | None, estimate=compare_policies) -> dict:
    metrics = {primary: estimate(target, new, old, outcome_cols[primary], data.propensity).to_dict()}
    cost_col = data.outcomes.get("coupon_cost")
    if money and (money != primary or expected_cost is not None):
        if cost_col:
            net_frame = target.copy()
            net_frame["__net"] = net_frame[outcome_cols[money]]
            if money == "revenue" or not data.manifest.get("margin_includes_coupon_cost"):
                net_frame["__net"] -= net_frame[cost_col]
            metrics["net"] = estimate(net_frame, new, old, "__net", data.propensity).to_dict()
        elif expected_cost is not None:
            effect = estimate(target, new, old, outcome_cols[money], data.propensity).to_dict()
            if money == "revenue" or not data.manifest.get("margin_includes_coupon_cost"):
                assumed = data.manifest["fixed_send_cost"] * float((new.astype(int) - old.astype(int)).mean())
                for key in ("mean", "lower", "upper"):
                    effect[key] -= assumed
            metrics["net"] = effect
        else:
            metrics[money] = estimate(target, new, old, outcome_cols[money], data.propensity).to_dict()
    return metrics


def _render(report: dict) -> str:
    lines = ["# Coupon Uplift 实验报告", "", f"运行 ID：`{report['run_id']}`", "",
             f"数据 SHA-256：`{report['dataset_sha256']}`", "",
             f"候选代码 SHA-256：`{report['candidate_sha256']}`", "",
             f"预测 SHA-256：`{report['prediction_sha256']}`", "",
             f"干预前特征：{', '.join(report['features'])}", "",
             "结果字段：" + "；".join(f"{name}={column}" for name, column in report["outcome_columns"].items()), "",
             f"切分：{report['holdout']}；预算：{report['budget']['kind']} = {report['budget']['value']}", "",
             f"模型设备：{report['model_device']}", "",
             "## 策略结果", ""]
    if report.get("agent"):
        agent = report["agent"]
        detail = f"Agent：{agent['provider']} / {agent['model']}；前一轮：`{agent['previous_run_id']}`"
        if "iteration_effort" in agent:
            detail += f"；迭代：{agent['iteration_effort']}；复核：{agent['review_effort']}"
        lines[2:2] = [detail, ""]
    for name, result in report["policies"].items():
        lines.append(f"### {name}")
        lines.append(f"选中 {result['selected_count']} / {report['holdout_size']} 个单位。")
        for outcome, estimate in result["effects"].items():
            lines.append(f"- {outcome} 增量/单位：{estimate['mean']:.6g}（95% 区间 {estimate['lower']:.6g} 至 {estimate['upper']:.6g}）")
        if "cost" in result:
            lines.append(f"- 成本/单位：{result['cost']['mean']:.6g}（95% 区间 {result['cost']['lower']:.6g} 至 {result['cost']['upper']:.6g}）")
        if "assumed_cost" in result:
            lines.append(f"- 假设成本/全体评估用户：{result['assumed_cost']:.6g}")
        if "net" in result:
            lines.append(f"- {result['net_label']}/单位：{result['net']['mean']:.6g}（95% 区间 {result['net']['lower']:.6g} 至 {result['net']['upper']:.6g}）")
        if "delta_vs_random" in result:
            lines.append("- 相对随机发券的点估计差值/单位：" +
                         "；".join(f"{key} {value:.6g}" for key, value in result["delta_vs_random"].items()))
        lines.append("")
    lines.extend(["## Uplift 排序诊断", "", "Qini/AUUC 只描述当前留出集的排序，不作为预算策略效果。", ""])
    for name, values in report["ranking_diagnostics"].items():
        lines.append(f"- {name}：AUUC {values['auuc']:.6g}；Qini {values['qini']:.6g}")
    lines.append("")
    if "paired_vs_baseline" in report:
        lines.extend(["## 新旧策略配对差值", "", "同一留出集、同一用户；正值表示当前候选策略更好。", ""])
        for name, metrics in report["paired_vs_baseline"].items():
            for metric, estimate in metrics.items():
                lines.append(f"- {name} / {metric}：{estimate['mean']:.6g}（95% 区间 {estimate['lower']:.6g} 至 {estimate['upper']:.6g}）")
        lines.append("")
    lines.extend(["## 相对随机策略的配对差值", "", "同一留出集、同一用户；正值表示当前候选策略更好。", ""])
    for name, metrics in report["paired_vs_random"].items():
        for metric, estimate in metrics.items():
            lines.append(f"- {name} / {metric}：{estimate['mean']:.6g}（95% 区间 {estimate['lower']:.6g} 至 {estimate['upper']:.6g}）")
    lines.append("")
    for field, title in (("paired_vs_baseline_bootstrap", "新旧策略配对 bootstrap 区间"),
                         ("paired_vs_random_bootstrap", "相对随机策略配对 bootstrap 区间")):
        if field in report:
            lines.extend([f"## {title}", ""])
            for name, metrics in report[field].items():
                for metric, estimate in metrics.items():
                    lines.append(f"- {name} / {metric}：{estimate['mean']:.6g}（95% 区间 {estimate['lower']:.6g} 至 {estimate['upper']:.6g}）")
            lines.append("")
    lines.extend(["## 数据与特征缺口", "", report.get("feature_gaps") or "当前轮未提出新字段。Agent 可根据指标、可用特征和干预前时点在下一轮补充建议。", "",
                  "## 数据证据", "", f"特征时点：{report['data_validation']['feature_timing']}；分组概率：{report['data_validation']['assignment_probability']}", "",
                  "## 口径与限制", "", *[f"- {note}" for note in report["notes"]], ""])
    return "\n".join(lines)


def run_experiment(manifest_path: Path, budget: Budget, *, seed: int, output: Path,
                   candidate_path: Path | None = None, final: bool = False,
                   feature_gaps: Path | None = None, agent_info: dict | None = None,
                   device: str = "cpu", compare_candidate_path: Path | None = None,
                   strict_data: bool = False, bootstrap_reps: int = 0,
                   sandbox_image: str | None = None, timeout_seconds: int = 3600) -> dict:
    if bootstrap_reps and (not final or bootstrap_reps < 2):
        raise ValueError("bootstrap intervals require --final and at least two resamples")
    manifest_path, output = Path(manifest_path), Path(output)
    candidate_path = Path(candidate_path or Path(__file__).with_name("candidate.py"))
    sandbox = DockerSandbox(sandbox_image, timeout_seconds=timeout_seconds) if sandbox_image else None
    manifest_bytes, candidate_bytes = manifest_path.read_bytes(), candidate_path.read_bytes()
    compare_candidate_path = Path(compare_candidate_path) if compare_candidate_path else None
    compare_bytes = compare_candidate_path.read_bytes() if compare_candidate_path else None
    gap_bytes = Path(feature_gaps).read_bytes() if feature_gaps is not None else None
    data = load_dataset(manifest_path, strict=strict_data)
    parts = split_dataset(data, seed)
    target = parts.test if final else parts.validation
    outcome_cols = {name: column for name, column in data.outcomes.items() if name != "coupon_cost"}
    cost_col = data.outcomes.get("coupon_cost")
    candidate_args = {"features": data.features, "treatment": "__treatment",
                      "outcomes": outcome_cols, "cost": cost_col}
    if device == "cuda":
        candidate_args["device"] = "cuda"
    train_columns = list(dict.fromkeys([*data.features, "__treatment", *outcome_cols.values(),
                                        *([cost_col] if cost_col else [])]))
    candidate_train = parts.train[train_columns].reset_index(drop=True)
    candidate_target = target[data.features].reset_index(drop=True)
    predictions = (sandbox.predict(candidate_path, candidate_train, candidate_target, candidate_args, seed, device)
                   if sandbox else _predict(candidate_path, candidate_train, candidate_target, candidate_args, seed, device))
    expected_columns = {f"{name}_uplift" for name in outcome_cols}
    if cost_col:
        expected_columns.add("expected_cost")
    if not isinstance(predictions, pd.DataFrame) or not expected_columns <= set(predictions) or len(predictions) != len(target):
        raise ValueError("candidate returned invalid prediction columns or row count")
    if device == "cuda" and predictions.attrs.get("model_device") != "cuda":
        raise ValueError("candidate did not confirm CUDA model training and prediction")
    if sandbox and device == "cuda" and predictions.attrs.get("cuda_peak_bytes", 0) <= 0:
        raise ValueError("candidate did not allocate a CUDA tensor in the sandbox")
    prediction_sha = _sha(predictions[sorted(expected_columns)].to_numpy(dtype="<f8").tobytes())
    ranking = {name: ranking_diagnostic(target, predictions[f"{name}_uplift"].to_numpy(),
                                        column, data.propensity)
               for name, column in outcome_cols.items()}
    primary = next((name for name in ("active", "visit", "click", "conversion") if name in outcome_cols), None)
    if primary is None:
        primary = next(iter(outcome_cols))
    money = "gross_margin" if "gross_margin" in outcome_cols else "revenue" if "revenue" in outcome_cols else None
    policies, expected_cost = _candidate_policies(predictions, data, budget, primary, money,
                                                   candidate_path, seed, sandbox, device)
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
    paired_random = {name: _paired_metrics(target, policy, policies["random"], data,
                                            outcome_cols, primary, money, expected_cost)
                     for name, policy in policies.items() if name != "random"}
    paired = None
    baseline_prediction_sha = None
    if compare_candidate_path:
        baseline = (sandbox.predict(compare_candidate_path, candidate_train, candidate_target,
                                    candidate_args, seed, device) if sandbox else
                    _predict(compare_candidate_path, candidate_train, candidate_target, candidate_args, seed, device))
        if not isinstance(baseline, pd.DataFrame) or not expected_columns <= set(baseline) or len(baseline) != len(target):
            raise ValueError("comparison candidate returned invalid predictions")
        if device == "cuda" and baseline.attrs.get("model_device") != "cuda":
            raise ValueError("comparison candidate did not confirm CUDA training and prediction")
        if sandbox and device == "cuda" and baseline.attrs.get("cuda_peak_bytes", 0) <= 0:
            raise ValueError("comparison candidate did not allocate a CUDA tensor in the sandbox")
        baseline_prediction_sha = _sha(baseline[sorted(expected_columns)].to_numpy(dtype="<f8").tobytes())
        old_policies, _ = _candidate_policies(baseline, data, budget, primary, money,
                                              compare_candidate_path, seed, sandbox, device)
        paired = {}
        for name, policy in policies.items():
            if name == "random" or name not in old_policies:
                continue
            old = old_policies[name]
            paired[name] = _paired_metrics(target, policy, old, data, outcome_cols, primary,
                                           money, expected_cost)
    unavailable = []
    if "active" not in outcome_cols:
        unavailable.append("active")
    if "gross_margin" not in outcome_cols or not cost_col:
        unavailable.append("net_margin")
    notes = []
    if data.validation["feature_timing"] != "checked":
        notes.append("特征时点只由 manifest 声明，缺少逐行时间证据；不能据此确认无干预后特征。")
    else:
        notes.append("特征记录时间早于分组时间已逐行检查；特征来源和计算口径仍需人工审计。")
    notes.append("随机分组概率已记录来源引用，但引用内容与试验执行情况仍需人工核对。")
    if not final:
        notes.append("验证集可用于多轮选择；其配对区间不能单独作为最终提升证据。")
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
        notes.append(f"成本来自每位选中用户 {data.manifest['fixed_send_cost']} 的固定发券费用假设；"
                     "策略成本按全体评估用户平均，不是实际核销成本。")
    if budget.kind == "cost":
        notes.append("发券选择使用预测成本；报告中的观测成本及区间可能超过预算。")
    if paired is not None:
        notes.append("配对区间基于同一批留出用户的逐用户 IPW 差值；多轮调参后的验证集区间不作为最终提升证据。")
    if bootstrap_reps:
        notes.append(f"最终留出集的配对差值另用 {bootstrap_reps} 次用户级有放回抽样计算百分位区间。")
    framework_sha = _sha(b"".join(Path(__file__).with_name(name).read_bytes()
                              for name in ("cli.py", "data.py", "evaluate.py", "sandbox.py",
                                           "sandbox_worker.py", "experience.py")))
    identity = {"dataset": data.source_sha256, "manifest": _sha(manifest_bytes),
                "framework": framework_sha,
                "candidate": _sha(candidate_bytes), "split": _split_sha(parts),
                "prediction": prediction_sha, "policy": _policy_shas(policies),
                "budget": {"kind": budget.kind, "value": budget.value}, "seed": seed, "final": final,
                "model_device": device, "strict_data": strict_data, "bootstrap_reps": bootstrap_reps,
                "execution": f"docker:{sandbox_image}" if sandbox else "trusted_local",
                "feature_gaps": _sha(gap_bytes) if gap_bytes is not None else None}
    if compare_bytes is not None:
        identity["compare_candidate"] = _sha(compare_bytes)
        identity["compare_prediction"] = baseline_prediction_sha
        identity["compare_policy"] = _policy_shas(old_policies)
    if agent_info is not None:
        identity["agent"] = agent_info
    run_id = _sha(json.dumps(identity, sort_keys=True).encode())[:16]
    report = {"run_id": run_id, "dataset_sha256": data.source_sha256,
              "framework_sha256": framework_sha,
              "manifest_sha256": identity["manifest"], "candidate_sha256": identity["candidate"],
              "prediction_sha256": prediction_sha,
              "split_sha256": identity["split"], "budget": identity["budget"], "seed": seed,
              "model_device": device,
              "cuda_peak_bytes": predictions.attrs.get("cuda_peak_bytes") if sandbox else None,
              "execution": identity["execution"],
              "holdout": "test" if final else "validation", "holdout_size": len(target),
              "features": data.features, "outcome_columns": data.outcomes,
              "data_validation": data.validation,
              "available_outcomes": list(outcome_cols), "unavailable_objectives": unavailable,
              "policies": results, "ranking_diagnostics": ranking,
              "policy_sha256": _policy_shas(policies),
              "paired_vs_random": paired_random, "notes": notes,
              "feature_gaps": gap_bytes.decode("utf-8") if gap_bytes is not None else None}
    if paired is not None:
        report["compare_candidate_sha256"] = identity["compare_candidate"]
        report["compare_prediction_sha256"] = baseline_prediction_sha
        report["paired_vs_baseline"] = paired
        report["baseline_policy_sha256"] = _policy_shas(old_policies)
    if bootstrap_reps:
        def bootstrap(frame, new, old, outcome, propensity):
            return bootstrap_policy_difference(frame, new, old, outcome, propensity,
                                               reps=bootstrap_reps, seed=seed)
        report["paired_vs_random_bootstrap"] = {
            name: _paired_metrics(target, policy, policies["random"], data,
                                  outcome_cols, primary, money, expected_cost, estimate=bootstrap)
            for name, policy in policies.items() if name != "random"}
        if paired is not None:
            report["paired_vs_baseline_bootstrap"] = {
                name: _paired_metrics(target, policy, old_policies[name], data,
                                      outcome_cols, primary, money, expected_cost, estimate=bootstrap)
                for name, policy in policies.items() if name != "random" and name in old_policies}
    if agent_info is not None:
        report["agent"] = agent_info
    if (_sha(data.dataset_path.read_bytes()) != data.source_sha256 or
            _sha(manifest_path.read_bytes()) != identity["manifest"] or
            _sha(candidate_path.read_bytes()) != identity["candidate"] or
            (compare_bytes is not None and _sha(compare_candidate_path.read_bytes()) != identity["compare_candidate"]) or
            (gap_bytes is not None and _sha(Path(feature_gaps).read_bytes()) != identity["feature_gaps"])):
        raise RuntimeError("dataset, manifest, candidate, or feature notes changed during the run")
    directory = output / run_id
    directory.mkdir(parents=True, exist_ok=True)
    split_manifest = {name: sorted(map(int, frame.index)) for name, frame in
                      (("train", parts.train), ("validation", parts.validation), ("test", parts.test))}
    (directory / "split_manifest.json").write_text(json.dumps(split_manifest, separators=(",", ":")) + "\n")
    (directory / "candidate.py").write_bytes(candidate_bytes)
    if compare_bytes is not None:
        (directory / "baseline_candidate.py").write_bytes(compare_bytes)
    if gap_bytes is not None:
        (directory / "feature_gaps.md").write_bytes(gap_bytes)
    (directory / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    (directory / "report.md").write_text(_render(report))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run offline coupon uplift experiments")
    parser.add_argument("command", choices=["run", "agent", "search", "finalize"])
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--budget-kind", choices=["count", "cost"], required=True)
    parser.add_argument("--budget", type=float, required=True, help="fraction of units, or cost per eligible unit")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("runs"))
    parser.add_argument("--candidate", type=Path, default=Path(__file__).with_name("candidate.py"))
    parser.add_argument("--compare-candidate", type=Path, help="frozen baseline candidate for paired policy differences")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu", help="model training and prediction device")
    parser.add_argument("--final", action="store_true", help="evaluate the final test split")
    parser.add_argument("--strict-data", action="store_true", help="require checked pre-assignment feature times")
    parser.add_argument("--bootstrap-reps", type=int, default=0, help="paired bootstrap interval on final holdout")
    parser.add_argument("--feature-gaps", type=Path, help="existing human or Agent feature suggestions")
    parser.add_argument("--objective", help="fixed policy objective for Agent search, such as active or net_margin")
    parser.add_argument("--max-steps", type=int, default=3, help="maximum Agent proposals in a search")
    parser.add_argument("--search-id", help="directory name for one resumable search")
    parser.add_argument("--resume", action="store_true", help="continue an existing search journal")
    parser.add_argument("--timeout-seconds", type=int, default=3600, help="per-candidate evaluation timeout")
    parser.add_argument("--sandbox-image", help="Docker image for isolated candidate execution")
    parser.add_argument("--unsafe-local-execution", action="store_true", help="run generated code without isolation")
    parser.add_argument("--experience-dir", type=Path, help="completed search journals for cross-task lessons")
    parser.add_argument("--agent-provider", choices=["codex", "deepseek", "api"], default="codex")
    parser.add_argument("--agent-config", type=Path, help="JSON configuration for an OpenAI-compatible API provider")
    parser.add_argument("--agent-provider-url", help="API base URL or full chat/completions endpoint")
    parser.add_argument("--agent-api-key-env", help="environment variable holding the API key")
    parser.add_argument("--agent-model", help="Agent model override; DeepSeek defaults to deepseek-flash")
    args = parser.parse_args()
    if args.command in {"agent", "search"} and args.final:
        parser.error("agent revisions and search must use validation; run --final separately after selection")
    if args.command in {"search", "finalize"} and (not args.objective or not args.search_id):
        parser.error("search and finalize need --objective and --search-id")
    api_mode = args.command in {"agent", "search"} and (args.agent_provider != "codex" or
                                                         args.agent_config is not None or args.agent_provider_url is not None)
    if args.command == "search" and not api_mode:
        parser.error("search needs an API Agent provider URL/config or --agent-provider deepseek")
    if args.command in {"agent", "search"} and not args.sandbox_image and not args.unsafe_local_execution:
        parser.error("Agent code requires --sandbox-image, or explicit --unsafe-local-execution")
    provider_name = ("api" if args.agent_provider == "codex" and api_mode else args.agent_provider)
    api_provider = None
    if api_mode:
        config = json.loads(args.agent_config.read_text()) if args.agent_config else {}
        if not isinstance(config, dict):
            parser.error("agent configuration must be a JSON object")
        extra = set(config) - {"provider_url", "model", "api_key_env", "thinking",
                               "iteration_effort", "review_effort"}
        if extra:
            parser.error(f"unknown agent configuration fields: {', '.join(sorted(extra))}")
        url = args.agent_provider_url or config.get("provider_url") or (
            "https://api.deepseek.com" if provider_name == "deepseek" else None)
        model = args.agent_model or config.get("model") or (
            "deepseek-flash" if provider_name == "deepseek" else None)
        key_env = args.agent_api_key_env or config.get("api_key_env") or (
            "DEEPSEEK_API_KEY" if provider_name == "deepseek" else "AGENT_API_KEY")
        if not url or not model:
            parser.error("API agent needs provider_url and model")
        api_key = os.environ.pop(key_env, None)
        if not api_key:
            parser.error(f"{key_env} is required for the API agent")
        try:
            api_provider = ApiProvider(url, model, api_key,
                                       thinking=config.get("thinking", "enabled" if provider_name == "deepseek" or
                                                           urlsplit(url).hostname == "api.deepseek.com" else "omit"),
                                       iteration_effort=config.get("iteration_effort", "high"),
                                       review_effort=config.get("review_effort", "max"))
        except ValueError as error:
            parser.error(str(error))
    budget = Budget(args.budget_kind, args.budget)
    if args.command == "finalize":
        report = finalize_search(args.manifest, budget, seed=args.seed, output=args.output,
                                 initial_candidate=args.candidate, objective=args.objective,
                                 search_id=args.search_id, device=args.device,
                                 strict_data=args.strict_data, timeout_seconds=args.timeout_seconds,
                                 bootstrap_reps=args.bootstrap_reps or 2000,
                                 sandbox_image=args.sandbox_image,
                                 experience_dir=args.experience_dir)
        print(args.output / args.search_id / "final" / report["run_id"] / "report.md")
        return
    if args.command == "search":
        journal = run_search(args.manifest, budget, seed=args.seed, output=args.output,
                             initial_candidate=args.candidate, objective=args.objective,
                             max_steps=args.max_steps, search_id=args.search_id,
                             proposer=lambda context: propose_search_candidate(api_provider, context),
                             device=args.device, strict_data=args.strict_data,
                             resume=args.resume, timeout_seconds=args.timeout_seconds,
                             sandbox_image=args.sandbox_image,
                             experience_dir=args.experience_dir,
                             agent_info={"provider_url": api_provider.url, "model": api_provider.model,
                                         "thinking": api_provider.thinking,
                                         "iteration_effort": api_provider.iteration_effort,
                                         "review_effort": api_provider.review_effort},
                             analyzer=lambda prior, revised, candidate, directory:
                             analyze_reports_deepseek(prior, revised, candidate_path=candidate,
                                                      output_dir=directory, provider=api_provider))
        print(args.output / args.search_id / "journal.json")
        return
    original_candidate = args.candidate.read_bytes() if args.command == "agent" else None
    try:
        if args.command == "agent":
            prior = run_experiment(args.manifest, budget, seed=args.seed, output=args.output,
                                   candidate_path=args.candidate, device=args.device,
                                   strict_data=args.strict_data, sandbox_image=args.sandbox_image,
                                   timeout_seconds=args.timeout_seconds)
            args.feature_gaps = args.output / "agent-feature-gaps.md"
            args.feature_gaps.unlink(missing_ok=True)
            prior_report = args.output / prior["run_id"] / "report.md"
            model = api_provider.model if api_mode else args.agent_model or "default"
            if api_mode:
                revise_candidate_deepseek(args.candidate, prior_report,
                                          feature_gaps_path=args.feature_gaps, provider=api_provider)
            else:
                revise_candidate(args.candidate, prior_report,
                                 feature_gaps_path=args.feature_gaps, model=args.agent_model)
            if not args.feature_gaps.exists():
                args.feature_gaps = None
            agent_info = {"provider": provider_name, "model": model,
                          "previous_run_id": prior["run_id"]}
            if api_mode:
                agent_info.update({"provider_url": api_provider.url,
                                   "iteration_effort": api_provider.iteration_effort,
                                   "review_effort": api_provider.review_effort})
        else:
            agent_info = None
        report = run_experiment(args.manifest, budget, seed=args.seed, output=args.output,
                                candidate_path=args.candidate, final=args.final,
                                feature_gaps=args.feature_gaps, agent_info=agent_info,
                                device=args.device, strict_data=args.strict_data,
                                bootstrap_reps=args.bootstrap_reps,
                                compare_candidate_path=(args.output / prior["run_id"] / "candidate.py"
                                                        if args.command == "agent" else args.compare_candidate),
                                sandbox_image=args.sandbox_image, timeout_seconds=args.timeout_seconds)
        if (args.command == "agent" and prior["prediction_sha256"] == report["prediction_sha256"] and
                prior["policy_sha256"] == report["policy_sha256"]):
            raise ValueError("Agent revision left predictions and policies unchanged")
        if api_mode:
            analyze_reports_deepseek(prior, report, candidate_path=args.candidate,
                                     output_dir=args.output / report["run_id"],
                                     provider=api_provider)
    except Exception:
        if original_candidate is not None:
            args.candidate.write_bytes(original_candidate)
        raise
    print(args.output / report["run_id"] / "report.md")


if __name__ == "__main__":
    main()
