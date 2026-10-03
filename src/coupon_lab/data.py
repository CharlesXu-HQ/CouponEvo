"""Load one frozen experiment and keep its randomization units together."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ALLOWED_OUTCOMES = {"active", "visit", "click", "conversion", "revenue", "gross_margin", "coupon_cost"}


@dataclass(frozen=True)
class Dataset:
    frame: pd.DataFrame
    features: list[str]
    outcomes: dict[str, str]
    propensity: float
    manifest: dict
    dataset_path: Path
    source_sha256: str
    has_unit_id: bool
    validation: dict[str, str]


@dataclass(frozen=True)
class Split:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def load_dataset(manifest_path: Path, *, strict: bool = False) -> Dataset:
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    dataset_path = (manifest_path.parent / manifest["dataset"]).resolve()
    raw = dataset_path.read_bytes()
    frame = pd.read_csv(dataset_path)
    treatment = manifest["treatment"]
    treatment_col = treatment["column"]
    outcomes = manifest["outcomes"]
    features = manifest["features"]
    unit_col = manifest.get("unit_id")
    if not outcomes or not set(outcomes) <= ALLOWED_OUTCOMES:
        raise ValueError("outcomes must name at least one supported result")
    if not any(name != "coupon_cost" for name in outcomes):
        raise ValueError("at least one non-cost outcome is required")
    if not features or len(features) != len(set(features)):
        raise ValueError("features must be a nonempty unique list")
    if manifest.get("feature_timing") != "pre_treatment":
        raise ValueError("feature_timing must assert pre_treatment availability")
    forbidden = set(outcomes.values()) | {treatment_col, unit_col}
    if set(features) & forbidden:
        raise ValueError("feature overlaps treatment, outcome, cost, or unit ID")
    required = set(features) | set(outcomes.values()) | {treatment_col}
    if unit_col:
        required.add(unit_col)
    missing = required - set(frame)
    if missing:
        raise ValueError(f"missing dataset columns: {sorted(missing)}")
    fixed_cost = manifest.get("fixed_send_cost")
    if "coupon_cost" in outcomes and fixed_cost is not None:
        raise ValueError("choose coupon_cost or fixed_send_cost, not both")
    if "gross_margin" in outcomes and ("coupon_cost" in outcomes or fixed_cost is not None) and "margin_includes_coupon_cost" not in manifest:
        raise ValueError("margin_includes_coupon_cost must be stated when margin and cost are present")
    frame = frame[frame[treatment_col].isin([treatment["control"], treatment["treated"]])].copy()
    frame["__treatment"] = (frame[treatment_col] == treatment["treated"]).astype(int)
    if frame.empty or frame["__treatment"].nunique() != 2:
        raise ValueError("selected comparison needs both treatment arms")
    checked = [treatment_col, *outcomes.values()]
    if unit_col:
        checked.append(unit_col)
    if frame[checked].isna().any().any():
        raise ValueError("treatment, outcome, cost, and unit ID cannot be missing")
    for name, column in outcomes.items():
        values = pd.to_numeric(frame[column], errors="coerce")
        if not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ValueError(f"{name} must be finite and numeric")
        frame[column] = values
    if unit_col:
        if frame.groupby(unit_col)["__treatment"].nunique().gt(1).any():
            raise ValueError("conflicting treatment assignment for one unit")
        if frame[unit_col].duplicated().any():
            raise ValueError("one row per unit is required when unit_id is present")
        frame["__unit_id"] = frame[unit_col]
    else:
        frame["__unit_id"] = frame.index
    probability = treatment.get("probability")
    probability_source = treatment.get("probability_source")
    probability_reference = treatment.get("probability_reference")
    if not isinstance(probability_source, str) or probability_source not in {"protocol", "assignment_log", "empirical"} or not isinstance(probability_reference, str) or not probability_reference.strip():
        raise ValueError("treatment needs probability_source and probability_reference")
    if (probability == "empirical") != (probability_source == "empirical"):
        raise ValueError("empirical probability requires empirical probability_source")
    if probability == "empirical":
        if not (treatment.get("simple_randomized") and treatment.get("sampling_preserves_arms")):
            raise ValueError("empirical probability requires simple_randomized and sampling_preserves_arms assertions")
        probability = float(frame["__treatment"].mean())
    if not isinstance(probability, (int, float)) or not 0 < probability < 1:
        raise ValueError("treatment probability must be in (0, 1) or verified empirical")
    if "coupon_cost" in outcomes:
        cost = frame[outcomes["coupon_cost"]]
        if (cost < 0).any():
            raise ValueError("coupon_cost cannot be negative")
        if (cost[frame["__treatment"] == 0] != 0).any():
            raise ValueError("control coupon_cost must be zero")
    if fixed_cost is not None and (not isinstance(fixed_cost, (int, float)) or
                                   not np.isfinite(fixed_cost) or fixed_cost < 0):
        raise ValueError("fixed_send_cost must be finite and nonnegative")
    assignment_time = manifest.get("assignment_time_column")
    feature_times = manifest.get("feature_time_columns")
    if strict and not assignment_time:
        raise ValueError("strict data validation needs assignment_time_column")
    if strict and not feature_times:
        raise ValueError("strict data validation needs feature_time_columns for every feature")
    if assignment_time or feature_times:
        if not assignment_time or not isinstance(feature_times, dict) or set(feature_times) != set(features):
            raise ValueError("assignment_time_column and feature_time_columns for every feature are required together")
        timing_columns = {assignment_time, *feature_times.values()}
        if timing_columns - set(frame):
            raise ValueError(f"missing timing columns: {sorted(timing_columns - set(frame))}")
        assignment = pd.to_datetime(frame[assignment_time], errors="coerce", utc=True)
        if assignment.isna().any():
            raise ValueError("assignment time must be present and parseable")
        for feature, column in feature_times.items():
            recorded = pd.to_datetime(frame[column], errors="coerce", utc=True)
            if recorded.isna().any():
                raise ValueError(f"feature time for {feature} must be present and parseable")
            if (recorded >= assignment).any():
                raise ValueError(f"feature {feature} was recorded at or after assignment")
    validation = {"feature_timing": "checked" if assignment_time else "declared_only",
                  "assignment_probability": "documented"}
    return Dataset(frame.reset_index(drop=True), list(features), dict(outcomes), float(probability),
                   manifest, dataset_path, hashlib.sha256(raw).hexdigest(), bool(unit_col), validation)


def split_dataset(dataset: Dataset, seed: int) -> Split:
    frame = dataset.frame
    unit_arms = frame.groupby("__unit_id", sort=False)["__treatment"].first()
    rng = np.random.default_rng(seed)
    assignments = {}
    for arm in (0, 1):
        units = unit_arms[unit_arms == arm].index.to_numpy(copy=True)
        if len(units) < 5:
            raise ValueError("each treatment arm needs at least five randomization units")
        rng.shuffle(units)
        boundaries = (int(len(units) * 0.6), int(len(units) * 0.8))
        for name, part in zip(("train", "validation", "test"), np.split(units, boundaries)):
            assignments.update({unit: name for unit in part})
    split_names = frame["__unit_id"].map(assignments)
    return Split(*(frame.loc[split_names == name].copy() for name in ("train", "validation", "test")))
