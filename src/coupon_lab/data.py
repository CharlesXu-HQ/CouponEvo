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


@dataclass(frozen=True)
class Split:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def load_dataset(manifest_path: Path) -> Dataset:
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
    return Dataset(frame.reset_index(drop=True), list(features), dict(outcomes), float(probability),
                   manifest, dataset_path, hashlib.sha256(raw).hexdigest(), bool(unit_col))


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
