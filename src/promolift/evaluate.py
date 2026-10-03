"""Budget selection and held-out randomized policy estimates."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Budget:
    kind: str  # count fraction, or expected cost per eligible unit
    value: float

    def __post_init__(self):
        if not np.isfinite(self.value):
            raise ValueError("budget must be finite")
        if self.kind == "count" and not 0 <= self.value <= 1:
            raise ValueError("count budget must be a fraction in [0, 1]")
        if self.kind == "cost" and self.value < 0:
            raise ValueError("cost budget cannot be negative")
        if self.kind not in {"count", "cost"}:
            raise ValueError("budget kind must be count or cost")


@dataclass(frozen=True)
class Estimate:
    mean: float
    lower: float
    upper: float
    se: float

    def to_dict(self) -> dict:
        return asdict(self)


def select_policy(scores: np.ndarray, costs: np.ndarray | None, budget: Budget) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or not np.isfinite(scores).all():
        raise ValueError("scores must be a finite one-dimensional array")
    policy = np.zeros(len(scores), dtype=bool)
    if budget.kind == "count":
        limit = int(np.floor(len(scores) * budget.value))
        order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
        policy[[i for i in order[:limit] if scores[i] > 0]] = True
        return policy
    if costs is None:
        raise ValueError("cost budget requires expected costs")
    costs = np.asarray(costs, dtype=float)
    if costs.shape != scores.shape or not np.isfinite(costs).all() or (costs < 0).any():
        raise ValueError("costs must be finite and nonnegative for each score")
    remaining = len(scores) * budget.value
    order = sorted(range(len(scores)), key=lambda i: (-scores[i] / costs[i] if costs[i] else -float("inf"), -scores[i], i))
    for i in order:
        if scores[i] > 0 and costs[i] <= remaining + 1e-12:
            policy[i] = True
            remaining -= costs[i]
    return policy


def _estimate(contributions: np.ndarray) -> Estimate:
    mean = float(np.mean(contributions))
    se = float(np.std(contributions, ddof=1) / np.sqrt(len(contributions))) if len(contributions) > 1 else 0.0
    return Estimate(mean, mean - 1.96 * se, mean + 1.96 * se, se)


def evaluate_policy(frame: pd.DataFrame, policy: np.ndarray, outcome: str, propensity: float) -> Estimate:
    policy = np.asarray(policy, dtype=bool)
    if len(policy) != len(frame) or not 0 < propensity < 1:
        raise ValueError("policy length or propensity is invalid")
    treated = frame["__treatment"].to_numpy(dtype=float)
    observed = frame[outcome].to_numpy(dtype=float)
    if not np.isfinite(observed).all():
        raise ValueError(f"{outcome} must be finite and numeric")
    influence = policy * (treated * observed / propensity - (1 - treated) * observed / (1 - propensity))
    return _estimate(influence)


def compare_policies(frame: pd.DataFrame, new: np.ndarray, old: np.ndarray,
                     outcome: str, propensity: float) -> Estimate:
    """Paired IPW difference on the same randomized holdout units."""
    return _estimate(_policy_difference_contributions(frame, new, old, outcome, propensity))


def _policy_difference_contributions(frame: pd.DataFrame, new: np.ndarray, old: np.ndarray,
                                     outcome: str, propensity: float) -> np.ndarray:
    new, old = np.asarray(new, dtype=bool), np.asarray(old, dtype=bool)
    if len(new) != len(frame) or len(old) != len(frame) or not 0 < propensity < 1:
        raise ValueError("policy length or propensity is invalid")
    treated = frame["__treatment"].to_numpy(dtype=float)
    observed = frame[outcome].to_numpy(dtype=float)
    if not np.isfinite(observed).all():
        raise ValueError(f"{outcome} must be finite and numeric")
    influence = (new.astype(float) - old.astype(float)) * (
        treated * observed / propensity - (1 - treated) * observed / (1 - propensity))
    return influence


def bootstrap_policy_difference(frame: pd.DataFrame, new: np.ndarray, old: np.ndarray,
                                outcome: str, propensity: float, *, reps: int, seed: int) -> Estimate:
    """Percentile interval from resampling paired holdout users."""
    if reps < 2:
        raise ValueError("bootstrap reps must be at least two")
    influence = _policy_difference_contributions(frame, new, old, outcome, propensity)
    rng = np.random.default_rng(seed)
    means = np.array([influence[rng.integers(0, len(influence), len(influence))].mean()
                      for _ in range(reps)])
    lower, upper = np.quantile(means, [0.025, 0.975])
    return Estimate(float(influence.mean()), float(lower), float(upper), float(means.std(ddof=1)))


def ranking_diagnostic(frame: pd.DataFrame, scores: np.ndarray, outcome: str,
                       propensity: float) -> dict[str, float]:
    """IPW cumulative uplift area, with Qini measured above a random ranking."""
    scores = np.asarray(scores, dtype=float)
    if scores.shape != (len(frame),) or not np.isfinite(scores).all() or not 0 < propensity < 1:
        raise ValueError("ranking scores or propensity are invalid")
    observed = frame[outcome].to_numpy(dtype=float)
    if not np.isfinite(observed).all():
        raise ValueError(f"{outcome} must be finite and numeric")
    treated = frame["__treatment"].to_numpy(dtype=float)
    contributions = treated * observed / propensity - (1 - treated) * observed / (1 - propensity)
    ordered = contributions[np.argsort(-scores, kind="stable")]
    reach = np.arange(len(frame) + 1) / len(frame)
    curve = np.r_[0, np.cumsum(ordered) / len(frame)]
    auuc = float(np.trapezoid(curve, reach))
    return {"auuc": auuc, "qini": auuc - float(curve[-1]) / 2}


def estimate_cost(frame: pd.DataFrame, policy: np.ndarray, cost_column: str, propensity: float) -> Estimate:
    policy = np.asarray(policy, dtype=bool)
    if len(policy) != len(frame) or not 0 < propensity < 1:
        raise ValueError("policy length or propensity is invalid")
    cost = frame[cost_column].to_numpy(dtype=float)
    if not np.isfinite(cost).all() or (cost < 0).any():
        raise ValueError("cost must be finite and nonnegative")
    influence = policy * frame["__treatment"].to_numpy(dtype=float) * cost / propensity
    return _estimate(influence)
