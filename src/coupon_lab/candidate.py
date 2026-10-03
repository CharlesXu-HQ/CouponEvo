"""Editable baseline: separate ridge outcome models for treatment and control."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _matrix(train: pd.DataFrame, target: pd.DataFrame, features: list[str]) -> tuple[np.ndarray, np.ndarray]:
    train_x = pd.get_dummies(train[features], dummy_na=True).astype(float)
    target_x = pd.get_dummies(target[features], dummy_na=True).astype(float)
    target_x = target_x.reindex(columns=train_x.columns, fill_value=0)
    means = train_x.mean().fillna(0)
    train_x = train_x.fillna(means)
    target_x = target_x.fillna(means)
    scale = train_x.std(ddof=0).replace(0, 1).fillna(1)
    train_values = ((train_x - means) / scale).to_numpy()
    target_values = ((target_x - means) / scale).to_numpy()
    return np.column_stack([np.ones(len(train)), train_values]), np.column_stack([np.ones(len(target)), target_values])


def _ridge_predict(x: np.ndarray, y: np.ndarray, target: np.ndarray) -> np.ndarray:
    penalty = np.eye(x.shape[1]) * 1e-3
    penalty[0, 0] = 0
    weights = np.linalg.solve(x.T @ x + penalty, x.T @ y)
    return target @ weights


def fit_predict(train: pd.DataFrame, target: pd.DataFrame, *, features: list[str], treatment: str,
                outcomes: dict[str, str], cost: str | None) -> pd.DataFrame:
    """Return uplift estimates for available outcomes and treatment cost estimates."""
    x_train, x_target = _matrix(train, target, features)
    arm = train[treatment].to_numpy(dtype=int)
    if not (np.any(arm == 0) and np.any(arm == 1)):
        raise ValueError("training set requires both arms")
    result = pd.DataFrame(index=target.index)
    for name, column in outcomes.items():
        y = train[column].to_numpy(dtype=float)
        treated = _ridge_predict(x_train[arm == 1], y[arm == 1], x_target)
        control = _ridge_predict(x_train[arm == 0], y[arm == 0], x_target)
        if name not in {"revenue", "gross_margin"}:
            treated, control = np.clip(treated, 0, 1), np.clip(control, 0, 1)
        result[f"{name}_uplift"] = treated - control
    if cost:
        y = train[cost].to_numpy(dtype=float)
        result["expected_cost"] = np.maximum(0, _ridge_predict(x_train[arm == 1], y[arm == 1], x_target))
    return result
