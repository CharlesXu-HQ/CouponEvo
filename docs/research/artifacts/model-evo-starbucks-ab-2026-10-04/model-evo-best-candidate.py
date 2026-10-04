"""Crossover candidate: shared-trunk per-arm nonlinear outcome heads (nonlinear T-learner)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

_MAX_LEVELS = 64


def _is_categorical(train: pd.DataFrame, feature: str) -> bool:
    column = train[feature]
    if not pd.api.types.is_float_dtype(column):
        return True
    return column.nunique(dropna=True) <= _MAX_LEVELS


def _design(train: pd.DataFrame, target: pd.DataFrame, features: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """One-hot low-cardinality fields and standardize continuous fields using train-only statistics."""
    categorical = [f for f in features if _is_categorical(train, f)]
    numeric = [f for f in features if f not in categorical]
    train_parts: list[pd.DataFrame] = []
    target_parts: list[pd.DataFrame] = []
    for feature in categorical:
        train_dummy = pd.get_dummies(train[feature].astype(object), prefix=feature)
        target_dummy = pd.get_dummies(target[feature].astype(object), prefix=feature)
        target_dummy = target_dummy.reindex(columns=train_dummy.columns, fill_value=0)
        train_parts.append(train_dummy.astype(np.float64))
        target_parts.append(target_dummy.astype(np.float64))
    for feature in numeric:
        mean = float(train[feature].mean())
        scale = float(train[feature].std(ddof=0)) or 1.0
        train_parts.append(pd.DataFrame({feature: (train[feature] - mean) / scale}, index=train.index).astype(np.float64))
        target_parts.append(pd.DataFrame({feature: (target[feature] - mean) / scale}, index=target.index).astype(np.float64))
    train_x = pd.concat(train_parts, axis=1) if train_parts else pd.DataFrame(index=train.index)
    target_x = pd.concat(target_parts, axis=1) if target_parts else pd.DataFrame(index=target.index)
    target_x = target_x.reindex(columns=train_x.columns, fill_value=0.0)
    return train_x.to_numpy(np.float64), target_x.to_numpy(np.float64)


def _with_intercept(x: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(len(x)), x])


class _TorchRidge:
    """Closed-form ridge whose linear algebra runs on the requested torch device."""

    def __init__(self, device: str, alpha: float = 1e-3):
        self.device = device
        self.alpha = alpha

    def fit(self, x: np.ndarray, y: np.ndarray):
        x_tensor = torch.as_tensor(np.asarray(x), dtype=torch.float64, device=self.device)
        y_tensor = torch.as_tensor(np.asarray(y), dtype=torch.float64, device=self.device)
        penalty = torch.eye(x_tensor.shape[1], dtype=torch.float64, device=self.device) * self.alpha
        penalty[0, 0] = 0.0
        self.weights_ = torch.linalg.solve(x_tensor.T @ x_tensor + penalty, x_tensor.T @ y_tensor)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        x_tensor = torch.as_tensor(np.asarray(x), dtype=torch.float64, device=self.device)
        return (x_tensor @ self.weights_).cpu().numpy()


class _SharedArmNet(torch.nn.Module):
    """Shared nonlinear trunk with separate control/treated outcome heads."""

    def __init__(self, width: int, hidden: tuple[int, ...] = (64, 32)):
        super().__init__()
        blocks: list[torch.nn.Module] = []
        previous = width
        for size in hidden:
            blocks.extend([torch.nn.Linear(previous, size), torch.nn.ReLU()])
            previous = size
        self.trunk = torch.nn.Sequential(*blocks)
        self.control_head = torch.nn.Linear(previous, 1)
        self.treated_head = torch.nn.Linear(previous, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.trunk(x)
        return self.control_head(z).squeeze(-1), self.treated_head(z).squeeze(-1)


def _fit_arm_net(x: np.ndarray, y: np.ndarray, arm: np.ndarray, device: str, epochs: int = 40,
                 batch_size: int = 4096, learning_rate: float = 3e-3, weight_decay: float = 1e-4,
                 seed: int = 0) -> _SharedArmNet:
    """Fit E[Y|X,T] as two nonlinear arm heads sharing a trunk, on pooled randomized rows."""
    torch.manual_seed(seed)
    x_tensor = torch.as_tensor(x, dtype=torch.float32, device=device)
    y_tensor = torch.as_tensor(y, dtype=torch.float32, device=device)
    arm_tensor = torch.as_tensor(arm, dtype=torch.float32, device=device)
    model = _SharedArmNet(x_tensor.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    rows = x_tensor.shape[0]
    for _ in range(epochs):
        order = torch.randperm(rows, device=device)
        for start in range(0, rows, batch_size):
            index = order[start:start + batch_size]
            optimizer.zero_grad()
            control_out, treated_out = model(x_tensor[index])
            prediction = arm_tensor[index] * treated_out + (1.0 - arm_tensor[index]) * control_out
            loss = torch.mean((prediction - y_tensor[index]) ** 2)
            loss.backward()
            optimizer.step()
    return model


def _predict_tau(model: _SharedArmNet, x: np.ndarray, device: str, batch_size: int = 8192) -> np.ndarray:
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            chunk = torch.as_tensor(x[start:start + batch_size], dtype=torch.float32, device=device)
            control_out, treated_out = model(chunk)
            outputs.append((treated_out - control_out).cpu().numpy())
    return np.concatenate(outputs) if outputs else np.zeros(0, dtype=np.float64)


def fit_predict(train: pd.DataFrame, target: pd.DataFrame, *, features: list[str], treatment: str,
                outcomes: dict[str, str], cost: str | None, device: str = "cpu") -> pd.DataFrame:
    """Return nonlinear T-learner uplift estimates and expected treatment cost estimates."""
    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu or cuda")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA model training requires an available GPU")
    x_train, x_target = _design(train, target, features)
    arm = train[treatment].to_numpy(dtype=float)
    if not (np.any(arm == 1) and np.any(arm == 0)):
        raise ValueError("training set requires both arms")
    result = pd.DataFrame(index=target.index)
    for name, column in outcomes.items():
        y = train[column].to_numpy(dtype=float)
        model = _fit_arm_net(x_train, y, arm, device)
        result[f"{name}_uplift"] = _predict_tau(model, x_target, device)
    if cost:
        cost_y = train[cost].to_numpy(dtype=float)
        treated = arm == 1
        cost_model = _TorchRidge(device).fit(_with_intercept(x_train[treated]), cost_y[treated])
        result["expected_cost"] = np.maximum(0.0, cost_model.predict(_with_intercept(x_target)))
    result.attrs["model_device"] = device
    result.attrs["causal_framework"] = "crossover: shared-trunk per-arm torch MLP outcome heads (nonlinear T-learner)"
    result.attrs["propensity"] = float(np.clip(arm.mean(), 0.02, 0.98))
    return result
