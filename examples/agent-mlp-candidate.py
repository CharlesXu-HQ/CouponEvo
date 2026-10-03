'''Editable EconML T-learner with PyTorch MLP outcome models.'''

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from econml.metalearners import TLearner
from sklearn.base import BaseEstimator, RegressorMixin


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


class TorchMLPRegressor(RegressorMixin, BaseEstimator):
    '''Scikit-learn compatible outcome model; all linear algebra runs in PyTorch.'''

    def __init__(self, device: str = 'cpu', hidden: tuple[int, ...] = (64, 32), alpha: float = 1e-4,
                 epochs: int = 100, lr: float = 1e-2, batch_size: int | None = 256):
        self.device = device
        self.hidden = hidden
        self.alpha = alpha
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size

    def _build(self, input_dim: int) -> torch.nn.Module:
        layers = []
        dims = [input_dim, *self.hidden]
        for i in range(len(dims) - 1):
            layers.append(torch.nn.Linear(dims[i], dims[i + 1]))
            layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Linear(dims[-1], 1))
        return torch.nn.Sequential(*layers)

    def fit(self, x: np.ndarray, y: np.ndarray):
        if self.device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA model training requires an available GPU')
        x_tensor = torch.as_tensor(np.asarray(x), dtype=torch.float32, device=self.device)
        y_tensor = torch.as_tensor(np.asarray(y), dtype=torch.float32, device=self.device).view(-1, 1)
        n, d = x_tensor.shape
        model = self._build(d).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.alpha)
        loss_fn = torch.nn.MSELoss()
        if self.batch_size is None:
            for _ in range(self.epochs):
                model.train()
                optimizer.zero_grad()
                pred = model(x_tensor)
                loss = loss_fn(pred, y_tensor)
                loss.backward()
                optimizer.step()
        else:
            for _ in range(self.epochs):
                model.train()
                perm = torch.randperm(n, device=self.device)
                for i in range(0, n, self.batch_size):
                    idx = perm[i:i + self.batch_size]
                    optimizer.zero_grad()
                    pred = model(x_tensor[idx])
                    loss = loss_fn(pred, y_tensor[idx])
                    loss.backward()
                    optimizer.step()
        self.model_ = model
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        x_tensor = torch.as_tensor(np.asarray(x), dtype=torch.float32, device=self.device)
        self.model_.eval()
        with torch.no_grad():
            pred = self.model_(x_tensor).view(-1)
        return pred.cpu().numpy()


def fit_predict(train: pd.DataFrame, target: pd.DataFrame, *, features: list[str], treatment: str,
                outcomes: dict[str, str], cost: str | None, device: str = 'cpu') -> pd.DataFrame:
    '''Return uplift estimates for available outcomes and treatment cost estimates.'''
    if device not in {'cpu', 'cuda'}:
        raise ValueError('device must be cpu or cuda')
    x_train, x_target = _matrix(train, target, features)
    arm = train[treatment].to_numpy(dtype=int)
    if not (np.any(arm == 0) and np.any(arm == 1)):
        raise ValueError('training set requires both arms')
    result = pd.DataFrame(index=target.index)
    for name, column in outcomes.items():
        y = train[column].to_numpy(dtype=float)
        learner = TLearner(models=TorchMLPRegressor(device=device))
        learner.fit(y, arm, X=x_train)
        result[f'{name}_uplift'] = learner.effect(x_target)
    if cost:
        y = train[cost].to_numpy(dtype=float)
        cost_model = TorchMLPRegressor(device=device).fit(x_train[arm == 1], y[arm == 1])
        result['expected_cost'] = np.maximum(0, cost_model.predict(x_target))
    result.attrs['model_device'] = device
    result.attrs['causal_framework'] = 'econml.TLearner'
    return result
