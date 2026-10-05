"""Bounded observations of candidate execution and prediction semantics."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


_MAX_EVENTS = 32
_MAX_COLUMNS = 64


def _append_unique(items: list, value) -> None:
    if value not in items and len(items) < _MAX_EVENTS:
        items.append(value)


def _source() -> str:
    frame = sys._getframe(2)
    return f"{Path(frame.f_code.co_filename).name}:{frame.f_lineno}"


class _CandidateObservation:
    def __init__(self, train: pd.DataFrame, features: list[str]):
        columns = [name for name in features if name in train]
        self.preprocessing = {
            "raw_feature_count": len(features),
            "raw_feature_columns": features[:_MAX_COLUMNS],
            "raw_feature_dtypes": {name: str(train[name].dtype) for name in columns[:_MAX_COLUMNS]},
            "get_dummies": [],
            "pandas_concats": [],
            "dataframe_arrays": [],
        }
        self.events = {"torch_tensors": [], "torch_modules": [], "losses": []}
        self._originals = {}
        self._module_handle = None
        self._seen_modules = set()

    def __enter__(self):
        original_dummies = pd.get_dummies
        self._originals["get_dummies"] = original_dummies

        def observed_dummies(data, *args, **kwargs):
            output = original_dummies(data, *args, **kwargs)
            if isinstance(data, (pd.DataFrame, pd.Series)):
                names = list(data.columns) if isinstance(data, pd.DataFrame) else [data.name]
                event = {"source": _source(), "input_columns": [str(name) for name in names[:_MAX_COLUMNS]],
                         "input_dtypes": [str(dtype) for dtype in (
                             data.dtypes if isinstance(data, pd.DataFrame) else [data.dtype])][:_MAX_COLUMNS],
                         "output_width": len(output.columns),
                         "output_columns": [str(name) for name in output.columns[:_MAX_COLUMNS]]}
                _append_unique(self.preprocessing["get_dummies"], event)
            return output

        pd.get_dummies = observed_dummies
        original_concat = pd.concat
        self._originals["concat"] = original_concat

        def observed_concat(*args, **kwargs):
            output = original_concat(*args, **kwargs)
            if isinstance(output, pd.DataFrame):
                _append_unique(self.preprocessing["pandas_concats"], {
                    "source": _source(), "rows": len(output), "width": len(output.columns),
                    "columns": [str(name) for name in output.columns[:_MAX_COLUMNS]]})
            return output

        pd.concat = observed_concat
        original_to_numpy = pd.DataFrame.to_numpy
        self._originals["to_numpy"] = original_to_numpy

        def observed_to_numpy(frame, *args, **kwargs):
            output = original_to_numpy(frame, *args, **kwargs)
            _append_unique(self.preprocessing["dataframe_arrays"], {
                "source": _source(), "rows": len(frame), "width": len(frame.columns),
                "columns": [str(name) for name in frame.columns[:_MAX_COLUMNS]]})
            return output

        pd.DataFrame.to_numpy = observed_to_numpy
        for name in ("as_tensor", "tensor", "from_numpy"):
            original = getattr(torch, name)
            self._originals[name] = original

            def observed_tensor(*args, _original=original, _name=name, **kwargs):
                tensor = _original(*args, **kwargs)
                if isinstance(tensor, torch.Tensor) and tensor.ndim == 2:
                    event = {"operation": _name, "source": _source(),
                             "shape": list(tensor.shape), "device": str(tensor.device),
                             "dtype": str(tensor.dtype)}
                    _append_unique(self.events["torch_tensors"], event)
                return tensor

            setattr(torch, name, observed_tensor)

        for name in ("mse_loss", "binary_cross_entropy", "binary_cross_entropy_with_logits",
                     "cross_entropy", "huber_loss", "smooth_l1_loss"):
            original = getattr(torch.nn.functional, name)
            self._originals[f"functional.{name}"] = original

            def observed_loss(*args, _original=original, _name=name, **kwargs):
                _append_unique(self.events["losses"], _name)
                return _original(*args, **kwargs)

            setattr(torch.nn.functional, name, observed_loss)

        def module_hook(module, args):
            if isinstance(module, torch.nn.modules.loss._Loss):
                _append_unique(self.events["losses"], type(module).__name__)
            tensor = next((arg for arg in args if isinstance(arg, torch.Tensor) and arg.ndim == 2), None)
            if tensor is None:
                return
            key = (id(module), tuple(tensor.shape), str(tensor.device))
            if key in self._seen_modules or len(self.events["torch_modules"]) >= _MAX_EVENTS:
                return
            self._seen_modules.add(key)
            if any(parameter.requires_grad for parameter in module.parameters(recurse=True)):
                event = {"source": "torch_forward_pre_hook", "class": type(module).__name__,
                         "input_shape": list(tensor.shape),
                         "input_device": str(tensor.device),
                         "parameter_devices": sorted({str(parameter.device)
                                                      for parameter in module.parameters(recurse=True)})}
                _append_unique(self.events["torch_modules"], event)

        self._module_handle = torch.nn.modules.module.register_module_forward_pre_hook(module_hook)
        return self

    def __exit__(self, *_):
        if self._module_handle is not None:
            self._module_handle.remove()
        pd.get_dummies = self._originals["get_dummies"]
        pd.concat = self._originals["concat"]
        pd.DataFrame.to_numpy = self._originals["to_numpy"]
        for name in ("as_tensor", "tensor", "from_numpy"):
            setattr(torch, name, self._originals[name])
        for name in ("mse_loss", "binary_cross_entropy", "binary_cross_entropy_with_logits",
                     "cross_entropy", "huber_loss", "smooth_l1_loss"):
            setattr(torch.nn.functional, name, self._originals[f"functional.{name}"])

    def to_dict(self) -> dict:
        return {"preprocessing": self.preprocessing, "events": self.events,
                "coverage": {"column_lineage": "not_verified",
                             "pandas_get_dummies": "observed" if self.preprocessing["get_dummies"] else "not_observed",
                             "torch_input": "observed" if (self.events["torch_tensors"] or
                                                            self.events["torch_modules"]) else "not_observed",
                             "loss": "observed" if self.events["losses"] else "not_observed"}}


def observe_candidate(train: pd.DataFrame, features: list[str]) -> _CandidateObservation:
    """Observe executed operations without claiming complete feature-column lineage."""
    return _CandidateObservation(train, features)


def prediction_fingerprint(predictions: pd.DataFrame, required_columns: set[str],
                           outcomes: dict[str, str]) -> str:
    """Fingerprint policy inputs and any reported per-arm outcomes."""
    digest = hashlib.sha256(predictions[sorted(required_columns)]
                            .to_numpy(dtype="<f8").tobytes())
    optional = sorted(f"{name}_{arm}" for name in outcomes for arm in ("mu0", "mu1")
                      if f"{name}_{arm}" in predictions)
    for column in optional:
        digest.update(column.encode())
        digest.update(b"\0")
        try:
            digest.update(predictions[column].to_numpy(dtype="<f8").tobytes())
        except (TypeError, ValueError):
            digest.update(json.dumps(predictions[column].tolist(), default=str).encode())
    return digest.hexdigest()


def check_prediction_contract(predictions: pd.DataFrame, outcomes: dict[str, str],
                              train: pd.DataFrame) -> dict:
    """Check optional potential outcomes while allowing direct CATE scores."""
    checks = {}
    for name, label in outcomes.items():
        left, right = f"{name}_mu0", f"{name}_mu1"
        if left not in predictions and right not in predictions:
            checks[name] = {"status": "not_observable", "reason": "per-arm predictions absent"}
            continue
        if left not in predictions or right not in predictions:
            checks[name] = {"status": "inconsistent", "reason": "only one per-arm prediction column"}
            continue
        try:
            mu0 = predictions[left].to_numpy(dtype=float)
            mu1 = predictions[right].to_numpy(dtype=float)
            uplift = predictions[f"{name}_uplift"].to_numpy(dtype=float)
        except (TypeError, ValueError, KeyError):
            checks[name] = {"status": "inconsistent", "reason": "non-numeric prediction column"}
            continue
        if not (np.isfinite(mu0).all() and np.isfinite(mu1).all() and np.isfinite(uplift).all()):
            checks[name] = {"status": "inconsistent", "reason": "non-finite prediction"}
            continue
        labels = pd.to_numeric(train[label], errors="coerce")
        binary = bool(labels.notna().all() and labels.isin([0, 1]).all())
        if binary and ((mu0 < 0).any() or (mu0 > 1).any() or
                       (mu1 < 0).any() or (mu1 > 1).any()):
            checks[name] = {"status": "inconsistent", "reason": "binary probability outside [0,1]"}
            continue
        error = float(np.max(np.abs(uplift - (mu1 - mu0)))) if len(predictions) else 0.0
        checks[name] = {"status": "consistent" if error <= 1e-5 else "inconsistent",
                        "max_abs_error": error, "binary_outcome": binary}
    statuses = {item["status"] for item in checks.values()}
    status = ("inconsistent" if "inconsistent" in statuses else
              "not_observable" if "not_observable" in statuses else "consistent")
    return {"status": status, "outcomes": checks,
            "evidence": ("per-arm predictions agree with reported uplift" if status == "consistent" else
                         "per-arm predictions conflict with reported uplift or outcome range" if
                         status == "inconsistent" else "per-arm predictions were not provided")}
