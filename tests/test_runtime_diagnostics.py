import json
import hashlib
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from couponevo.runtime_diagnostics import (check_prediction_contract, observe_candidate,
                                           prediction_fingerprint)
from couponevo.sandbox_worker import main as sandbox_worker_main


class RuntimeObservationTests(unittest.TestCase):
    def test_numeric_get_dummies_noop_and_actual_torch_input_are_observed(self):
        train = pd.DataFrame({"v1": [0, 1, 0, 1], "v2": [0.1, 0.2, 0.3, 0.4]})
        original_dummies = pd.get_dummies
        original_as_tensor = torch.as_tensor
        with observe_candidate(train, ["v1", "v2"]) as observation:
            matrix = pd.get_dummies(train)
            inputs = torch.as_tensor(matrix.to_numpy(dtype=np.float32))
            model = torch.nn.Linear(2, 1)
            loss = torch.nn.BCEWithLogitsLoss()(model(inputs).flatten(),
                                                 torch.tensor([0., 1., 0., 1.]))
            loss.backward()
        diagnostics = observation.to_dict()
        self.assertIs(pd.get_dummies, original_dummies)
        self.assertIs(torch.as_tensor, original_as_tensor)
        self.assertEqual(diagnostics["preprocessing"]["raw_feature_count"], 2)
        self.assertEqual(diagnostics["preprocessing"]["get_dummies"][0]["output_width"], 2)
        self.assertEqual(diagnostics["preprocessing"]["get_dummies"][0]["output_columns"],
                         ["v1", "v2"])
        self.assertTrue(any(event["shape"] == [4, 2] for event in
                            diagnostics["events"]["torch_tensors"]))
        self.assertTrue(any(event["class"] == "Linear" and event["input_shape"] == [4, 2]
                            for event in diagnostics["events"]["torch_modules"]))
        self.assertTrue(all(event["source"] == "torch_forward_pre_hook"
                            for event in diagnostics["events"]["torch_modules"]))
        self.assertIn("BCEWithLogitsLoss", diagnostics["events"]["losses"])
        self.assertEqual(diagnostics["coverage"]["column_lineage"], "not_verified")

    def test_functional_loss_is_observed(self):
        train = pd.DataFrame({"x": [0.0, 1.0]})
        with observe_candidate(train, ["x"]) as observation:
            torch.nn.functional.mse_loss(torch.tensor([0.0, 1.0]),
                                         torch.tensor([1.0, 1.0]))
        self.assertIn("mse_loss", observation.to_dict()["events"]["losses"])

    def test_dataframe_preprocessing_views_record_columns_before_tensor_conversion(self):
        train = pd.DataFrame({"a": [0, 1], "b": [2.0, 3.0]})
        with observe_candidate(train, ["a", "b"]) as observation:
            matrix = pd.concat([train[["a"]], train[["b"]]], axis=1)
            matrix.to_numpy(dtype=float)
        preprocessing = observation.to_dict()["preprocessing"]
        self.assertTrue(any(event["columns"] == ["a", "b"] and event["width"] == 2
                            for event in preprocessing["pandas_concats"]))
        self.assertTrue(any(event["columns"] == ["a", "b"] and event["width"] == 2
                            for event in preprocessing["dataframe_arrays"]))


class PredictionContractTests(unittest.TestCase):
    def setUp(self):
        self.train = pd.DataFrame({"purchase": [0, 1, 0, 1]})
        self.outcomes = {"conversion": "purchase"}

    def test_binary_potential_outcomes_match_probability_uplift(self):
        predictions = pd.DataFrame({"conversion_uplift": [0.2, 0.1],
                                    "conversion_mu0": [0.1, 0.4],
                                    "conversion_mu1": [0.3, 0.5]})
        result = check_prediction_contract(predictions, self.outcomes, self.train)
        self.assertEqual(result["status"], "consistent")
        self.assertEqual(result["outcomes"]["conversion"]["status"], "consistent")

    def test_logit_shift_is_flagged_without_blocking_evaluation(self):
        predictions = pd.DataFrame({"conversion_uplift": [2.0, 1.0],
                                    "conversion_mu0": [0.02, 0.5],
                                    "conversion_mu1": [0.12, 0.73]})
        result = check_prediction_contract(predictions, self.outcomes, self.train)
        self.assertEqual(result["status"], "inconsistent")
        self.assertGreater(result["outcomes"]["conversion"]["max_abs_error"], 0.7)

    def test_direct_cate_without_per_arm_outputs_remains_available(self):
        predictions = pd.DataFrame({"conversion_uplift": [-0.2, 0.5]})
        result = check_prediction_contract(predictions, self.outcomes, self.train)
        self.assertEqual(result["status"], "not_observable")

    def test_partial_or_out_of_range_binary_potential_outcomes_are_inconsistent(self):
        partial = pd.DataFrame({"conversion_uplift": [0.1], "conversion_mu0": [0.2]})
        outside = pd.DataFrame({"conversion_uplift": [0.1],
                                "conversion_mu0": [-0.1], "conversion_mu1": [0.0]})
        self.assertEqual(check_prediction_contract(partial, self.outcomes, self.train)["status"],
                         "inconsistent")
        self.assertEqual(check_prediction_contract(outside, self.outcomes, self.train)["status"],
                         "inconsistent")

    def test_potential_outcomes_affect_prediction_fingerprint(self):
        first = pd.DataFrame({"conversion_uplift": [0.2], "conversion_mu0": [0.1],
                              "conversion_mu1": [0.3]})
        second = pd.DataFrame({"conversion_uplift": [0.2], "conversion_mu0": [0.2],
                               "conversion_mu1": [0.4]})
        required = {"conversion_uplift"}
        self.assertNotEqual(prediction_fingerprint(first, required, self.outcomes),
                            prediction_fingerprint(second, required, self.outcomes))
        direct = pd.DataFrame({"conversion_uplift": [0.2]})
        previous = hashlib.sha256(direct[["conversion_uplift"]]
                                  .to_numpy(dtype="<f8").tobytes()).hexdigest()
        self.assertEqual(prediction_fingerprint(direct, required, self.outcomes), previous)


class RuntimeBoundaryTests(unittest.TestCase):
    def test_sandbox_worker_returns_observed_operations(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "input", root / "output"
            source.mkdir()
            output.mkdir()
            (source / "candidate.py").write_text(
                "import pandas as pd\nimport torch\n"
                "def fit_predict(train, target, *, features, treatment, outcomes, cost=None):\n"
                "    data = pd.get_dummies(train[features])\n"
                "    x = torch.as_tensor(data.to_numpy(dtype=float).copy(), dtype=torch.float32)\n"
                "    model = torch.nn.Linear(x.shape[1], 1)\n"
                "    torch.nn.MSELoss()(model(x), torch.zeros((len(x), 1)))\n"
                "    result = pd.DataFrame({'conversion_uplift': [0.1] * len(target)})\n"
                "    result.attrs['model_device'] = 'cpu'\n"
                "    return result\n")
            train = pd.DataFrame({"v": [1.0, 2.0], "__treatment": [0, 1],
                                  "purchase": [0, 1]})
            target = pd.DataFrame({"v": [3.0]})
            (source / "train.json").write_text(train.to_json(orient="table"))
            (source / "target.json").write_text(target.to_json(orient="table"))
            (source / "job.json").write_text(json.dumps({
                "operation": "predict", "device": "cpu", "seed": 42,
                "args": {"features": ["v"], "treatment": "__treatment",
                         "outcomes": {"conversion": "purchase"}, "cost": None}}))
            with patch.object(sys, "argv", ["worker", str(source), str(output)]):
                sandbox_worker_main()
            response = json.loads((output / "response.json").read_text())
            self.assertEqual(response["runtime_diagnostics"]["preprocessing"]["get_dummies"][0]
                             ["output_width"], 1)
            self.assertIn("MSELoss", response["runtime_diagnostics"]["events"]["losses"])

    def test_cli_reports_inconsistent_probability_claim_without_aborting_evaluation(self):
        from couponevo.cli import run_experiment
        from couponevo.evaluate import Budget

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frame = pd.DataFrame({"id": range(100), "arm": [i % 2 for i in range(100)],
                                  "v": [float(i) / 100 for i in range(100)],
                                  "purchase": [i % 2 for i in range(100)]})
            frame.to_csv(root / "data.csv", index=False)
            manifest = {"dataset": "data.csv", "unit_id": "id",
                        "treatment": {"column": "arm", "control": 0, "treated": 1,
                                      "probability": 0.5, "probability_source": "protocol",
                                      "probability_reference": "fixture"},
                        "features": ["v"], "feature_timing": "pre_treatment",
                        "outcomes": {"conversion": "purchase"}}
            (root / "manifest.json").write_text(json.dumps(manifest))
            candidate = root / "candidate.py"
            candidate.write_text(
                "import pandas as pd\n"
                "def fit_predict(train, target, *, features, treatment, outcomes, cost=None):\n"
                "    return pd.DataFrame({'conversion_uplift': [2.0] * len(target),\n"
                "                         'conversion_mu0': [0.1] * len(target),\n"
                "                         'conversion_mu1': [0.2] * len(target)})\n")
            report = run_experiment(root / "manifest.json", Budget("count", 0.2), seed=7,
                                    output=root / "runs", candidate_path=candidate)
            self.assertEqual(report["runtime_diagnostics"]["prediction_contract"]["status"],
                             "inconsistent")
            self.assertEqual(report["holdout"], "validation")
            markdown = (root / "runs" / report["run_id"] / "report.md").read_text()
            self.assertIn("预测语义校验：inconsistent", markdown)
            alternate = root / "alternate.py"
            alternate.write_text(candidate.read_text().replace("[0.1] * len(target)",
                                                               "[0.2] * len(target)")
                                 .replace("[0.2] * len(target)})", "[0.3] * len(target)})"))
            second = run_experiment(root / "manifest.json", Budget("count", 0.2), seed=7,
                                    output=root / "runs", candidate_path=alternate)
            self.assertNotEqual(report["prediction_sha256"], second["prediction_sha256"])


if __name__ == "__main__":
    unittest.main()
