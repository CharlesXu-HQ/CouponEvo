import unittest
from unittest.mock import patch

import pandas as pd

from couponevo.candidate import fit_predict

try:
    import torch
except ImportError:
    torch = None


class TorchCandidateTests(unittest.TestCase):
    @unittest.skipUnless(torch is not None, "PyTorch unavailable")
    def test_t_learner_uses_pytorch_on_cpu(self):
        train = pd.DataFrame({"x": [0., 1., 2., 3., 4., 5.],
                              "__treatment": [0, 1, 0, 1, 0, 1],
                              "conversion": [0, 1, 0, 1, 0, 1]})
        with patch("numpy.linalg.solve", side_effect=AssertionError("NumPy model fit used")):
            result = fit_predict(train, pd.DataFrame({"x": [1.5, 3.5]}),
                                 features=["x"], treatment="__treatment",
                                 outcomes={"conversion": "conversion"}, cost=None)
        self.assertEqual(result.attrs["model_device"], "cpu")
        self.assertTrue((result["conversion_uplift"] > 0).all())

    @unittest.skipUnless(torch is not None and torch.cuda.is_available(), "CUDA PyTorch unavailable")
    def test_t_learner_trains_and_predicts_on_cuda(self):
        train = pd.DataFrame({"x": [0., 1., 2., 3., 4., 5.],
                              "__treatment": [0, 1, 0, 1, 0, 1],
                              "conversion": [0, 1, 0, 1, 0, 1]})
        target = pd.DataFrame({"x": [1.5, 3.5]})
        result = fit_predict(train, target, features=["x"], treatment="__treatment",
                             outcomes={"conversion": "conversion"}, cost=None,
                             device="cuda")
        self.assertEqual(result.attrs["model_device"], "cuda")
        self.assertTrue((result["conversion_uplift"] > 0).all())
        self.assertGreater(torch.cuda.max_memory_allocated(), 0)
