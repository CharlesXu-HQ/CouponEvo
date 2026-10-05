import unittest

from couponevo.search import _validate_proposal


class ParallelImportsTests(unittest.TestCase):
    def proposal(self, module):
        return {"operator": "draft", "parent_ids": [], "hypothesis": "Route multiple branches",
                "candidate_py": f"from {module} import ParallelBranches\n"
                                "def fit_predict(train, valid, config):\n    return None\n"}

    def test_native_composition_is_available_to_model_evo_candidates(self):
        module = "model_evo_harness.models.pytorch.composition"
        result = _validate_proposal(self.proposal(module), {}, {"source": "ModelEvoHarness"})
        self.assertIn(module, result["candidate_py"])

    def test_allowing_composition_does_not_enable_unrelated_harness_imports(self):
        for module, harness in [
            ("model_evo_harness.models.pytorch.composition", None),
            ("model_evo_harness.models.tensorflow.composition", {"source": "ModelEvoHarness"}),
            ("model_evo_harness.engine", {"source": "ModelEvoHarness"}),
        ]:
            with self.subTest(module=module, harness=harness):
                with self.assertRaisesRegex(ValueError, "unsupported modules"):
                    _validate_proposal(self.proposal(module), {}, harness)
