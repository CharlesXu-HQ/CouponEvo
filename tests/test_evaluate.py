import unittest

import numpy as np
import pandas as pd

from promolift.candidate import fit_predict
from promolift.evaluate import Budget, bootstrap_policy_difference, compare_policies, estimate_cost, evaluate_policy, ranking_diagnostic, select_policy


class EvaluationTests(unittest.TestCase):
    def test_nonfinite_budget_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            Budget("count", float("nan"))

    def test_count_budget_selects_highest_positive_scores(self):
        policy = select_policy(np.array([0.2, -1, 0.8, 0.5]), None, Budget("count", 0.5))
        self.assertEqual(policy.tolist(), [False, False, True, True])

    def test_cost_budget_uses_predicted_cost(self):
        scores = np.array([5.0, 3.0, 2.0, 1.0])
        costs = np.array([2.0, 1.0, 1.0, 1.0])
        policy = select_policy(scores, costs, Budget("cost", 0.5))
        self.assertEqual(policy.tolist(), [False, True, True, False])

    def test_randomized_policy_effect_and_cost(self):
        n = 100
        frame = pd.DataFrame({"__treatment": np.arange(n) % 2})
        frame["active"] = frame["__treatment"]
        frame["cost"] = frame["__treatment"] * 2.0
        policy = np.ones(n, dtype=bool)
        effect = evaluate_policy(frame, policy, "active", 0.5)
        cost = estimate_cost(frame, policy, "cost", 0.5)
        self.assertAlmostEqual(effect.mean, 1.0)
        self.assertAlmostEqual(cost.mean, 2.0)
        self.assertLess(effect.lower, effect.mean)
        self.assertGreater(effect.upper, effect.mean)

    def test_empty_policy_has_zero_effect(self):
        frame = pd.DataFrame({"__treatment": [0, 1], "active": [0, 1]})
        result = evaluate_policy(frame, np.array([False, False]), "active", 0.5)
        self.assertEqual((result.mean, result.lower, result.upper), (0, 0, 0))

    def test_paired_difference_uses_per_unit_policy_contrast(self):
        frame = pd.DataFrame({"__treatment": [0, 1, 0, 1], "active": [0, 1, 1, 1]})
        old = np.array([True, False, True, False])
        new = np.array([False, True, True, False])
        estimate = compare_policies(frame, new, old, "active", 0.5)
        self.assertAlmostEqual(estimate.mean, 0.5)
        self.assertAlmostEqual(estimate.se, np.std([0, 2, 0, 0], ddof=1) / 2)
        same = compare_policies(frame, new, new, "active", 0.5)
        self.assertEqual((same.mean, same.lower, same.upper, same.se), (0, 0, 0, 0))

    def test_bootstrap_policy_difference_uses_paired_users(self):
        frame = pd.DataFrame({"__treatment": [0, 1, 0, 1], "active": [0, 1, 1, 1]})
        old = np.array([True, False, True, False])
        new = np.array([False, True, True, False])
        estimate = bootstrap_policy_difference(frame, new, old, "active", 0.5, reps=500, seed=7)
        self.assertAlmostEqual(estimate.mean, 0.5)
        self.assertGreaterEqual(estimate.lower, 0)
        self.assertLessEqual(estimate.upper, 2)
        same = bootstrap_policy_difference(frame, new, new, "active", 0.5, reps=500, seed=7)
        self.assertEqual((same.mean, same.lower, same.upper), (0, 0, 0))

    def test_ranking_diagnostic_rewards_early_positive_increment(self):
        frame = pd.DataFrame({"__treatment": [1, 0, 1, 0], "active": [1, 0, 0, 0]})
        good = ranking_diagnostic(frame, np.array([4, 3, 2, 1]), "active", 0.5)
        bad = ranking_diagnostic(frame, np.array([1, 2, 3, 4]), "active", 0.5)
        self.assertAlmostEqual(good["qini"], 0.1875)
        self.assertAlmostEqual(bad["qini"], -0.1875)
        self.assertGreater(good["auuc"], bad["auuc"])

    def test_candidate_handles_optional_cost_and_categories(self):
        rows = []
        for i in range(40):
            t = i % 2
            rows.append({"__treatment": t, "x": i / 40, "segment": "a" if i < 20 else "b",
                         "active": t, "cost": 2 * t})
        train = pd.DataFrame(rows)
        target = train.iloc[:4].copy()
        result = fit_predict(train, target, features=["x", "segment"],
                             treatment="__treatment", outcomes={"active": "active"}, cost="cost")
        self.assertEqual(list(result), ["active_uplift", "expected_cost"])
        self.assertTrue((result.active_uplift > 0).all())
        self.assertTrue((result.expected_cost >= 0).all())
        self.assertEqual(result.attrs["causal_framework"], "econml.TLearner")
        no_cost = fit_predict(train, target, features=["x", "segment"],
                              treatment="__treatment", outcomes={"active": "active"}, cost=None)
        self.assertNotIn("expected_cost", no_cost)


if __name__ == "__main__":
    unittest.main()
