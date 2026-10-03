import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from coupon_lab.data import load_dataset, split_dataset


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        rows = []
        for user in range(20):
            rows.append({
                "user": user, "arm": "coupon" if user % 2 else "none",
                "x": user / 20, "segment": "new" if user < 10 else "old",
                "active": user % 2, "margin": float(user),
                "cost": 2.0 if user % 2 else 0.0,
            })
        pd.DataFrame(rows).to_csv(self.root / "data.csv", index=False)
        self.manifest = {
            "dataset": "data.csv", "unit_id": "user",
            "treatment": {"column": "arm", "control": "none", "treated": "coupon", "probability": 0.5},
            "features": ["x", "segment"], "feature_timing": "pre_treatment",
            "outcomes": {"active": "active", "gross_margin": "margin", "coupon_cost": "cost"},
            "margin_includes_coupon_cost": False,
        }

    def load(self):
        path = self.root / "manifest.json"
        path.write_text(json.dumps(self.manifest))
        return load_dataset(path)

    def test_optional_outcomes_and_arm_mapping(self):
        data = self.load()
        self.assertEqual(set(data.outcomes), {"active", "gross_margin", "coupon_cost"})
        self.assertEqual(data.frame["__treatment"].sum(), 10)
        self.assertEqual(data.propensity, 0.5)
        self.assertEqual(len(data.source_sha256), 64)

    def test_missing_monetary_and_cost_outcomes_are_allowed(self):
        self.manifest["outcomes"] = {"active": "active"}
        self.manifest.pop("margin_includes_coupon_cost")
        data = self.load()
        self.assertEqual(data.outcomes, {"active": "active"})

    def test_outcome_as_feature_is_rejected(self):
        self.manifest["features"].append("active")
        with self.assertRaisesRegex(ValueError, "feature.*outcome"):
            self.load()

    def test_unknown_propensity_is_rejected(self):
        self.manifest["treatment"].pop("probability")
        with self.assertRaisesRegex(ValueError, "probability"):
            self.load()

    def test_empirical_propensity_requires_randomization_and_sampling_assertions(self):
        self.manifest["treatment"]["probability"] = "empirical"
        with self.assertRaisesRegex(ValueError, "simple_randomized"):
            self.load()
        self.manifest["treatment"].update(simple_randomized=True, sampling_preserves_arms=True)
        self.assertEqual(self.load().propensity, 0.5)

    def test_same_user_stays_in_one_split(self):
        split = split_dataset(self.load(), seed=7)
        ids = [set(part.user) for part in (split.train, split.validation, split.test)]
        self.assertFalse(ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
        self.assertEqual(sum(map(len, (split.train, split.validation, split.test))), 20)

    def test_conflicting_assignment_for_same_user_is_rejected(self):
        frame = pd.read_csv(self.root / "data.csv")
        duplicate = frame.iloc[[0]].copy()
        duplicate.loc[:, "arm"] = "coupon"
        frame = pd.concat([frame, duplicate], ignore_index=True)
        frame.to_csv(self.root / "data.csv", index=False)
        with self.assertRaisesRegex(ValueError, "conflicting"):
            self.load()

    def test_duplicate_user_rows_are_rejected(self):
        frame = pd.read_csv(self.root / "data.csv")
        frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
        frame.to_csv(self.root / "data.csv", index=False)
        with self.assertRaisesRegex(ValueError, "one row per unit"):
            self.load()

    def test_non_numeric_outcome_is_rejected_at_load(self):
        frame = pd.read_csv(self.root / "data.csv")
        frame["active"] = frame["active"].astype(object)
        frame.loc[0, "active"] = "unknown"
        frame.to_csv(self.root / "data.csv", index=False)
        with self.assertRaisesRegex(ValueError, "numeric"):
            self.load()


if __name__ == "__main__":
    unittest.main()
