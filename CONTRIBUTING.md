# Contributing to CouponEvo

Thank you for improving reproducible marketing experiments.

1. Open an issue or pull request describing the behavior and the fixed dataset/task assumptions.
2. Keep changes focused. Add a test when behavior or a data guard changes; run `uv run --frozen python -m unittest discover -s tests -v`.
3. Do not commit API keys, raw user data, downloaded datasets, or generated `runs/` files. Public examples should point to the upstream dataset and state any licensing or measurement limits.
4. For evaluation changes, preserve the validation/test boundary and document the estimand, budget, and uncertainty calculation.

The project code is licensed under [Apache-2.0](LICENSE). Dataset licenses remain with their publishers.
