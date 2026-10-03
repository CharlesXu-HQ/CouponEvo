# Coupon Uplift Lab Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Build a runnable local first version that accepts a frozen randomized dataset, trains an editable uplift candidate, evaluates available business outcomes under a budget, and records each experiment.

**Architecture:** A JSON manifest maps source CSV columns to treatment, pre-treatment features, outcomes, and assignment probability. A fixed runner owns loading, splitting, policy selection, off-policy evaluation, and reports; the Agent may edit only the candidate module in a temporary workspace before the runner re-evaluates it.

**Tech Stack:** Python 3.12+, pandas, NumPy, standard-library `unittest`, optional installed Codex CLI for automated candidate revisions.

**Spec:** `docs/superpowers/specs/2026-10-03-coupon-uplift-lab-design.md`

## Global Constraints

- One binary comparison: control versus one selected treatment arm.
- The input dataset is fixed; data collection and online deployment are out of scope.
- Activity, revenue, margin, and actual coupon cost are independent optional capabilities.
- The Agent can revise candidate modeling code; manifests, data, splits, evaluator, and historical reports remain fixed.

## Review Focus

- An unknown or nonuniform treatment propensity must not silently produce causal policy value.
- An outcome, cost, ID, or treatment column listed as a feature must be rejected.
- Duplicate users must not be split across train, validation, and test when an ID exists.
- Revenue minus cost must not be labeled profit; margin already net of cost must not be charged twice.
- A selected policy can exceed the estimated budget on observed holdout outcomes; report estimated cost and its uncertainty.

---

### Task 1: Manifest and frozen dataset

**Files:** Create `src/coupon_lab/data.py`, `tests/test_data.py`, `pyproject.toml`.

**Interfaces:** `load_dataset(manifest_path: Path) -> Dataset`; `Dataset` contains a normalized frame, feature names, available outcome names, propensity, metadata, and source SHA-256. `split_dataset(dataset: Dataset, seed: int) -> Split` returns disjoint train/validation/test frames.

- [x] Add failing tests for optional outcomes, arm filtering, feature leakage, propensity, and ID-level splits.
- [x] Run `python -m unittest tests.test_data -v`; observe failure.
- [x] Implement manifest validation, CSV loading, stable row identity, dataset hash, and seeded 60/20/20 splits.
- [x] Run the same command; observe pass.

### Task 2: Uplift candidate and budgeted evaluation

**Files:** Create `src/coupon_lab/candidate.py`, `src/coupon_lab/evaluate.py`, `tests/test_evaluate.py`.

**Interfaces:** `fit_predict(train, target, *, features, treatment, outcomes, cost) -> DataFrame` emits one `<name>_uplift` per available outcome and `expected_cost` when cost is present. `select_policy(scores, costs, budget) -> ndarray[bool]`; `evaluate_policy(frame, policy, outcome, propensity) -> Estimate` reports mean effect and 95% interval.

- [x] Add failing tests with a deterministic randomized fixture for count and cost budgets, positive and zero effects, and an optional cost outcome.
- [x] Run `python -m unittest tests.test_evaluate -v`; observe failure.
- [x] Implement a small ridge T-learner in editable candidate code, deterministic selection, and held-out inverse-propensity estimates with standard-error intervals.
- [x] Run the same command; observe pass.

### Task 3: Reproducible run and Agent revision

**Files:** Create `src/coupon_lab/cli.py`, `src/coupon_lab/agent.py`, `tests/test_cli.py`, `README.md`, `examples/README.md`.

**Interfaces:** `python -m coupon_lab.cli run MANIFEST --budget-kind count|cost --budget VALUE --seed N --output DIR` writes a report JSON and Markdown with dataset, candidate, split, and config hashes. `python -m coupon_lab.cli agent ...` asks the installed Codex CLI to edit only a temporary copy of candidate code, copies it back after validation, then runs the same experiment.

- [x] Add failing end-to-end tests for a manifest with active + margin + cost and one with only conversion, and a mocked Agent CLI edit.
- [x] Run `python -m unittest tests.test_cli -v`; observe failure.
- [x] Implement CLI, fixed evaluator guard, artifacts, capability-specific reporting, and isolated Agent revision.
- [x] Run the same command; observe pass.
- [x] Run the complete test suite and a two-run smoke experiment; verify identical metrics and distinct candidate code hashes after an edit.

## Self-review

Each spec requirement maps to a task: data contract and split (1), candidate/policy/evaluation (2), iterative Agent and report (3). The first release uses local files and a single treatment arm. Public datasets are documented adapters rather than bundled data because source licenses and data sizes vary.
