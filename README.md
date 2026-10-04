# CouponEvo

[English](README.md) | [简体中文](README.zh-CN.md)

**Agent-led uplift experiments for budgeted coupon targeting.**

A coupon model should answer a causal question: *who will change their behavior because of an offer?* A high purchase probability alone does not answer it. CouponEvo gives an Agent a controlled loop to improve the uplift model **and** the allocation policy on a fixed randomized experiment, then checks whether the resulting policy improves the outcome under a coupon budget.

## The experiment contract

| The Agent can change | The runner keeps fixed |
| --- | --- |
| EconML `TLearner` candidate code, including its PyTorch outcome models and an optional `choose_policy` allocation function | Dataset and manifest hashes, treatment probability, train/validation/test split, objective, budget, and evaluator |
| The next hypothesis and search move (`draft`, `improve`, `debug`, or `crossover`) based on validation feedback | Independent policy estimates, budget checks, paired comparisons, and the final holdout |
| A `feature_gaps.md` note when a useful pretreatment feature is missing | The dataset itself; missing features are proposed for a future dataset version, never invented for the current experiment |

Generated candidates run in an isolated Docker container. Each search retains its hypotheses, code snapshots, reports, and failure history, and can resume with the same task definition. Optional experience reuse reads **validation history only** from completed searches with the same dataset bytes, manifest, objective, budget, split seed, and evaluator. Changing the data creates a new experience scope.

The primary question is **policy value**, not just uplift ranking: the evaluator estimates incremental outcomes from randomized assignments with inverse probability weighting (IPW), compares policies on the same users, and reports paired uncertainty intervals. Qini/AUUC remain ranking diagnostics. After exploration, `finalize` freezes the validation champion and compares it with the seed policy once on the held-out test set, including a paired bootstrap interval.

The paired estimator averages `(policy_new − policy_seed) × [treatment × outcome / p − control × outcome / (1 − p)]` over evaluation users, where `p` is the documented treatment probability.

```text
fixed randomized data → Agent proposes code → sandboxed training and validation
                      ↑                         ↓
                      └── hypotheses and feedback ┘
                                      ↓
                         freeze candidate → final holdout
```

## What the public experiment showed

On the [84,534-row Starbucks randomized promotion dataset](examples/README.md), a two-step DeepSeek Agent search ran EconML/PyTorch candidates on an RTX 5090. Its objective was incremental purchase conversion under an **assumed** send-cost budget.

| Candidate | Validation IPW incremental conversions per eligible user |
| --- | ---: |
| Seed: T-learner with PyTorch ridge outcome models | 0.003312 |
| Agent proposal: PyTorch MLP outcome models | **0.004140** |
| Agent proposal: MLP ensemble | 0.002957 |

The validation winner's **held-out paired difference from the seed** was `+0.000118` conversions per user; its 95% bootstrap interval was `[-0.001065, +0.001301]`. The interval crosses zero, so this run **does not establish an improvement**. That distinction between a promising search result and a supported final claim is central to the project. See the [full search record](docs/research/agent-search-gpu-2026-10-03.md).

Starbucks has neither App reactivation labels nor actual coupon redemption cost or user-level margin. Criteo and X5 RetailHero have also been used for [GPU model checks](docs/research/gpu-validation-2026-10-03.md), not for causal policy-value claims where the published data do not support the required treatment probability.

## Dataset manifest

CouponEvo expects a fixed randomized dataset with one row per assignment unit (usually a user), a treatment/control assignment, a documented assignment probability, pretreatment features, and at least one measured outcome. A user ID lets the loader check for duplicate users; without one, it reports that limitation. A manifest such as [examples/starbucks.json](examples/starbucks.json) declares that contract. The evaluator rejects outcome or treatment columns used as features and can check row-level feature timestamps when provided.

A count budget needs no cost label. A cost budget uses observed coupon cost or an explicitly marked fixed send-cost assumption; real net-margin claims additionally require suitable margin and cost definitions. See the [business RCT contract](docs/business-rct-contract.md) and [public dataset notes](examples/README.md). Public datasets and generated runs are not committed to this repository.

## Try an Agent search

Use Python 3.12+, [uv](https://docs.astral.sh/uv/), and the [Starbucks CSV](examples/README.md) at `data/starbucks-training.csv`. For the CUDA example, use a Linux GPU host with a matching PyTorch CUDA build and Docker. Set `AGENT_API_KEY` in the environment; [the example provider config](examples/agent.deepseek.json) supplies the URL and model without storing a key.

```bash
uv sync --frozen --python 3.12
docker build -f Dockerfile.sandbox -t couponevo-sandbox:py312-cuda128 .
export SANDBOX_IMAGE="$(docker image inspect couponevo-sandbox:py312-cuda128 --format '{{.Id}}')"
uv run --frozen python -m couponevo.cli search examples/starbucks.json \
  --budget-kind cost --budget 0.03 --objective conversion --seed 42 \
  --max-steps 3 --search-id starbucks-01 --device cuda \
  --agent-config examples/agent.deepseek.json --sandbox-image "$SANDBOX_IMAGE"
uv run --frozen python -m couponevo.cli finalize examples/starbucks.json \
  --budget-kind cost --budget 0.03 --objective conversion --seed 42 \
  --search-id starbucks-01 --device cuda --bootstrap-reps 2000 \
  --sandbox-image "$SANDBOX_IMAGE"
```

The Agent provider is configurable by URL, model, and API-key environment variable. The example requests `high` reasoning for iteration and analysis, and `max` review for flagged anomalies; the endpoint must support those settings. Search results and candidate snapshots are written under ignored `runs/`. Run `python -m couponevo.cli --help` for local baseline evaluation, resuming a search, and other options.

## Scope and contribution

This is **Agent-directed improvement of candidate algorithms**, not training the Agent's own weights or a demonstrated full recursive self-improvement system. The current implementation supports one treatment versus control, one targeting decision, fixed CSV datasets, and offline RCT evaluation. It does not send coupons or establish production uplift.

Start with the [architecture](docs/architecture.md), [dataset research](docs/research/open-uplift-datasets.md), or [contribution guide](CONTRIBUTING.md). Run tests with `uv run --frozen python -m unittest discover -s tests -v`. Code is [Apache-2.0](LICENSE); upstream datasets retain their own terms.
