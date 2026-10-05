# CouponEvo

[English](README.md) | [简体中文](README.zh-CN.md)

**Agent-led uplift experiments for budgeted coupon targeting.**

A coupon model should answer a causal question: *who will change their behavior because of an offer?* A high purchase probability alone does not answer it. CouponEvo gives an Agent a controlled loop to improve the uplift model **and** the allocation policy on a fixed randomized experiment, then checks whether the resulting policy improves the outcome under a coupon budget.

## Design: an RSI-inspired loop

[OpenRSI](https://github.com/FrontisAI/OpenRSI) connects executable program search, experience, and training of the improver on a path toward recursive self-improvement. CouponEvo takes the **executable search loop** into a narrower domain: an Agent edits an uplift model or coupon allocation policy, runs it against a fixed randomized task, reads independent validation feedback, and decides what to try next. Completed searches can supply experience only for that same dataset and task.

The evolving artifact is the **candidate algorithm**, not the Agent. This version does not train the Agent's weights or its proposal policy, so an improving candidate score is not evidence of full RSI. Keeping the dataset, objective, budget, and evaluator outside Agent-editable code makes proposals comparable. Validation feedback drives exploration; one frozen winner reaches the independent test. The separation matters because repeated validation wins can disappear on the holdout, as the public experiment below shows.

## The experiment contract

| The Agent can change | The runner keeps fixed |
| --- | --- |
| Candidate-side uplift estimation, training, available pretreatment feature transformations, and optional `choose_policy` allocation code | Dataset and manifest hashes, treatment probability, train/validation/test split, objective, budget, and evaluator |
| Read same-task experience and validation history, then diagnose, experiment, request data, or stop; an experiment states a hypothesis and expected result | Independent policy estimates, budget checks, paired comparisons, and the final holdout |
| A `feature_gaps.md` note during an experiment, or a standalone structured data request when a useful pretreatment feature is missing | The dataset itself; missing features are proposed for a future dataset version, never invented for the current experiment |

Generated candidates run in an isolated Docker container. The candidate interface and requested device remain fixed, but the Agent is not tied to the seed's uplift estimator or model structure. After each experiment, the Agent compares the validation result with its expectation and records evidence, a dataset-bound lesson, and a next direction. The next decision receives that reflection, the recent candidate report, prior failures, and optional same-task experience. A diagnosis interprets existing validation evidence without running new candidate code; at most one diagnosis is allowed between experiments. The Agent may stop early. If a proposed feature needs data that the fixed dataset lacks, the Agent can save `feature_request.json` with its definition, source, pretreatment timing, supporting evidence, and validation plan. A human must publish a new dataset version before testing that hypothesis; it starts a new search, not a resume of the old one. Each search retains code snapshots, reports, decisions, and failures for resumption. Optional experience reuse reads **validation history only** from finalized searches with the same dataset bytes, manifest, objective, budget, split seed, and evaluator. Changing the data creates a new experience scope. A reviewed feature-leakage flag, failed report analysis, or invalid reflection prevents a candidate from becoming the champion.

The primary question is **policy value**, not just uplift ranking: the evaluator estimates incremental outcomes from randomized assignments with inverse probability weighting (IPW), compares policies on the same users, and reports paired uncertainty intervals. Qini/AUUC remain ranking diagnostics. After exploration, `finalize` freezes the validation champion and compares it with the seed policy once on the held-out test set, including a paired bootstrap interval.

The paired estimator averages `(policy_new − policy_seed) × [treatment × outcome / p − control × outcome / (1 − p)]` over evaluation users, where `p` is the documented treatment probability.

```text
fixed randomized data → Agent reads experience and validation history
                              ↓
              diagnose / experiment / request data / stop
                              ↓
      experiment → sandboxed training and validation → hypothesis reflection
                              └──── next Agent decision ────┘
                              ↓
       request data → human publishes new dataset → new search
                    stop → freeze candidate → final holdout
```

## What the public experiment showed

On the [84,534-row Starbucks randomized promotion dataset](examples/README.md), an earlier two-step DeepSeek Agent search ran EconML/PyTorch candidates on an RTX 5090. That run preceded the explicit reflection and stop actions described above and used the former T-learner-only search contract. Its objective was incremental purchase conversion under an **assumed** send-cost budget.

| Candidate | Validation IPW incremental conversions per eligible user |
| --- | ---: |
| Seed: T-learner with PyTorch ridge outcome models | 0.003312 |
| Agent proposal: PyTorch MLP outcome models | **0.004140** |
| Agent proposal: MLP ensemble | 0.002957 |

The validation winner's **held-out paired difference from the seed** was `+0.000118` conversions per user; its 95% bootstrap interval was `[-0.001065, +0.001301]`. The interval crosses zero, so this run **does not establish an improvement**. That distinction between a promising search result and a supported final claim is central to the project. See the [full search record](docs/research/agent-search-gpu-2026-10-03.md).

The current open-candidate harness also ran an [EconML S-Learner with a PyTorch CUDA model](docs/research/open-candidate-gpu-validation-2026-10-04.md) on the full Starbucks dataset. This was a deterministic compatibility check, not a new DeepSeek proposal or a performance gain.

Starbucks has neither App reactivation labels nor actual coupon redemption cost or user-level margin. Criteo and X5 RetailHero have also been used for [GPU model checks](docs/research/gpu-validation-2026-10-03.md), not for causal policy-value claims where the published data do not support the required treatment probability.

## Plug in a research harness

Add `--harness harnesses/coupon-research.json` to `search`, `--resume`, and `finalize` to enable the [research harness plugin](docs/harness.md). It supplies categorized, non-exhaustive research guidance, a training-partition feature profile, and a required experiment design covering mechanism, alternatives, available inputs and falsification. The design follows the candidate through reflection and dataset-bound experience. Copy the JSON to add your own directions without modifying the runner. The guide draws on [FunRec](https://github.com/datawhalechina/fun-rec) and adapts recommendation mechanisms to causal coupon experiments; it is not a bundled implementation of every referenced model.

The independent [ModelEvoHarness](https://github.com/CharlesXu-HQ/ModelEvoHarness) project is available as a Git submodule at `third_party/model-evo-harness`. Initialize the submodule and install its package from that checkout into the host environment. Each fresh `search --harness model-evo` automatically updates the submodule to the latest `main` before loading it; `--resume` and `finalize` keep the search's recorded revision. CouponEvo still owns candidate execution and evaluation, and records the external source revision and catalog/implementation digests with the search. See the [integration guide](docs/model-evo-harness.md) and [full-dataset GPU A/B](docs/research/model-evo-harness-starbucks-ab-2026-10-04.md). The diagnostic paired test interval includes zero, so this run does not establish a performance gain.

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

### Evidence before mechanism claims

With ModelEvoHarness, research proposals cite host observations and explain why the next test is worth its budget. Runtime reports record executed preprocessing widths, tensor shapes, devices and losses. Analysis checks the hypothesis against the parent code, the candidate diff and these observations; a contradicted implementation cannot become the champion. Probability-difference proposals expose both potential-outcome predictions for numerical checks, while direct CATE and explicitly named ranking scores keep their own contracts.

These checks do not prove complete feature or gradient lineage. Reflections retain that uncertainty and distinguish joint changes from isolated attribution. Timing or cost concerns can remain nonblocking audit recommendations; an experimental feature request needs measured, trial-specific gap evidence across distinct mechanisms. Invalid proposals at budget exhaustion preserve completed experiments and close the search with the error recorded.

Claims about added model capacity must be checked against the parent's prediction functions. See the [full-data GPU behavior audit](docs/research/harness-evidence-gpu-validation-2026-10-05.md) for the Agent's actual experiment sequence, failed reviews, and paired uncertainty. A promising validation score remains exploratory.

The current implementation supports one treatment versus control, one targeting decision, fixed CSV datasets, and offline RCT evaluation. It does not send coupons or establish production uplift.

Start with the [architecture](docs/architecture.md), [dataset research](docs/research/open-uplift-datasets.md), or [contribution guide](CONTRIBUTING.md). Run tests with `uv run --frozen python -m unittest discover -s tests -v`. Code is [Apache-2.0](LICENSE); upstream datasets retain their own terms.
