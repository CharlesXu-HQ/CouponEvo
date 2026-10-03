# CouponEvo

[English](README.md) | [简体中文](README.zh-CN.md)

CouponEvo is an Agent-guided offline experiment runner for cost-aware coupon targeting. The name pairs *coupon* with the evolution of candidate algorithms across experiments. For a fixed randomized dataset, the Agent proposes model or targeting-code changes, the runner evaluates them under a frozen objective and budget, and a final holdout compares the chosen candidate with the seed policy.

## Status

- Implemented: an EconML `TLearner` candidate with PyTorch outcome models; Agent-led `draft`, `improve`, `debug`, and `crossover` proposals; resumable search; isolated candidate execution; paired policy comparisons and bootstrap intervals.
- Experience is tied to the dataset bytes and evaluation task. Search steps use their own validation history; optional reuse reads only finalized searches with matching dataset SHA-256, manifest, objective, budget, split seed, and evaluator version. Changed data starts a new experience scope.
- The Agent itself is **not trained or improved** by this project. This is an iterative algorithm search loop, not a demonstrated full recursive self-improvement (RSI) system. OpenRSI's broader training loop is described in its [project README](https://github.com/FrontisAI/OpenRSI).
- No production targeting, online coupon delivery, or business-data validation is included. The public datasets do not establish App reactivation or real coupon profit.

## Quick start

Python 3.12+, [uv](https://docs.astral.sh/uv/), and the [Starbucks randomized promotion CSV](examples/README.md) are required. Place the CSV at `data/starbucks-training.csv` as described in the dataset guide. For GPU runs, install a CUDA-capable PyTorch build and use a Linux CUDA host.

```bash
uv sync --frozen --python 3.12
uv run --frozen python -m couponevo.cli run examples/starbucks.json \
  --budget-kind cost --budget 0.03 --seed 42 --device cuda
```

To let an API Agent search, set `AGENT_API_KEY` in your shell, then build and pin the sandbox image:

```bash
docker build -f Dockerfile.sandbox -t couponevo-sandbox:py312-cuda128 .
export SANDBOX_IMAGE="$(docker image inspect couponevo-sandbox:py312-cuda128 --format '{{.Id}}')"
uv run --frozen python -m couponevo.cli search examples/starbucks.json \
  --budget-kind cost --budget 0.03 --objective conversion --seed 42 \
  --max-steps 3 --search-id starbucks-01 --device cuda \
  --agent-config examples/agent.deepseek.json --sandbox-image "$SANDBOX_IMAGE"
```

The example config uses an OpenAI-compatible provider URL, model name, and API-key environment-variable name. It requests `high` reasoning for proposals and analysis, with `max` review when configured anomaly checks trigger. Your endpoint must support these settings. Keys are read from the environment; do not put them in tracked files.

After the search, freeze its validation champion and run the independent test once:

```bash
uv run --frozen python -m couponevo.cli finalize examples/starbucks.json \
  --budget-kind cost --budget 0.03 --objective conversion --seed 42 \
  --search-id starbucks-01 --device cuda --bootstrap-reps 2000 \
  --sandbox-image "$SANDBOX_IMAGE"
```

Reports and candidate snapshots go under `runs/`, which is ignored by Git. See the [Chinese guide](README.zh-CN.md) for data contracts, costs, feature timing, sandbox behavior, and interpretation of intervals.

## Repository layout

| Path | Purpose |
| --- | --- |
| `src/couponevo/` | Dataset validation, candidate model, Agent, search, evaluation, sandbox, and task-bound experience |
| `examples/` | Public-data manifests, frozen task specs, and Agent configuration; no raw data |
| `tests/` | Unit and integration tests |
| `docs/` | Business data contract and historical experiment records |
| `.github/workflows/` | Continuous integration |

## Reproduce and contribute

```bash
uv run --frozen python -m unittest discover -s tests -v
```

Read [CONTRIBUTING.md](CONTRIBUTING.md) before proposing changes. Project code is [Apache-2.0 licensed](LICENSE); datasets have separate terms and are not redistributed here. Historical GPU results are in [docs/research](docs/research), including a **superseded cross-dataset experience experiment**. Current experience reuse is dataset-bound.
