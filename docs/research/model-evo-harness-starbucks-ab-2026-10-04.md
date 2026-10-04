# ModelEvoHarness A/B on the full Starbucks dataset

This experiment compares CouponEvo's baseline Agent search with the same search using the external ModelEvoHarness research guide. It tests whether the guide changes the Agent's model choices and whether the resulting frozen policy improves purchase conversion on an untouched test split. Both searches requested new data after their allotted experiments, so the test comparison below is **diagnostic**, not a finalized search result.

## Fixed protocol and provenance

| Item | Setting |
| --- | --- |
| Dataset | Full public Starbucks randomized promotion table, 84,534 rows; SHA-256 `4d48190fd0d6a65d3874fa9a9ac79d89007579716366c2cfe140ae999088aa4f` |
| Split | Seed 42; 50,720 train, 16,907 validation, 16,907 test |
| Objective | IPW estimated incremental purchase conversions per eligible user; budget `0.03` assumed coupon cost per eligible user |
| Search | DeepSeek `deepseek-flash`, `thinking=enabled`, iteration effort `high`, review effort `max`; at most three experiments per arm |
| Controlled difference | Only `--harness model-evo` in the guided arm; no `--experience-dir` in either arm |
| Compute | NVIDIA GeForce RTX 5090, PyTorch `2.14.1+cu130`, EconML `0.17.0`, CUDA Docker image `sha256:ec2754fb9f85d20060360fad9165fd272ea3d84c07e9301757c7bb42c62d6095` |
| Source | CouponEvo [`144ee7c`](https://github.com/CharlesXu-HQ/CouponEvo/commit/144ee7c1796d36fc5f82c64e9880f50d77277bbf); ModelEvoHarness submodule [`bc36900`](https://github.com/CharlesXu-HQ/ModelEvoHarness/commit/bc3690079ccebcdb224e7ba653d406b5a731058d) |

Both arms used the same manifest, initial candidate, validation split, objective, budget, Agent configuration, and sandbox image. The whole dataset was split once by the fixed seed; no sample was taken. The dataset and manifest hashes, package identity, and GPU environment are in the [artifacts](artifacts/model-evo-starbucks-ab-2026-10-04/). The configured Agent efforts are recorded in each journal; this report does not independently establish how the provider implemented those settings.

## Search trajectory

The validation score is the estimated conversion effect of the selected policy, not a test result. Both arms started from the same linear PyTorch T-learner score of `0.003312`.

| Arm | Step | Tested direction | Status | Validation score |
| --- | --- | --- | --- | ---: |
| Baseline | 1 | EconML X-learner | Failed: unsupported `random_state` argument in EconML 0.17.0 | — |
| Baseline | 2 | Corrected nonlinear X-learner | Evaluated | 0.002484 |
| Baseline | 3 | R-learner crossover | Evaluated; same policy score as seed | 0.003312 |
| ModelEvoHarness | 1 | Residualized R-learner with a nonlinear PyTorch effect head | Evaluated | 0.005323 |
| ModelEvoHarness | 2 | Shared-trunk nonlinear T-learner with separate arm heads | Evaluated; frozen best | **0.005560** |
| ModelEvoHarness | 3 | Cross-fitted AIPW pseudo-outcome learner | Evaluated; rejected by validation | 0.003312 |

The baseline arm kept its seed. The guided arm kept step 2. Its step 1 and step 2 proposals explicitly compared distinct causal estimators, explained the mechanism and available fields, named alternatives, and gave falsification criteria. Step 3 tested an alternative from the previous reflection and lost on validation. These are observations about this one Agent run; the [journals](artifacts/model-evo-starbucks-ab-2026-10-04/) preserve the proposals, candidate states, scores, and reflections.

Both Agents then requested row-level measurement times for `V1`–`V7` relative to randomized assignment. The guided Agent also asked for realized coupon cost and net margin if available. The manifest only declares features pre-treatment, and this public table uses a fixed `0.15` send-cost assumption; neither timing leakage nor the true cost tradeoff can be resolved from these data. CouponEvo's finalization guard forbids `finalize` when a journal contains a data request, so formal finalization was not attempted for these searches.

## One diagnostic paired test

Before reading the test split, the script froze each journal's validation `best_id`, verified that the task identities differed only by harness, and made one direct paired evaluation of the guided candidate against the baseline seed. The candidate ran in the CUDA sandbox (`model_device=cuda`, peak allocation `84,723,200` bytes); the runner also required CUDA allocation from the comparison candidate. The report has `holdout=test` and `holdout_size=16,907`. The interval resamples the same test users 2,000 times and estimates the paired IPW policy difference.

| Guided minus baseline, conversions per eligible user | Estimate | 95% bootstrap interval |
| --- | ---: | ---: |
| Test split, seed 42 | +0.000946 | [−0.001065, +0.002957] |

The point estimate is +0.0946 percentage points, with an interval from −0.1065 to +0.2957 percentage points. **The interval includes zero: this run does not establish a test-set improvement.** The result is diagnostic because the data requests prevented formal search finalization. It does show that the external harness led this Agent run to try several model mechanisms and select a different candidate on validation.

The table measures purchase conversion under an assumed send cost. Starbucks has no App reactivation label, user-level margin, or realized redemption cost, and pre-assignment feature timing has not been checked per row. A single seed and one stochastic Agent run cannot establish a general performance advantage. Further model selection would require a new untouched holdout or fresh randomized data; the current test split has now been viewed.

## Reproduction artifacts

- [Baseline journal](artifacts/model-evo-starbucks-ab-2026-10-04/plain-journal.json) and [guided journal](artifacts/model-evo-starbucks-ab-2026-10-04/model-evo-journal.json): all three candidate decisions, errors, reflections, and data requests.
- [Frozen guided candidate](artifacts/model-evo-starbucks-ab-2026-10-04/model-evo-best-candidate.py) and [paired CUDA report](artifacts/model-evo-starbucks-ab-2026-10-04/paired-report.json): the evaluated code and full test metrics.
- [Paired summary](artifacts/model-evo-starbucks-ab-2026-10-04/paired-summary.json), [environment](artifacts/model-evo-starbucks-ab-2026-10-04/environment.json), and [input hashes](artifacts/model-evo-starbucks-ab-2026-10-04/input-sha256.txt).
- [Benchmark procedure](../model-evo-harness.md): commands and the predeclared data-request fallback. The archived candidate and journals are evidence from the published source commit above; they are not library defaults.
