# Local-iteration reliability validation — 2026-10-05

## Scope

Fix the failures observed during the earlier compositional GPU validation: model identity coupled to editable descriptions, truncated or malformed Agent responses, short fixed request budgets, and loss of a rejected proposal during repair. This run checks consecutive experiments within a tracked backbone. Cross-backbone migration is outside this validation.

## Frozen implementation

- CouponEvo: `4515d3dd32176acedc06317c2aa0d1aab2ee9ea5`.
- ModelEvoHarness submodule: `508781f107831a810ec2a556c58a0410c52080f9`.
- Stable `estimator_id` / `backbone_id` are separate from editable `estimator` / `backbone` descriptions. Local proposals inherit omitted IDs. Explicit identity changes cannot be labeled local. Legacy records resolve deterministic IDs without rewriting journals.
- Source selection groups by stable identities. Component inheritance and source/runtime verification still apply.
- Request deadlines and stage output budgets come from `examples/agent.deepseek.json`: 600 seconds; proposal 65,536, analysis 20,000, review 32,768, reflection 20,000, diagnosis 10,000 tokens. High analysis and conditional max review remain enabled.
- Response recovery allows up to three attempts, at most one timeout retry. Truncation can increase the token budget to at most twice its initial value (or 32,768 for smaller defaults). Reference reads do not reset recovery. Final provider failure stops outer retries.
- Invalid JSON and nonempty truncated final content are returned to the Agent with the error and full-response contract. No JSON completion is accepted locally. Host validation persists the rejected proposal and exposes it once in retry/resume context. All proposal/code checks still apply.

## Regression evidence

- Harness: 86 targeted tests passed.
- CouponEvo full suite: 227 tests, 226 passed and one CUDA-dependent test skipped on the local host. A subsequent targeted run including the additional truncated-content regression passed 67 tests.
- The prior real `call-009` local ablation is replayed unchanged by `test_recorded_gpu_local_ablation_keeps_identity_despite_changed_description`; it now passes the complete research validator and inherits the parent identity.
- Prior malformed final replies (`call-002`, `call-007`, `call-010`) and truncated `call-008` were replayed through mocked HTTP. The original final text reaches repair context; persistent errors remain rejected within the bounded recovery budget. These tests establish recovery wiring, not an actual provider's likelihood of repairing a reply.
- Independent code review found no remaining blocker after the submodule and test-loader initialization fixes.

## GPU protocol

- Full Starbucks source: 84,534 rows, SHA-256 `4d48190fd0d6a65d3874fa9a9ac79d89007579716366c2cfe140ae999088aa4f`.
- Seed 42; 50,720 training, 16,907 validation, 16,907 test rows. No row sampling; no test evaluation in this run. The test split was used in earlier project work and is not claimed to be a new untouched holdout.
- Fixed conversion objective and cost budget 0.03 per source user; cost 0.15 per treatment is assumed, not an observed coupon-redemption cost.
- RTX 5090; host PyTorch 2.14.1+cu130; Docker image `coupon-lab-sandbox:py312-cuda128`, digest `sha256:ec2754fb9f85d20060360fad9165fd272ea3d84c07e9301757c7bb42c62d6095`.
- Real DeepSeek Agent, `deepseek-flash`, normal search entry point with two candidate slots. No candidate edits, forced backbone switch, provider monkeypatches, or manual proposal acceptance. Passive HTTP/evaluation instrumentation records only final response content and metadata, never reasoning text or credentials.
- A preflight attempt stopped before any API/GPU experiment because the offline Git mirror needed file-protocol permission. The rerun supplies process-scoped Git configuration for the verified local mirror; project sources and request budgets are unchanged.

## Run outcome

The run completed two candidate evaluations, high analysis, independent max review, component reflection and a budget-limited stop. It required **one explicit operator resume after host proposal validation exhausted its two attempts**. It was not an uninterrupted autonomous success. No proposal text or candidate source was manually edited.

| Trial | Change | Policy conversion effect | Paired difference vs its parent (95% CI) | Decision |
| --- | --- | ---: | --- | --- |
| Seed | Existing per-arm affine ridge | 0.00331224 | — | Remained champion |
| step-001 | Initialize per-arm MLP with BCE and probability-difference outputs | 0.00248418 | −0.00082806 [−0.00207662, 0.00042050] vs seed | Eligible; no demonstrated gain |
| step-002 | Local ablation to one affine logit per arm | 0.00343053 | 0.00094635 [−0.00032353, 0.00221624] vs step-001 | Blocked for an implementation/claim mismatch |

These are conversion-policy effects under the assumed-cost task, not observed financial returns. Both paired intervals cross zero. The second comparison is against step-001, not against the seed.

### What was observed

1. **Stable identity worked in the real local edit.** step-001 recorded `estimator_id=t_learner`, `backbone_id=tab_mlp`. The next proposal omitted IDs, changed the backbone description from a two-hidden-layer MLP to a single affine logit, and declared `local`. The host inherited both parent IDs. Component validation, training and review proceeded; description changes did not trigger an identity mismatch.
2. **Malformed JSON recovered automatically.** Calls 006 and 007 returned malformed final JSON despite `finish_reason=stop`; call 008 was valid on the third response attempt. The hypothesis and entire `candidate_py` are byte-for-byte equal across initial proposal 005, repaired proposal 008 and resumed proposal 009. Raw candidate source SHA-256: `3080658d4633bb8a7008e111063a1bfe8695b9c20d2353863b846c31517d3fd9`.
3. **Host errors remained bounded and resumable.** Proposal 005 cited invalid evidence IDs; 008 repaired the evidence but conflicted with `retain` metadata. The host saved each original proposal and stopped at its configured limit. Call 009, after one operator resume, received `rejected_proposal` and `proposal_error`; the Agent corrected the inheritance declarations without changing the candidate or hypothesis.
4. **A higher score did not bypass review.** The second candidate also changed epochs 300→500, patience 25→50 and early-stopping improvement tolerance 1e−6→1e−7, while claiming only the arm network changed. High and max review marked this contradiction; reflection called the experiment invalid for its claimed control. The candidate was excluded from promotion and the seed remained champion.
5. **The loop produced a concrete next experiment.** Its stop message recommends restoring the parent training controls and rerunning the network-only ablation, or explicitly declaring those controls as additional factors. It records feature timing and assumed cost as audit notes, rather than asserting missing features without evidence.

All three evaluated models used CUDA: seed peak allocated 35,390,976 bytes, step-001 71,858,688, step-002 69,100,032. All split manifests have the same hash and cover every source row with disjoint partitions. Thirteen API requests completed; two final replies were malformed JSON. No timeout or `length` response occurred in this run; timeout/truncation behavior is covered by the regression/replay tests above. The longest completed API call took 123.194 seconds.

### Follow-up prompted by this run

Harness commit `702c273` improves the two host errors encountered here: unknown evidence IDs now report a bounded list of offending and available IDs; retain conflicts report the source trial, component and actual mismatched fields. Validation rules are unchanged. This additional error-text improvement passed the 86-test Harness regression set and 104 CouponEvo integration/provider tests after updating the submodule. It was integrated after the GPU run; it was **not** loaded into the frozen run above.

### Limits

- One resume was needed. The evidence does not establish reliable unattended completion across arbitrary long searches.
- No statistically supported gain or clean single-factor architecture benefit was demonstrated.
- This run did not read Harness model reference files and did not test feature-block composition coverage or cross-backbone migration. It validates the specific identity and response-recovery changes.
- Experience remains bound to this dataset/task. The next hypothesis has been proposed but not evaluated in this two-slot run.

[Machine-readable evidence](local-iteration-reliability-gpu-validation-2026-10-05.json) includes the final provider replies, both research designs, component assessments, execution diagnostics, policy metrics, error records and split hashes. API reasoning text and credentials are excluded.
