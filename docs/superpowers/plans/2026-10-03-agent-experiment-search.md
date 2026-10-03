# Agent-led experiment search

## Scope

Add a bounded, resumable offline search over the existing fixed dataset and evaluator. The Agent chooses an experiment operator and writes a complete candidate; the runner, objective, budget, and final test remain fixed. This is test-time search, not training the Agent model.

## Files and interfaces

- `src/coupon_lab/search.py`: `run_search(...)` owns the journal, candidate snapshots, parent context, provisional champion, failure recovery, and resume. A proposal callable is the seam for real API and deterministic tests.
- `src/coupon_lab/agent.py`: `propose_search_candidate(...)` asks the configured provider for a structured hypothesis, operator, parent IDs, and candidate source. Validate the returned structure and the existing EconML/PyTorch candidate contract before execution.
- `src/coupon_lab/cli.py`: add `search` with fixed objective, step cap, search ID, resume, and existing provider options. Run candidate evaluations in separate processes on the requested device. Add a separate `finalize` command for the frozen champion. Never pass `--final` during search.
- `tests/test_search.py`: test multiple steps, selection, failure retention, resume without repeated proposal, unchanged task configuration, and API proposal validation.
- `README.md`: document the command, artifacts, interpretation limits, and process-isolation limit.

## Execution order and verification

1. Write failing search tests for journal, next-step context, and selection. Implement the smallest loop. Verify targeted tests.
2. Write failing tests for proposal schema and retry. Implement API proposal. Verify targeted tests.
3. Write failing CLI tests for `search`, resume, and held-out test protection. Wire the entrypoint and subprocess execution. Verify the complete suite.
4. Run the fixed public Starbucks experiment on the target GPU with a bounded Agent search. Check CUDA reports and journal ancestry; compare the frozen champion once on the final test only if the search is complete.

## Interpretation

The search score is exploratory because the Agent reuses feedback. Report a final improvement only from a frozen candidate on the independent test set. Public Starbucks has conversion and an assumed send cost, so it cannot establish App activation or real net value.

Separate-process execution bounds a crashed or stalled candidate; it is not a security sandbox. Container-level file and network isolation is required before giving generated code access to sensitive business data.
