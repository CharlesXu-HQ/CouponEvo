# Experiment architecture

```mermaid
flowchart LR
    A[Fixed RCT CSV and manifest] --> B[Dataset checks and frozen split]
    B --> C[Agent proposes candidate code]
    C --> D[Isolated candidate evaluation]
    D --> E[Validation report and search journal]
    E --> C
    E --> F[Freeze validation champion]
    F --> G[One final holdout comparison]
```

`src/couponevo/data.py` validates the randomized data and declared pre-treatment features. `search.py` keeps the objective, budget, seed, manifest, evaluator, and candidate snapshots fixed for a resumable search. `agent.py` chooses among draft, improve, debug, and crossover proposals; `sandbox.py` executes generated code in a restricted Docker container. `evaluate.py` and `cli.py` produce policy estimates, cost checks, uncertainty intervals, and reports independently of the proposed candidate.

The journal contains within-search history. `experience.py` may summarize **validation-only** steps from earlier finalized searches when the raw dataset SHA-256, manifest SHA-256, objective, budget, split seed, evaluator hash, and strict-data setting match. A changed CSV or manifest does not inherit that experience. No final holdout metric enters Agent context. A new run may intentionally start without reused experience by omitting `--experience-dir`.

This loop improves *candidate algorithms* through search. It does not update the Agent's weights, prompt policy, or search operator selection from training data, so the repository does not claim full RSI. The current scope is a single treatment versus control, one decision, and offline RCT evaluation; production deployment and real App reactivation data are outside the present implementation.
