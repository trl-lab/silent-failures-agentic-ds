# DSTS: curated failure-mode test suite

Subset of the full test-suite repository (reference withheld for anonymous review), included so the `scripts/dsts_*` adapters run without a separate checkout.

Included: task workloads (`src/agents/dsguru/workload/*.json`), per-task input data (`data/*/input`), metric scripts (`scripts/fm0*_metric.py`, `loader.py`), and the judge (`llmjudge/judge.py`).
Not included: agent implementations, raw/processed source datasets, and thesis material.
