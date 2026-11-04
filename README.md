# The Silent Failures in Agentic Data Science

[![Paper](https://img.shields.io/badge/arXiv-2610.14115-b31b1b.svg)](https://arxiv.org/pdf/2610.14115)

This repository provides the code, prompts, and released outputs needed to reproduce the agentic data-science failure-mode study presented in [arXiv:2610.14115](https://arxiv.org/pdf/2610.14115). The study evaluates coding agents (AIDE, MLEvolve, and a zero-shot ReAct baseline) on tabular ML tasks and uses an LLM judge to evaluate their generated pipelines against a 15-mode failure taxonomy.

**What's included:**
* Run scripts for the agents and the LLM judge.
* Scripts for the re-usable testing suite.
* The LLM judge prompts, structure schema, and failure-mode taxonomy criteria.
* Dataset fetching scripts.
* Analysis and figure-generation scripts.
* Final generated pipelines and LLM judge results for the runs (`outputs/`).

**What's not included:**
* **Raw benchmark datasets:** Fetch scripts are provided instead. 
* **Raw run outputs (logs, agent transcripts, raw predictions):** Raw results are large.

## Repository Structure

```text
scripts/
  config/                  Model registry, task lists, few-shot examples
  patches/                 Patches for AIDE / MLEvolve (for Qwen-27B runs)
  fetch_*.py               Dataset pull scripts (Kaggle, OpenML, UCI, STRABLE, etc.)
  run_*.py                 Agent runners (AIDE, MLEvolve, Vanilla/ReAct) for the main sweep
  dsts_*.py, dsts_run_*    Adapters for running agents over the curated test suite
  judge_spec.py            Failure-mode taxonomy, structure schema, gates, and judge prompts
  judge_common.py          Shared judge I/O (prompt assembly, result loading)
  run_judge.py             LLM judge runner
  analyze_memorization_*.py, probe_*.py, select_memorization_tasks.py, build_altered_tasks.py
                           Scripts for the memorization check (Table 5)
  run_*_hsc.py, run_autogluon_hsc.py
                           Scripts for Kaggle first-place comparisons (Table 4)
  visualization/           Data loaders and scripts for figures (plot_*.py) and tables (build_*.py)
analysis/hsc/              Task list for Kaggle first-place comparisons
judge_criteria/            Test-suite per-test judge eval criteria (TF01-TF06)
dsts_leakfree_data/        Leak-free data for the TF04 and TF06 test-suite tasks
outputs/                   Released pipelines, judge verdicts, and human annotations
docs/                      Dataset registry, source metadata, and licensing
requirements*.txt          Python dependencies
```

## Released Outputs (`outputs/`)

The `outputs/` directory contains the final pipelines and LLM-judge (v3) verdicts for the main runs. This covers 3 agents (ReAct, AIDE, MLEvolve) and 4 models (Gemini-3.6-flash, Gemini-3.7-flash, Qwen3.5-4B, Qwen3.8-27B) across 220 tabular tasks. 

* **`outputs/pipelines/<agent>/<model>/<task>.py`**: The final solution code exactly as the judge saw it. 
* **`outputs/judge_verdicts/<agent>/<model>/<task>.json`**: The complete judge output (per-failure-mode verdict, severity, reasoning, evidence).
* **`outputs/judge_verdicts.csv`**: A flattened overview with one row per judged pipeline. 
* **`outputs/dsts_judge_results/`**: Per-check judge results for the curated failure-mode test suite (used in Figs. 3-4).
* **`outputs/validation_study/`**: The `human_annotations.csv` and `llm_judge_verdicts.csv` used for the agreement table (Table 9). Annotator notes and identities are excluded. 

## Setup

Install the required dependencies:

```bash
pip install -r requirements.txt            # Core: judge, analysis, figures
pip install -r requirements-data.txt       # Datasets: only needed for fetch_*.py
pip install -r requirements-agents.txt     # Agents: only needed to run agents
```

### Environment Variables

Configure these based on the models and paths you intend to use. See `scripts/config/model_registry.json` for the complete mapping.

| Variable | Description |
|---|---|
| `GOOGLE_API_KEY` | Required for Gemini models |
| `OPENAI_API_KEY` | Required for GPT models |
| `OPENROUTER_API_KEY` | Required for Qwen or other OpenRouter models |
| `GEMINI_JUDGE_MODEL_ID` / `OPENAI_JUDGE_MODEL_ID` / `GEMINI_MODEL_ID` | Override default model IDs |
| `MLEVOLVE_ROOT` | Path to the MLEvolve clone (default: `external/agents/MLEvolve`) |
| `DSTS_ROOT` | Path to the DSTS test suite checkout (default: `./dsts_suite`) |
| `DS_DATASETS_DIR` | Override the default `datasets/` download directory |

### Kaggle Credentials
To fetch Kaggle datasets (`scripts/fetch_kaggle.py`), you need a `~/.kaggle/kaggle.json` file or the `KAGGLE_USERNAME` and `KAGGLE_KEY` environment variables. Some competitions require to accept the terms first.

### External Agents
The agents frameworks are not included in this repo. Install them locally before running the pipeline:
* **AIDE**: Run `pip install aideml==0.2.2`. This specific pinned version is imported by `run_aide.py`.
* **MLEvolve**: Clone the repository (`git clone https://github.com/InternScience/MLEvolve.git`), checkout commit `7d8403c899c40f01941c0429f1c4ef51e82ae41c`, and point `MLEVOLVE_ROOT` to it. Follow their repo's setup instructions.

## Reproducing the Pipeline

1. **Fetch Datasets:** Run the relevant fetch scripts (`fetch_kaggle.py`, `fetch_openml.py`, etc.) to populate your local `datasets/` directory. 
2. **Configure Models:** Add your target models to `scripts/config/model_registry.json` and export the corresponding API keys.
3. **Run the Agents:** Execute `run_aide.py`, `run_mlevolve.py`, or `run_vanilla.py` for the main sweep. For the DSTS hint-level probe suite, use the `dsts_run_*.py` equivalents.
4. **Run the Judge:** Execute `run_judge.py` to evaluate the agent outputs. 
5. **Regenerate Figures and Tables:** Use the scripts in `scripts/visualization/` to parse the verdicts and output LaTeX tables to `results/` and figures to `analysis/figures/`.

## Credits & License

* **Code License:** MIT (`LICENSE`)
* **Outputs License:** CC BY 4.0 (`outputs/LICENSE`)
* **Data License:** Derived datasets (like Transfermarkt) retain their original licenses. See `docs/` for specifics. 
* **Test Suite:** The curated failure-mode test suite (DSTS) was developed by Hongqian. This repository adapts it and releases the corresponding judge results.

## Citation

If you use this code, the released outputs, or the test suite, please cite the paper:

```bibtex
@article{li2026silent,
  title         = {The Silent Failures in Agentic Data Science},
  author        = {Li, Xue and Xia, Hongqian and Gomm, Daniel and Santos, A{\'e}cio and Wolff, Cornelius and Hulsebos, Madelon},
  year          = {2026},
  eprint        = {2610.14115},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI},
  url           = {https://arxiv.org/abs/2610.14115}
}
```
