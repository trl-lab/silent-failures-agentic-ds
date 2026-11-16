#!/usr/bin/env bash
# Runs the DSTS unit-test suite across all 3 of our agents (vanilla,
# AIDE, MLEvolve) and all 6 TF0X failure modes, for whichever models you
# list below, then judges every run with her own unmodified judge.py and
# computes the final per-failure-mode scores with her own fm0X_metric.py.
#
# Edit the MODELS/WORKLOADS/AGENTS arrays below to narrow what runs.
#
# Usage:
#   ./scripts/dsts_run_all.sh              # generate + judge + score everything below
#   ./scripts/dsts_run_all.sh --skip-run   # skip generation, just (re-)judge + score existing runs
#   ./scripts/dsts_run_all.sh --skip-judge # only generate, skip judging (e.g. to run judge later once a key is ready)
set -euo pipefail
cd "$(dirname "$0")/.."   # project root

MODELS=(
    "gemini/gemini-3.6-flash"
    # "openrouter/claude-opus-5"
    # "qwen2.5-coder:14b"
)
WORKLOADS=(
    TF01_goal_imbalance
    TF02_player_rep_norm
    TF03_correlation_causation
    TF04_balance_torque_leakfree   # leak-free data and prompts (dsts_leakfree_data/), as reported in the paper
    TF05_credit_card_leakage
    TF06_metric_select_leakfree
)
AGENTS=(vanilla aide mlevolve)   # each has its own dsts_run_<agent>.py
AIDE_STEPS=5
MLEVOLVE_STEPS=5
REPEATS=1   # set >1 for real pass-rate stats (e.g. 5) -- each task runs REPEATS independent
            # times as <task_id>_rep1..N. NOTE: with REPEATS>1, the test suite's own fm0X_metric.py
            # scoring step below is skipped -- its variant-index parsing assumes exactly one
            # run per task_id and would misparse the _repN suffix. Use the project's own
            # binary-checks analysis (docs/dsts_*_results.md generation) instead, which reads
            # judge.py's raw per-task CSVs directly and doesn't care about repeat suffixes.

# Judge model (defaults to Gemini).
# judge.py needs GEMINI_API_KEY specifically (not GOOGLE_API_KEY):
export GEMINI_API_KEY="${GEMINI_API_KEY:-$GOOGLE_API_KEY}"
JUDGE_MODEL="gemini-3.6-flash"

SKIP_RUN=false
SKIP_JUDGE=false
for arg in "$@"; do
    case "$arg" in
        --skip-run) SKIP_RUN=true ;;
        --skip-judge) SKIP_JUDGE=true ;;
    esac
done

model_slug() { echo "$1" | tr ':/' '--'; }

# Model name -> judge_results subfolder label (readable, no slashes)
judge_subfolder() {
    local agent="$1" model="$2"
    echo "${agent^}_$(model_slug "$model")"
}

if [ "$SKIP_RUN" = false ]; then
    for model in "${MODELS[@]}"; do
        for workload in "${WORKLOADS[@]}"; do
            echo ""
            echo "=================================================================="
            echo "RUN  workload=$workload  model=$model"
            echo "=================================================================="
            for agent in "${AGENTS[@]}"; do
                echo "--- agent=$agent ---"
                case "$agent" in
                    vanilla)
                        python3 scripts/dsts_run_vanilla.py --workload "$workload" --model "$model" --repeats "$REPEATS"
                        ;;
                    aide)
                        python3 scripts/dsts_run_aide.py --workload "$workload" --model "$model" --steps "$AIDE_STEPS" --repeats "$REPEATS"
                        ;;
                    mlevolve)
                        python3 scripts/dsts_run_mlevolve.py --workload "$workload" --model "$model" --steps "$MLEVOLVE_STEPS" --repeats "$REPEATS"
                        ;;
                esac
            done
        done
    done
fi

# Project root (where this is run from) and the DSTS test-suite checkout (override via env).
PROJECT_DIR="$PWD"
DSTS_ROOT="${DSTS_ROOT:-$PROJECT_DIR/dsts_suite}"

if [ "$SKIP_JUDGE" = false ]; then
    for model in "${MODELS[@]}"; do
        for workload in "${WORKLOADS[@]}"; do
            criteria="$PROJECT_DIR/judge_criteria/${workload%_leakfree}_eval.json"
            if [ ! -f "$criteria" ]; then
                echo "[SKIP judge] criteria not found: $criteria"
                continue
            fi
            for agent in "${AGENTS[@]}"; do
                scratch_dir="$PROJECT_DIR/dsts_runs/${agent}/$(model_slug "$model")"
                subfolder="$(judge_subfolder "$agent" "$model")"
                [[ "$workload" == *_leakfree ]] && subfolder="Leakfree_$subfolder"
                echo ""
                echo ">>> judge  agent=$agent  workload=$workload  model=$model"
                (
                    cd "$DSTS_ROOT/llmjudge"
                    python3 judge.py --scratch_dir "$scratch_dir" \
                        --criteria "$criteria" --filter "$workload" \
                        --subfolder "$subfolder" --model "$JUDGE_MODEL"
                )
            done
        done
    done

    # Per-FM metric scripts -- fm01 first (creates/resets the shared table.csv),
    # the rest append. Not all TF0X have a metric.py yet in the test-suite repo;
    # skip quietly if missing. Skipped entirely when REPEATS>1 -- see note above.
    if [ "$REPEATS" -gt 1 ]; then
        echo ""
        echo "REPEATS=$REPEATS -- skipping fm0X_metric.py (variant-index parsing assumes 1 run/task)."
        echo "Use the project's own binary-checks analysis on the raw judge_results/ CSVs instead."
    else
        echo ""
        echo "=================================================================="
        echo "SCORING"
        echo "=================================================================="
        for i in 01 02 03 04 05 06; do
            script_rel="scripts/fm${i}_metric.py"
            if [ -f "$DSTS_ROOT/$script_rel" ]; then
                # fm0X_metric.py's own default paths (llmjudge/judge_results,
                # llmjudge/table.csv) are relative to ITS project root, not ours --
                # must actually run it from there, not just reference it by path.
                (cd "$DSTS_ROOT" && python3 "$script_rel") \
                    || echo "[WARN] fm${i}_metric.py failed, continuing"
            fi
        done
        echo ""
        echo "Final table: $DSTS_ROOT/llmjudge/table.csv"
    fi
fi
