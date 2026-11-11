"""
LLM-as-a-judge for analyzing pipeline logs (dsguru or dsstar).

Run from the llmjudge/ directory:

  python judge.py \\
    --scratch_dir ../src/agents/dsguru/system_scratch/BaselineLLMSystemGPT4oOneShot \\
    --criteria judge_criteria/FM04_balance_torque_eval.json

  python judge.py \\
    --scratch_dir ../src/agents/dsstar/runs \\
    --criteria judge_criteria/FM04_balance_torque_eval.json \\
    --filter FM04_balance_torque --verbose

  # Use a specific model:
  python judge.py ... --model gemini-2.0-flash
"""

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from openai import OpenAI

# Load .env from project root (two levels up from this file)
load_dotenv(Path(__file__).parent.parent / ".env")


JUDGE_SYSTEM_PROMPT = (
    "You are an expert ML code reviewer evaluating data science pipeline implementations. "
    "You will be given a Python pipeline script and its execution output. Read the code carefully "
    "and answer specific questions about the implementation using the record_analysis tool. "
    "Base your answers only on what is explicitly visible in the code and output — do not infer "
    "or assume anything not directly observable."
)


CHAR_BUDGET = 16000

# Path substrings ranked by relevance (lower index = higher priority)
_PRIORITY_PATTERNS = [
    "final_output",   # dsstar: final JSON result
    "answer",         # dsguru: answer.json — load before intermediate files
    "_intermediate",  # dsguru: pipeline code + output JSON
    "exec_env",       # dsstar: executed Python scripts
]


def _file_priority(path: Path, root: Path) -> int:
    rel = str(path.relative_to(root)).lower()
    for i, pattern in enumerate(_PRIORITY_PATTERNS):
        if pattern in rel:
            return i
    return len(_PRIORITY_PATTERNS)


def _deduplicate_intermediate_files(candidates: list[Path], task_dir: Path) -> list[Path]:
    """For dsguru multi-retry pipelines, keep only the newest of each repeated file type.

    When a pipeline retries, each attempt N writes pipeline-N_out.json and pipeline-N.py.
    Earlier iterations contain stale metrics and superseded code that confuse the judge.
    Keeping only the newest of each type eliminates conflicting values and frees budget
    for the files that actually matter (answer.json and the final pipeline output).
    """
    def is_intermediate_out(p: Path) -> bool:
        rel = str(p.relative_to(task_dir))
        return "_intermediate" in rel and p.name.endswith("_out.json")

    def is_intermediate_pipeline_py(p: Path) -> bool:
        rel = str(p.relative_to(task_dir))
        return "_intermediate" in rel and p.suffix == ".py" and p.stem.startswith("pipeline-")

    result = list(candidates)

    out_jsons = [p for p in result if is_intermediate_out(p)]
    if len(out_jsons) > 1:
        newest = max(out_jsons, key=lambda p: p.stat().st_mtime)
        result = [p for p in result if not is_intermediate_out(p) or p == newest]

    pipeline_pys = [p for p in result if is_intermediate_pipeline_py(p)]
    if len(pipeline_pys) > 1:
        newest_py = max(pipeline_pys, key=lambda p: p.stat().st_mtime)
        result = [p for p in result if not is_intermediate_pipeline_py(p) or p == newest_py]

    return result


def _has_final_answer(path: Path) -> bool:
    """Return True if the file references 'final_answer' (dsstar output scripts)."""
    try:
        return "final_answer" in path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return False


def _is_error_file(path: Path) -> bool:
    """Return True for error/warning log files (e.g. errors-0.txt) that should be loaded last."""
    return path.name.startswith("errors") and path.suffix == ".txt"


def load_task_files_auto(task_dir: Path, char_budget: int = CHAR_BUDGET) -> dict[str, str]:
    """Scan .py, .json, and .txt files, rank by relevance, fill up to char_budget."""
    candidates = (
        list(task_dir.rglob("*.py"))
        + list(task_dir.rglob("*.json"))
        + list(task_dir.rglob("*.txt"))
    )
    candidates = _deduplicate_intermediate_files(candidates, task_dir)
    candidates.sort(key=lambda p: (
        _file_priority(p, task_dir),
        len(p.relative_to(task_dir).parts),
        not _has_final_answer(p),  # final_answer scripts first within same priority group
        _is_error_file(p),         # error/warning logs last within same group
        -p.stat().st_mtime,
    ))

    result = {}
    remaining = char_budget
    for path in candidates:
        if remaining <= 0:
            break
        try:
            content = path.read_text(encoding="utf-8", errors="ignore").strip()
        except Exception:
            continue
        if not content:
            continue
        key = str(path.relative_to(task_dir))
        result[key] = content[:remaining]
        remaining -= len(result[key])

    return result


def build_context(task_id: str, files: dict[str, str]) -> str:
    parts = [f"Task ID: {task_id}"]
    for rel_path, content in files.items():
        parts.append(f"=== {rel_path} ===\n{content}")
    return "\n\n".join(parts)


def build_tool(criteria: list[dict]) -> dict:
    """Build an OpenAI function tool definition from criteria."""
    return {
        "type": "function",
        "function": {
            "name": "record_analysis",
            "description": "Record the structured analysis results for this pipeline",
            "parameters": {
                "type": "object",
                "properties": {c["key"]: c["schema"] for c in criteria},
                "required": [c["key"] for c in criteria],
            },
        },
    }


def judge_task(
    client: OpenAI,
    task_id: str,
    files: dict[str, str],
    criteria: list[dict],
    model: str,
) -> dict[str, Any]:
    context = build_context(task_id, files)
    tool = build_tool(criteria)

    questions = "\n".join(f"- {c['key']}: {c['question']}" for c in criteria)
    user_content = (
        f"Analyze this ML pipeline and answer each question using the record_analysis tool.\n\n"
        f"Questions to answer:\n{questions}\n\n"
        f"---\n{context}"
    )

    response = client.chat.completions.create(
        model=model,
        stream=False,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user",   "content": user_content},
        ],
        tools=[tool],  # type: ignore[arg-type]
        tool_choice={"type": "function", "function": {"name": "record_analysis"}},  # type: ignore[arg-type]
    )

    message = response.choices[0].message
    if message.tool_calls:
        arguments = json.loads(message.tool_calls[0].function.arguments)
        arguments["_input_tokens"]  = response.usage.prompt_tokens if response.usage else 0
        arguments["_output_tokens"] = response.usage.completion_tokens if response.usage else 0
        return arguments

    return {"_error": "no tool_call in response"}


def serialize_for_csv(value: Any) -> str:
    if isinstance(value, list):
        return json.dumps(value, ensure_ascii=False)
    if value is None:
        return ""
    return str(value)


def main():
    parser = argparse.ArgumentParser(description="LLM-as-a-judge for pipeline logs")
    parser.add_argument(
        "--config", default=None,
        help="Path to YAML config file (e.g. config.yaml). CLI args override config values.",
    )
    parser.add_argument("--scratch_dir", default=None,
        help="Path to directory containing per-task subdirectories")
    parser.add_argument("--criteria", default=None,
        help="Path to JSON criteria file")
    parser.add_argument("--output", default=None,
        help="Output CSV path. Defaults to judge_results/<criteria_name>.csv")
    parser.add_argument("--subfolder", default=None,
        help="Subfolder under judge_results/ for the output CSV (e.g. GPT4oFewShot)")
    parser.add_argument("--filter", default=None,
        help="Only process task directories whose name contains this substring")
    parser.add_argument("--model", default=None,
        help="Model to use (default: gemini-2.0-flash)")
    parser.add_argument("--verbose", action="store_true",
        help="Print full results for each task")
    args = parser.parse_args()

    # Load config file and apply as defaults (CLI args take precedence)
    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        if args.scratch_dir is None: args.scratch_dir = cfg.get("scratch_dir")
        if args.criteria    is None: args.criteria    = cfg.get("criteria")
        if args.output      is None: args.output      = cfg.get("output")
        if args.subfolder   is None: args.subfolder   = cfg.get("subfolder")
        if args.filter      is None: args.filter      = cfg.get("filter")
        if args.model       is None: args.model       = cfg.get("model")
        if not args.verbose:         args.verbose      = bool(cfg.get("verbose", False))

    if args.model is None:
        args.model = "gpt-4.1"

    if not args.scratch_dir or not args.criteria:
        parser.error("--scratch_dir and --criteria are required (via CLI or config file)")

    # Load criteria
    criteria_path = Path(args.criteria)
    with open(criteria_path) as f:
        criteria_config = json.load(f)
    criteria: list[dict] = criteria_config["criteria"]
    print(f"Loaded {len(criteria)} criteria from {criteria_path.name}")

    # Default output path
    if args.output:
        output_path = Path(args.output)
    elif args.subfolder:
        output_path = Path("judge_results") / args.subfolder / f"{criteria_path.stem}.csv"
    else:
        output_path = Path("judge_results") / f"{criteria_path.stem}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Discover task directories
    scratch_dir = Path(args.scratch_dir)
    task_dirs = sorted(d for d in scratch_dir.iterdir() if d.is_dir())
    if args.filter:
        task_dirs = [d for d in task_dirs if args.filter in d.name]
    print(f"Found {len(task_dirs)} task(s) to judge in {scratch_dir}")

    if args.model.startswith("gemini") and os.environ.get("GEMINI_API_KEY"):
        client = OpenAI(
            api_key=os.environ["GEMINI_API_KEY"],
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
    else:
        # OpenAI / UVA proxy (reads OPENAI_API_KEY and OPENAI_BASE_URL from env)
        client = OpenAI(base_url=os.environ.get("OPENAI_BASE_URL"))

    criteria_keys = [c["key"] for c in criteria]
    token_keys = ["_input_tokens", "_output_tokens"]
    fieldnames = ["task_id"] + criteria_keys + token_keys

    rows: list[dict] = []
    total_input = total_output = 0

    for task_dir in task_dirs:
        task_id = task_dir.name
        print(f"  Judging {task_id}...", end=" ", flush=True)

        files = load_task_files_auto(task_dir)
        if not files:
            print("SKIP (no files)")
            continue

        result = judge_task(client, task_id, files, criteria, model=args.model)
        result["task_id"] = task_id

        inp = result.get("_input_tokens", 0)
        out = result.get("_output_tokens", 0)
        total_input += inp
        total_output += out
        print(f"in={inp} out={out}")

        if args.verbose:
            for k in criteria_keys:
                print(f"    {k}: {result.get(k)}")

        rows.append(result)

    if not rows:
        print("No results to save.")
        return

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: serialize_for_csv(row.get(k)) for k in fieldnames})

    print(f"\nResults saved to {output_path}")
    print(f"Total tokens — input: {total_input}, output: {total_output}")

    # Quick summary table
    print("\n--- Summary ---")
    print(f"{'task_id':<35}" + "".join(f"{k:<22}" for k in criteria_keys))
    for row in rows:
        line = f"{row['task_id']:<35}"
        for k in criteria_keys:
            v = row.get(k, "")
            if isinstance(v, list):
                v = f"[{len(v)} features]"
            line += f"{str(v):<22}"
        print(line)


if __name__ == "__main__":
    main()