#!/usr/bin/env python3
"""Vanilla single-shot LLM baseline for the DS agent failure analysis benchmark.

Run directory is auto-named  runs/vanilla__<model-slug>__<variant>/
unless --run-name is passed explicitly.

Usage:
  python scripts/run_vanilla.py                                  # pilot tasks, zero_shot
  python scripts/run_vanilla.py --variant one_shot               # one-shot variant
  python scripts/run_vanilla.py --variant few_shot               # few-shot + retries
  python scripts/run_vanilla.py --model gemini-2.5-pro           # different model
  python scripts/run_vanilla.py --all                            # all 212 tasks
  python scripts/run_vanilla.py --task titanic                   # single task
  python scripts/run_vanilla.py --resume                         # skip completed tasks
  python scripts/run_vanilla.py --run-name my_custom_name        # override run name
"""

import argparse
import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR   = PROJECT_ROOT / "scripts" / "config"
# DS_DATASETS_DIR lets memorization-alteration runs point at datasets_altered/ (default: unchanged)
DATASETS_ROOT = PROJECT_ROOT / os.environ.get("DS_DATASETS_DIR", "datasets")
SOURCES      = ["strable", "openml", "uci", "kaggle", "mlebench"] + (["altered"] if os.environ.get("DS_DATASETS_DIR") else [])


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_registry() -> dict:
    return json.loads((CONFIG_DIR / "model_registry.json").read_text())


def load_run_config(path: Path) -> dict:
    return json.loads(path.read_text())


def resolve_model_cfg(model_name: str, registry: dict) -> dict:
    if model_name not in registry:
        raise ValueError(f"Model '{model_name}' not in model_registry.json. "
                         f"Available: {list(registry)}")
    cfg = dict(registry[model_name])
    # model_name in the config overrides the registry key (used for proxy aliasing)
    if "model_name" not in cfg:
        cfg["model_name"] = model_name
    if "api_key_env" in cfg:
        env_val = os.environ.get(cfg["api_key_env"], "")
        if not env_val:
            raise EnvironmentError(f"Env var {cfg['api_key_env']} not set for model {model_name}")
        cfg["api_key"] = env_val
    return cfg


def model_slug(model_name: str) -> str:
    return model_name.replace(":", "-").replace("/", "-")


def auto_run_name(model_name: str, variant: str) -> str:
    return f"vanilla__{model_slug(model_name)}__{variant}"


def load_shot_examples(n: int) -> list:
    if n == 0:
        return []
    path = CONFIG_DIR / "shot_examples.json"
    if not path.exists():
        return []
    all_examples = json.loads(path.read_text())
    return all_examples[:n]


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

def build_prompt(task_dir: Path, task_meta: dict,
                 examples: list, supply_data_snippet: bool) -> str:
    desc = (task_dir / "description.md").read_text(encoding="utf-8").strip() \
        if (task_dir / "description.md").exists() else ""
    train_path = (task_dir / "data" / "train.parquet").resolve()
    test_path  = (task_dir / "data" / "test.parquet").resolve()
    target = task_meta["target_column"]
    metric = task_meta.get("metric", "rmse")

    parts = []

    # Few-shot examples
    for i, ex in enumerate(examples, 1):
        parts.append(f"=== EXAMPLE {i} ===")
        parts.append(ex.get("description", "").strip())
        parts.append(f"\nData files:\n  train: {ex['train_path']}\n  test:  {ex['test_path']}")
        parts.append(f"Target column: {ex['target_column']}")
        parts.append(f"Evaluation metric: {ex['metric']}")
        if ex.get("data_snippet"):
            parts.append(f"\nData sample (first 5 rows of train):\n{ex['data_snippet']}")
        if ex.get("column_dtypes"):
            parts.append(f"\nColumn dtypes:\n{ex['column_dtypes']}")
        parts.append(f"\n=== SOLUTION {i} ===\n```python\n{ex['code'].strip()}\n```\n")

    # Current task header
    if examples:
        parts.append("=== YOUR TASK ===")

    parts.append(desc)
    parts.append(f"\nData files:\n  train: {train_path}\n  test:  {test_path}")
    parts.append(f"Target column: {target}")
    parts.append(f"Evaluation metric: {metric}")

    if supply_data_snippet:
        try:
            df = pd.read_parquet(train_path)
            parts.append(f"\nData sample (first 5 rows of train):\n{df.head(5).to_string()}")
            parts.append(f"\nColumn dtypes:\n{df.dtypes.to_string()}")
        except Exception:
            pass

    parts.append(f"""
Write a complete, self-contained Python script that:
1. Loads the train and test data from the parquet files above using pandas
2. Trains a model to predict "{target}"
3. Saves predictions to predictions.csv in the current working directory,
   with exactly two columns: "id" (integer row index starting from 0) and "{target}"

Requirements:
- The script must run without any user input or internet access
- Use only standard libraries plus: pandas, numpy, scikit-learn, lightgbm, xgboost, catboost (all pre-installed)
- Handle missing values and categorical columns automatically
- Do not use pip install or access the internet
- Output only the Python code, no explanation or markdown fences
""")

    return "\n".join(parts).strip()


def build_retry_message(stderr: str, stdout: str, score_error: str | None = None) -> str:
    parts = []

    if stderr and stderr.strip():
        parts.append(f"--- stderr ---\n{stderr[-2000:]}")

    if stdout and stdout.strip():
        parts.append(f"--- stdout ---\n{stdout[-1000:]}")

    if score_error:
        parts.append(f"--- scoring error ---\n{score_error}")

    if not parts:
        parts.append("The code ran without errors but did not produce predictions.csv.")

    error_block = "\n\n".join(parts)
    return (
        f"The code failed. Here is the error output:\n\n{error_block}\n\n"
        "Please fix the code and try again. Output only the corrected Python code, "
        "no explanation or markdown fences."
    )


# ---------------------------------------------------------------------------
# Code extraction
# ---------------------------------------------------------------------------

def extract_code(text: str) -> str:
    # Reasoning models (e.g. Qwen3.5-4B) emit a <think>...</think> preamble
    # that can itself contain illustrative ```python fences from mid-reasoning
    # (abandoned approaches sketched aloud) -- only text after </think> is the
    # actual final answer. No-op for models that never emit this tag.
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    match = re.search(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def call_llm(messages: list, model_cfg: dict, system_prompt: str | None = None) -> tuple[str, float]:
    t0 = time.time()

    # Gemini's base_url in model_registry.json is the native REST endpoint
    # (https://generativelanguage.googleapis.com/), not the OpenAI-compatible
    # shim path — the generic OpenAI client below would 404 against it. Route
    # through the native google.genai SDK instead, same as AIDE/MLEvolve do.
    if model_cfg.get("provider") == "gemini":
        from google import genai
        from google.genai import types

        client = genai.Client(
            api_key=model_cfg.get("api_key", ""),
            http_options={"base_url": model_cfg["base_url"]},
        )
        parts = [system_prompt] if system_prompt else []
        for m in messages:
            role = "Assistant" if m["role"] == "assistant" else "User"
            parts.append(f"{role}: {m['content']}")
        contents = "\n\n".join(parts)
        if "flash" in model_cfg["model_name"].lower():
            thinking_config = types.ThinkingConfig(thinking_budget=0)
        else:
            thinking_config = types.ThinkingConfig(thinking_level="high")
        config = types.GenerateContentConfig(
            temperature=model_cfg.get("temperature", 0.1),
            max_output_tokens=model_cfg.get("max_tokens", 4096),
            thinking_config=thinking_config,
        )
        response = client.models.generate_content(
            model=model_cfg["model_name"],
            contents=contents,
            config=config,
        )
        return response.text, time.time() - t0

    import openai
    client_kwargs = {"base_url": model_cfg["base_url"], "api_key": model_cfg.get("api_key", "EMPTY")}
    if "request_timeout_s" in model_cfg:
        client_kwargs["timeout"] = model_cfg["request_timeout_s"]
    client = openai.OpenAI(**client_kwargs)
    full_messages = []
    if system_prompt:
        full_messages.append({"role": "system", "content": system_prompt})
    full_messages.extend(messages)
    # vLLM's OpenAI-compatible server accepts extra (non-standard) sampling
    # params via extra_body -- e.g. repetition_penalty, which some models
    # (confirmed: Qwen3.5-4B self-hosted via vLLM) need explicitly set to
    # avoid degenerating into an infinite repetition loop under low
    # temperature. Only applied when the registry entry actually sets one of
    # these fields, so this is a no-op for every other model's config.
    extra_body = {k: model_cfg[k] for k in ("repetition_penalty", "presence_penalty", "top_p", "top_k")
                  if k in model_cfg}
    if "openrouter.ai" in model_cfg.get("base_url", ""):
        extra_body["provider"] = {"sort": "price"}
        if "reasoning_max_tokens" in model_cfg:
            extra_body["reasoning"] = {"max_tokens": model_cfg["reasoning_max_tokens"]}
    response = client.chat.completions.create(
        model=model_cfg["model_name"],
        messages=full_messages,
        max_tokens=model_cfg.get("max_tokens", 4096),
        temperature=model_cfg.get("temperature", 0.1),
        extra_body=extra_body or None,
    )
    return response.choices[0].message.content, time.time() - t0


# ---------------------------------------------------------------------------
# Code execution
# ---------------------------------------------------------------------------

def execute_code(code: str, run_dir: Path, timeout: int) -> tuple[str, str, int]:
    with tempfile.NamedTemporaryFile(
        suffix=".py", mode="w", encoding="utf-8", delete=False, dir=run_dir
    ) as f:
        f.write(code)
        tmpfile = f.name
    try:
        proc = subprocess.run(
            [sys.executable, tmpfile],
            capture_output=True, text=True,
            timeout=timeout,
            cwd=str(run_dir),
        )
        return proc.stdout, proc.stderr, proc.returncode
    except subprocess.TimeoutExpired:
        return "", f"TIMEOUT after {timeout}s", -1
    finally:
        try:
            os.unlink(tmpfile)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_predictions(task_dir: Path, run_dir: Path, target_col: str) -> dict:
    pred_path   = run_dir / "predictions.csv"
    labels_path = task_dir / "evaluation" / "labels.parquet"
    metric_path = task_dir / "evaluation" / "metric.py"

    if not pred_path.exists():
        return {"success": False, "error": "predictions.csv not produced"}

    try:
        labels_df = pd.read_parquet(labels_path)
        preds_df  = pd.read_csv(pred_path)
        label_cols = [c for c in labels_df.columns if c != "id"]

        task_meta = json.loads((task_dir / "task.json").read_text())
        if task_meta.get("multi_output"):
            # True multi-output regression: the agent must predict every label
            # column itself (unlike e.g. survival analysis, where multiple
            # label columns exist but only one is actually predicted).
            missing = [c for c in label_cols if c not in preds_df.columns]
            if missing:
                return {"success": False,
                        "error": f"Missing {len(missing)} target columns in predictions.csv, "
                                 f"e.g. {missing[:5]}"}
            merged = labels_df.merge(preds_df, on="id", how="inner", suffixes=("_true", "_pred"))
            if len(merged) == 0:
                return {"success": False, "error": "No rows matched on 'id'"}
            y_true = merged[[c + "_true" for c in label_cols]]
            y_pred = merged[[c + "_pred" for c in label_cols]]

            spec = importlib.util.spec_from_file_location("metric", metric_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            result = module.evaluate(y_true, y_pred)
            result["success"] = True
            result["n_predicted"] = len(merged)
            return result

        if target_col not in preds_df.columns:
            return {"success": False,
                    "error": f"Column '{target_col}' missing from predictions.csv. "
                             f"Found: {list(preds_df.columns)}"}

        preds_slim = preds_df[["id", target_col]].rename(columns={target_col: "_pred"})
        merged = labels_df[["id"] + label_cols].merge(preds_slim, on="id", how="inner")

        if len(merged) == 0:
            return {"success": False, "error": "No rows matched on 'id'"}

        # Multi-column labels (e.g. survival analysis needs event + time)
        y_true = merged[label_cols] if len(label_cols) > 1 else merged[target_col]
        y_pred = merged["_pred"]

        if isinstance(y_true, pd.Series) and isinstance(y_true.dtype, pd.CategoricalDtype):
            cats = y_true.cat.categories
            y_true = y_true.astype(cats.dtype if len(cats) else object)

        # Coerce types if needed (e.g. model output int but labels are str)
        # Only applies for single-column y_true (Series); skip for multi-column (DataFrame)
        if isinstance(y_true, pd.Series) and y_true.dtype == object and y_pred.dtype != object:
            unique_true = sorted(y_true.unique())
            unique_pred = sorted(y_pred.unique())
            if len(unique_true) == len(unique_pred):
                mapping = {p: t for p, t in zip(unique_pred, unique_true)}
                y_pred = y_pred.map(mapping)

        spec = importlib.util.spec_from_file_location("metric", metric_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        result = module.evaluate(y_true, y_pred)
        result["success"] = True
        result["n_predicted"] = len(merged)
        return result

    except Exception as exc:
        return {"success": False, "error": str(exc)}


# ---------------------------------------------------------------------------
# Task runner
# ---------------------------------------------------------------------------

def run_task(task_id: str, source: str, run_cfg: dict,
             model_cfg: dict, variant_cfg: dict, examples: list,
             run_root: Path, task_ts: str | None = None) -> dict:

    task_dir = DATASETS_ROOT / source / task_id
    if not (task_dir / "task.json").exists():
        return {"task_id": task_id, "source": source,
                "success": False, "error": "task.json not found"}

    task_meta  = json.loads((task_dir / "task.json").read_text())
    target_col = task_meta["target_column"]
    max_retries = variant_cfg.get("max_retries", 0)
    supply_snippet = variant_cfg.get("supply_data_snippet", False)
    system_prompt = variant_cfg.get("system_prompt") or None
    timeout = run_cfg.get("timeout_seconds", 600)

    folder_name = f"{task_id}_{task_ts}" if task_ts else task_id
    run_dir = run_root / folder_name
    run_dir.mkdir(parents=True, exist_ok=True)

    log_path = run_dir / "task.log"
    llm_io_path = run_dir / "llm_io.jsonl"

    def _log(msg: str) -> None:
        print(msg, flush=True)
        ts = datetime.now(timezone.utc).isoformat()
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"[{ts}] {msg}\n")

    if system_prompt:
        (run_dir / "system_prompt.txt").write_text(system_prompt, encoding="utf-8")

    t_start = time.time()

    # Build initial prompt and messages
    prompt = build_prompt(task_dir, task_meta, examples, supply_snippet)
    (run_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
    messages = [{"role": "user", "content": prompt}]

    attempts = []
    final_score = None

    for attempt in range(max_retries + 1):
        _log(f"  attempt {attempt + 1}/{max_retries + 1} — calling LLM ...")

        ts_call = datetime.now(timezone.utc).isoformat()
        try:
            response, llm_time = call_llm(messages, model_cfg, system_prompt)
        except Exception as exc:
            _log(f"  LLM error: {exc}")
            result = {
                "task_id": task_id, "source": source,
                "model": model_cfg["model_name"],
                "variant": run_cfg.get("variant", "zero_shot"),
                "success": False, "error": f"LLM error: {exc}",
                "wall_time_s": round(time.time() - t_start, 1),
                "n_attempts": attempt + 1,
            }
            (run_dir / "result.json").write_text(json.dumps(result, indent=2))
            return result

        with open(llm_io_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "timestamp": ts_call,
                "attempt": attempt,
                "model": model_cfg["model_name"],
                "system": system_prompt,
                "messages": messages,
                "response": response,
                "latency_s": round(llm_time, 3),
            }) + "\n")

        if not response:
            _log(f"  LLM returned empty/no content (attempt {attempt + 1}) -- "
                 f"likely exhausted max_tokens on reasoning before answering")
            attempts.append({
                "attempt": attempt,
                "llm_time_s": round(llm_time, 1),
                "returncode": None,
                "success": False,
                "error": "LLM returned empty/no content",
            })
            if attempt < max_retries:
                messages.append({"role": "user",
                                 "content": "Your previous response was empty. Please provide "
                                            "your solution again, making sure to include the "
                                            "final Python code block."})
            continue

        code = extract_code(response)

            # Save per-attempt files
        attempt_dir = run_dir / f"attempt_{attempt}"
        attempt_dir.mkdir(exist_ok=True)
        (attempt_dir / "generated_code.py").write_text(code, encoding="utf-8")

        _log(f"  attempt {attempt + 1}/{max_retries + 1} — executing ...")
        stdout, stderr, returncode = execute_code(code, run_dir, timeout)
        (attempt_dir / "stdout.txt").write_text(stdout, encoding="utf-8")
        (attempt_dir / "stderr.txt").write_text(stderr, encoding="utf-8")

        score = score_predictions(task_dir, run_dir, target_col)
        attempt_rec = {
            "attempt": attempt,
            "llm_time_s": round(llm_time, 1),
            "returncode": returncode,
            **score,
        }
        attempts.append(attempt_rec)

        if score.get("success"):
            final_score = score
            _log(f"  score={score.get('score'):.4f} ({task_meta.get('metric')})")
            break

        _log(f"  FAILED: {score.get('error', '')[:80]}")

        # Prepare retry message
        if attempt < max_retries:
            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "user",
                             "content": build_retry_message(
                                 stderr, stdout,
                                 score_error=score.get("error") if not score.get("success") else None,
                             )})

    # Copy final attempt files to run_dir root for easy inspection
    last_dir = run_dir / f"attempt_{len(attempts) - 1}"
    for fname in ["generated_code.py", "stdout.txt", "stderr.txt"]:
        src = last_dir / fname
        dst = run_dir / fname
        if src.exists() and not dst.exists():
            dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")

    result = {
        "task_id":       task_id,
        "source":        source,
        "model":         model_cfg["model_name"],
        "variant":       run_cfg.get("variant", "zero_shot"),
        "metric":        task_meta.get("metric"),
        "target_column": target_col,
        "n_attempts":    len(attempts),
        "wall_time_s":   round(time.time() - t_start, 1),
        "attempts":      attempts,
        **(final_score or {"success": False,
                           "error": attempts[-1].get("error", "all attempts failed")}),
    }
    (run_dir / "result.json").write_text(json.dumps(result, indent=2))
    return result



OPENROUTER_BATCH_URL = "https://openrouter.ai/api/beta/batches"


def submit_openrouter_batch(api_key: str, model_name: str, entries: list) -> str:
    resp = requests.post(
        OPENROUTER_BATCH_URL,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        json={"endpoint": "/v1/chat/completions", "model": model_name, "requests": entries},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["id"]


def poll_openrouter_batch(api_key: str, batch_id: str, poll_interval: int = 20) -> dict:
    url = f"{OPENROUTER_BATCH_URL}/{batch_id}"
    terminal = {"completed", "failed", "expired", "cancelled"}
    while True:
        resp = requests.get(url, headers={"Authorization": f"Bearer {api_key}"}, timeout=60)
        resp.raise_for_status()
        batch = resp.json()
        status = batch.get("status")
        counts = batch.get("request_counts", {})
        print(f"    batch {batch_id}: {status} "
              f"({counts.get('completed', 0)}/{counts.get('total', '?')} done, "
              f"{counts.get('failed', 0)} failed)", flush=True)
        if status in terminal:
            if status != "completed":
                print(f"    WARNING: batch {batch_id} ended with status={status}")
            return batch
        time.sleep(poll_interval)


def run_batch_tasks(tasks: list, run_cfg: dict, model_cfg: dict, variant_cfg: dict,
                     examples: list, run_root: Path) -> list:
    if not model_cfg.get("base_url", "").startswith("https://openrouter.ai"):
        raise ValueError(
            f"--batch is only supported for OpenRouter-backed models "
            f"(got base_url={model_cfg.get('base_url')!r})"
        )

    api_key = model_cfg["api_key"]
    max_retries = variant_cfg.get("max_retries", 0)
    supply_snippet = variant_cfg.get("supply_data_snippet", False)
    system_prompt = variant_cfg.get("system_prompt") or None
    timeout = run_cfg.get("timeout_seconds", 600)

    state = {}
    for t in tasks:
        task_dir = DATASETS_ROOT / t["source"] / t["id"]
        task_meta = json.loads((task_dir / "task.json").read_text())
        run_dir = run_root / t["id"]
        run_dir.mkdir(parents=True, exist_ok=True)
        if system_prompt:
            (run_dir / "system_prompt.txt").write_text(system_prompt, encoding="utf-8")
        prompt = build_prompt(task_dir, task_meta, examples, supply_snippet)
        (run_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
        state[t["id"]] = {
            "task": t, "task_dir": task_dir, "task_meta": task_meta, "run_dir": run_dir,
            "messages": [{"role": "user", "content": prompt}],
            "attempts": [], "done": False, "final_score": None, "t_start": time.time(),
        }

    for round_idx in range(max_retries + 1):
        pending_ids = [tid for tid, s in state.items() if not s["done"]]
        if not pending_ids:
            break
        print(f"\n=== Batch round {round_idx + 1}/{max_retries + 1}: "
              f"{len(pending_ids)} task(s) ===")

        entries = []
        for tid in pending_ids:
            s = state[tid]
            body_messages = list(s["messages"])
            if system_prompt:
                body_messages = [{"role": "system", "content": system_prompt}] + body_messages
            entries.append({
                "custom_id": tid,
                "body": {
                    "messages": body_messages,
                    "max_tokens": model_cfg.get("max_tokens", 4096),
                    "temperature": model_cfg.get("temperature", 0.1),
                },
            })

        batch_id = submit_openrouter_batch(api_key, model_cfg["model_name"], entries)
        print(f"  submitted batch {batch_id}")
        batch = poll_openrouter_batch(api_key, batch_id)
        results_by_id = {r["custom_id"]: r for r in batch.get("results", [])}

        for tid in pending_ids:
            s = state[tid]
            r = results_by_id.get(tid)
            attempt_idx = len(s["attempts"])
            attempt_dir = s["run_dir"] / f"attempt_{attempt_idx}"
            attempt_dir.mkdir(exist_ok=True)

            status_code = (r or {}).get("response", {}).get("status_code") if r else None
            if r is None or r.get("error") or status_code != 200:
                err = (r or {}).get("error") or f"batch item failed (status={status_code})"
                s["attempts"].append({
                    "attempt": attempt_idx, "llm_time_s": None, "returncode": None,
                    "success": False, "error": f"batch error: {err}",
                })
                print(f"  [{tid}] batch error: {str(err)[:80]}")
                if round_idx == max_retries:
                    s["done"] = True
                continue

            response_text = r["response"]["body"]["choices"][0]["message"]["content"]

            with open(s["run_dir"] / "llm_io.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "attempt": attempt_idx, "model": model_cfg["model_name"],
                    "system": system_prompt, "messages": s["messages"],
                    "response": response_text, "latency_s": None, "batch_id": batch_id,
                }) + "\n")

            code = extract_code(response_text)
            (attempt_dir / "generated_code.py").write_text(code, encoding="utf-8")

            stdout, stderr, returncode = execute_code(code, s["run_dir"], timeout)
            (attempt_dir / "stdout.txt").write_text(stdout, encoding="utf-8")
            (attempt_dir / "stderr.txt").write_text(stderr, encoding="utf-8")

            score = score_predictions(s["task_dir"], s["run_dir"], s["task_meta"]["target_column"])
            s["attempts"].append({
                "attempt": attempt_idx, "llm_time_s": None, "returncode": returncode, **score,
            })

            if score.get("success"):
                s["final_score"] = score
                s["done"] = True
                print(f"  [{tid}] score={score.get('score'):.4f} ({s['task_meta'].get('metric')})")
            else:
                print(f"  [{tid}] FAILED: {str(score.get('error', ''))[:80]}")
                if round_idx < max_retries:
                    s["messages"].append({"role": "assistant", "content": response_text})
                    s["messages"].append({"role": "user", "content": build_retry_message(
                        stderr, stdout,
                        score_error=score.get("error") if not score.get("success") else None,
                    )})
                else:
                    s["done"] = True

    results = []
    for tid, s in state.items():
        if s["attempts"]:
            last_dir = s["run_dir"] / f"attempt_{len(s['attempts']) - 1}"
            for fname in ["generated_code.py", "stdout.txt", "stderr.txt"]:
                src = last_dir / fname
                dst = s["run_dir"] / fname
                if src.exists() and not dst.exists():
                    dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")

        result = {
            "task_id": tid, "source": s["task"]["source"],
            "model": model_cfg["model_name"], "variant": run_cfg.get("variant", "zero_shot"),
            "metric": s["task_meta"].get("metric"), "target_column": s["task_meta"]["target_column"],
            "n_attempts": len(s["attempts"]), "wall_time_s": round(time.time() - s["t_start"], 1),
            "attempts": s["attempts"],
            **(s["final_score"] or {
                "success": False,
                "error": s["attempts"][-1].get("error", "all attempts failed") if s["attempts"] else "no attempts",
            }),
        }
        (s["run_dir"] / "result.json").write_text(json.dumps(result, indent=2))
        results.append(result)
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model",    help="Model name (must be in model_registry.json)")
    parser.add_argument("--variant",  help="Variant: zero_shot | one_shot | few_shot")
    parser.add_argument("--all",      action="store_true", help="Run all 214 tasks")
    parser.add_argument("--task",     help="Run a single task by ID")
    parser.add_argument("--tasks",    action="append", dest="extra_tasks", metavar="TASK_ID",
                        help="Run specific task(s) by ID (repeatable). Combines with --task.")
    parser.add_argument("--resume",    action="store_true",
                        help="Skip tasks that already have a result.json")
    parser.add_argument("--timestamp", action="store_true",
                        help="Append UTC timestamp to each task folder so re-runs don't overwrite old results.")
    parser.add_argument("--batch", action="store_true",
                        help="Use OpenRouter's batch API (~50%% cheaper, up to 24h turnaround). "
                             "OpenRouter-backed models only; not compatible with --timestamp.")
    parser.add_argument("--run-name", help="Override auto-generated run directory name")
    parser.add_argument("--runs-dir", default="runs",
                        help="Parent directory for run outputs (default: runs/)")
    parser.add_argument("--config",   default=str(CONFIG_DIR / "run_config.json"))
    args = parser.parse_args()

    run_cfg  = load_run_config(Path(args.config))
    registry = load_registry()

    model_name = args.model   or run_cfg.get("model",   "qwen2.5-coder:7b")
    variant    = args.variant or run_cfg.get("variant", "zero_shot")

    model_cfg   = resolve_model_cfg(model_name, registry)
    variant_cfg = run_cfg["variants"][variant]
    examples    = load_shot_examples(variant_cfg.get("n_examples", 0))

    run_name = args.run_name or auto_run_name(model_name, variant)
    run_root = PROJECT_ROOT / args.runs_dir / run_name
    run_root.mkdir(parents=True, exist_ok=True)

    print(f"Model  : {model_name}")
    print(f"Variant: {variant}  (n_examples={variant_cfg['n_examples']}, "
          f"max_retries={variant_cfg['max_retries']}, "
          f"data_snippet={variant_cfg['supply_data_snippet']})")
    print(f"Output : {run_root}\n")

    # Determine task list
    task_ids = list(args.extra_tasks or [])
    if args.task:
        task_ids.insert(0, args.task)

    if task_ids:
        tasks = []
        for tid in task_ids:
            found = False
            for src in SOURCES:
                if (DATASETS_ROOT / src / tid / "task.json").exists():
                    tasks.append({"id": tid, "source": src})
                    found = True
                    break
            if not found:
                print(f"Task '{tid}' not found.")
                sys.exit(1)
    elif args.all:
        tasks = []
        for src in SOURCES:
            for p in sorted((DATASETS_ROOT / src).glob("*/task.json")):
                tasks.append({"id": p.parent.name, "source": src})
    else:
        tasks = run_cfg.get("pilot_tasks", [])

    if args.resume:
        before = len(tasks)
        # A machine that sleeps mid-run doesn't pause wall-clock deadlines --
        # time.time() keeps advancing while suspended, so on wake every
        # per-attempt timeout looks instantly, wildly overdue. Legit worst-case
        # is (max_retries + 1) attempts x timeout_seconds each; anything well
        # beyond that is a sleep/suspend artifact, not a real timeout, so
        # don't trust it as "done" even if result.json exists.
        wall_cap = (variant_cfg.get("max_retries", 0) + 1) * run_cfg.get("timeout_seconds", 600)
        suspect_threshold = wall_cap + 600
        n_suspect = 0
        kept = []
        for t in tasks:
            result_path = run_root / t["id"] / "result.json"
            if not result_path.exists():
                kept.append(t)
                continue
            try:
                existing = json.loads(result_path.read_text())
            except Exception:
                kept.append(t)
                continue
            wall_time = existing.get("wall_time_s")
            if wall_time is not None and wall_time > suspect_threshold:
                n_suspect += 1
                kept.append(t)
            elif not existing.get("success"):
                kept.append(t)
        tasks = kept
        suffix = f" ({n_suspect} re-queued as sleep-corrupted)" if n_suspect else ""
        print(f"Resume: {before - len(tasks)} already done, {len(tasks)} remaining{suffix}\n")

    print(f"Running {len(tasks)} tasks\n")

    if args.batch:
        if args.timestamp:
            print("--batch does not support --timestamp; ignoring --timestamp.")
        results = run_batch_tasks(tasks, run_cfg, model_cfg, variant_cfg, examples, run_root)
    else:
        results = []
        for i, t in enumerate(tasks, 1):
            task_ts = _ts() if args.timestamp else None
            print(f"[{i:3d}/{len(tasks)}] {t['source']}/{t['id']}" +
                  (f"  [{task_ts}]" if task_ts else ""))
            try:
                r = run_task(t["id"], t["source"], run_cfg, model_cfg,
                             variant_cfg, examples, run_root, task_ts=task_ts)
            except Exception as exc:
                print(f"  UNEXPECTED ERROR, skipping: {type(exc).__name__}: {exc}", flush=True)
                r = {"task_id": t["id"], "source": t["source"], "success": False,
                     "error": f"unexpected error: {type(exc).__name__}: {exc}"}
            results.append(r)
            print()

    n_ok = sum(1 for r in results if r.get("success"))
    print("=" * 60)
    print(f"Done. {n_ok}/{len(results)} tasks scored successfully.")

    summary_path = run_root / "summary.json"
    summary_path.write_text(json.dumps(results, indent=2))
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
