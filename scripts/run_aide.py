#!/usr/bin/env python3
"""AIDE agent runner for the DS agent failure analysis benchmark.

Run directory: aide_runs/aide__<model-slug>__<steps>steps/<task_id>/

Usage:
  python scripts/run_aide.py --task equity-post-HCT-survival-predictions
  python scripts/run_aide.py --tasks store-sales-time-series-forecasting --tasks house-prices-advanced-regression-techniques
  python scripts/run_aide.py --hsc                            # all HSC tasks
  python scripts/run_aide.py --all                            # all benchmark tasks
  python scripts/run_aide.py --model qwen2.5-coder:14b       # model (must be in model_registry.json)
  python scripts/run_aide.py --steps 10                      # tree-search iterations
  python scripts/run_aide.py --resume                        # skip already-completed tasks
  python scripts/run_aide.py --runs-dir aide_runs            # output root (default: aide_runs/)
"""

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT   = Path(__file__).resolve().parent.parent
CONFIG_DIR     = PROJECT_ROOT / "scripts" / "config"
# DS_DATASETS_DIR lets memorization-alteration runs point at datasets_altered/ (default: unchanged)
DATASETS_ROOT  = PROJECT_ROOT / os.environ.get("DS_DATASETS_DIR", "datasets")
SOURCES        = ["strable", "openml", "uci", "kaggle", "mlebench"] + (["altered"] if os.environ.get("DS_DATASETS_DIR") else [])
HSC_TASKS_FILE = PROJECT_ROOT / "analysis" / "hsc" / "hsc_tasks.json"

# Python binary inside the conda env where aideml is installed.
# Called directly to avoid conda run overhead and rich TUI hangs.
# sys.executable, not a hardcoded path -- always resolves to whatever
# interpreter is actually running this script, so it works regardless of
# where conda/miniconda happens to be installed on a given machine (local
# Mac dev vs. Snellius's module-provided Miniconda, different install paths).
AIDE_PYTHON = sys.executable

DEFAULT_MODEL = "qwen2.5-coder:14b"
DEFAULT_STEPS = 10
WALL_TIMEOUT  = 7200  # per-task wall-clock cap in seconds, default for fast hosted APIs


def resolve_wall_timeout_s(model_cfg: dict) -> int:
    """Per-model override (model_registry.json's time_cap_hours.aide) if present, else the module
    default.
    """
    override_h = (model_cfg.get("time_cap_hours") or {}).get("aide")
    return int(override_h * 3600) if override_h else WALL_TIMEOUT

# Data scientist persona prepended to every task description.
# Aligns AIDE with our vanilla few_shot_role system prompt.
DS_PERSONA = (
    "You are an experienced data scientist working on a predictive modelling competition. "
    "You follow best practices: always create a validation split to estimate generalisation "
    "performance, clean data before modelling, choose metrics appropriate for the task, "
    "and avoid any form of data leakage. Write clean, well-structured Python code."
)


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_registry() -> dict:
    return json.loads((CONFIG_DIR / "model_registry.json").read_text())


def resolve_model_cfg(model_name: str, registry: dict) -> dict:
    if model_name not in registry:
        raise ValueError(
            f"Model '{model_name}' not in model_registry.json. "
            f"Available: {list(registry)}"
        )
    cfg = dict(registry[model_name])
    if "model_name" not in cfg:
        cfg["model_name"] = model_name
    if "api_key_env" in cfg:
        env_val = os.environ.get(cfg["api_key_env"], "")
        if not env_val:
            raise EnvironmentError(
                f"Env var {cfg['api_key_env']} not set for model {model_name}"
            )
        cfg["api_key"] = env_val
    return cfg


def model_slug(model_name: str) -> str:
    return model_name.replace(":", "-").replace("/", "-")


def load_hsc_task_ids() -> list[str]:
    data = json.loads(HSC_TASKS_FILE.read_text())
    return [t["id"] for t in data["tasks"]]


# ---------------------------------------------------------------------------
# Task description builder
# ---------------------------------------------------------------------------

def build_task_desc(task_dir: Path, task_meta: dict) -> str:
    """
    Build the plaintext task description that AIDE's agent receives.
    AIDE adds its own implementation guidelines on top of this, so we focus on
    task context, persona, data paths, and output format.
    """
    desc = ""
    if (task_dir / "description.md").exists():
        desc = (task_dir / "description.md").read_text(encoding="utf-8").strip()

    target = task_meta["target_column"]
    metric = task_meta.get("metric", "rmse")
    higher = task_meta.get("metric_higher_is_better", True)
    direction = "higher is better" if higher else "lower is better"

    lines = [DS_PERSONA, ""]

    if desc:
        lines += [desc, ""]

    lines += [
        f"**Target column**: `{target}`",
        f"**Evaluation metric**: `{metric}` ({direction})",
        "",
        "**Data**: Train data is at `./input/train.parquet` and test data at "
        "`./input/test.parquet`. Read them with `pd.read_parquet()`.",
        "",
        "**Installed package versions** (use APIs compatible with these):",
        "- `lightgbm==4.6.0`: `early_stopping_rounds` and `verbose_eval` were removed "
        "from `lgb.train()`. Use callbacks: "
        "`callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)]`",
        "- `scikit-learn==1.5.0`, `pandas==2.1.4`",
        "",
        "**Output format**: Save test predictions as `submission.csv` in `./working/` "
        f"with exactly two columns: `id` (integer row index starting from 0) and "
        f"`{target}` (predicted values).",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Post-process AIDE output → our predictions.csv
# ---------------------------------------------------------------------------

def find_submission(workspace_dir: Path, log_dir: Path) -> Path | None:
    """Find submission.csv — first in workspace, then in logs (copied by bootstrap)."""
    matches = sorted(workspace_dir.glob("*/working/submission.csv"))
    if matches:
        return matches[-1]
    direct = workspace_dir / "working" / "submission.csv"
    if direct.exists():
        return direct
    # Fallback: bootstrap copies it to logs dir when workspace survives
    log_matches = sorted(log_dir.glob("*/submission.csv"))
    return log_matches[-1] if log_matches else None


def post_process_submission(submission_csv: Path, target_col: str,
                             task_dir: Path | None = None) -> pd.DataFrame | None:
    """
    Convert AIDE's submission.csv to our standard predictions.csv format:
      id (int, 0-based) | <target_col> (predicted value)

    For multi-output tasks (task.json "multi_output": true), instead expects
    id + every label column from evaluation/labels.parquet, since there is no
    single target_col to rename to.
    """
    try:
        df = pd.read_csv(submission_csv)
    except Exception as e:
        print(f"  Could not read submission.csv: {e}", flush=True)
        return None

    if df.empty or len(df.columns) < 2:
        print(
            f"  submission.csv has unexpected shape {df.shape}, "
            f"columns: {list(df.columns)}",
            flush=True,
        )
        return None

    if task_dir is not None:
        task_meta = json.loads((task_dir / "task.json").read_text())
        if task_meta.get("multi_output"):
            labels_df = pd.read_parquet(task_dir / "evaluation" / "labels.parquet")
            expected_cols = [c for c in labels_df.columns if c != "id"]
            if "id" not in df.columns:
                df.insert(0, "id", range(len(df)))
            missing = [c for c in expected_cols if c not in df.columns]
            if missing:
                print(f"  submission.csv missing {len(missing)} expected columns, "
                      f"e.g. {missing[:5]}", flush=True)
                return None
            return df[["id"] + expected_cols]

    # Prefer exact target name match, then common synonyms, then column[1]
    pred_col = None
    for c in df.columns:
        if c.lower() == target_col.lower():
            pred_col = c
            break
    if pred_col is None:
        for c in df.columns[1:]:
            if c.lower() in ("prediction", "pred", "score", "target", "label", "output"):
                pred_col = c
                break
    if pred_col is None:
        pred_col = df.columns[1]

    return pd.DataFrame({"id": range(len(df)), target_col: df[pred_col].values})


# ---------------------------------------------------------------------------
# Scoring (mirrors run_vanilla.py)
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
            return {
                "success": False,
                "error": (
                    f"Column '{target_col}' missing from predictions.csv. "
                    f"Found: {list(preds_df.columns)}"
                ),
            }

        preds_slim = preds_df[["id", target_col]].rename(columns={target_col: "_pred"})
        merged = labels_df[["id"] + label_cols].merge(preds_slim, on="id", how="inner")

        if len(merged) == 0:
            return {"success": False, "error": "No rows matched on 'id'"}

        y_true = merged[label_cols] if len(label_cols) > 1 else merged[target_col]
        y_pred = merged["_pred"]

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
        result["success"]     = True
        result["n_predicted"] = len(merged)
        return result

    except Exception as exc:
        return {"success": False, "error": str(exc)}


# ---------------------------------------------------------------------------
# AIDE bootstrap script (run inside the ds_failures Python env)
# ---------------------------------------------------------------------------

def _bootstrap_script(data_dir: Path, desc_file: Path, model: str,
                      steps: int, log_dir: Path, workspace_dir: Path,
                      max_tokens: int = 16384, request_timeout_s: int | None = None) -> str:
    """
    Generate a self-contained Python script that invokes AIDE via its Python API.
    Runs the agent loop with plain print() progress instead of rich's TUI.
    """
    llm_log_path = log_dir / "llm_io.jsonl"
    return f"""\
import sys, os, re, multiprocessing, json as _json, time as _time
from pathlib import Path
from datetime import datetime, timezone as _timezone
from omegaconf import OmegaConf

# Patch AIDE's OpenAI backend to always use chat completions when OPENAI_BASE_URL
# is set. By default AIDE routes gpt-* models to the Responses API at the
# hardcoded api.openai.com, which breaks when using an OpenAI-compatible proxy.
#
# AIDE's own request/response logging (aide/backend/backend_openai.py) never
# fires for two reasons: this proxy replaces that function entirely, and
# AIDE's logger is configured at WARNING level by default anyway (see
# aide/utils/config.py) so its INFO-level "OpenAI API request/response" logs
# are dropped even when the real backend runs. So we log LLM I/O ourselves.
import openai as _openai
import aide.backend as _aide_backend
import aide.backend.backend_openai as _aide_be
import aide.backend.backend_gemini as _aide_be_gemini

_LLM_LOG_PATH = {str(llm_log_path)!r}

def _parse_qwen_xml_tool_call(text, func_spec):
    if not text or "<tool_call>" not in text:
        return None
    m = re.search(r"<function=([^>]+)>(.*?)</function>", text, re.DOTALL)
    if not m:
        return None
    props = (getattr(func_spec, "json_schema", None) or {{}}).get("properties", {{}})
    params = {{}}
    for pm in re.finditer(r"<parameter=([^>]+)>\\n?(.*?)\\n?</parameter>", m.group(2), re.DOTALL):
        key, raw_val = pm.group(1), pm.group(2)
        ptype = (props.get(key) or {{}}).get("type")
        if ptype == "number":
            try: raw_val = float(raw_val)
            except ValueError: pass
        elif ptype == "integer":
            try: raw_val = int(raw_val)
            except ValueError: pass
        elif ptype == "boolean":
            raw_val = raw_val.strip().lower() == "true"
        params[key] = raw_val
    return params


def _proxy_openai_query(system_message=None, user_message=None,
                        func_spec=None, **kw):
    base_url = os.environ.get("OPENAI_BASE_URL")
    api_key  = os.environ.get("OPENAI_API_KEY")
    kw = {{k: v for k, v in kw.items() if v is not None}}
    # Responses API uses max_output_tokens; chat API uses max_tokens
    if "max_output_tokens" in kw:
        kw["max_tokens"] = kw.pop("max_output_tokens")
    # AIDE's own agent.py never passes max_tokens to query() for any model --
    # left unset, providers default to the model's own ceiling (65536 for
    # Claude Opus 5 via OpenRouter), which 402s the instant the account's
    # remaining balance can't cover that worst-case allocation regardless of
    # prompt size. Default it here to our registry's configured value.
    kw.setdefault("max_tokens", {max_tokens!r})
    _client_kwargs = dict(api_key=api_key, base_url=base_url, max_retries=5)
    if {request_timeout_s!r} is not None:
        _client_kwargs["timeout"] = {request_timeout_s!r}
    client = _openai.OpenAI(**_client_kwargs)
    if system_message is not None and user_message is None:
        system_message, user_message = user_message, system_message
    msgs = []
    if system_message: msgs.append({{"role": "system", "content": system_message}})
    if user_message:   msgs.append({{"role": "user",   "content": user_message}})
    if func_spec is not None:
        kw["tools"]       = [func_spec.as_openai_tool_dict]
        kw["tool_choice"] = func_spec.openai_tool_choice_dict
    ts_start = datetime.now(_timezone.utc).isoformat()
    t0 = _time.time()
    # Note: OpenRouter's 402 in_flight_budget_exhausted isn't
    # in the OpenAI SDK's default retryable-status set (only 408/409/429/5xx
    # are), so max_retries alone doesn't cover it -- confirmed 4/15 failures
    # in the same retry batch were exactly this 402, each carrying a
    # Retry-After header telling us precisely how long to wait. Retry it
    # explicitly here, honoring Retry-After when present.
    _max_402_retries = 5
    for _attempt_402 in range(_max_402_retries + 1):
        try:
            resp = client.chat.completions.create(messages=msgs, **kw)
            break
        except _openai.APIStatusError as _e:
            if _e.status_code != 402 or _attempt_402 >= _max_402_retries:
                raise
            _wait_s = 120
            try:
                _wait_s = int(_e.response.headers.get("Retry-After", 120))
            except Exception:
                pass
            print("402 in_flight_budget_exhausted, waiting %ds (attempt %d/%d) ..."
                  % (_wait_s, _attempt_402 + 1, _max_402_retries), flush=True)
            _time.sleep(_wait_s)
    req_time = _time.time() - t0
    msg = resp.choices[0].message
    if func_spec and getattr(msg, "tool_calls", None):
        output = _json.loads(msg.tool_calls[0].function.arguments)
    elif func_spec and msg.content and "<tool_call>" in msg.content:
        output = _parse_qwen_xml_tool_call(msg.content, func_spec)
        if output is None:
            output = msg.content or ""
    else:
        output = msg.content or ""
    try:
        with open(_LLM_LOG_PATH, "a") as _f:
            _f.write(_json.dumps({{
                "timestamp": ts_start,
                "model": kw.get("model"),
                "system": system_message,
                "user": user_message,
                "response": output if isinstance(output, str) else _json.dumps(output),
                "latency_s": round(req_time, 3),
                "prompt_tokens": resp.usage.prompt_tokens,
                "completion_tokens": resp.usage.completion_tokens,
            }}) + "\\n")
    except Exception:
        pass
    return output, req_time, resp.usage.prompt_tokens, resp.usage.completion_tokens, {{}}

_aide_be.query = _proxy_openai_query
_aide_backend.provider_to_query_func["openai"] = _proxy_openai_query

# AIDE's own installed backend_gemini.py talks to Gemini via the
# OpenAI-compatible shim endpoint -- confirmed (raw curl, bypassing AIDE
# entirely) that this project's current GOOGLE_API_KEY gets a 401
# UNAUTHENTICATED / ACCESS_TOKEN_TYPE_UNSUPPORTED from that endpoint for
# BOTH gemini-3.6-flash and gemini-3.7-flash -- not a per-model issue.
# vanilla/run_vanilla.py and MLEvolve both authenticate successfully against
# the SAME key right now via the native google.genai SDK (a different auth
# path), so this replaces AIDE's Gemini backend entirely with that same
# native-SDK approach rather than trying to fix the shim's auth. Mirrors
# run_vanilla.py's own native-Gemini branch (client construction, contents
# building) so it's a proven-working pattern, not a new one; adds
# func_spec support on top since AIDE (unlike vanilla) needs structured
# JSON output for several of its calls, via Gemini's own response_schema
# support rather than emulating OpenAI-style tool calls.
from google import genai as _genai
from google.genai import types as _genai_types

def _proxy_gemini_query(system_message=None, user_message=None, func_spec=None, **kw):
    ts_start = datetime.now(_timezone.utc).isoformat()
    t0 = _time.time()
    model_name = kw.get("model")
    try:
        client = _genai.Client(
            api_key=os.environ.get("GOOGLE_API_KEY", ""),
            http_options={{"base_url": "https://generativelanguage.googleapis.com/"}},
        )
        parts = []
        if system_message: parts.append(f"System: {{system_message}}")
        if user_message:   parts.append(f"User: {{user_message}}")
        contents = "\\n\\n".join(parts)

        config_kwargs = {{
            "temperature": kw.get("temperature", 0.1),
            "max_output_tokens": kw.get("max_tokens") or 4096,
        }}
        if func_spec is not None:
            config_kwargs["response_mime_type"] = "application/json"
            config_kwargs["response_schema"] = func_spec.json_schema
        config = _genai_types.GenerateContentConfig(**config_kwargs)

        response = client.models.generate_content(
            model=model_name, contents=contents, config=config,
        )
        req_time = _time.time() - t0

        # Same defensive None-guard as the OpenAI/OpenRouter proxy above --
        # response.text can be None (no text part, e.g. empty candidates or
        # a safety block) rather than raising, which would otherwise crash
        # AIDE's regex-based extraction the same way.
        _resp_text = response.text or ""
        output = _json.loads(_resp_text) if func_spec is not None else _resp_text

        usage = getattr(response, "usage_metadata", None)
        in_tok = getattr(usage, "prompt_token_count", 0) or 0
        out_tok = getattr(usage, "candidates_token_count", 0) or 0
        info = {{"model": model_name}}

        try:
            with open(_LLM_LOG_PATH, "a") as _f:
                _f.write(_json.dumps({{
                    "timestamp": ts_start,
                    "backend": "gemini",
                    "model": model_name,
                    "system": system_message,
                    "user": user_message,
                    "response": output if isinstance(output, str) else _json.dumps(output),
                    "latency_s": round(req_time, 3),
                    "prompt_tokens": in_tok,
                    "completion_tokens": out_tok,
                }}) + "\\n")
        except Exception:
            pass
        return output, req_time, in_tok, out_tok, info
    except Exception as exc:
        try:
            with open(_LLM_LOG_PATH, "a") as _f:
                _f.write(_json.dumps({{
                    "timestamp": ts_start,
                    "backend": "gemini",
                    "model": model_name,
                    "system_len": len(system_message) if system_message else 0,
                    "user_len": len(user_message) if user_message else 0,
                    "system_preview": (system_message or "")[:2000],
                    "user_preview": (user_message or "")[:2000],
                    "error": str(exc),
                    "latency_s": round(_time.time() - t0, 3),
                }}) + "\\n")
        except Exception:
            pass
        raise

_aide_be_gemini.query = _proxy_gemini_query
_aide_backend.provider_to_query_func["gemini"] = _proxy_gemini_query

# Must guard with __main__ check: macOS 'spawn' start method re-imports this
# script in each child process, which would re-run the whole agent loop.
if __name__ == "__main__":
    multiprocessing.set_start_method("fork")  # avoid spawn re-import issues on macOS

    from aide.utils.config import _load_cfg, prep_cfg, load_task_desc, prep_agent_workspace, save_run
    from aide.journal import Journal
    from aide.agent import Agent
    from aide.interpreter import Interpreter

    cfg = _load_cfg(use_cli_args=False)
    cfg.data_dir      = {str(data_dir)!r}
    cfg.desc_file     = {str(desc_file)!r}
    cfg.log_dir       = {str(log_dir)!r}
    cfg.workspace_dir = {str(workspace_dir)!r}
    cfg.agent.code.model     = {model!r}
    cfg.agent.feedback.model = {model!r}
    cfg.agent.steps          = {steps}
    cfg.generate_report  = False
    cfg.preprocess_data  = False
    cfg.copy_data        = False
    cfg = prep_cfg(cfg)

    print(f"Experiment: {{cfg.exp_name}}", flush=True)
    print("Preparing workspace ...", flush=True)
    prep_agent_workspace(cfg)

    task_desc   = load_task_desc(cfg)
    journal     = Journal()
    agent       = Agent(task_desc=task_desc, cfg=cfg, journal=journal)
    interpreter = Interpreter(cfg.workspace_dir, **OmegaConf.to_container(cfg.exec))

    for i in range({steps}):
        print(f"Step {{i+1}}/{steps} ...", flush=True)
        agent.step(exec_callback=interpreter.run)
        save_run(cfg, journal)
        best = journal.get_best_node(only_good=False)
        if best and best.metric and best.metric.value is not None:
            print(f"  best so far: {{best.metric.value:.4f}}", flush=True)

    best_node = journal.get_best_node(only_good=False)
    if best_node and best_node.code:
        sol_path = Path(cfg.log_dir) / "best_solution.py"
        sol_path.parent.mkdir(parents=True, exist_ok=True)
        sol_path.write_text(best_node.code)
        print(f"Best solution saved: {{sol_path}}", flush=True)
        if best_node.metric and best_node.metric.value is not None:
            print(f"Best metric: {{best_node.metric.value:.4f}}", flush=True)
        submission_matches = sorted(
            Path(cfg.workspace_dir).glob("*/working/submission.csv"),
            key=lambda p: p.stat().st_mtime,
        )
        if submission_matches:
            import shutil
            shutil.copy(submission_matches[-1], sol_path.parent / "submission.csv")
            print(f"submission.csv saved to logs dir", flush=True)
        else:
            print("WARNING: best_node found but no submission.csv in any node workspace", flush=True)
    else:
        print("No valid solution found.", flush=True)
        sys.exit(1)

    interpreter.cleanup_session()
"""


# ---------------------------------------------------------------------------
# Task runner
# ---------------------------------------------------------------------------

def run_task(task_id: str, source: str, model_cfg: dict,
             steps: int, run_root: Path) -> dict:
    task_dir = DATASETS_ROOT / source / task_id
    if not (task_dir / "task.json").exists():
        return {"task_id": task_id, "source": source,
                "success": False, "error": "task.json not found"}

    task_meta  = json.loads((task_dir / "task.json").read_text())
    target_col = task_meta["target_column"]
    data_dir   = (task_dir / "data").resolve()

    run_dir = run_root / task_id
    run_dir.mkdir(parents=True, exist_ok=True)

    aide_log_dir  = run_dir / "aide_logs"
    aide_work_dir = run_dir / "aide_workspace"
    if aide_log_dir.exists():
        shutil.rmtree(aide_log_dir)
    if aide_work_dir.exists():
        shutil.rmtree(aide_work_dir)
    aide_log_dir.mkdir(exist_ok=True)
    aide_work_dir.mkdir(exist_ok=True)
    for _stale_name in ("best_solution.py", "submission.csv", "predictions.csv"):
        _stale_f = run_dir / _stale_name
        if _stale_f.exists():
            _stale_f.unlink()

    # Write task description for AIDE's desc_file parameter
    task_desc = build_task_desc(task_dir, task_meta)
    desc_file = run_dir / "task_desc.md"
    desc_file.write_text(task_desc, encoding="utf-8")

    model_name = model_cfg["model_name"]
    base_url   = model_cfg.get("base_url", "http://localhost:11434/v1")
    api_key    = model_cfg.get("api_key", "EMPTY")

    # Write a self-contained bootstrap that uses AIDE's Python API directly.
    # This avoids conda run overhead and rich's alternate-screen TUI hang.
    bootstrap = run_dir / "_aide_bootstrap.py"
    bootstrap.write_text(
        _bootstrap_script(data_dir, desc_file, model_name, steps, aide_log_dir, aide_work_dir,
                          max_tokens=model_cfg.get("max_tokens") or 16384,
                          request_timeout_s=model_cfg.get("request_timeout_s")),
        encoding="utf-8",
    )

    env = os.environ.copy()
    env["OPENAI_BASE_URL"] = base_url
    env["OPENAI_API_KEY"]  = api_key

    wall_timeout_s = resolve_wall_timeout_s(model_cfg)
    print(f"  Running AIDE ({steps} steps, model={model_name}) ...", flush=True)
    t_start = time.time()

    output_lines = []
    returncode   = -1
    proc = subprocess.Popen(
        [str(AIDE_PYTHON), str(bootstrap)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )

    def _drain_stdout():
        for line in proc.stdout:
            print(f"    {line}", end="", flush=True)
            ts = datetime.now(timezone.utc).isoformat()
            output_lines.append(f"[{ts}] {line}" if line.strip() else line)

    reader = threading.Thread(target=_drain_stdout, daemon=True)
    reader.start()

    try:
        proc.wait(timeout=wall_timeout_s)
        returncode = proc.returncode
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)
        output_lines.append(f"\nTIMEOUT after {wall_timeout_s}s\n")
        returncode = -1
    reader.join(timeout=5)

    wall_time = round(time.time() - t_start, 1)
    (run_dir / "aide_stdout.txt").write_text("".join(output_lines), encoding="utf-8")

    if returncode != 0:
        print(f"  AIDE exited with code {returncode}", flush=True)

    # Locate and copy AIDE artefacts
    best_solutions = sorted(aide_log_dir.glob("**/best_solution.py"))
    if best_solutions:
        shutil.copy(best_solutions[-1], run_dir / "best_solution.py")

    submission = find_submission(aide_work_dir, aide_log_dir)
    if submission:
        shutil.copy(submission, run_dir / "submission.csv")
        preds_df = post_process_submission(submission, target_col, task_dir)
        if preds_df is not None:
            preds_df.to_csv(run_dir / "predictions.csv", index=False)
    else:
        print("  No submission.csv found in AIDE workspace.", flush=True)

    score = score_predictions(task_dir, run_dir, target_col)
    if score.get("success"):
        print(f"  score={score.get('score'):.4f} ({task_meta.get('metric')})", flush=True)
    else:
        print(f"  FAILED: {score.get('error', '')[:80]}", flush=True)

    result = {
        "task_id":       task_id,
        "source":        source,
        "agent":         "aide",
        "model":         model_name,
        "aide_steps":    steps,
        "metric":        task_meta.get("metric"),
        "target_column": target_col,
        "wall_time_s":   wall_time,
        "returncode":    returncode,
        **score,
    }
    (run_dir / "result.json").write_text(json.dumps(result, indent=2))
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def collect_tasks(args) -> list[dict]:
    if args.hsc:
        task_ids = load_hsc_task_ids()
    elif args.all:
        tasks = []
        for src in SOURCES:
            src_dir = DATASETS_ROOT / src
            if src_dir.exists():
                for td in sorted(src_dir.iterdir()):
                    if (td / "task.json").exists():
                        tasks.append({"id": td.name, "source": src})
        return tasks
    else:
        task_ids = list(args.extra_tasks or [])
        if args.task:
            task_ids.insert(0, args.task)
        if not task_ids:
            return []

    tasks = []
    for tid in task_ids:
        found = False
        for src in SOURCES:
            if (DATASETS_ROOT / src / tid / "task.json").exists():
                tasks.append({"id": tid, "source": src})
                found = True
                break
        if not found:
            print(f"Task '{tid}' not found in any dataset source.")
            sys.exit(1)
    return tasks


def main():
    ap = argparse.ArgumentParser(description="Run AIDE on benchmark tasks")
    ap.add_argument("--task",     metavar="TASK_ID", help="Single task ID")
    ap.add_argument("--tasks",    action="append", dest="extra_tasks", metavar="TASK_ID",
                    help="Task ID (repeatable)")
    ap.add_argument("--hsc",      action="store_true", help="Run all HSC tasks")
    ap.add_argument("--all",      action="store_true", help="Run all benchmark tasks")
    ap.add_argument("--model",    default=DEFAULT_MODEL,
                    help=f"Model name from model_registry.json (default: {DEFAULT_MODEL})")
    ap.add_argument("--steps",    type=int, default=DEFAULT_STEPS,
                    help=f"AIDE tree-search iterations (default: {DEFAULT_STEPS})")
    ap.add_argument("--runs-dir", default="aide_runs",
                    help="Parent directory for run outputs (default: aide_runs/)")
    ap.add_argument("--resume",   action="store_true",
                    help="Skip tasks that already have a successful result.json")
    args = ap.parse_args()

    registry  = load_registry()
    model_cfg = resolve_model_cfg(args.model, registry)
    slug      = model_slug(args.model)
    run_root  = PROJECT_ROOT / args.runs_dir / f"aide__{slug}__{args.steps}steps"

    tasks = collect_tasks(args)
    if not tasks:
        ap.error("Specify --task, --tasks, --hsc, or --all")

    wall_timeout_s = resolve_wall_timeout_s(model_cfg)

    print(f"AIDE runner — model={args.model}, steps={args.steps}, tasks={len(tasks)}")
    print(f"Run root: {run_root}\n")

    results = []
    for i, t in enumerate(tasks, 1):
        print(f"[{i}/{len(tasks)}] {t['id']} ({t['source']})", flush=True)

        if args.resume:
            result_path = run_root / t["id"] / "result.json"
            if result_path.exists():
                existing = json.loads(result_path.read_text())
                wall_time = existing.get("wall_time_s")
                # A machine that sleeps mid-run doesn't pause wall-clock deadlines --
                # time.time() keeps advancing while suspended, so on wake every
                # timeout looks instantly, wildly overdue. Legit timeout-kills
                # cluster tightly at WALL_TIMEOUT + ~0-40s (thread-join overhead);
                # anything well beyond that is a sleep/suspend artifact, not a
                # real timeout, so don't trust it even if it's marked "success".
                suspect = wall_time is not None and wall_time > wall_timeout_s + 600
                if suspect:
                    print(f"  Existing result has wall_time_s={wall_time:.0f}s, "
                          f"over {wall_timeout_s + 600}s -- likely a sleep/suspend "
                          f"artifact rather than a real timeout. Re-running instead "
                          f"of trusting it.", flush=True)
                elif existing.get("success"):
                    print(f"  Skipping (score={existing.get('score')})", flush=True)
                    results.append(existing)
                    print()
                    continue

        r = run_task(t["id"], t["source"], model_cfg, args.steps, run_root)
        results.append(r)
        print()

    successes = [r for r in results if r.get("success")]
    print(f"=== Done: {len(successes)}/{len(results)} successful ===")
    for r in results:
        if r.get("success"):
            print(f"  {r['task_id']}: score={r['score']:.4f} ({r.get('metric')})")
        else:
            print(f"  {r['task_id']}: FAILED — {r.get('error', '')[:60]}")

    run_root.mkdir(parents=True, exist_ok=True)
    summary = run_root / "results_summary.json"
    summary.write_text(json.dumps(results, indent=2))
    print(f"\nSummary saved: {summary}")


if __name__ == "__main__":
    main()
