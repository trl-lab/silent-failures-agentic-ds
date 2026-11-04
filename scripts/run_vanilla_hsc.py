#!/usr/bin/env python3
"""
Vanilla runner variant combining both special-case prompts needed by the HSC
set: multi_table (home-credit) and multi_output (single-cell perturbations).
Same rationale as run_aide_hsc.py — reuses the builder functions from
run_vanilla_multitable.py / run_vanilla_multioutput.py, adding only the
three-way dispatch. Every task without either flag falls through to the
original, byte-identical build_prompt().

Usage: identical to run_vanilla.py, e.g.
  python scripts/run_vanilla_hsc.py --model <model> --variant few_shot_role --tasks <id> [--tasks <id> ...]
"""

import importlib.util
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_as_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


run_vanilla = _load_as_module("run_vanilla", PROJECT_ROOT / "scripts" / "run_vanilla.py")
_original_build_prompt = run_vanilla.build_prompt

_multitable  = _load_as_module("_vanilla_multitable_impl", PROJECT_ROOT / "scripts" / "run_vanilla_multitable.py")
_multioutput = _load_as_module("_vanilla_multioutput_impl", PROJECT_ROOT / "scripts" / "run_vanilla_multioutput.py")


def build_prompt_dispatch(task_dir: Path, task_meta: dict,
                           examples: list, supply_data_snippet: bool) -> str:
    if task_meta.get("multi_table"):
        return _multitable.build_prompt_multitable(task_dir, task_meta, examples, supply_data_snippet)
    if task_meta.get("multi_output"):
        return _multioutput.build_prompt_multioutput(task_dir, task_meta, examples, supply_data_snippet)
    return _original_build_prompt(task_dir, task_meta, examples, supply_data_snippet)


run_vanilla.build_prompt = build_prompt_dispatch


if __name__ == "__main__":
    run_vanilla.main()
