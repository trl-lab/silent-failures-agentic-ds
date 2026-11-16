#!/usr/bin/env python3
"""
MLEvolve runner variant combining both special-case task descriptions needed
by the HSC set: multi_table (home-credit) and multi_output (single-cell
perturbations). Same rationale as run_aide_hsc.py — reuses the builder
functions from run_mlevolve_multitable.py / run_mlevolve_multioutput.py,
adding only the three-way dispatch. Every task without either flag falls
through to the original, byte-identical build_task_desc().

Usage: identical to run_mlevolve.py, e.g.
  python scripts/run_mlevolve_hsc.py --hsc --model gemini/gemini-3.6-flash --steps 10 --resume
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


run_mlevolve = _load_as_module("run_mlevolve", PROJECT_ROOT / "scripts" / "run_mlevolve.py")
_original_build_task_desc = run_mlevolve.build_task_desc

_multitable  = _load_as_module("_mlevolve_multitable_impl", PROJECT_ROOT / "scripts" / "run_mlevolve_multitable.py")
_multioutput = _load_as_module("_mlevolve_multioutput_impl", PROJECT_ROOT / "scripts" / "run_mlevolve_multioutput.py")


def build_task_desc_dispatch(task_dir: Path, task_meta: dict) -> str:
    if task_meta.get("multi_table"):
        return _multitable.build_task_desc_multitable(task_dir, task_meta)
    if task_meta.get("multi_output"):
        return _multioutput.build_task_desc_multioutput(task_dir, task_meta)
    return _original_build_task_desc(task_dir, task_meta)


run_mlevolve.build_task_desc = build_task_desc_dispatch


if __name__ == "__main__":
    run_mlevolve.main()
