"""Export final pipelines + final judge verdicts for the 4 models x 3 agents in the paper.
Run from the original ds_failures project; writes into this folder. Read-only on the source."""
import csv, json, re, sys
from pathlib import Path
import os
SRC = Path(os.environ["DS_FAILURES_SRC"])  # path to the original project checkout with raw runs
OUT = Path(__file__).resolve().parent.parent / "outputs"
sys.path.insert(0, str(SRC / "scripts"))
import judge_common as jc
from judge_viz_common import MODELS, AGENTS, result_dir

# The original runs used internal failure-mode ids; the release uses the paper's FM01-FM15.
# F11, F13 and F20 are not part of the paper's taxonomy and are dropped.
LEGACY_TO_FM = {"F_TASKFORM": "FM01", "F03": "FM02", "F04": "FM03", "F12": "FM04", "F14": "FM05",
                "F26": "FM06", "F_FEATENG": "FM07", "F09": "FM08", "F10": "FM09", "F16": "FM10",
                "F21": "FM11", "F_ENSEMBLE": "FM12", "F27": "FM13", "F29": "FM14", "F31": "FM15"}
FM_IDS = sorted(LEGACY_TO_FM.values())
_LEGACY_RX = re.compile(r"(?<![A-Za-z0-9_])(" + "|".join(sorted(LEGACY_TO_FM, key=len, reverse=True)) + r")(?![A-Za-z0-9_])")
_DROPPED_RX = re.compile(r"\s*\(F(?:11|13|20)\)|F(?:11|13|20)(?:,\s*|/)|,\s*F(?:11|13|20)(?![A-Za-z0-9_])")
def convert(j):
    j = dict(j)
    # "no code" = every applicable verdict is cannot_determine, evaluated on the original run's
    # modes (including the three dropped ones) so the paper's pipeline counts stay reproducible.
    cells = [v for v in j.get("failure_modes", {}).values() if isinstance(v, dict) and v.get("applicable") is not False]
    j["no_code"] = bool(cells) and all(v.get("present") == "cannot_determine" for v in cells)
    j["failure_modes"] = {LEGACY_TO_FM[k]: v for k, v in j.get("failure_modes", {}).items() if k in LEGACY_TO_FM}
    return json.loads(_DROPPED_RX.sub('', _LEGACY_RX.sub(lambda m: LEGACY_TO_FM[m.group(1)], json.dumps(j))))

SCRUB = [(re.compile(r"/(?:Users|home|gpfs/home\d*)/[A-Za-z0-9_.-]+"), "/path/to/user"),
         (re.compile(re.escape(os.environ.get("SCRUB_USERNAME", "<no-username-set>")), re.I), "user"),  # set SCRUB_USERNAME to your login
         (re.compile(r"(sk-[A-Za-z0-9_-]{16,}|AIza[0-9A-Za-z_-]{30,}|hf_[A-Za-z0-9]{20,})"), "REDACTED_KEY")]
def scrub(t):
    for p, r in SCRUB: t = p.sub(r, t)
    return t

# Optional fallback: a directory holding run folders pulled from the cluster
# (same layout as the originals: runs/, aide_runs/, mlevolve_runs/), used when a
# run's code is missing locally. Applies the same file-selection rule as
# judge_common.load_code (MLEvolve: latest mlevolve_logs/*/best_solution.py).
FALLBACK = Path(os.environ["DS_FAILURES_REMOTE_CODE"]) if os.environ.get("DS_FAILURES_REMOTE_CODE") else None
def fallback_code(agent, mk, task):
    if not FALLBACK: return None
    rd = jc.run_dir(agent, mk); d = FALLBACK / rd.parent.name / rd.name / task
    if agent == "mlevolve":
        m = sorted(d.glob("mlevolve_logs/*/best_solution.py")); f = m[-1] if m else None
    else:
        f = d / jc.AGENT_FILES[agent]["code"]
    return f.read_text(encoding="utf-8", errors="replace") if f and f.exists() else None

rows, nscrub = [], 0
for agent in AGENTS:
    for m in MODELS:
        mk = m["key"]; d = result_dir(agent, mk)
        for jf in sorted(d.glob("*.json")):
            task = jf.stem; j = json.loads(jf.read_text())
            if not j.get("failure_modes"): continue          # parse errors: not a judged result
            j = convert(j)
            code = jc.load_code(agent, task, mk) or fallback_code(agent, mk, task)
            res = jc.load_result(agent, task, mk)
            base = f"{agent}/{mk}/{task}"
            (OUT/"judge_verdicts"/agent/mk).mkdir(parents=True, exist_ok=True)
            (OUT/"judge_verdicts"/agent/mk/f"{task}.json").write_text(scrub(json.dumps(j, indent=1)))
            if code:
                (OUT/"pipelines"/agent/mk).mkdir(parents=True, exist_ok=True)
                s = scrub(code); nscrub += s != code
                (OUT/"pipelines"/agent/mk/f"{task}.py").write_text(s)
            row = {"pipeline_id": base, "agent": agent, "model": mk, "task": task,
                   "source": res.get("source"), "metric": res.get("metric"), "score": res.get("score"),
                   "run_success": res.get("success"), "has_code": bool(code)}
            for fid in FM_IDS:
                e = j["failure_modes"].get(fid) or {}
                v = "N/A" if e.get("applicable") is False else {True:"Present",False:"Absent"}.get(e.get("present"), "Unsure" if e else "")
                row[fid] = v
            rows.append(row)
cols = list(rows[0]); cols = cols[:9] + sorted(cols[9:])
with open(OUT/"judge_verdicts.csv", "w", newline="") as f:
    w = csv.DictWriter(f, cols); w.writeheader(); w.writerows(rows)
print(len(rows), "judged pipelines;", sum(r["has_code"] for r in rows), "with code;", nscrub, "files scrubbed")
