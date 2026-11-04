"""Memorization-intervention pilot: does altering column names hurt more where exposure is high?

Usage: python scripts/analyze_memorization_delta.py --agent vanilla|aide|mlevolve
Output: analysis/exposure_probe/delta_table_<agent>.csv
"""
import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from skill_score import skill  # noqa: E402

SEL = ROOT / "analysis" / "exposure_probe" / "selection.json"

# Tasks whose vanilla delta is dominated by a mechanism found by reading the generated code -- neither of
# them is memorization. Excluding them is a post-hoc, mechanism-based choice (NOT "drop the biggest deltas"),
# so it is off by default: pass --exclude-explained. See analysis/exposure_probe/five_tasks_headers.md.
EXPLAINED = {
    "bike-sharing-demand": "benchmark leak: test.parquet keeps casual+registered (= target); altered run used them",
    "osic-pulmonary-fibrosis-progression": "benchmark leak: every test patient is in train, FVC/Percent constant per patient; altered run used patient-group stats",
    "sf-building-permits": "semantics lost: unaltered run built date-difference features from the date column names",
    "mlr-summary-reports": "semantics lost: unaltered run built MLR Numerator / MLR Denominator",
    "food-enforcement": "semantics lost: unaltered run selected the text columns carrying the recall reason",
}
AGENT = "vanilla"       # set from --agent in main()
MODEL_SLUG = None       # set from --model-slug in main(); None = original gemini-3.6-flash pilot paths


def run_result_path(arm: str, folder: str) -> Path:
    """Where the pilot jobs (cluster job scripts) wrote result.json.

    Pass --model-slug (the exact on-disk slug, not a registry key) to replicate the pilot for another model; it is used verbatim across all three agents."""
    if MODEL_SLUG is not None:
        if AGENT == "aide":
            return ROOT / f"aide_memo_{arm}" / f"aide__{MODEL_SLUG}__10steps" / folder / "result.json"
        if AGENT == "mlevolve":
            return ROOT / f"mlevolve_memo_{arm}" / f"mlevolve__{MODEL_SLUG}__10steps" / folder / "result.json"
        return ROOT / "runs_memo" / f"vanilla__{MODEL_SLUG}__few_shot_role__{arm}" / folder / "result.json"
    if AGENT == "aide":
        return ROOT / f"aide_memo_{arm}" / "aide__gemini-3.6-flash__10steps" / folder / "result.json"
    if AGENT == "mlevolve":
        return ROOT / f"mlevolve_memo_{arm}" / "mlevolve__gemini-gemini-3.6-flash__10steps" / folder / "result.json"
    return ROOT / "runs_memo" / f"vanilla__gemini-3.6-flash__few_shot_role__{arm}" / folder / "result.json"

RNG = np.random.default_rng(0)
CLIP = (-1.0, 1.0)


def load_skill(arm: str, folder: str, tid: str, metric: str):
    """(status, skill): status in ok / failed / missing."""
    p = run_result_path(arm, folder)
    if not p.exists():
        return "missing", np.nan
    r = json.loads(p.read_text())
    s, hib = r.get("score"), r.get("higher_is_better", True)
    if not r.get("success") or s is None or (hib and abs(s) < 1e-6):
        return "failed", np.nan
    k = skill(tid, metric, s)
    return ("ok", float(np.clip(k, *CLIP))) if k is not None else ("failed", np.nan)


def boot_ci(x, n=5000):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return np.nan, np.nan
    b = [RNG.choice(x, len(x)).mean() for _ in range(n)]
    return tuple(np.percentile(b, [2.5, 97.5]))


def main() -> None:
    global AGENT, MODEL_SLUG
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", choices=["vanilla", "aide", "mlevolve"], default="vanilla")
    ap.add_argument("--model-slug", default=None,
                    help="exact on-disk model slug (e.g. gemini-gemini-3.7-flash) -- see run_result_path's "
                         "docstring. Default (unset) keeps the original hardcoded gemini-3.6-flash paths.")
    ap.add_argument("--exclude-explained", action="store_true",
                    help="drop the tasks in EXPLAINED (leak / lost-semantics cases) before all statistics")
    args = ap.parse_args()
    AGENT = args.agent
    MODEL_SLUG = args.model_slug
    model_label = MODEL_SLUG if MODEL_SLUG else "gemini-3.6-flash"
    print(f"=== agent: {AGENT} ({model_label}) ===\n")
    sel = json.loads(SEL.read_text())
    if args.exclude_explained:
        print("Excluding tasks with an identified non-memorization mechanism:")
        for t, why in EXPLAINED.items():
            print(f"  - {t}: {why}")
        print()
        sel = [r for r in sel if r["task"] not in EXPLAINED]
    rows = []
    for r in sel:
        st = {}
        for arm, folder in (("orig", r["task"]), ("orig2", r["task"]), ("alt", r["alt_id"])):
            st[arm] = load_skill(arm, folder, r["task"], r["metric"])
        rows.append({"group": r["group"], "task": r["task"], "set": r["set"], "task_type": r["task_type"],
                     "strict": r["strict"], "relaxed": r["relaxed"],
                     **{f"st_{a}": st[a][0] for a in st}, **{f"s_{a}": st[a][1] for a in st}})
    df = pd.DataFrame(rows)
    base = df[["s_orig", "s_orig2"]].mean(axis=1, skipna=True)
    df["delta_alt"] = base - df["s_alt"]
    df["delta_placebo"] = df["s_orig"] - df["s_orig2"]
    model_suffix = f"_{MODEL_SLUG}" if MODEL_SLUG else ""  # avoid overwriting the gemini-3.6-flash tables
    df.to_csv(ROOT / "analysis" / "exposure_probe" /
              f"delta_table_{AGENT}{model_suffix}{'_excl_explained' if args.exclude_explained else ''}.csv", index=False)

    print("Run status per arm and group (ok / failed / missing):")
    for arm in ("orig", "orig2", "alt"):
        t = df.groupby(["group", f"st_{arm}"]).size().unstack(fill_value=0)
        print(f"  {arm:5s} " + "; ".join(f"{g}: " + "/".join(f"{int(t.loc[g].get(k, 0))}" for k in ("ok", "failed", "missing"))
                                          for g in ("high", "low") if g in t.index))
    v = df.dropna(subset=["delta_alt"])
    print(f"\nValid altered-vs-baseline pairs: high={int((v.group == 'high').sum())}, low={int((v.group == 'low').sum())} of 20 each")
    if len(v) < 6:
        print("too few valid pairs for statistics yet")
        return

    print("\n[1] Mean skill and delta by group (bootstrap 95% CI over tasks)")
    print("| Group | n | baseline skill | altered skill | delta_alt [95% CI] | mean of tasks with delta_alt > 0 |\n|---|---|---|---|---|---|")
    for g in ("high", "low"):
        x = v[v.group == g]
        b = x[["s_orig", "s_orig2"]].mean(axis=1, skipna=True)
        lo, hi = boot_ci(x.delta_alt)
        print(f"| {g} | {len(x)} | {b.mean():+.3f} | {x.s_alt.mean():+.3f} | {x.delta_alt.mean():+.3f} [{lo:+.3f}, {hi:+.3f}] | "
              f"{(x.delta_alt > 0).mean():.2f} |")
    hi_, lo_ = v[v.group == "high"].delta_alt, v[v.group == "low"].delta_alt
    diffs = [RNG.choice(hi_, len(hi_)).mean() - RNG.choice(lo_, len(lo_)).mean() for _ in range(5000)]
    print(f"\nHIGH - LOW delta_alt = {hi_.mean() - lo_.mean():+.3f}  [95% CI {np.percentile(diffs, 2.5):+.3f}, "
          f"{np.percentile(diffs, 97.5):+.3f}]   Mann-Whitney p = {stats.mannwhitneyu(hi_, lo_).pvalue:.3g}")
    for g, x in (("high", hi_), ("low", lo_)):
        if len(x) > 5:
            print(f"  {g}: Wilcoxon delta_alt > 0: p = {stats.wilcoxon(x, alternative='greater').pvalue:.3g}")

    print("\n[2] Correlation of exposure (strict column-completion score) with delta_alt")
    rho, p = stats.spearmanr(v.strict, v.delta_alt)
    print(f"  all valid pairs: Spearman rho = {rho:+.3f} (p = {p:.3g}, n = {len(v)})")
    for tt in ("classification", "regression"):
        x = v[v.task_type == tt]
        if len(x) > 5:
            r2, p2 = stats.spearmanr(x.strict, x.delta_alt)
            print(f"  {tt:15s} Spearman rho = {r2:+.3f} (p = {p2:.3g}, n = {len(x)})")
    print("  (exposure is nearly binary here, so this is essentially the high-vs-low contrast above)")

    print("\n[3] Noise floor: unaltered rerun (orig - orig2)")
    pl = df.dropna(subset=["delta_placebo"])
    print(f"  n = {len(pl)}; mean = {pl.delta_placebo.mean():+.3f}; SD = {pl.delta_placebo.std():.3f}; "
          f"mean |diff| = {pl.delta_placebo.abs().mean():.3f}")
    for g in ("high", "low"):
        x = pl[pl.group == g].delta_placebo
        if len(x) > 2:
            print(f"  {g}: mean {x.mean():+.3f}, mean |diff| {x.abs().mean():.3f} (n = {len(x)})")
    print(f"  altered-vs-baseline mean |delta| = {v.delta_alt.abs().mean():.3f} (compare with the |diff| above)")

    print("\n[4] Failure rate under alteration (run invalid) vs unaltered, by group")
    print("| Group | orig failed | orig2 failed | alt failed | n |\n|---|---|---|---|---|")
    for g in ("high", "low"):
        x = df[df.group == g]
        print(f"| {g} | {(x.st_orig == 'failed').sum()} | {(x.st_orig2 == 'failed').sum()} | {(x.st_alt == 'failed').sum()} | {len(x)} |")

    print("\n[5] Sensitivity: treat a failed altered run as skill 0 (baseline valid)")
    z = df.dropna(subset=["s_orig"]).copy()
    z["s_alt0"] = z.s_alt.fillna(0.0).where(z.st_alt != "missing")
    z["delta0"] = z.s_orig - z.s_alt0
    for g in ("high", "low"):
        x = z[z.group == g].delta0.dropna()
        print(f"  {g}: mean delta = {x.mean():+.3f} (n = {len(x)})")

    print("\n[6] Is the altered delta different from the placebo delta? (tasks with all three arms valid)")
    print("    d_alt = skill(orig) - skill(alt) [same single-run baseline as the placebo]; d_plc = skill(orig) - skill(orig2);")
    print("    paired difference d_alt - d_plc = skill(orig2) - skill(alt). Signed = does alteration shift scores;")
    print("    magnitude = does it move scores MORE than a rerun does (either direction). Primary: signed paired test and the")
    print("    high-vs-low comparison; magnitude tests are secondary/exploratory (overlapping tests, no multiplicity correction).")
    a3 = df.dropna(subset=["s_orig", "s_orig2", "s_alt"]).copy()
    a3["d_alt1"] = a3.s_orig - a3.s_alt
    a3["d_plc"] = a3.s_orig - a3.s_orig2
    print("| Subset | n | mean d_alt | mean d_plc | paired t p | Wilcoxon p | mean abs d_alt | mean abs d_plc | abs Wilcoxon p | Brown-Forsythe p |\n|---|---|---|---|---|---|---|---|---|---|")
    for name, x in (("all", a3), ("high", a3[a3.group == "high"]), ("low", a3[a3.group == "low"])):
        if len(x) < 6:
            continue
        pt = stats.ttest_rel(x.d_alt1, x.d_plc).pvalue
        pw = stats.wilcoxon(x.d_alt1, x.d_plc).pvalue
        pm = stats.wilcoxon(x.d_alt1.abs(), x.d_plc.abs()).pvalue
        pl = stats.levene(x.d_alt1, x.d_plc, center="median").pvalue
        print(f"| {name} | {len(x)} | {x.d_alt1.mean():+.3f} | {x.d_plc.mean():+.3f} | {pt:.3f} | {pw:.3f} | "
              f"{x.d_alt1.abs().mean():.3f} | {x.d_plc.abs().mean():.3f} | {pm:.3f} | {pl:.3f} |")
    h3, l3 = a3[a3.group == "high"], a3[a3.group == "low"]
    if len(h3) > 3 and len(l3) > 3:
        u, w = h3.d_alt1 - h3.d_plc, l3.d_alt1 - l3.d_plc
        print(f"  alteration effect relative to placebo, high vs low: {u.mean():+.3f} vs {w.mean():+.3f}  "
              f"Welch p = {stats.ttest_ind(u, w, equal_var=False).pvalue:.3f}, Mann-Whitney p = {stats.mannwhitneyu(u, w).pvalue:.3f}")

    print("\nPer-task table (sorted by exposure, then group):")
    cols = ["group", "task", "set", "task_type", "strict", "s_orig", "s_orig2", "s_alt", "delta_alt"]
    print(df.sort_values(["group", "strict"], ascending=[True, False])[cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
