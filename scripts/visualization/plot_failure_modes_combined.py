"""Combines two ALREADY-GENERATED figures side by side into one (a)/(b) composite: (a)
failure_modes_nohint.png (scripts/visualization/plot_failure_modes.py's 2x3 grid) (b) failure_modes_by_hint.png
(scripts/visualization/plot_failure_modes_by_hint.py's 2-row stack)

Usage: python scripts/visualization/plot_failure_modes_combined.py
-> analysis/figures/failure_modes_combined.{pdf,png}
"""
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
FIG_DIR = PROJECT_ROOT / "analysis" / "figures"
PANEL_A = FIG_DIR / "failure_modes_nohint.png"
PANEL_B = FIG_DIR / "failure_modes_by_hint.png"
OUT_STEM = "failure_modes_combined"
PAD_PX = 8  # small breathing room left around the cropped content, in source pixels

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})


def autocrop(img: np.ndarray, bg_threshold: float = 0.98) -> np.ndarray:
    """Crop to the bounding box of non-near-white pixels (a small PAD_PX margin
    kept around it) -- removes each source script's own incidental tight_layout
    whitespace so only real plot content gets compared/aligned below."""
    rgb = img[..., :3]
    is_content = (rgb < bg_threshold).any(axis=-1)
    rows = np.where(is_content.any(axis=1))[0]
    cols = np.where(is_content.any(axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        return img  # entirely blank -- nothing to crop to, return unchanged
    r0, r1 = max(rows[0] - PAD_PX, 0), min(rows[-1] + PAD_PX + 1, img.shape[0])
    c0, c1 = max(cols[0] - PAD_PX, 0), min(cols[-1] + PAD_PX + 1, img.shape[1])
    return img[r0:r1, c0:c1]


def main() -> None:
    for p in (PANEL_A, PANEL_B):
        if not p.exists():
            raise SystemExit(f"{p} not found -- run its own script first "
                              f"({'plot_failure_modes.py' if p == PANEL_A else 'plot_failure_modes_by_hint.py'})")

    img_a, img_b = autocrop(plt.imread(PANEL_A)), autocrop(plt.imread(PANEL_B))
    ratio_a = img_a.shape[1] / img_a.shape[0]  # width / height, in pixels, post-crop
    ratio_b = img_b.shape[1] / img_b.shape[0]

    # Give each panel canvas width proportional to its own CROPPED image's
    # aspect ratio, at one shared height -- keeps both images undistorted
    # (no stretching) while their actual content now sits on one common row
    # at matching scale (both were cropped to content, so "shared height"
    # here means shared CONTENT height, not shared incidental-margin height).
    HEIGHT_IN = 3.0
    width_a, width_b = HEIGHT_IN * ratio_a, HEIGHT_IN * ratio_b
    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(width_a + width_b, HEIGHT_IN),
        gridspec_kw={"width_ratios": [width_a, width_b]},
    )

    for ax, img, label in ((ax_a, img_a, "(a)"), (ax_b, img_b, "(b)")):
        ax.imshow(img)
        ax.axis("off")
        ax.text(0.0, 1.0, label, transform=ax.transAxes, fontsize=13,
                fontweight="bold", va="bottom", ha="left")

    # Explicit, identical axes positions (not tight_layout, which is free to
    # give the two subplots different heights/vertical offsets) -- this is
    # what actually guarantees both images' top edges align.
    fig.subplots_adjust(left=0.01, right=0.99, top=0.90, bottom=0.02, wspace=0.04)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        p = FIG_DIR / f"{OUT_STEM}.{ext}"
        fig.savefig(p, dpi=300)
        print(f"Wrote {p}")


if __name__ == "__main__":
    main()
