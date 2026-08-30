# -*- coding: utf-8 -*-
"""Shared plotting style used by the study analysis scripts.
Local workstation provenance paths were removed for the public release."""
from pathlib import Path
import matplotlib.pyplot as plt

MODEL_COLORS = {
    "LR": "#9467bd", "SVM": "#1f77b4", "ANN": "#ff7f0e", "KNN": "#d62728",
    "RF": "#17becf", "AdaBoost": "#8c564b", "XGBoost": "#2ca02c",
    "LightGBM": "#bcbd22", "DT": "#7f7f7f", "NB": "navy",
}
COLOR_MEAN = "#2ca02c"
COLOR_ALT = "#1f77b4"
FOLD_COLORS = ["#1f77b4", "#ff7f0e", "#d62728", "#9467bd", "#8c564b"]
FONT = "Times New Roman"


def set_paper_style(font: str = FONT, base_fontsize: int = 16,
                    figsize=(8, 6), dpi: int = 300):
    plt.rcParams.update(plt.rcParamsDefault)
    plt.rcParams.update({
        "font.family": "serif", "font.serif": [font], "font.size": base_fontsize,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "figure.figsize": tuple(figsize), "figure.dpi": dpi,
        "axes.linewidth": 2.0,
        "axes.labelsize": base_fontsize + 2, "axes.labelweight": "bold",
        "xtick.labelsize": base_fontsize, "ytick.labelsize": base_fontsize,
        "xtick.direction": "in", "ytick.direction": "in",
        "axes.unicode_minus": False,
        "xtick.major.width": 2.0, "ytick.major.width": 2.0,
        "xtick.major.size": 6, "ytick.major.size": 6,
        "legend.fontsize": base_fontsize - 3, "legend.frameon": False,
        "axes.spines.top": False, "axes.spines.right": False,
    })


def bold_ticks(ax):
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")


def savefig(fig, out_path, formats=("tiff", "pdf"), dpi: int = 300):
    """用户约定：只出 300dpi TIFF(LZW) + 矢量 PDF，不出 PNG。"""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stem = out_path.with_suffix("")
    saved = []
    for fmt in formats:
        p = f"{stem}.{fmt}"
        kw = dict(format=fmt, dpi=dpi, bbox_inches="tight")
        if fmt in ("tif", "tiff"):
            kw["pil_kwargs"] = {"compression": "tiff_lzw"}
        else:
            kw["transparent"] = True
        fig.savefig(p, **kw)
        saved.append(p)
    plt.close(fig)
    return saved
