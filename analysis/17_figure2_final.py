# -*- coding: utf-8 -*-
"""Figure 2 final layout (uses the authoritative recovered Aim 1 result object)

  A  上排左 —— OR 与 ΔAUROC 合成一个 aligned forest-table，18 个指标名只出现一次
  B  上排右 —— 特征域逐级叠加，双轨累积信息图（AUPRC 一轨、AUROC 一轨），不用柱+折线
  C  下排整行 —— 三个对照亚型并列成三列小森林，共用 18 个行标签，各自有 OR=1 参考线

一次性 matplotlib 出图，不再用 PIL 拼，避免上宽下窄、重心断裂。
"""
from public_paths import WORK_ROOT, FONT_PATH
import os, pickle
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

OUT = str(WORK_ROOT / "结核-Aim2" / "outputs")
EXT_OUT = WORK_ROOT / "结核-外部验证" / "outputs"
PAN = os.path.join(OUT, "panels", "Fig2")
os.makedirs(PAN, exist_ok=True)
FONT = "Times New Roman"
C_LOW, C_HIGH = "#1f77b4", "#c0392b"        # OR<1 / OR>1
C_SLATE = "#37474F"                          # C 三列统一低饱和
C_AP, C_AU = "#2e7d32", "#1565c0"

rec = pd.read_pickle(EXT_OUT / "_aim1_recovered.pkl")
main = rec["main"]
strat = rec["strat"]
AUC0 = float(rec["auc_base"])
S13 = pickle.load(open(os.path.join(OUT, "_cache", "s13.pkl"), "rb"))
M = S13['metrics']

SUBS = ['Acute infection', 'Chronic airway disease', 'Other non-infectious']
HEAD = ['Acute infection\n(n = 3185)', 'Chronic airway disease\n(n = 259)',
        'Other non-infectious\n(n = 325)']

plt.rcParams.update(plt.rcParamsDefault)
plt.rcParams.update({
    "font.family": "serif", "font.serif": [FONT], "font.size": 16,
    "pdf.fonttype": 42, "ps.fonttype": 42, "figure.dpi": 300,
    "axes.linewidth": 1.8, "axes.labelweight": "bold",
    "xtick.direction": "in", "ytick.direction": "in", "axes.unicode_minus": False,
    "xtick.major.width": 1.8, "ytick.major.width": 1.8,
    "xtick.major.size": 5, "ytick.major.size": 5,
    "legend.frameon": False, "axes.spines.top": False, "axes.spines.right": False})

d = main.iloc[::-1].reset_index(drop=True)
order = list(d['指标']); n = len(d); yp = np.arange(n)

fig = plt.figure(figsize=(22, 19.5))
gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.06], hspace=0.16,
                      left=0.055, right=0.985, top=0.955, bottom=0.045)
gt = gs[0].subgridspec(1, 2, width_ratios=[1.52, 1.0], wspace=0.20)
ga = gt[0].subgridspec(1, 2, wspace=0.05)
a1 = fig.add_subplot(ga[0]); a2 = fig.add_subplot(ga[1], sharey=a1)
b = fig.add_subplot(gt[1])
gc = gs[1].subgridspec(1, 3, wspace=0.05)
cs = [fig.add_subplot(gc[0])]
cs += [fig.add_subplot(gc[i], sharey=cs[0]) for i in (1, 2)]


def bold(ax):
    for t in ax.get_xticklabels() + ax.get_yticklabels():
        t.set_fontweight("bold")


# ============ A：aligned forest-table ============
for i, r in d.iterrows():
    c = C_HIGH if r['校正OR'] > 1 else C_LOW
    a1.plot([r['lo'], r['hi']], [i, i], color=c, lw=2.8, solid_capstyle='butt', zorder=2)
    a1.plot(r['校正OR'], i, 'o', color=c, ms=9.5, zorder=3)
    a2.plot([r['ΔAUC_lo'], r['ΔAUC_hi']], [i, i], color=c, lw=2.8,
            solid_capstyle='butt', zorder=2)
    a2.plot(r['ΔAUC'], i, 'o', color=c, ms=9.5, zorder=3)
a1.axvline(1, color='0.35', ls='--', lw=1.8); a2.axvline(0, color='0.35', ls='--', lw=1.8)
a1.set_xscale('log'); a1.set_xticks([0.5, 0.7, 1.0, 1.4])
a1.set_xticklabels(['0.5', '0.7', '1.0', '1.4'])
a1.xaxis.set_minor_locator(plt.NullLocator())
a1.set_yticks(yp); a1.set_yticklabels(order, fontsize=16)
a1.set_ylim(-0.9, n - 0.1)
a1.set_title('Adjusted OR per 1-SD (95% CI)', fontsize=18, fontweight='bold', pad=12)
a2.set_title(r'$\Delta$AUROC over age + sex (95% CI)', fontsize=18,
             fontweight='bold', pad=12)
a2.set_xlim(-0.008, 0.058)
plt.setp(a2.get_yticklabels(), visible=False)
a2.tick_params(axis='y', length=0)
a2.text(0.98, 0.012, f'age + sex model AUROC = {AUC0:.3f}', transform=a2.transAxes,
        ha='right', va='bottom', fontsize=13.5, style='italic', color='0.3')
for ax in (a1, a2):
    bold(ax)

# ============ B：双轨累积信息图 ============
keys = ['M1', 'M2', 'M3', 'M4']
STAGE = ['Age + sex\n(2)', '+ Blood count\n(14)', '+ Biochemistry\n(27)',
         '+ Derived indices\n(38)']
ap = [M[k][0].mean() for k in keys]
au = [M[k][1].mean() for k in keys]
x = np.arange(4)
TRACK = {'AUPRC': 1.0, 'AUROC': 0.0}
b.plot(x, [TRACK['AUPRC']]*4, '-', color=C_AP, lw=2.2, zorder=1)
b.plot(x, [TRACK['AUROC']]*4, '-', color=C_AU, lw=2.2, zorder=1)
for i in range(4):
    b.plot(x[i], TRACK['AUPRC'], 'o', color=C_AP, ms=17, zorder=3)
    b.plot(x[i], TRACK['AUROC'], 'o', color=C_AU, ms=17, zorder=3)
    b.text(x[i], TRACK['AUPRC'] - 0.13, f'{ap[i]:.3f}', ha='center',
           fontsize=17, fontweight='bold', color=C_AP)
    b.text(x[i], TRACK['AUROC'] - 0.13, f'{au[i]:.3f}', ha='center',
           fontsize=17, fontweight='bold', color=C_AU)
    if i:
        dp, du = ap[i] - ap[i-1], au[i] - au[i-1]
        for val, ty, col in [(dp, TRACK['AUPRC'], C_AP), (du, TRACK['AUROC'], C_AU)]:
            flat = abs(val) < 0.01
            b.text(x[i] - 0.5, ty + 0.11, f'{val:+.3f}' if abs(val) >= 0.001 else f'{val:+.4f}',
                   ha='center', fontsize=16, fontweight='bold',
                   color=('0.55' if flat else col))
b.text(-0.62, TRACK['AUPRC'], 'AUPRC', ha='right', va='center',
       fontsize=19, fontweight='bold', color=C_AP)
b.text(-0.62, TRACK['AUROC'], 'AUROC', ha='right', va='center',
       fontsize=19, fontweight='bold', color=C_AU)
b.set_xticks(x); b.set_xticklabels(STAGE, fontsize=16)
b.set_xlim(-0.95, 3.45); b.set_ylim(-0.42, 1.42)
b.set_yticks([]); b.spines['left'].set_visible(False)
b.spines['bottom'].set_visible(True)
b.set_title('Cumulative information across routine feature domains',
            fontsize=18, fontweight='bold', pad=12)
b.text(3.42, TRACK['AUPRC'] + 0.30,
       'derived indices add\nessentially no information',
       ha='right', fontsize=14.5, style='italic', color='0.35')
bold(b)

# ============ C：三列并列森林 ============
piv = {k: strat[k].reindex(order) for k in SUBS}
for j, (ax, k) in enumerate(zip(cs, SUBS)):
    t = piv[k]
    for i in range(n):
        ax.plot([t['lo'].iloc[i], t['hi'].iloc[i]], [i, i], color=C_SLATE,
                lw=2.6, solid_capstyle='butt', zorder=2, alpha=0.85)
    ax.plot(t['校正OR'].values, yp, 'o', color=C_SLATE, ms=9, zorder=3)
    ax.axvline(1, color='#c0392b', ls='--', lw=1.8, zorder=1)
    ax.set_xscale('log'); ax.set_xticks([0.3, 0.5, 0.7, 1.0, 1.5, 2.0])
    ax.set_xticklabels(['0.3', '0.5', '0.7', '1.0', '1.5', '2.0'])
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.set_xlim(0.25, 2.3); ax.set_ylim(-0.9, n - 0.1)
    ax.set_title(HEAD[j], fontsize=18, fontweight='bold', pad=12)
    if j == 0:
        ax.set_yticks(yp); ax.set_yticklabels(order, fontsize=16)
    else:
        plt.setp(ax.get_yticklabels(), visible=False)
        ax.tick_params(axis='y', length=0)
    bold(ax)
cs[1].set_xlabel('Adjusted OR per 1-SD (95% CI)', fontsize=18, fontweight='bold',
                 labelpad=10)

for txt, xx, yy in [("A", 0.012, 0.982), ("B", 0.615, 0.982), ("C", 0.012, 0.505)]:
    fig.text(xx, yy, txt, fontsize=32, fontweight='bold', va='top')

for ext, kw in [("tiff", dict(pil_kwargs={"compression": "tiff_lzw"})),
                ("pdf", dict(transparent=True))]:
    fig.savefig(os.path.join(OUT, f"Figure2_hostresponse_combined.{ext}"),
                format=ext, dpi=300, bbox_inches="tight", **kw)
plt.close(fig)
print("Figure 2 重排完成")
print(f"  A 18 指标 aligned forest-table（OR | ΔAUROC 共用行标签）")
print(f"  B 双轨累积：AUPRC {ap[0]:.3f}→{ap[1]:.3f}→{ap[2]:.3f}→{ap[3]:.3f}")
print(f"           AUROC {au[0]:.3f}→{au[1]:.3f}→{au[2]:.3f}→{au[3]:.3f}")
print(f"  C 三列并列森林，统一低饱和配色，靠列标题区分")
