# -*- coding: utf-8 -*-
"""Supplementary Figure S5 —— 外部验证（第二中心，n = 1149）

A  ROC：外部主分析 vs 推导集 OOF 参照
B  PR：外部主分析 vs 推导集 OOF 参照，各自随机基线
C  校准曲线 + CITL / 斜率
D  决策曲线（外部）
E  预测概率分布：结核 / 肺炎 / 肺癌，标注冻结阈值
F  各分析集的 AUROC 与 AUPRC 森林图

单格进 panels\\Fig7\\，组图进 outputs 根目录。
"""
from public_paths import WORK_ROOT, FONT_PATH
import sys, pickle, warnings
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib import gridspec
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.style import savefig

DESK = WORK_ROOT
OUT = DESK / "结核-外部验证" / "outputs"
PAN = OUT / "panels" / "FigS5"
PAN.mkdir(parents=True, exist_ok=True)

C_EXT, C_INT = "#c0392b", "#1565c0"
C_TB, C_PN, C_LC = "#c0392b", "#2e7d32", "#8e6bbf"
FONT = "Times New Roman"

R = pickle.load(open(OUT / "_external_results.pkl", "rb"))
ext, THR = R['ext'], R['THR']
res = pd.DataFrame(R['res']).set_index('set')
dev = pickle.load(open(DESK / "结核-Aim2" / "outputs" / "_cache" / "s03b.pkl", "rb"))
p_dev = dev['oof_threshold_dev']
y_dev = pd.read_excel(DESK / "结核数据-25指标-建模集.xlsx")['Group'].values

m = ext.complete_10 == 1
yE, pE = ext.loc[m, 'Outcome'].values, ext.loc[m, 'p'].values
PRIM = 'Primary: complete cases (10 variables)'
rp = res.loc[PRIM]

plt.rcParams.update(plt.rcParamsDefault)
plt.rcParams.update({
    "font.family": "serif", "font.serif": [FONT], "font.size": 16,
    "pdf.fonttype": 42, "ps.fonttype": 42, "figure.dpi": 300,
    "axes.linewidth": 2.0, "axes.labelweight": "bold", "axes.labelsize": 18,
    "xtick.direction": "in", "ytick.direction": "in", "axes.unicode_minus": False,
    "xtick.major.width": 2.0, "ytick.major.width": 2.0,
    "xtick.major.size": 6, "ytick.major.size": 6,
    "legend.frameon": False, "axes.spines.top": False, "axes.spines.right": False})


def bold(ax):
    for t in ax.get_xticklabels() + ax.get_yticklabels():
        t.set_fontweight("bold")


fig = plt.figure(figsize=(22, 13.4))
gs = gridspec.GridSpec(2, 3, figure=fig, wspace=0.36, hspace=0.32,
                       left=0.05, right=0.985, top=0.945, bottom=0.065)
A, B, C = [fig.add_subplot(gs[0, i]) for i in range(3)]
D, E, F = [fig.add_subplot(gs[1, i]) for i in range(3)]

# ---------------- A  ROC ----------------
fi, ti, _ = roc_curve(y_dev, p_dev)
fe, te, _ = roc_curve(yE, pE)
A.plot(fi, ti, color=C_INT, lw=3.0, alpha=0.85,
       label=f"Derivation, cross-fitted (AUC {roc_auc_score(y_dev, p_dev):.3f})")
A.plot(fe, te, color=C_EXT, lw=3.6,
       label=f"External validation (AUC {rp.AUROC:.3f}, 95% CI {rp.AUROC_lo:.3f}\u2013{rp.AUROC_hi:.3f})")
A.plot([0, 1], [0, 1], ls='--', color='0.55', lw=2.0)
A.set_xlabel('1 \u2212 Specificity'); A.set_ylabel('Sensitivity')
A.set_xlim(-0.02, 1.02); A.set_ylim(-0.02, 1.02)
A.legend(loc='lower right', fontsize=13.5)
A.set_title('Discrimination', fontsize=19, fontweight='bold', pad=10)
bold(A)

# ---------------- B  PR ----------------
pi, ri, _ = precision_recall_curve(y_dev, p_dev)
pe, re_, _ = precision_recall_curve(yE, pE)
B.plot(ri, pi, color=C_INT, lw=3.0, alpha=0.85,
       label=f"Derivation (AP {average_precision_score(y_dev, p_dev):.3f}, baseline {y_dev.mean():.3f})")
B.plot(re_, pe, color=C_EXT, lw=3.6,
       label=f"External (AP {rp.AUPRC:.3f}, 95% CI {rp.AUPRC_lo:.3f}\u2013{rp.AUPRC_hi:.3f})")
B.axhline(y_dev.mean(), color=C_INT, ls=':', lw=2.2)
B.axhline(yE.mean(), color=C_EXT, ls=':', lw=2.2)
B.text(0.50, yE.mean() - 0.055, f"external baseline = {yE.mean():.3f}", ha='left',
       fontsize=13, style='italic', color=C_EXT)
B.text(0.02, y_dev.mean() + 0.025, f"derivation baseline = {y_dev.mean():.3f}", ha='left',
       fontsize=13, style='italic', color=C_INT)
B.set_xlabel('Recall'); B.set_ylabel('Precision')
B.set_xlim(-0.02, 1.02); B.set_ylim(0, 1.02)
B.legend(loc='upper right', fontsize=13.5)
B.set_title('Precision\u2013recall', fontsize=19, fontweight='bold', pad=10)
B.text(0.97, 0.60, f"{rp.AUPRC/yE.mean():.2f}\u00d7 random baseline (external)\n"
                   f"{average_precision_score(y_dev, p_dev)/y_dev.mean():.2f}\u00d7 random baseline (derivation)",
       ha='right', fontsize=13.5, style='italic', color='0.25')
bold(B)

# ---------------- C  Calibration ----------------
q = pd.qcut(pE, 10, duplicates='drop')
cal = pd.DataFrame({'p': pE, 'y': yE, 'q': q}).groupby('q', observed=True).agg(
    pm=('p', 'mean'), om=('y', 'mean'), n=('y', 'size'))
se = np.sqrt(cal.om * (1 - cal.om) / cal.n)
C.plot([0, cal.pm.max() * 1.15], [0, cal.pm.max() * 1.15], ls='--', color='0.5', lw=2.2,
       label='Perfect calibration')
C.errorbar(cal.pm, cal.om, yerr=1.96 * se, fmt='o', color=C_EXT, ms=11, lw=2.2,
           capsize=5, capthick=2.2, label='External, decile of predicted risk')
z = np.polyfit(cal.pm, cal.om, 1)
xs = np.linspace(0, cal.pm.max() * 1.15, 50)
C.plot(xs, np.polyval(z, xs), color=C_EXT, lw=2.4, alpha=0.55)
C.set_xlabel('Predicted probability'); C.set_ylabel('Observed proportion')
C.set_xlim(0, cal.pm.max() * 1.15); C.set_ylim(0, max(cal.om.max(), cal.pm.max()) * 1.25)
C.legend(loc='upper left', fontsize=13.5)
C.set_title('Calibration', fontsize=19, fontweight='bold', pad=10)
C.text(0.97, 0.06, f"slope = {rp.Slope:.3f} (95% CI {rp.Slope_lo:.3f}\u2013{rp.Slope_hi:.3f})\n"
                   f"CITL = {rp.CITL:+.3f} (95% CI {rp.CITL_lo:+.3f}\u2013{rp.CITL_hi:+.3f})\n"
                   f"Brier = {rp.Brier:.4f}",
       transform=C.transAxes, ha='right', va='bottom', fontsize=13.5, style='italic')
bold(C)

# ---------------- D  DCA ----------------
ths = np.linspace(0.005, 0.45, 180)
prev = yE.mean()
nb_m, nb_all = [], []
for t in ths:
    yh = pE >= t
    tp = np.sum(yh & (yE == 1)) / len(yE)
    fp = np.sum(yh & (yE == 0)) / len(yE)
    nb_m.append(tp - fp * t / (1 - t))
    nb_all.append(prev - (1 - prev) * t / (1 - t))
D.plot(ths, nb_m, color=C_EXT, lw=3.6, label='Frozen model (external)')
D.plot(ths, nb_all, color='0.35', lw=2.4, ls='--', label='Treat all')
D.axhline(0, color='0.35', lw=2.4, ls=':', label='Treat none')
D.axvline(THR, color='#e08214', lw=2.4, ls='-.')
D.text(THR * 1.25, max(nb_m) * 0.62, f'operating\nthreshold\n= {THR:.4f}',
       fontsize=13, style='italic', color='#a35c00')
D.set_xlabel('Threshold probability'); D.set_ylabel('Net benefit')
D.set_xlim(0, 0.45); D.set_ylim(-0.03, max(nb_m) * 1.15)
D.legend(loc='upper right', fontsize=13.5)
D.set_title('Decision curve, external cohort', fontsize=19, fontweight='bold', pad=10)
bold(D)

# ---------------- E  概率分布 ----------------
sub = ext[m]
grp = [('Tuberculosis', sub.loc[sub.Outcome == 1, 'p'], C_TB),
       ('Pneumonia', sub.loc[sub.Control_subgroup == 'Pneumonia', 'p'], C_PN),
       ('Lung cancer', sub.loc[sub.Control_subgroup == 'LungCancer', 'p'], C_LC)]
pos = np.arange(len(grp))
for i, (lab, v, c) in enumerate(grp):
    x = np.random.default_rng(7).normal(pos[i], 0.075, len(v))
    E.scatter(x, v, s=16, color=c, alpha=0.35, edgecolors='none', zorder=2)
    bp = E.boxplot([v.values], positions=[pos[i]], widths=0.44, showfliers=False,
                   patch_artist=True, zorder=3)
    bp['boxes'][0].set(facecolor='white', edgecolor=c, linewidth=2.6, alpha=0.9)
    for k in ('whiskers', 'caps'):
        for e in bp[k]:
            e.set(color=c, linewidth=2.2)
    bp['medians'][0].set(color=c, linewidth=3.0)
    E.text(pos[i], 1.045, f"n = {len(v)}\nmedian {v.median():.3f}", ha='center',
           fontsize=13.5, fontweight='bold', color=c)
E.axhline(THR, color='#e08214', lw=2.6, ls='-.')
E.text(2.34, THR * 1.22, 'threshold' + chr(10) + f'{THR:.4f}', ha='left', va='bottom',
       fontsize=13, style='italic', color='#a35c00')
E.set_yscale('log'); E.set_ylim(2e-4, 1.0)
E.set_xticks(pos); E.set_xticklabels([g[0] for g in grp])
E.set_ylabel('Predicted probability (log scale)')
E.set_xlim(-0.62, len(grp) + 0.32)
E.set_title('Predicted risk by final diagnosis', fontsize=19, fontweight='bold', pad=34)
bold(E)

# ---------------- F  分析集森林 ----------------
# 82 例外部结核已全部人工核查为涂片阴性，原 smear-verified 子集只是结构化
# 字段的记录完整性差异，无科学含义，按三方讨论删除；原 3a/3b 顺延为 2a/2b
LAB = {
 'Primary: complete cases (10 variables)': 'Primary (n = 776)',
 'Sensitivity 1: full cohort, frozen imputation': 'Full cohort (n = 1149)',
 'Sensitivity 3a: tuberculosis vs pneumonia': 'vs pneumonia (n = 596)',
 'Sensitivity 3b: tuberculosis vs lung cancer': 'vs lung cancer (n = 241)'}
ks = list(LAB)[::-1]
yy = np.arange(len(ks))
for i, k in enumerate(ks):
    r = res.loc[k]
    F.plot([r.AUROC_lo, r.AUROC_hi], [i + 0.16] * 2, color=C_INT, lw=3.0,
           solid_capstyle='butt', zorder=2)
    F.plot(r.AUROC, i + 0.16, 'o', color=C_INT, ms=11, zorder=3)
    F.plot([r.AUPRC_lo, r.AUPRC_hi], [i - 0.16] * 2, color=C_EXT, lw=3.0,
           solid_capstyle='butt', zorder=2)
    F.plot(r.AUPRC, i - 0.16, 's', color=C_EXT, ms=10, zorder=3)
    F.plot(r.AUPRC_baseline, i - 0.16, '|', color='0.35', ms=16, mew=2.6, zorder=4)
F.axvline(0.5, color='0.6', ls=':', lw=2.0)
F.axvline(0.8259, color=C_INT, ls='--', lw=2.0, alpha=0.6)
F.axvline(0.4991, color=C_EXT, ls='--', lw=2.0, alpha=0.6)
F.text(0.8259, -0.72, 'internal\nAUROC', ha='center', va='top', fontsize=12,
       style='italic', color=C_INT)
F.text(0.4991, -0.72, 'internal\nAUPRC', ha='center', va='top', fontsize=12,
       style='italic', color=C_EXT)
F.set_yticks(yy); F.set_yticklabels([LAB[k] for k in ks], fontsize=15)
F.set_ylim(-1.25, len(ks) + 0.45)
F.set_xlim(0, 1.0)
F.set_xlabel('Performance (95% CI)')
F.set_title('Consistency across analysis sets', fontsize=19, fontweight='bold', pad=10)
h = [plt.Line2D([], [], color=C_INT, marker='o', lw=3, ms=11, label='AUROC'),
     plt.Line2D([], [], color=C_EXT, marker='s', lw=3, ms=10, label='AUPRC'),
     plt.Line2D([], [], color='0.35', marker='|', lw=0, ms=16, mew=2.6,
                label='AUPRC random baseline')]
F.legend(handles=h, loc='upper center', bbox_to_anchor=(0.5, 1.005), ncol=3,
         fontsize=12.5, columnspacing=1.4, handletextpad=0.5)
F.text(0.5, -0.145, 'dashed lines: internal performance from nested cross-validation',
       transform=F.transAxes, ha='center', fontsize=12.5, style='italic', color='0.35')
bold(F)

for txt, ax in zip("ABCDEF", [A, B, C, D, E, F]):
    ax.text(-0.155, 1.06, txt, transform=ax.transAxes, fontsize=30,
            fontweight='bold', va='top', ha='left')

savefig(fig, OUT / "FigureS5_external_validation", formats=("png", "tiff"))
print("Supplementary Figure S5 已出")
print(f"  A/B  外部 AUROC {rp.AUROC:.3f}，AUPRC {rp.AUPRC:.3f}（基线 {yE.mean():.3f}，"
      f"{rp.AUPRC/yE.mean():.2f}x）")
print(f"  C    斜率 {rp.Slope:.3f}，CITL {rp.CITL:+.3f}，Brier {rp.Brier:.4f}")
print(f"  E    结核中位概率 {grp[0][1].median():.3f}，肺炎 {grp[1][1].median():.3f}，"
      f"肺癌 {grp[2][1].median():.3f}")
