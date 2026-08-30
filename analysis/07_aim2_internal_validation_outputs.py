# -*- coding: utf-8 -*-
"""Aim 2 · 步骤 4：Figure 2D / 2E / 2F + 校准 + DCA + Table S4

全部内部评价一律来自 nested CV：
  · 2D/2E/TableS4 —— s02 的各 k 曲线（5 次重复 mean ± SD；2E 阈值来自 outer-training）
  · 2F/校准/DCA  —— seed=42 那一次重复、精简流程的 outer-fold OOF
    图注统一写 "reference nested-CV out-of-fold predictions"
"""
from public_paths import WORK_ROOT, FONT_PATH
import sys, pickle, warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.style import set_paper_style, savefig, bold_ticks, MODEL_COLORS, COLOR_MEAN, FOLD_COLORS, FONT
from sklearn.metrics import roc_curve, roc_auc_score, average_precision_score
from docx import Document
from docx.shared import Pt
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

OUT = WORK_ROOT / "结核-Aim2" / "outputs"
CACHE = OUT / "_cache"
S1 = pickle.load(open(CACHE / "s01.pkl", "rb"))
S2 = pickle.load(open(CACHE / "s02.pkl", "rb"))
S3 = pickle.load(open(CACHE / "s03.pkl", "rb"))
y, SPLITS, KGRID = S1['y'], S1['splits'], S2['KGRID']
PREV = float(y.mean())
KFIN = S3['K_FINAL']
CX, CL = S2['results']['XGBoost']['curve'], S2['results']['LightGBM']['curve']
oof = S2['results']['XGBoost']['proc_oof'][0]          # seed=42
AP_M, AP_S = S2['results']['XGBoost']['proc_AUPRC']
AU_M, AU_S = S2['results']['XGBoost']['proc_AUC']

# ================= Figure 2D：AUPRC vs 特征数 =================
set_paper_style()
fig, ax = plt.subplots(figsize=(8, 6))
for nm, c, col in [('XGBoost', CX, MODEL_COLORS['XGBoost']),
                   ('LightGBM', CL, MODEL_COLORS['LightGBM'])]:
    ax.errorbar(c['k'], c['AUPRC_mean'], yerr=c['AUPRC_sd'], marker='o', ms=8,
                lw=2.4, capsize=4, color=col, label=nm)
ax.axvline(KFIN, color='k', ls='--', lw=1.8)
ax.axhline(PREV, color='0.55', ls=':', lw=1.6)
ax.text(KFIN + 0.6, ax.get_ylim()[1] - 0.035, f'k = {KFIN}', fontsize=14,
        fontweight='bold', style='italic', va='top')
ax.text(3.4, PREV + 0.008, 'prevalence', fontsize=11, style='italic', color='0.4')
ax.set_xlabel("Number of Features"); ax.set_ylabel("AUPRC")
ax.set_xticks(KGRID)
ax.legend(loc='lower right')
bold_ticks(ax); fig.tight_layout()
savefig(fig, OUT / "Figure2D_AUPRC_vs_features")
print("Figure 2D 已出")

# ================= Figure 2E：各指标 vs 特征数 =================
set_paper_style()
fig, ax = plt.subplots(figsize=(8, 6))
spec = [('AUC_mean', 'AUC', '#2ca02c', 'o'), ('Spec', 'Specificity', '#1f77b4', 's'),
        ('Sens', 'Sensitivity', '#d62728', '^'), ('NPV', 'NPV', '#9467bd', 'D'),
        ('F1', 'F1 Score', '#ff7f0e', 'v')]
for col, lab, c, mk in spec:
    ax.plot(CX['k'], CX[col], marker=mk, ms=8, lw=2.4, color=c, label=lab)
ax.axvline(KFIN, color='k', ls='--', lw=1.8)
ax.set_xlabel("Number of Features"); ax.set_ylabel("Performance Metrics")
ax.set_xticks(KGRID); ax.set_ylim(0, 1.22)
ax.legend(loc='upper center', ncol=3, fontsize=12, columnspacing=1.2,
          handletextpad=0.4)
bold_ticks(ax); fig.tight_layout()
savefig(fig, OUT / "Figure2E_metrics_vs_features")
print("Figure 2E 已出")

# ================= Figure 2F：seed=42 五折 ROC =================
set_paper_style()
fig, ax = plt.subplots(figsize=(8, 7))
grid = np.linspace(0, 1, 200); tprs = []
for i, (tr, te) in enumerate(SPLITS[0]):
    fpr, tpr, _ = roc_curve(y[te], oof[te])
    tprs.append(np.interp(grid, fpr, tpr)); tprs[-1][0] = 0.0
    ax.plot(fpr, tpr, color=FOLD_COLORS[i], lw=1.5, alpha=0.75,
            label=f"ROC fold {i+1} (AUC = {roc_auc_score(y[te], oof[te]):.3f})")
mt = np.mean(tprs, axis=0); mt[-1] = 1.0
ax.plot(grid, mt, color=COLOR_MEAN, lw=3.4,
        label=f"Mean ROC (AUC = {roc_auc_score(y, oof):.3f})")
ax.plot([0, 1], [0, 1], ls='--', color='0.6', lw=1.8)
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
ax.legend(loc="lower right", fontsize=11.5)
bold_ticks(ax); fig.tight_layout()
savefig(fig, OUT / "Figure2F_final_5fold_ROC")
print("Figure 2F 已出")

# ================= 校准曲线 =================
set_paper_style()
fig, ax = plt.subplots(figsize=(8, 7))
q = pd.qcut(oof, 10, labels=False, duplicates='drop')
xs = [oof[q == i].mean() for i in range(q.max() + 1)]
ys = [y[q == i].mean() for i in range(q.max() + 1)]
ax.plot([0, max(max(xs), max(ys)) * 1.05], [0, max(max(xs), max(ys)) * 1.05],
        ls='--', color='0.6', lw=1.8, label='Ideal')
ax.plot(xs, ys, 'o-', color=COLOR_MEAN, ms=9, lw=2.6, label='Observed (deciles)')
from sklearn.linear_model import LogisticRegression
lo = np.clip(oof, 1e-6, 1-1e-6); lg = np.log(lo/(1-lo))
cal = LogisticRegression().fit(lg.reshape(-1, 1), y)
ax.set_xlabel("Predicted probability"); ax.set_ylabel("Observed frequency")
ax.legend(loc='upper left')
ax.text(0.97, 0.05, f"Calibration slope = {cal.coef_[0][0]:.3f}\n"
                    f"Intercept = {cal.intercept_[0]:.3f}",
        transform=ax.transAxes, ha='right', fontsize=13, style='italic')
bold_ticks(ax); fig.tight_layout()
savefig(fig, OUT / "FigureS_calibration")
print(f"校准曲线已出  slope={cal.coef_[0][0]:.3f} intercept={cal.intercept_[0]:.3f}")

# ================= DCA =================
set_paper_style()
fig, ax = plt.subplots(figsize=(8, 6))
ths = np.linspace(0.01, 0.60, 120)
n = len(y)
nb_model = [((oof >= t) & (y == 1)).sum()/n - ((oof >= t) & (y == 0)).sum()/n * (t/(1-t))
            for t in ths]
nb_all = [y.mean() - (1-y.mean()) * (t/(1-t)) for t in ths]
ax.plot(ths, nb_model, color=COLOR_MEAN, lw=3.0, label='Host-response model')
ax.plot(ths, nb_all, color='#1f77b4', lw=2.0, ls='-', label='Treat all')
ax.axhline(0, color='k', lw=2.0, ls='--', label='Treat none')
ax.set_xlabel("Threshold probability"); ax.set_ylabel("Net benefit")
ax.set_xlim(0, 0.6); ax.set_ylim(-0.05, max(nb_model) * 1.25)
ax.legend(loc='upper right'); bold_ticks(ax); fig.tight_layout()
savefig(fig, OUT / "FigureS_DCA")
print("DCA 已出")


# ================= Table S4 =================
def _bd(el, e_, sz):
    tag = qn("w:"+e_); e = el.find(tag)
    if e is None:
        e = OxmlElement("w:"+e_); el.append(e)
    e.set(qn("w:val"), "single"); e.set(qn("w:sz"), str(sz))
    e.set(qn("w:space"), "0"); e.set(qn("w:color"), "000000")


def threeline(df, stem, title, note, fs=9):
    doc = Document(); st = doc.styles["Normal"]
    st.font.name = FONT; st.font.size = Pt(fs)
    st.element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
    p = doc.add_paragraph(); r = p.add_run(title); r.bold = True; r.font.size = Pt(fs)
    tb = doc.add_table(rows=1, cols=len(df.columns)); tb.autofit = True
    for i, c in enumerate(df.columns):
        cell = tb.rows[0].cells[i]; cell.text = str(c)
        for pr in cell.paragraphs:
            for run in pr.runs:
                run.bold = True; run.font.size = Pt(fs)
    for _, row in df.iterrows():
        cs = tb.add_row().cells
        for i, c in enumerate(df.columns):
            cs[i].text = str(row[c])
            for pr in cs[i].paragraphs:
                for run in pr.runs:
                    run.font.size = Pt(fs)
    pr_ = tb._tbl.tblPr; b = pr_.find(qn("w:tblBorders"))
    if b is None:
        b = OxmlElement("w:tblBorders"); pr_.append(b)
    _bd(b, "top", 12); _bd(b, "bottom", 12)
    for e_ in ("left", "right", "insideH", "insideV"):
        x = OxmlElement("w:"+e_); x.set(qn("w:val"), "none"); b.append(x)
    for cell in tb.rows[0].cells:
        tp = cell._tc.get_or_add_tcPr(); bb = tp.find(qn("w:tcBorders"))
        if bb is None:
            bb = OxmlElement("w:tcBorders"); tp.append(bb)
        _bd(bb, "bottom", 6)
    for line in note.split("\n"):
        pp = doc.add_paragraph(); rr = pp.add_run(line); rr.font.size = Pt(fs-1)
    doc.save(f"{stem}.docx"); df.to_excel(f"{stem}.xlsx", index=False)


t4 = pd.DataFrame({
    'No. of features': CX['k'],
    'AUPRC (mean ± SD)': CX.apply(lambda r: f"{r['AUPRC_mean']:.3f} ± {r['AUPRC_sd']:.3f}", axis=1),
    'AUC (mean ± SD)': CX.apply(lambda r: f"{r['AUC_mean']:.3f} ± {r['AUC_sd']:.3f}", axis=1),
    'Sensitivity': CX['Sens'].round(3), 'Specificity': CX['Spec'].round(3),
    'PPV': CX['PPV'].round(3), 'NPV': CX['NPV'].round(3), 'F1': CX['F1'].round(3),
    'AUPRC retained (%)': (CX['AUPRC_mean'] / CX['AUPRC_mean'].iloc[-1] * 100).round(1)})
threeline(t4, OUT / "TableS4_performance_by_feature_number",
          "Table S5. Performance of the XGBoost host-response model with different numbers of features.",
          f"All values are out-of-fold estimates from five repetitions of nested stratified "
          f"five-fold cross-validation. Within every outer training fold, features were ranked by "
          f"recursive feature elimination and the feature subset was re-derived; the outer test fold "
          f"was used only for evaluation. Threshold-dependent metrics were obtained by applying, to "
          f"each outer test fold, the high-sensitivity threshold (sensitivity ≥ 90%) derived solely "
          f"from that fold's outer-training inner cross-validation. The final number of features "
          f"(k = {KFIN}) was determined on the complete development cohort by the one-standard-error "
          f"rule applied to inner cross-validated AUPRC. The random-classifier AUPRC equals the "
          f"outcome prevalence ({PREV:.3f}).\n"
          "AUPRC, area under the precision-recall curve; AUC, area under the receiver operating "
          "characteristic curve; PPV, positive predictive value; NPV, negative predictive value.")
print("Table S4 已出")
print(t4.to_string(index=False))

pickle.dump({'cal_slope': float(cal.coef_[0][0]), 'cal_intercept': float(cal.intercept_[0]),
             'nb_model': nb_model, 'nb_all': nb_all, 'ths': ths},
            open(CACHE / "s04.pkl", "wb"))
print(f"\n参考重复(seed=42) AUPRC={average_precision_score(y, oof):.4f} "
      f"AUC={roc_auc_score(y, oof):.4f}；正文报告 nested CV 均值 "
      f"AUPRC={AP_M:.4f}±{AP_S:.4f} AUC={AU_M:.4f}±{AU_S:.4f}")
