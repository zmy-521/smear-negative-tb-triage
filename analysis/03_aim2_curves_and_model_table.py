# -*- coding: utf-8 -*-
"""Aim 2 · 步骤 1b：Figure 2A（9 模型合并 ROC）+ Figure S（合并 PR）+ Table S3 性能表

锁定：ROC/PR 与阈值相关指标一律用 **seed=42 那一次重复**的 OOF 预测；
      AUPRC/AUC 的点估计与离散度报 5 次重复的 mean ± SD。
"""
from public_paths import WORK_ROOT, FONT_PATH
import sys, pickle
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.style import set_paper_style, savefig, bold_ticks, MODEL_COLORS, FONT
from common.models import MODEL_ORDER
from sklearn.metrics import roc_curve, roc_auc_score, average_precision_score, precision_recall_curve
from docx import Document
from docx.shared import Pt
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

OUT = WORK_ROOT / "结核-Aim2" / "outputs"
CACHE = OUT / "_cache"
PAN = OUT / "panels" / "Fig3"          # 单格图一律进 panels，根目录只放组图与最终件
PAN.mkdir(parents=True, exist_ok=True)
S = pickle.load(open(CACHE / "s01.pkl", "rb"))
y, oof, res = S['y'], S['oof'], S['res']
PREV = float(y.mean())
REF = 0                                   # 第 0 次重复 = seed 42

order = res['Model'].tolist()             # 按平均 AUPRC 降序

# ---------------- Figure 2A：合并 ROC ----------------
set_paper_style()
fig, ax = plt.subplots(figsize=(8, 7))
for m in order:
    s = oof[m][REF]
    fpr, tpr, _ = roc_curve(y, s)
    ax.plot(fpr, tpr, color=MODEL_COLORS[m], lw=2.2,
            label=f"{m} (AUC = {roc_auc_score(y, s):.3f})")
ax.plot([0, 1], [0, 1], 'k--', lw=1.8, label="Random (AUC = 0.500)")
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
ax.legend(loc="lower right", fontsize=11.5)
bold_ticks(ax)
fig.tight_layout()
savefig(fig, OUT / "Figure2A_combined_ROC")
print("Figure 2A 已出")

# ---------------- Figure S：合并 PR ----------------
set_paper_style()
fig, ax = plt.subplots(figsize=(8, 7))
for m in order:
    s = oof[m][REF]
    pr, rc, _ = precision_recall_curve(y, s)
    ax.plot(rc, pr, color=MODEL_COLORS[m], lw=2.2,
            label=f"{m} (AUPRC = {average_precision_score(y, s):.3f})")
ax.axhline(PREV, color='k', ls='--', lw=1.8, label=f"Random (AUPRC = {PREV:.3f})")
ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
ax.set_xlim(-0.02, 1.02); ax.set_ylim(0, 1.0)
ax.legend(loc="upper right", fontsize=11.5)
bold_ticks(ax)
fig.tight_layout()
savefig(fig, OUT / "FigureS2_combined_PR")
print("Figure S2 已出")


# ---------------- Table S3 ----------------
def thr_youden(yt, sc):
    fpr, tpr, th = roc_curve(yt, sc)
    return float(th[np.argmax(tpr - fpr)])


def thr_sens(yt, sc, target=0.90):
    ps = np.sort(sc[yt == 1])[::-1]
    k = int(np.ceil(target * len(ps)))
    return float(ps[min(k, len(ps)) - 1])


def mets(yt, sc, t):
    p = (sc >= t).astype(int)
    tp = int(((p == 1) & (yt == 1)).sum()); fp = int(((p == 1) & (yt == 0)).sum())
    fn = int(((p == 0) & (yt == 1)).sum()); tn = int(((p == 0) & (yt == 0)).sum())
    sens = tp/max(tp+fn, 1); spec = tn/max(tn+fp, 1)
    ppv = tp/max(tp+fp, 1); npv = tn/max(tn+fn, 1)
    f1 = 2*ppv*sens/max(ppv+sens, 1e-12); acc = (tp+tn)/len(yt)
    lrn = (1-sens)/max(spec, 1e-12)
    return sens, spec, ppv, npv, f1, acc, lrn


rows = []
for m in order:
    r = res[res['Model'] == m].iloc[0]
    s = oof[m][REF]
    ty, ts = thr_youden(y, s), thr_sens(y, s)
    a = mets(y, s, ty); b = mets(y, s, ts)
    rows.append({
        'Model': m,
        'AUPRC (mean ± SD)': f"{r['AUPRC_mean']:.3f} ± {r['AUPRC_sd']:.3f}",
        'AUC (mean ± SD)': f"{r['AUC_mean']:.3f} ± {r['AUC_sd']:.3f}",
        'Youden threshold': f"{ty:.3f}",
        'Sens.a': f"{a[0]:.3f}", 'Spec.a': f"{a[1]:.3f}", 'PPV.a': f"{a[2]:.3f}",
        'NPV.a': f"{a[3]:.3f}", 'F1.a': f"{a[4]:.3f}", 'Accuracy.a': f"{a[5]:.3f}",
        'High-sens. threshold': f"{ts:.3f}",
        'Spec.b': f"{b[1]:.3f}", 'PPV.b': f"{b[2]:.3f}", 'NPV.b': f"{b[3]:.3f}",
        'LR-.b': f"{b[6]:.3f}"})
# 追加 demographic-only 基准行（不是第 10 种算法，仅作对照）
try:
    S12 = pickle.load(open(CACHE / "s12.pkl", "rb"))
    ap, au = S12['metrics']['Age + Sex (demographic-only comparator)']
    s = S12['oof_agesex'][REF]
    ty, ts = thr_youden(y, s), thr_sens(y, s)
    a, b = mets(y, s, ty), mets(y, s, ts)
    rows.append({'Model': 'Age + Sex †',
                 'AUPRC (mean ± SD)': f"{ap.mean():.3f} ± {ap.std(ddof=1):.3f}",
                 'AUC (mean ± SD)': f"{au.mean():.3f} ± {au.std(ddof=1):.3f}",
                 'Youden threshold': f"{ty:.3f}",
                 'Sens.a': f"{a[0]:.3f}", 'Spec.a': f"{a[1]:.3f}", 'PPV.a': f"{a[2]:.3f}",
                 'NPV.a': f"{a[3]:.3f}", 'F1.a': f"{a[4]:.3f}", 'Accuracy.a': f"{a[5]:.3f}",
                 'High-sens. threshold': f"{ts:.3f}",
                 'Spec.b': f"{b[1]:.3f}", 'PPV.b': f"{b[2]:.3f}", 'NPV.b': f"{b[3]:.3f}",
                 'LR-.b': f"{b[6]:.3f}"})
    print("已追加 Age+Sex 基准行")
except FileNotFoundError:
    print("⚠ 未找到 s12.pkl，跳过 Age+Sex 基准行")
t3 = pd.DataFrame(rows)


def _bd(el, edge, sz):
    tag = qn("w:"+edge); e = el.find(tag)
    if e is None:
        e = OxmlElement("w:"+edge); el.append(e)
    e.set(qn("w:val"), "single"); e.set(qn("w:sz"), str(sz))
    e.set(qn("w:space"), "0"); e.set(qn("w:color"), "000000")


def threeline(df, stem, title, note, fs=9):
    doc = Document()
    st = doc.styles["Normal"]; st.font.name = FONT; st.font.size = Pt(fs)
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
    for e in ("left", "right", "insideH", "insideV"):
        x = OxmlElement("w:"+e); x.set(qn("w:val"), "none"); b.append(x)
    for cell in tb.rows[0].cells:
        tp = cell._tc.get_or_add_tcPr(); bb = tp.find(qn("w:tcBorders"))
        if bb is None:
            bb = OxmlElement("w:tcBorders"); tp.append(bb)
        _bd(bb, "bottom", 6)
    for line in note.split("\n"):
        pp = doc.add_paragraph(); rr = pp.add_run(line); rr.font.size = Pt(fs-1)
    doc.save(f"{stem}.docx"); df.to_excel(f"{stem}.xlsx", index=False)


threeline(t3, OUT / "TableS3_model_performance",
          "Table S4. Internal performance of the nine machine-learning models for discriminating "
          "smear-negative pulmonary tuberculosis from non-tuberculous respiratory disease.",
          f"AUPRC and AUC are reported as the mean ± SD across five repetitions of nested "
          f"stratified five-fold cross-validation; all algorithms used identical data splits. "
          f"Threshold-dependent metrics were computed from the out-of-fold predictions of the "
          f"pre-specified reference repetition (seed = 42). Superscript a, metrics at the "
          f"Youden-optimal threshold; superscript b, metrics at the pre-specified high-sensitivity "
          f"operating point (the highest probability threshold achieving sensitivity ≥ 90%). "
          f"The random-classifier AUPRC equals the outcome prevalence ({PREV:.3f}). "
          f"Development cohort n = {len(y)} ({int(y.sum())} tuberculosis, {int((1-y).sum())} controls). "
          f"† The age- and sex-only model is not a tenth algorithm but a demographic-only comparator, "
          f"fitted with the identical learner, hyperparameter grid and data partitions.\n"
          "AUPRC, area under the precision-recall curve; AUC, area under the receiver operating "
          "characteristic curve; PPV, positive predictive value; NPV, negative predictive value; "
          "LR−, negative likelihood ratio.")
print("Table S3 已出")
print(t3[['Model', 'AUPRC (mean ± SD)', 'AUC (mean ± SD)', 'Sens.a', 'Spec.a',
          'Spec.b', 'NPV.b']].to_string(index=False))
