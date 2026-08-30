# -*- coding: utf-8 -*-
"""Aim 3 出表出图

Figure 3  A 五模型性能森林图 | B 增量森林图 | C  B vs D 的 DCA（重叠即为"无增量"的可视化证据）
Table     主表 A/B/T/D/D7；补充表 C1–C4
纪律：AUPRC 随机基线 = 0.314（子队列患病率），全程标注；主指标 AUPRC，不用 AUC 翻盘
"""
from public_paths import WORK_ROOT, FONT_PATH
import sys, pickle, warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.style import set_paper_style, savefig, bold_ticks, FONT
from sklearn.metrics import average_precision_score, roc_auc_score
from docx import Document
from docx.shared import Pt
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

OUT = WORK_ROOT / "结核-Aim2" / "outputs"
S = pickle.load(open(OUT / "_cache" / "s07.pkl", "rb"))
oof, R, D, y, PREV = S['oof'], S['res'], S['delta'], S['y'], S['PREV']
R = R.set_index('Model')
Dm = D.set_index('Comparison')
C_AP, C_AU = "#2ca02c", "#1f77b4"
MAIN = ['A', 'B', 'T', 'D', 'D7']
# T 模型含 Age + Sex + 4 项 T 细胞，写 "T-cell only" 是事实错误
LBL = {'A': 'A  Demographics', 'B': 'B  Routine (10)', 'T': 'T  Demographics + T-cell',
       'D': 'D  Routine + T-cell', 'D7': 'D7  Routine + extended T-cell'}

# ===================== Figure 3A：五模型性能 =====================
set_paper_style()
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
ypos = np.arange(len(MAIN))[::-1]
for ax, col, sdc, c, lab, ref in [
        (a1, 'AUPRC_mean', 'AUPRC_sd', C_AP, 'AUPRC', PREV),
        (a2, 'AUC_mean', 'AUC_sd', C_AU, 'AUROC', 0.5)]:
    for i, m in enumerate(MAIN):
        v, s = R.loc[m, col], R.loc[m, sdc]
        ax.plot([v - s, v + s], [ypos[i]] * 2, color=c, lw=2.6, solid_capstyle='butt')
        ax.plot(v, ypos[i], 'o', color=c, ms=10)
    ax.axvline(ref, color='0.4', ls='--', lw=1.8)
    ax.set_xlabel(lab)
    ax.set_ylim(-0.6, len(MAIN) - 0.4)
    bold_ticks(ax)
a1.set_yticks(ypos); a1.set_yticklabels([LBL[m] for m in MAIN])
a1.text(PREV, len(MAIN) - 0.55, f'  random = {PREV:.3f}', fontsize=12,
        style='italic', color='0.35', va='top')
a2.text(0.5, len(MAIN) - 0.55, '  random = 0.500', fontsize=12,
        style='italic', color='0.35', va='top')
fig.tight_layout()
savefig(fig, OUT / "Figure3A_aim3_model_performance")
print("Figure 3A 已出")

# ===================== Figure 3B：增量森林图 =====================
CMP = [('B vs A', 'B  vs  A'), ('T vs A', 'T  vs  A'),
       ('D vs B', 'D  vs  B'), ('D7 vs B', 'D7  vs  B')]
set_paper_style()
fig, (b1, b2) = plt.subplots(1, 2, figsize=(13, 5.0), sharey=True)
yp = np.arange(len(CMP))[::-1]
for ax, (pt, lo, hi), c, lab in [
        (b1, ('dAUPRC', 'dAUPRC_lo', 'dAUPRC_hi'), C_AP, r'$\Delta$AUPRC (95% CI)'),
        (b2, ('dAUC', 'dAUC_lo', 'dAUC_hi'), C_AU, r'$\Delta$AUROC (95% CI)')]:
    for i, (k, _) in enumerate(CMP):
        r = Dm.loc[k]
        sig = r[lo] > 0
        ax.plot([r[lo], r[hi]], [yp[i]] * 2, color=c, lw=2.8 if sig else 2.0,
                alpha=1.0 if sig else 0.55, solid_capstyle='butt')
        ax.plot(r[pt], yp[i], 'o' if sig else 'o', color=c, ms=11 if sig else 9,
                mfc=c if sig else 'white', mew=2.2)
    ax.axvline(0, color='0.35', ls='--', lw=1.8)
    ax.axhline(1.5, color='0.85', lw=1.2)          # 分隔"参照 A"与"参照 B"
    ax.set_xlabel(lab); ax.set_ylim(-0.6, len(CMP) - 0.4)
    bold_ticks(ax)
b1.set_yticks(yp); b1.set_yticklabels([t for _, t in CMP])
# 参照系已写在行标签里（"B vs A" / "D vs B"），不再另加文字，避免压住刻度
fig.tight_layout()
savefig(fig, OUT / "Figure3B_aim3_incremental")
print("Figure 3B 已出")

# ===================== Figure 3C：B vs D 的 DCA =====================
set_paper_style()
fig, ax = plt.subplots(figsize=(8, 6))
ths = np.linspace(0.05, 0.80, 150)
n = len(y)


def nb(sc):
    return [((sc >= t) & (y == 1)).sum()/n - ((sc >= t) & (y == 0)).sum()/n * (t/(1-t))
            for t in ths]


ax.axvspan(0.10, 0.50, color='0.90', zorder=0)      # 临床相关分诊阈值区间
ax.plot(ths, nb(oof['B'][0]), color=C_AP, lw=3.2, label='B  Routine (10 variables)')
ax.plot(ths, nb(oof['D'][0]), color="#d62728", lw=2.6, ls='--',
        label='D  Routine + T-cell')
ax.plot(ths, [y.mean() - (1-y.mean())*(t/(1-t)) for t in ths], color='#1f77b4',
        lw=1.8, label='Treat all')
ax.axhline(0, color='k', lw=1.8, ls=':', label='Treat none')
ax.set_xlabel("Threshold probability"); ax.set_ylabel("Net benefit")
ax.set_xlim(0.05, 0.8); ax.set_ylim(-0.05, max(nb(oof['D'][0])) * 1.25)
ax.legend(loc='upper right', fontsize=12)
ax.text(0.30, ax.get_ylim()[0] + 0.012, 'clinically relevant triage range',
        ha='center', fontsize=12.5, style='italic', color='0.35')
bold_ticks(ax); fig.tight_layout()
savefig(fig, OUT / "Figure3C_aim3_DCA")
print("Figure 3C 已出")


# ===================== 表 =====================
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


VARS = {'A': 'Age, sex', 'B': '10 routine clinical laboratory features',
        'T': 'Age, sex, CD3#, CD4#, CD8#, CD4/CD8',
        'D': 'B + CD3#, CD4#, CD8#, CD4/CD8',
        'D7': 'B + CD3#, CD4#, CD8#, CD4/CD8, CD3%, CD4%, CD8%'}
REF = {'A': None, 'B': 'B vs A', 'T': 'T vs A', 'D': 'D vs B', 'D7': 'D7 vs B'}
rows = []
for m in MAIN:
    d = Dm.loc[REF[m]] if REF[m] else None
    rows.append({
        'Model': m, 'Variables': VARS[m], 'No.': R.loc[m, 'n_feat'],
        'AUPRC (mean ± SD)': f"{R.loc[m,'AUPRC_mean']:.3f} ± {R.loc[m,'AUPRC_sd']:.3f}",
        'AUROC (mean ± SD)': f"{R.loc[m,'AUC_mean']:.3f} ± {R.loc[m,'AUC_sd']:.3f}",
        'Reference': REF[m].split(' vs ')[1] if REF[m] else '—',
        'ΔAUPRC (95% CI)': (f"{d['dAUPRC']:+.3f} ({d['dAUPRC_lo']:+.3f} to {d['dAUPRC_hi']:+.3f})"
                            if REF[m] else '—'),
        'ΔAUROC (95% CI)': (f"{d['dAUC']:+.3f} ({d['dAUC_lo']:+.3f} to {d['dAUC_hi']:+.3f})"
                            if REF[m] else '—')})
T3 = pd.DataFrame(rows)
threeline(T3, OUT / "Table3_aim3_incremental_value",
          "Table 3. Incremental value of T-cell immunophenotyping over routine clinical variables "
          "for discriminating smear-negative pulmonary tuberculosis.",
          f"Analyses were restricted to the {len(y)} participants with a complete T-cell panel "
          f"({int(y.sum())} tuberculosis, {int((1-y).sum())} controls). Because the prevalence in this "
          f"subgroup was {PREV:.1%}, the AUPRC expected by chance is {PREV:.3f} rather than that of the "
          f"full development cohort. All models were fitted with identical preprocessing, the same "
          f"XGBoost algorithm and hyperparameter grid, and identical data partitions; feature sets were "
          f"pre-specified and no feature selection was performed. AUPRC and AUROC are the mean ± SD "
          f"across five repetitions of nested stratified five-fold cross-validation. Differences and "
          f"their 95% CIs were obtained by paired stratified bootstrap resampling (1000 replicates) of "
          f"the reference repetition's out-of-fold predictions. AUPRC was the pre-specified primary "
          f"endpoint. The clinical feature set was predefined in the full cohort before subgroup "
          f"analyses; because all compared models were evaluated within the same flow-cytometry subset "
          f"using identical clinical predictors, comparisons focused on differences in performance "
          f"rather than absolute model estimates.\n"
          "AUPRC, area under the precision-recall curve; AUROC, area under the receiver operating "
          "characteristic curve; CD3#/CD4#/CD8#, absolute counts.")
print("Table 3 已出")

sup = []
for m in ['C1', 'C2', 'C3', 'C4']:
    d = Dm.loc[f'{m} vs B']
    sup.append({'Model': m, 'Added parameter': S['DESC'][m].replace('B + ', ''),
                'AUPRC (mean ± SD)': f"{R.loc[m,'AUPRC_mean']:.3f} ± {R.loc[m,'AUPRC_sd']:.3f}",
                'AUROC (mean ± SD)': f"{R.loc[m,'AUC_mean']:.3f} ± {R.loc[m,'AUC_sd']:.3f}",
                'ΔAUPRC vs B (95% CI)': f"{d['dAUPRC']:+.3f} ({d['dAUPRC_lo']:+.3f} to {d['dAUPRC_hi']:+.3f})",
                'ΔAUROC vs B (95% CI)': f"{d['dAUC']:+.3f} ({d['dAUC_lo']:+.3f} to {d['dAUC_hi']:+.3f})"})
threeline(pd.DataFrame(sup), OUT / "TableS_aim3_single_Tcell_parameters",
          "Table S6. Incremental value of individual T-cell parameters added separately to the "
          "routine clinical model.",
          "Each model adds a single T-cell parameter to the 10-variable routine model (Model B). "
          "Individual T-cell parameters showed inconsistent incremental effects when added "
          "separately, and none was reproduced by the complete four-parameter panel (Model D). "
          "No formal correction for multiple comparisons was applied, as this analysis was not "
          "intended as a marker-discovery exercise; these comparisons are exploratory.")
print("Table S (C1–C4) 已出")
print("\n" + T3[['Model', 'AUPRC (mean ± SD)', 'AUROC (mean ± SD)',
                 'ΔAUPRC (95% CI)']].to_string(index=False))
nbB, nbD = np.array(nb(oof['B'][0])), np.array(nb(oof['D'][0]))
d_all = nbD - nbB
rel = (ths >= 0.10) & (ths <= 0.50)          # 临床相关阈值区间
print(f"\nDCA 净获益差 (D − B):")
print(f"   全区间 0.05–0.80  最大 {d_all.max():+.4f}（出现在阈值 {ths[d_all.argmax()]:.2f}）")
print(f"   临床相关 0.10–0.50 最大 {d_all[rel].max():+.4f}  中位 {np.median(d_all[rel]):+.4f}")
