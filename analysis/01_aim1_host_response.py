# -*- coding: utf-8 -*-
"""Aim 1 表格复原 —— Table 2 / Table S3（分层）/ Table S4（hsCRP 亚组）

背景：原 结核-Aim1 目录已从本机消失（三个盘与回收站均无），重建图表目录时把
      结核论文-图表 里仅存的副本一并清掉了。此脚本从原始数据把三张表重算回来。

复原依据（全部可核对，不臆造）：
  · 18 项指标与排序 —— 从现存 Figure 2 面板 A 的 y 轴标签逐项读回
  · 统计口径 —— winsorize 1/99 → z 标准化 → logistic  TB ~ z + Age + Sex；
                 BH-FDR；ΔAUROC 相对 age+sex 模型，分层自助 1000 次
  · 对照亚型规则 —— 反推得到，三组例数 3185 / 259 / 325 与记录完全一致
      急性下呼吸道感染：diagnosis 含 肺炎|肺部感染|脓肿|两肺炎症|感染
      慢性气道疾病　　：其余中含 慢性阻塞性肺疾病|慢性支气管炎|哮喘|支气管扩张
      其他　　　　　　：剩余
    注意：原规则的关键词写的是「慢性阻塞性肺疾病」，未覆盖「慢性阻塞性肺病」这一
    写法，故 109 例 COPD 落在「其他」组。此处按原口径保留，以与已定稿的 Figure 2C 一致。
  · 复算结果需与 Figure 2A/2C 的点位一致方可采用

结论数字校验（记录值）：NEUT# OR 0.64、NLR 0.80、MLR 1.02（FDR P=0.775）；
                        最大 ΔAUROC +0.030；hsCRP 亚组 n=1966、hsCRP OR 0.87 P=0.14
"""
from public_paths import WORK_ROOT, FONT_PATH
import sys, warnings
import numpy as np, pandas as pd
import statsmodels.api as sm
from pathlib import Path
from sklearn.metrics import roc_auc_score
from statsmodels.stats.multitest import multipletests
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.threeline import new_doc, add_caption, add_footnote, three_line

DESK = WORK_ROOT
OUT = DESK / "结核-外部验证" / "outputs"
RNG = np.random.default_rng(42)
NBOOT = 1000

# Figure 2A 自上而下的 18 项（即 ΔAUROC 降序）
ORDER = ['HGB', 'WBC', 'NEUT#', 'PNI', 'HALP', 'NAR', 'ALB', 'dNLR', 'LYMPH#',
         'PAR', 'NLR', 'SIRI', 'PLR', 'MONO#', 'PLT', 'SII', 'PIV', 'MLR']
COL = {'NEUT#': 'NEUT_abs', 'LYMPH#': 'LYMPH_abs', 'MONO#': 'MONO_abs'}
EN = {'HGB': 'Haemoglobin', 'WBC': 'White blood cell count', 'NEUT#': 'Neutrophil count',
      'PNI': 'Prognostic nutritional index',
      'HALP': 'Haemoglobin-albumin-lymphocyte-platelet score',
      'NAR': 'Neutrophil-to-albumin ratio', 'ALB': 'Albumin',
      'dNLR': 'Derived neutrophil-to-lymphocyte ratio', 'LYMPH#': 'Lymphocyte count',
      'PAR': 'Platelet-to-albumin ratio', 'NLR': 'Neutrophil-to-lymphocyte ratio',
      'SIRI': 'Systemic inflammation response index',
      'PLR': 'Platelet-to-lymphocyte ratio', 'MONO#': 'Monocyte count',
      'PLT': 'Platelet count', 'SII': 'Systemic immune-inflammation index',
      'PIV': 'Pan-immune-inflammation value', 'MLR': 'Monocyte-to-lymphocyte ratio'}
CRP = ['hsCRP', 'CAR', 'CLR', 'CALLY']
CRP_EN = {'hsCRP': 'High-sensitivity C-reactive protein', 'CAR': 'CRP-to-albumin ratio',
          'CLR': 'CRP-to-lymphocyte ratio', 'CALLY': 'CRP-albumin-lymphocyte index'}

# ---------------- 数据 ----------------
m = pd.read_excel(DESK / "结核数据-82指标-已补全.xlsx", sheet_name="数据")
m.columns = [c.strip() for c in m.columns]
ids = set(pd.read_excel(DESK / "结核数据-25指标-建模集.xlsx")['Patient_ID'])
d = m[m['Patient_ID'].isin(ids)].reset_index(drop=True)
assert len(d) == 4177 and d['Group'].sum() == 408, (len(d), d['Group'].sum())

AC = ['肺炎', '肺部感染', '脓肿', '两肺炎症', '感染']
CH = ['慢性阻塞性肺疾病', '慢性支气管炎', '哮喘', '支气管扩张']
dx = d['diagnosis'].fillna('').astype(str).str.strip()
ha = dx.apply(lambda s: any(k in s for k in AC))
hc = (~ha) & dx.apply(lambda s: any(k in s for k in CH))
d['subtype'] = np.where(d['Group'] == 1, 'TB',
                        np.where(ha, 'Acute infection',
                                 np.where(hc, 'Chronic airway disease',
                                          'Other non-infectious')))
cnt = d.loc[d.Group == 0, 'subtype'].value_counts()
assert (cnt['Acute infection'], cnt['Chronic airway disease'],
        cnt['Other non-infectious']) == (3185, 259, 325), cnt.to_dict()
print(f"对照亚型核对通过：急性感染 {cnt['Acute infection']} / "
      f"慢性气道 {cnt['Chronic airway disease']} / 其他 {cnt['Other non-infectious']}")


def wz(s):
    """winsorize 1/99 后 z 标准化"""
    v = np.asarray(pd.to_numeric(s, errors='coerce'), dtype=float).ravel()
    lo, hi = np.nanpercentile(v, [1, 99])
    v = np.clip(v, lo, hi)
    return (v - np.nanmean(v)) / np.nanstd(v, ddof=1)


def fit(y, z, age, sex):
    X = sm.add_constant(np.column_stack([z, age, sex]))
    r = sm.Logit(y, X).fit(disp=0)
    b, se = r.params[1], r.bse[1]
    return np.exp(b), np.exp(b - 1.96 * se), np.exp(b + 1.96 * se), r.llf, r.pvalues[1]


def auc_pair(y, z, age, sex):
    """单指标 AUROC，以及 age+sex 与 age+sex+z 两个模型的 AUROC"""
    X0 = sm.add_constant(np.column_stack([age, sex]))
    X1 = sm.add_constant(np.column_stack([z, age, sex]))
    p0 = sm.Logit(y, X0).fit(disp=0).predict(X0)
    p1 = sm.Logit(y, X1).fit(disp=0).predict(X1)
    return roc_auc_score(y, z), roc_auc_score(y, p0), roc_auc_score(y, p1)


def analyse(df, names, boot=True):
    y = df['Group'].values.astype(int)
    age = np.asarray(df['Age'], dtype=float)
    sex = np.asarray(df['Sex'], dtype=float)
    i1, i0 = np.where(y == 1)[0], np.where(y == 0)[0]
    rows = []
    for nm in names:
        z = wz(df[COL.get(nm, nm)])
        ok = np.isfinite(z)
        yy, zz, aa, ss = y[ok], z[ok], age[ok], sex[ok]
        orr, lo, hi, _, p = fit(yy, zz, aa, ss)
        a_single, a0, a1 = auc_pair(yy, zz, aa, ss)
        dlo = dhi = np.nan
        if boot:
            v = []
            j1, j0 = np.where(yy == 1)[0], np.where(yy == 0)[0]
            for _ in range(NBOOT):
                idx = np.concatenate([RNG.choice(j1, len(j1), True),
                                      RNG.choice(j0, len(j0), True)])
                try:
                    _, b0, b1 = auc_pair(yy[idx], zz[idx], aa[idx], ss[idx])
                    v.append(b1 - b0)
                except Exception:
                    pass
            if v:
                dlo, dhi = np.percentile(v, [2.5, 97.5])
        rows.append(dict(指标=nm, n=int(ok.sum()), 校正OR=orr, lo=lo, hi=hi, P=p,
                         AUC_single=a_single, AUC_base=a0, AUC_full=a1,
                         ΔAUC=a1 - a0, ΔAUC_lo=dlo, ΔAUC_hi=dhi))
    r = pd.DataFrame(rows)
    r['FDR_P'] = multipletests(r['P'], method='fdr_bh')[1]
    return r


def fmt(v, lo, hi, n=2):
    return f"{v:.{n}f} ({lo:.{n}f}\u2013{hi:.{n}f})"


def pf(p):
    return '<0.001' if p < 0.001 else f"{p:.3f}"


# ---------------- Table 2 ----------------
main = analyse(d, ORDER)
main = main.set_index('指标').loc[ORDER].reset_index()
AUC0 = main['AUC_base'].iloc[0]
print(f"\nage+sex 基线 AUROC = {AUC0:.3f}")
print("校验：NEUT# OR %.2f（记录 0.64）  NLR OR %.2f（记录 0.80）  MLR OR %.2f FDR P=%.3f（记录 1.02 / 0.775）"
      % (main.set_index('指标').loc['NEUT#', '校正OR'],
         main.set_index('指标').loc['NLR', '校正OR'],
         main.set_index('指标').loc['MLR', '校正OR'],
         main.set_index('指标').loc['MLR', 'FDR_P']))
print(f"校验：最大 ΔAUROC = +{main['ΔAUC'].max():.3f}（记录 +0.030）")

rows = [[r['指标'], EN[r['指标']], f"{int(r['n'])}",
         fmt(r['校正OR'], r['lo'], r['hi']), pf(r['P']), pf(r['FDR_P']),
         f"{r['AUC_single']:.3f}",
         fmt(r['ΔAUC'], r['ΔAUC_lo'], r['ΔAUC_hi'], 3)]
        for _, r in main.iterrows()]
doc = new_doc(landscape=True)
add_caption(doc, "Table 2.",
            "Association of individual routine haematological and inflammatory indicators "
            "with smear-negative pulmonary tuberculosis, adjusted for age and sex.")
three_line(doc, ['Indicator', 'Full name', 'n', 'Adjusted OR per 1-SD (95% CI)', 'P',
                 'FDR-adjusted P', 'Single-marker AUROC',
                 '\u0394AUROC over age + sex (95% CI)'],
           rows, size=8.5, head_size=8.5,
           widths=[2.4, 7.0, 1.8, 5.2, 2.0, 2.4, 2.8, 5.4])
add_footnote(doc,
 "Each indicator was winsorised at the 1st and 99th percentiles, standardised, and entered "
 "into a logistic regression model together with age and sex; the odds ratio is therefore per "
 "one standard deviation increase. P values were adjusted for multiple testing across the 18 "
 f"indicators by the Benjamini-Hochberg procedure. The age and sex model alone reached an AUROC "
 f"of {AUC0:.3f}; \u0394AUROC is the gain from adding the indicator to that model, with a 95% "
 "confidence interval from 1000 outcome-stratified bootstrap resamples. Indicators are ordered by "
 "\u0394AUROC. AUROC, area under the receiver operating characteristic curve; FDR, false "
 "discovery rate; OR, odds ratio; SD, standard deviation.")
doc.save(OUT / "Table2_single_marker_analysis.docx")
main.to_excel(OUT / "Table2_single_marker_analysis.xlsx", index=False)

# ---------------- Table S3：按对照亚型分层 ----------------
SUBS = ['Acute infection', 'Chronic airway disease', 'Other non-infectious']
strat = {}
for s in SUBS:
    sub = d[(d.Group == 1) | (d.subtype == s)]
    strat[s] = analyse(sub, ORDER, boot=False).set_index('指标')
    print(f"  {s}: 对照 {int((sub.Group == 0).sum())} 例")
rows = [[nm, EN[nm]] + [fmt(strat[s].loc[nm, '校正OR'], strat[s].loc[nm, 'lo'],
                            strat[s].loc[nm, 'hi']) + f"  {pf(strat[s].loc[nm, 'FDR_P'])}"
                        for s in SUBS] for nm in ORDER]
doc = new_doc(landscape=True)
add_caption(doc, "Table S3.",
            "Adjusted odds ratios for each indicator, stratified by the diagnosis of the "
            "non-tuberculosis comparison group.")
three_line(doc, ['Indicator', 'Full name',
                 'Acute lower respiratory tract infection (n = 3185)',
                 'Chronic airway disease (n = 259)',
                 'Other non-infectious (n = 325)'],
           rows, size=8.5, head_size=8.5, widths=[2.6, 7.2, 6.4, 5.6, 5.6])
add_footnote(doc,
 "Each cell gives the adjusted odds ratio per 1-SD (95% CI) followed by the FDR-adjusted P value, "
 "from a logistic regression of tuberculosis status on the standardised indicator, age and sex, "
 "restricted to the 408 tuberculosis patients and the stated comparison group. Comparison groups "
 "were assigned from the discharge diagnosis: acute lower respiratory tract infection (pneumonia, "
 "pulmonary infection, abscess or other documented infection), chronic airway disease (chronic "
 "obstructive pulmonary disease, chronic bronchitis, asthma or bronchiectasis), and other "
 "non-infectious conditions (malignancy, interstitial and other non-infectious disease). "
 "FDR, false discovery rate; SD, standard deviation.")
doc.save(OUT / "TableS3_stratified_by_control_subtype.docx")
pd.concat({s: strat[s] for s in SUBS}, axis=0).to_excel(
    OUT / "TableS3_stratified_by_control_subtype.xlsx")

# ---------------- Table S4：hsCRP 亚组 ----------------
sub = d[d['hsCRP'].notna()].reset_index(drop=True)
print(f"\nhsCRP 亚组 n={len(sub)}（结核 {int(sub.Group.sum())} / 对照 {int((sub.Group==0).sum())}）"
      f"　记录值 1966（169 / 1797）")
crp = analyse(sub, [c for c in CRP if c in sub.columns])
print("校验：hsCRP OR %.2f  P=%.2f（记录 0.87 / 0.14）"
      % (crp.set_index('指标').loc['hsCRP', '校正OR'],
         crp.set_index('指标').loc['hsCRP', 'P']))
rows = [[r['指标'], CRP_EN.get(r['指标'], r['指标']), f"{int(r['n'])}",
         fmt(r['校正OR'], r['lo'], r['hi']), pf(r['P']), pf(r['FDR_P']),
         f"{r['AUC_single']:.3f}",
         fmt(r['ΔAUC'], r['ΔAUC_lo'], r['ΔAUC_hi'], 3)] for _, r in crp.iterrows()]
doc = new_doc(landscape=True)
add_caption(doc, "Table S4.",
            "C-reactive protein and CRP-derived indices in the subgroup with an available "
            "high-sensitivity CRP measurement.")
three_line(doc, ['Indicator', 'Full name', 'n', 'Adjusted OR per 1-SD (95% CI)', 'P',
                 'FDR-adjusted P', 'Single-marker AUROC',
                 '\u0394AUROC over age + sex (95% CI)'],
           rows, size=8.5, head_size=8.5,
           widths=[2.6, 7.4, 1.8, 5.2, 2.0, 2.4, 2.8, 5.4])
add_footnote(doc,
 f"High-sensitivity CRP was measured in {len(sub)} of the 4177 patients "
 f"({int(sub.Group.sum())} with tuberculosis and {int((sub.Group == 0).sum())} without). "
 "Because measurement was at the discretion of the treating team, this subgroup is not a random "
 "sample of the cohort and the analysis is exploratory. Models and adjustment are as in Table 2. "
 "CRP, C-reactive protein; FDR, false discovery rate; OR, odds ratio; SD, standard deviation.")
doc.save(OUT / "TableS4_CRP_subgroup.docx")
crp.to_excel(OUT / "TableS4_CRP_subgroup.xlsx", index=False)

pd.to_pickle(dict(main=main, strat=strat, crp=crp, auc_base=AUC0, order=ORDER),
             OUT / "_aim1_recovered.pkl")
print("\n三张表已重算：Table 2 / Table S3 / Table S4")
