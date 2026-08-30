# -*- coding: utf-8 -*-
"""全部表格重排为参考件的房子风格

参考：原始项目的三线表样式（本地路径已移除）
  · 第一列「缩写, 单位」，取消 Full name 列
  · 全称只在表下缩写表出现一次
  · Table S2 与正文 Table 1 版式完全一致（一个训练集、一个验证集）

产出：正文 Table 2 单独 docx；补充表 S1–S8 单独 docx；外加一份合订
      Supplementary_Material.docx（表 + 全部图注），只出 Word，不出 Excel。
"""
from public_paths import WORK_ROOT, FONT_PATH
import sys, json, pickle, warnings
import numpy as np, pandas as pd
from pathlib import Path
from scipy import stats
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.housetable import new_doc, caption, note, abbrev, table

DESK = WORK_ROOT
OUT = DESK / "结核-外部验证" / "outputs" / "tables"
OUT.mkdir(parents=True, exist_ok=True)

# ---------- 显示名：缩写, 单位 ----------
U = {
 'NEUT': 'NEUT, %', 'NEUT_abs': 'NEUT#, ×10\u2079/L', 'LYMPH': 'LYMPH, %',
 'LYMPH_abs': 'LYMPH#, ×10\u2079/L', 'MONO': 'MONO, %', 'MONO_abs': 'MONO#, ×10\u2079/L',
 'EO': 'EO, %', 'EO_abs': 'EO#, ×10\u2079/L', 'BASO': 'BASO, %',
 'BASO_abs': 'BASO#, ×10\u2079/L', 'WBC': 'WBC, ×10\u2079/L', 'RBC': 'RBC, ×10\u00b9\u00b2/L',
 'HGB': 'HGB, g/L', 'HCT': 'HCT, %', 'MCV': 'MCV, fL', 'MCH': 'MCH, pg',
 'MCHC': 'MCHC, g/L', 'RDW_CV': 'RDW-CV, %', 'RDW_SD': 'RDW-SD, fL',
 'PLT': 'PLT, ×10\u2079/L', 'MPV': 'MPV, fL', 'PDW': 'PDW, fL',
 'PCT_plateletcrit': 'PCT, %', 'P_LCR': 'P-LCR, %',
 'ALB': 'ALB, g/L', 'GLB': 'GLB, g/L', 'A_G': 'A/G', 'TP': 'TP, g/L',
 'ALT': 'ALT, U/L', 'AST': 'AST, U/L', 'ALP': 'ALP, U/L', 'GGT': 'GGT, U/L',
 'TBIL': 'TBIL, \u03bcmol/L', 'DBIL': 'DBIL, \u03bcmol/L', 'IBIL': 'IBIL, \u03bcmol/L',
 'TBA': 'TBA, \u03bcmol/L', 'PAB': 'PAB, mg/L', 'BUN': 'BUN, mmol/L',
 'CREA': 'CREA, \u03bcmol/L', 'UA': 'UA, \u03bcmol/L', 'GLU': 'GLU, mmol/L',
 'K': 'K, mmol/L', 'Na': 'Na, mmol/L', 'Cl': 'Cl, mmol/L',
 'NLR': 'NLR', 'dNLR': 'dNLR', 'PLR': 'PLR', 'MLR': 'MLR', 'SII': 'SII',
 'SIRI': 'SIRI', 'PIV': 'PIV', 'PNI': 'PNI', 'HALP': 'HALP', 'NAR': 'NAR', 'PAR': 'PAR',
 'hsCRP': 'hsCRP, mg/L', 'CAR': 'CAR', 'CLR': 'CLR', 'CALLY': 'CALLY',
 'Age': 'Age, years', 'Sex': 'Sex (male), n (%)'}
# Aim 1 森林图里的写法 -> 数据列名
A1COL = {'NEUT#': 'NEUT_abs', 'LYMPH#': 'LYMPH_abs', 'MONO#': 'MONO_abs'}

DEF = {
 'A/G': 'albumin-to-globulin ratio', 'ALB': 'albumin', 'ALP': 'alkaline phosphatase',
 'ALT': 'alanine aminotransferase', 'AST': 'aspartate aminotransferase',
 'BASO': 'basophil percentage', 'BASO#': 'basophil count', 'BUN': 'blood urea nitrogen',
 'CALLY': 'C-reactive protein-albumin-lymphocyte index',
 'CAR': 'C-reactive protein-to-albumin ratio', 'Cl': 'chloride',
 'CLR': 'C-reactive protein-to-lymphocyte ratio', 'CREA': 'creatinine',
 'DBIL': 'direct bilirubin', 'dNLR': 'derived neutrophil-to-lymphocyte ratio',
 'EO': 'eosinophil percentage', 'EO#': 'eosinophil count',
 'GGT': 'gamma-glutamyl transferase', 'GLB': 'globulin', 'GLU': 'glucose',
 'HALP': 'haemoglobin-albumin-lymphocyte-platelet score', 'HCT': 'haematocrit',
 'HGB': 'haemoglobin', 'hsCRP': 'high-sensitivity C-reactive protein',
 'IBIL': 'indirect bilirubin', 'K': 'potassium', 'LYMPH': 'lymphocyte percentage',
 'LYMPH#': 'lymphocyte count', 'MCH': 'mean corpuscular haemoglobin',
 'MCHC': 'mean corpuscular haemoglobin concentration', 'MCV': 'mean corpuscular volume',
 'MLR': 'monocyte-to-lymphocyte ratio', 'MONO': 'monocyte percentage',
 'MONO#': 'monocyte count', 'MPV': 'mean platelet volume', 'Na': 'sodium',
 'NAR': 'neutrophil-to-albumin ratio', 'NEUT': 'neutrophil percentage',
 'NEUT#': 'neutrophil count', 'NLR': 'neutrophil-to-lymphocyte ratio',
 'PAB': 'prealbumin', 'PAR': 'platelet-to-albumin ratio', 'PCT': 'plateletcrit',
 'PDW': 'platelet distribution width', 'PIV': 'pan-immune-inflammation value',
 'P-LCR': 'platelet large cell ratio', 'PLR': 'platelet-to-lymphocyte ratio',
 'PLT': 'platelet count', 'PNI': 'prognostic nutritional index',
 'RBC': 'red blood cell count',
 'RDW-CV': 'red cell distribution width-coefficient of variation',
 'RDW-SD': 'red cell distribution width-standard deviation',
 'SII': 'systemic immune-inflammation index',
 'SIRI': 'systemic inflammation response index', 'TBA': 'total bile acid',
 'TBIL': 'total bilirubin', 'TP': 'total protein', 'UA': 'uric acid',
 'WBC': 'white blood cell count'}


def abbr_of(labels, extra=()):
    """只列出该表实际出现的缩写，按字母序"""
    keys = set()
    for s_ in labels:
        k = str(s_).split(',')[0].strip()
        if k in DEF:
            keys.add(k)
    items = [f"{k}, {DEF[k]}" for k in sorted(keys, key=lambda x: x.lower())]
    items += list(extra)
    return "; ".join(items) + "."


ABBRP = ["AUPRC, area under the precision-recall curve",
         "AUROC, area under the receiver operating characteristic curve"]
ABBR_PERF = (
 "AUPRC, area under the precision-recall curve; AUROC, area under the receiver operating "
 "characteristic curve; CI, confidence interval; NPV, negative predictive value; PPV, positive "
 "predictive value; SD, standard deviation.")

# ---------- 数据 ----------
EXT = pd.read_pickle(DESK / "结核-外部验证" / "outputs" / "_external.pkl")
A1 = pd.read_pickle(DESK / "结核-外部验证" / "outputs" / "_aim1_recovered.pkl")
XV = pickle.load(open(DESK / "结核-外部验证" / "outputs" / "_external_results.pkl", "rb"))
XRES = pd.DataFrame(XV['res'])
OLD = json.load(open(Path(__file__).resolve().parent.parent /
                     "outputs" / "_old_tables.json", encoding="utf-8"))
S1RAW = pd.read_excel(DESK / "补充表1-缺外部集.xlsx")
TRAIN = pd.read_excel(DESK / "结核数据-25指标-建模集.xlsx")

T1ROWS = ['Age', 'Sex', 'NEUT_abs', 'LYMPH_abs', 'MONO_abs', 'EO_abs', 'BASO_abs', 'PLT',
          'HGB', 'MCV', 'MCHC', 'RDW_CV', 'MPV', 'PDW', 'ALB', 'GLB', 'ALT', 'AST',
          'ALP', 'GGT', 'TBIL', 'TBA', 'PAB', 'BUN', 'CREA', 'UA', 'GLU']


def med(s):
    s = pd.to_numeric(s, errors='coerce').dropna()
    return '\u2014' if not len(s) else \
        f"{s.median():.2f} ({s.quantile(.25):.2f}, {s.quantile(.75):.2f})"


def npct(s):
    s = pd.to_numeric(s, errors='coerce').dropna()
    k = int((s == 1).sum())
    return f"{k} ({100*k/len(s):.1f}%)"


def pf(p):
    return '<0.001' if p < 0.001 else f"{p:.3f}"


def pval(a, b, binary=False):
    a = pd.to_numeric(a, errors='coerce').dropna()
    b = pd.to_numeric(b, errors='coerce').dropna()
    if len(a) < 3 or len(b) < 3:
        return '\u2014'
    if binary:
        p = stats.chi2_contingency([[int((a == 1).sum()), int((a == 0).sum())],
                                    [int((b == 1).sum()), int((b == 0).sum())]])[1]
    else:
        p = stats.mannwhitneyu(a, b, alternative='two-sided')[1]
    return pf(p)


def ci(v, lo, hi, n=2):
    return f"{v:.{n}f} ({lo:.{n}f}, {hi:.{n}f})"


BUILD = []          # (num, title, header, rows, widths, notes, landscape, sections)


# ============ 正文 Table 2 = T 细胞增量价值（原 Table 3）============
T2 = dict(num="Table 2.",
          title="Incremental value of T-cell immunophenotyping over routine clinical variables "
                "for discriminating smear-negative pulmonary tuberculosis.",
          header=[r for r in OLD["T3"][:1]][0],
          rows=[r for r in OLD["T3"][1:]],
          widths=[1.6, 5.0, 1.2, 3.0, 3.0, 1.8, 4.0, 4.0],
          notes=["Analyses were restricted to the 625 participants with a complete T-cell panel "
                 "(196 with tuberculosis, 429 without). Because prevalence in this subgroup was "
                 "31.4%, the AUPRC expected by chance is 0.314 rather than that of the full "
                 "training cohort. All models used identical preprocessing, the same XGBoost "
                 "algorithm and hyperparameter grid, and identical data partitions; only the "
                 "feature set differed. Differences are paired estimates with 95% CIs from 1000 "
                 "stratified bootstrap resamples.",
                 "AUPRC, area under the precision-recall curve; AUROC, area under the receiver "
                 "operating characteristic curve; CD3#/CD4#/CD8#, absolute counts; CI, confidence "
                 "interval; SD, standard deviation."],
          landscape=True, sections=set())

# ============ Table S3 = 单指标分析（原正文 Table 2，压缩为 4 列）============
main = A1['main'].set_index('指标')
rows = [[nm, ci(r['校正OR'], r['lo'], r['hi']), pf(r['FDR_P']),
         f"{r['ΔAUC']:+.3f} ({r['ΔAUC_lo']:+.3f}, {r['ΔAUC_hi']:+.3f})"]
        for nm, r in main.loc[A1['order']].iterrows()]
S3 = dict(num="Table S3.",
          title="Association of individual routine indicators with smear-negative pulmonary "
                "tuberculosis, adjusted for age and sex.",
          header=['Indicator', 'Adjusted OR (95% CI)', 'FDR-adjusted p-value',
                  'ΔAUROC (95% CI)'],
          rows=rows, widths=[3.4, 4.6, 3.6, 4.8],
          notes=["Each indicator was winsorised at the 1st and 99th percentiles and standardised; "
                 "odds ratios are therefore per 1-SD increase. p-values were adjusted across the "
                 "18 indicators using the Benjamini-Hochberg procedure. The age- and sex-only "
                 f"model reached an AUROC of {A1['auc_base']:.3f}; ΔAUROC is the gain "
                 "obtained by adding the indicator, with 95% CIs from 1000 outcome-stratified "
                 "bootstrap resamples. Indicators are ordered by ΔAUROC. These results are "
                 "shown graphically in Figure 2A.",
                 abbr_of([r[0] for r in rows], ABBRP[1:] + ['CI, confidence interval',
                          'FDR, false discovery rate', 'OR, odds ratio',
                          'SD, standard deviation'])],
          landscape=False, sections=set())

# ================= Table S1 =================
rows, secs = [], set()
CATMAP = {'血常规': 'Complete blood count', '生化': 'Clinical chemistry',
          '衍生指标': 'Derived immune-inflammatory index'}
for cn, en in CATMAP.items():
    secs.add(len(rows)); rows.append([en, '', '', '', ''])
    for _, r in S1RAW[S1RAW['类别'] == cn].iterrows():
        v = r['指标']
        ne = int(EXT[v].notna().sum()) if v in EXT.columns else 0
        rows.append([U.get(v, v), f"{int(r['可用例数'])}", f"{r['缺失率(%)']:.1f}",
                     f"{ne}", f"{100*(1-ne/len(EXT)):.1f}"])
S1 = dict(num="Table S1.",
          title="Candidate routine laboratory indicators and their availability in the training "
                "and external validation cohorts.",
          header=[['Variables', 'Training cohort', 'Percentage of missing data, %',
                   'External validation cohort', 'Percentage of missing data, %']],
          rows=rows, widths=[4.6, 3.2, 3.6, 3.6, 3.6],
          notes=["Fifty-five routine indicators available at the training centre were screened; "
                 "indicators with more than 40% missingness were removed beforehand. Potassium, "
                 "sodium and chloride (28.7% missing) were subsequently excluded and collinear "
                 "indicators (absolute Spearman rho > 0.6, Figure S1) were reduced to one member "
                 "per cluster, giving the 25 laboratory variables that, with age and sex, formed "
                 "the 27 candidate predictors. Missingness in the external cohort reflects "
                 "test-ordering practice at the second centre; the primary external analysis used "
                 "the 776 patients with complete data for all ten model variables.",
                 abbr_of([r[0] for r in rows])],
          landscape=False, sections=secs)

# ================= Table S2 =================
exT, exC = EXT[EXT.Outcome == 1], EXT[EXT.Outcome == 0]
rows = []
for v in T1ROWS:
    if v == 'Sex':
        rows.append([U[v], npct(exC.Sex), npct(exT.Sex), pval(exT.Sex, exC.Sex, binary=True)])
    else:
        rows.append([U[v], med(exC[v]), med(exT[v]), pval(exT[v], exC[v])])
S2 = dict(num="Table S2.",
          title="Baseline demographic and laboratory characteristics of the external validation "
                "cohort.",
          header=['Characteristic', 'Non-tuberculous respiratory disease (N=1067)',
                  'Pulmonary tuberculosis (N=82)', 'p-value'],
          rows=rows, widths=[4.4, 5.4, 4.6, 2.4],
          notes=["Continuous variables are presented as median (Q1, Q3) and compared using the "
                 "Mann-Whitney U test; categorical variables are presented as n (%) and compared "
                 "using the chi-square test. Analyses were performed on non-imputed data; "
                 "availability of each variable is given in Table S1. The external validation "
                 "cohort comprised 1149 patients consecutively identified at a second centre "
                 "between March 2024 and August 2026: 82 with pulmonary tuberculosis and 1067 "
                 "without (833 pneumonia, 234 lung cancer). All patients in the external cohort, "
                 "including all 82 with pulmonary tuberculosis, were smear-negative, as confirmed "
                 "by manual record review. Variables and units are identical to "
                 "Table 1, which reports the training cohort.",
                 abbr_of([r[0] for r in rows], ['Q1, first quartile', 'Q3, third quartile'])],
          landscape=False, sections=set())

# ================= Table S3 =================
SUBS = [('Acute infection', 'Acute lower respiratory tract infection (n=3185)'),
        ('Chronic airway disease', 'Chronic airway disease (n=259)'),
        ('Other non-infectious', 'Other non-infectious (n=325)')]
st = A1['strat']
rows = []
for nm in A1['order']:
    cells = []
    for k, _ in SUBS:
        r = st[k].loc[nm]
        star = '*' if r['FDR_P'] < 0.05 else ''
        cells.append(ci(r['校正OR'], r['lo'], r['hi']) + star)
    rows.append([nm] + cells)
S4 = dict(num="Table S4.",
          title="Adjusted odds ratios for each indicator, stratified by the diagnosis of the "
                "non-tuberculous comparison group.",
          header=['Variables'] + [t for _, t in SUBS],
          rows=rows, widths=[3.2, 5.4, 4.6, 4.6],
          notes=["Values are adjusted odds ratios per 1-SD (95% CI) from a logistic regression of "
                 "tuberculosis status on the standardised indicator, age and sex, restricted to "
                 "the 408 tuberculosis patients and the stated comparison group. Comparison groups "
                 "were assigned from the discharge diagnosis. *FDR-adjusted p < 0.05.",
                 abbr_of([r[0] for r in rows], ['CI, confidence interval', 'FDR, false discovery rate', 'SD, standard deviation'])],
          landscape=False, sections=set())

# ================= Table S4 =================
crp = A1['crp']
rows = [[U.get(r['指标'], r['指标']), ci(r['校正OR'], r['lo'], r['hi']), pf(r['P']),
         pf(r['FDR_P']), f"{r['AUC_single']:.3f}",
         f"{r['ΔAUC']:+.3f} ({r['ΔAUC_lo']:+.3f}, {r['ΔAUC_hi']:+.3f})"]
        for _, r in crp.iterrows()]
S5 = dict(num="Table S5.",
          title="C-reactive protein and CRP-derived indices in the subgroup with an available "
                "high-sensitivity CRP measurement.",
          header=['Variables', 'Adjusted OR (95% CI)', 'p-value', 'FDR p-value', 'AUROC',
                  '\u0394AUROC (95% CI)'],
          rows=rows, widths=[3.2, 4.4, 2.2, 2.4, 2.2, 4.4],
          notes=["High-sensitivity CRP was measured in 1966 of the 4177 patients (169 with "
                 "tuberculosis, 1797 without). Because measurement was at the discretion of the "
                 "treating team this subgroup is not a random sample and the analysis is "
                 "exploratory. Models and adjustment are as in Table S3.",
                 abbr_of([r[0] for r in rows], ABBRP + ['CI, confidence interval', 'CRP, C-reactive protein', 'FDR, false discovery rate', 'OR, odds ratio', 'SD, standard deviation'])],
          landscape=False, sections=set())

# ================= Table S5 =================
old5 = OLD['S5']   # 数据键沿用旧名，与表号无关
rows = [[r[0].replace(' \u2020', ''), r[1], r[2], r[9], r[4], r[5], r[6], r[7], r[8]]
        for r in old5[1:]]
S6 = dict(num="Table S6.",
          title="Internal performance of the nine machine-learning models for discriminating "
                "smear-negative pulmonary tuberculosis.",
          header=['Model', 'AUPRC (mean ± SD)', 'AUROC (mean ± SD)', 'Accuracy', 'Sensitivity',
                  'Specificity', 'PPV', 'NPV', 'F1 score'],
          rows=rows, widths=[3.0, 3.6, 3.6, 2.4, 2.6, 2.6, 2.2, 2.2, 2.4],
          notes=["AUPRC and AUROC are the mean ± SD across five repetitions of nested stratified "
                 "five-fold cross-validation; all algorithms used identical data splits. "
                 "Threshold-dependent metrics were computed from the out-of-fold predictions at "
                 "the Youden-optimal threshold of each model. The final row is the demographic-only "
                 "benchmark fitted with the same algorithm and splits.",
                 ABBR_PERF + " ANN, artificial neural network; DT, decision tree; KNN, k-nearest "
                 "neighbour; LightGBM, light gradient boosting machine; LR, logistic regression; "
                 "RF, random forest; SVM, support vector machine; XGBoost, extreme gradient "
                 "boosting."],
          landscape=False, sections=set())

# ================= Table S6 =================
S7 = dict(num="Table S7.",
          title="Performance of the XGBoost host-response model with different numbers of "
                "features.",
          header=['No. of features', 'AUPRC (mean ± SD)', 'AUROC (mean ± SD)', 'Sensitivity',
                  'Specificity', 'PPV', 'NPV', 'F1 score', 'AUPRC retained, %'],
          rows=[r for r in OLD['S6'][1:]], widths=[2.8, 3.6, 3.6, 2.6, 2.6, 2.2, 2.2, 2.2, 3.0],
          notes=["All values are out-of-fold estimates from five repetitions of nested stratified "
                 "five-fold cross-validation. Within every outer training fold, features were "
                 "ranked by recursive feature elimination and the subset re-derived; the outer "
                 "test fold was never used for selection. AUPRC retained is expressed relative to "
                 "the full 27-feature model.", ABBR_PERF],
          landscape=False, sections=set())

# ============ Table S8 = 外部验证性能（按三方讨论修订）============
# 修订点：删去内部参照列（内部对照已由 Figure S5 A/B/F 承担）；
#         删去 smear-verified 子集列（82 例外部结核已全部人工核查为涂片阴性，
#         该子集只是结构化字段的记录完整性差异，无科学含义）；
#         原 3a/3b 顺延为 2a/2b；截距重校准列只报判别与校准，
#         阈值类指标一律 "—"（重校准改变了概率尺度，0.0284 不再是同一操作点）。
LABX = {'Primary: complete cases (10 variables)': 'Primary (complete-case)',
        'Sensitivity 1: full cohort, frozen imputation': 'Sensitivity 1 (full cohort)',
        'Sensitivity 3a: tuberculosis vs pneumonia': 'Sensitivity 2a (vs pneumonia)',
        'Sensitivity 3b: tuberculosis vs lung cancer': 'Sensitivity 2b (vs lung cancer)',
        'Secondary: intercept-only recalibration (primary set)': 'Secondary (recalibrated)'}
RECAL = 'Secondary: intercept-only recalibration (primary set)'


def g(r, k, n=3, pm=False):
    v = r.get(k)
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return '—'
    lo, hi = r.get(k + '_lo'), r.get(k + '_hi')
    t = f"{v:+.{n}f}" if pm else f"{v:.{n}f}"
    if lo is None or (isinstance(lo, float) and np.isnan(lo)):
        return t
    return t + (f" ({lo:+.{n}f}, {hi:+.{n}f})" if pm else f" ({lo:.{n}f}, {hi:.{n}f})")


cols = [XRES[XRES['set'] == k].iloc[0].to_dict() for k in LABX]
ISREC = [k == RECAL for k in LABX]
SPEC = [('Patients, n', lambda r: f"{int(r['n'])}", False),
        ('Tuberculosis cases, n', lambda r: f"{int(r['events'])}", False),
        ('Prevalence, %', lambda r: f"{100*r['prevalence']:.2f}", False),
        ('__Discrimination', None, False),
        ('AUROC (95% CI)', lambda r: g(r, 'AUROC'), False),
        ('AUPRC (95% CI)', lambda r: g(r, 'AUPRC'), False),
        ('AUPRC random baseline', lambda r: f"{r['prevalence']:.3f}", False),
        ('AUPRC / random baseline', lambda r: f"{r['AUPRC']/r['prevalence']:.2f}", False),
        ('__Calibration', None, False),
        ('Brier score', lambda r: f"{r['Brier']:.4f}", False),
        ('Calibration-in-the-large', lambda r: g(r, 'CITL', pm=True), False),
        ('Calibration slope (95% CI)', lambda r: g(r, 'Slope'), False),
        ('__Performance at the pre-specified threshold of 0.0284', None, False),
        ('Sensitivity', lambda r: g(r, 'thr_Sens'), True),
        ('Specificity', lambda r: g(r, 'thr_Spec'), True),
        ('PPV', lambda r: g(r, 'thr_PPV'), True),
        ('NPV', lambda r: g(r, 'thr_NPV'), True),
        ('True positive / false positive, n',
         lambda r: f"{int(r['thr_TP'])} / {int(r['thr_FP'])}", True),
        ('False negative / true negative, n',
         lambda r: f"{int(r['thr_FN'])} / {int(r['thr_TN'])}", True),
        ('Patients flagged, %', lambda r: f"{100*r['thr_Flagged']:.1f}", True)]
rows, secs = [], set()
for name, f, thr_only in SPEC:
    if f is None:
        secs.add(len(rows)); rows.append([name[2:]] + [''] * len(cols)); continue
    out = [name]
    for c, isrec in zip(cols, ISREC):
        if thr_only and isrec:
            out.append('—'); continue
        try:
            out.append(f(c))
        except Exception:
            out.append('—')
    rows.append(out)
S8 = dict(num="Table S8.",
          title="Performance of the frozen ten-variable model in the external validation cohort, "
                "by analysis set.",
          header=['Metric'] + [LABX[k] for k in LABX],
          rows=rows, widths=[5.6, 4.0, 4.0, 4.0, 4.0, 3.6],
          notes=["The frozen model — comprising the same ten variables, hyperparameters, "
                 "training-fitted median imputer and operating threshold of 0.0284 — was "
                 "applied without refitting, variable re-selection or threshold re-optimisation. "
                 "All pulmonary tuberculosis cases in the external cohort were smear-negative, as "
                 "confirmed by manual record review. Confidence intervals are percentile intervals "
                 "from 2000 outcome-stratified bootstrap resamples (500 for calibration metrics "
                 "and 1000 for threshold-based metrics). Primary: 776 patients with complete data "
                 "for all ten model variables. Sensitivity 1: all 1149 patients, with missing "
                 "values filled using the training-fitted median imputer. Sensitivity 2a and 2b: "
                 "the primary set restricted to pneumonia and lung cancer controls, respectively. "
                 "Secondary: intercept-only recalibration of the primary set; because "
                 "recalibration changes the probability scale, threshold-based classification "
                 "metrics are not reported for this analysis. Internal performance for comparison "
                 "is shown in Figure S5A, S5B and S5F.", ABBR_PERF],
          landscape=True, sections=secs)

# ================= Table S8 =================
S9 = dict(num="Table S9.",
          title="Incremental value of individual T-cell parameters added separately to the "
                "routine clinical model.",
          header=['Model', 'Added parameter', 'AUPRC (mean ± SD)', 'AUROC (mean ± SD)',
                  '\u0394AUPRC vs B (95% CI)', '\u0394AUROC vs B (95% CI)'],
          rows=[r for r in OLD['S8'][1:]], widths=[2.0, 3.0, 3.4, 3.4, 4.4, 4.4],
          notes=["Each model adds a single T-cell parameter to the ten-variable routine model "
                 "(Model B), in the 625 participants with a complete T-cell panel. Individual "
                 "parameters showed inconsistent incremental effects and none was reproduced by "
                 "the complete four-parameter panel (Model D, Table 2). No formal multiplicity "
                 "correction was applied; these analyses are exploratory.",
                 ABBR_PERF + " CD3#/CD4#/CD8#, absolute counts."],
          landscape=False, sections=set())

# ================= 图注 =================
LEGENDS = [
 ("Figure S1.", "Heat map of Spearman correlation coefficients among the candidate indicators.",
  "Indicators listed in Table S1 were entered into pairwise Spearman correlation analysis in the "
  "training cohort. Ninety-two pairs exceeded an absolute coefficient of 0.6; one member of each "
  "correlated cluster was retained, giving the 25 laboratory variables used for modelling."),
 ("Figure S2.", "Precision-recall curves of the nine machine-learning models in the training "
  "cohort.",
  "Curves are pooled out-of-fold predictions from five repetitions of nested stratified five-fold "
  "cross-validation. The dashed line marks the precision expected by chance, which equals the "
  "prevalence of tuberculosis in the training cohort (0.098)."),
 ("Figure S3.", "Calibration of the final ten-variable model in the training cohort.",
  "Observed frequency against predicted probability by decile of predicted risk, from cross-fitted "
  "out-of-fold predictions. The calibration slope was 0.930 and the calibration intercept "
  "\u22120.010."),
 ("Figure S4.", "Decision curve analysis of the final ten-variable model in the training cohort.",
  "Net benefit of the model against treat-all and treat-none strategies across threshold "
  "probabilities. The model provided greater net benefit than either default strategy across the "
  "0.05 to 0.40 range."),
 ("Figure S5.", "Performance of the frozen ten-variable model in the external validation cohort.",
  "(A) Receiver operating characteristic curves for the external cohort and, for reference, the "
  "cross-fitted predictions of the training cohort. (B) Precision-recall curves with the "
  "corresponding random baselines. (C) Calibration by decile of predicted risk, with calibration "
  "slope, calibration-in-the-large and Brier score. (D) Decision curve for the external cohort; "
  "the vertical line marks the pre-specified operating threshold of 0.0284. (E) Distribution of "
  "predicted probability by final diagnosis. (F) AUROC and AUPRC with 95% confidence intervals "
  "across the pre-specified analysis sets; vertical dashed lines mark the corresponding internal "
  "estimates from nested cross-validation, and short grey marks the AUPRC expected by chance in "
  "each set."),
 ("Figure S6.", "Online triage tool implementing the final ten-variable model.",
  "(A) Input interface and probability output. (B) Feature contributions for an example patient, "
  "expressed as SHAP values on the log-odds scale. The tool reports the pre-specified "
  "high-sensitivity threshold of 2.84% alongside each prediction and is intended to support "
  "prioritisation for confirmatory testing rather than to replace microbiological or molecular "
  "diagnosis."),
]

# ================= 输出 =================
def emit(doc, spec):
    caption(doc, spec['num'], spec['title'])
    table(doc, spec['header'], spec['rows'], widths=spec['widths'],
          section_rows=spec['sections'])
    for t in spec['notes']:
        note(doc, t)


ALL = [S1, S2, S3, S4, S5, S6, S7, S8, S9]
for spec, fn in zip([T2] + ALL,
                    ["Table2_tcell_incremental"] + [f"TableS{i}" for i in range(1, 10)]):
    d = new_doc(landscape=spec['landscape'])
    emit(d, spec)
    d.save(OUT / f"{fn}.docx")

# 合订本
d = new_doc(landscape=False)
from common.housetable import _run
p = d.add_paragraph(); _run(p, "Supplementary Material", 14, bold=True)
p = d.add_paragraph()
_run(p, "Host response profiling and a routine-laboratory prediction model for smear-negative "
        "pulmonary tuberculosis", 11, italic=True)
for spec in ALL:
    sec = d.add_section(8 if spec['landscape'] else 0) if False else None
    emit(d, spec)
p = d.add_paragraph(); p.paragraph_format.space_before = __import__('docx').shared.Pt(16)
_run(p, "Supplementary Figure legends", 12, bold=True)
for num, title, body in LEGENDS:
    p = d.add_paragraph()
    p.paragraph_format.space_before = __import__('docx').shared.Pt(8)
    _run(p, f"{num} ", 10.5, bold=True); _run(p, title, 10.5, bold=True)
    p = d.add_paragraph(); _run(p, body, 10)
d.save(OUT / "Supplementary_Material.docx")

print("表格已重排（房子风格，无 Full name 列，单位并入指标名）")
for spec, fn in zip([T2] + ALL, ["Table 2"] + [f"Table S{i}" for i in range(1, 10)]):
    print(f"  {fn:9s} {len(spec['rows']):3d} 行 × {len(spec['header'][0]) if isinstance(spec['header'][0],(list,tuple)) else len(spec['header']):2d} 列")
print("  合订本 Supplementary_Material.docx（9 张补充表 + 6 条图注）")
print(f"输出目录：{OUT}")
