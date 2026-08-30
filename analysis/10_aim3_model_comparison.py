# -*- coding: utf-8 -*-
"""Aim 3：T 细胞流式在低成本常规模型之上的增量价值

队列：625 例流式子队列（结核 196 / 对照 429），**AUPRC 随机基线 = 0.314**
框架：完全沿用 Aim 2 —— 同一 5×5 重复嵌套 CV、同一 XGBoost、同一网格、同一预处理
      特征集**预先固定，不做 RFE**
比较：ΔAUPRC / ΔAUROC，paired stratified bootstrap 1000 次 95%CI，**不用 DeLong**
      （B 是 D 的嵌套模型，且模型高度相关）

模型：
  A    Age + Sex                                （人口学基线）
  B    10 个常规临床变量                          （Aim 2 在全队列预先选定）
  T    Age + Sex + 4 项 T 细胞                    （回答"流式本身有无信息"）
  C1–C4  B + 单个 T 细胞指标（分别建模）
  D    B + CD3# + CD4# + CD8# + CD4/CD8
  D7   B + 7 项 T 细胞（敏感性）
"""
from public_paths import WORK_ROOT, FONT_PATH
import sys, time, pickle, warnings
from pathlib import Path
import numpy as np, pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.models import build_models, TUNE_GRID, RANDOM_STATE
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import average_precision_score, roc_auc_score

OUT = WORK_ROOT / "结核-Aim2" / "outputs"
CACHE = OUT / "_cache"
FULL = str(WORK_ROOT / "结核数据-82指标-已补全.xlsx")
MOD = str(WORK_ROOT / "结核数据-25指标-建模集.xlsx")
MODEL, N_REPEAT, N_FOLD, NBOOT = "XGBoost", 5, 5, 1000
LOG = open(OUT / "_s07_log.txt", "w", encoding="utf-8")


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s); LOG.write(s + "\n"); LOG.flush()


f = pd.read_excel(FULL, dtype={'Patient_ID': str})
f.columns = [c.strip() for c in f.columns]
keep = set(pd.read_excel(MOD, dtype={'Patient_ID': str})['Patient_ID'])
f = f[f['Patient_ID'].isin(keep)].reset_index(drop=True)

CORE7 = ['CD3_T', 'CD4_T', 'CD8_T', 'CD4_CD8_ratio', 'CD3_T_abs', 'CD4_T_abs', 'CD8_T_abs']
sub = f[f[CORE7].notna().all(axis=1)].reset_index(drop=True)
y = (sub['Group'] == 1).astype(int).values
PREV = float(y.mean())
say(f"Aim 3 子队列 n={len(sub)}  结核 {int(y.sum())}  对照 {int((1-y).sum())}")
say(f"★ AUPRC 随机基线 = 患病率 = {PREV:.4f}  （不是全队列的 0.098）\n")

B10 = ['Sex', 'Age', 'UA', 'NEUT_abs', 'CREA', 'ALT', 'RDW_CV', 'PAB', 'EO_abs', 'ALP']
T4 = ['CD3_T_abs', 'CD4_T_abs', 'CD8_T_abs', 'CD4_CD8_ratio']
T7 = T4 + ['CD3_T', 'CD4_T', 'CD8_T']
SETS = {
    'A':  ['Age', 'Sex'],
    'B':  B10,
    'T':  ['Age', 'Sex'] + T4,
    'C1': B10 + ['CD3_T_abs'], 'C2': B10 + ['CD4_T_abs'],
    'C3': B10 + ['CD8_T_abs'], 'C4': B10 + ['CD4_CD8_ratio'],
    'D':  B10 + T4,
    'D7': B10 + T7,
}
DESC = {'A': 'Age + Sex', 'B': '10 routine clinical features', 'T': 'Age + Sex + 4 T-cell',
        'C1': 'B + CD3#', 'C2': 'B + CD4#', 'C3': 'B + CD8#', 'C4': 'B + CD4/CD8',
        'D': 'B + 4 T-cell', 'D7': 'B + 7 T-cell (sensitivity)'}

SPLITS = [list(StratifiedKFold(N_FOLD, shuffle=True,
                               random_state=RANDOM_STATE + r).split(np.zeros(len(y)), y))
          for r in range(N_REPEAT)]

oof, res = {}, []
for name, cols in SETS.items():
    t0 = time.time()
    X = sub[cols].astype(float).values
    O = np.zeros((N_REPEAT, len(y)))
    for r in range(N_REPEAT):
        for tr, te in SPLITS[r]:
            gs = GridSearchCV(build_models()[MODEL], TUNE_GRID[MODEL],
                              scoring='average_precision',
                              cv=StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE),
                              n_jobs=1)
            gs.fit(X[tr], y[tr])
            O[r, te] = gs.best_estimator_.predict_proba(X[te])[:, 1]
    oof[name] = O
    ap = np.array([average_precision_score(y, O[r]) for r in range(N_REPEAT)])
    au = np.array([roc_auc_score(y, O[r]) for r in range(N_REPEAT)])
    res.append({'Model': name, 'Variables': DESC[name], 'n_feat': len(cols),
                'AUPRC_mean': ap.mean(), 'AUPRC_sd': ap.std(ddof=1),
                'AUC_mean': au.mean(), 'AUC_sd': au.std(ddof=1)})
    say(f"{name:<3} {DESC[name]:<32} {len(cols):>2}var  "
        f"AUPRC {ap.mean():.4f} ± {ap.std(ddof=1):.4f}  "
        f"AUC {au.mean():.4f} ± {au.std(ddof=1):.4f}   [{time.time()-t0:.0f}s]")
R = pd.DataFrame(res)

# ---------- 配对分层 bootstrap ----------
rng = np.random.default_rng(42)
ip, ineg = np.where(y == 1)[0], np.where(y == 0)[0]
BOOT = [np.concatenate([rng.choice(ip, len(ip), True), rng.choice(ineg, len(ineg), True)])
        for _ in range(NBOOT)]


def delta(a, b):
    """a − b 的 ΔAUPRC / ΔAUC 与 95%CI（用 seed=42 那一次重复的 OOF，配对重抽样）"""
    sa, sb = oof[a][0], oof[b][0]
    dp = average_precision_score(y, sa) - average_precision_score(y, sb)
    du = roc_auc_score(y, sa) - roc_auc_score(y, sb)
    bp, bu = [], []
    for ii in BOOT:
        yy = y[ii]
        if yy.sum() == 0 or yy.sum() == len(yy):
            continue
        bp.append(average_precision_score(yy, sa[ii]) - average_precision_score(yy, sb[ii]))
        bu.append(roc_auc_score(yy, sa[ii]) - roc_auc_score(yy, sb[ii]))
    return (dp, np.percentile(bp, 2.5), np.percentile(bp, 97.5),
            du, np.percentile(bu, 2.5), np.percentile(bu, 97.5))


PAIRS = [('B', 'A', '常规模型 vs 人口学基线'), ('T', 'A', 'T细胞 vs 人口学基线（流式本身有无信息）'),
         ('C1', 'B', 'B+CD3# vs B'), ('C2', 'B', 'B+CD4# vs B'),
         ('C3', 'B', 'B+CD8# vs B'), ('C4', 'B', 'B+CD4/CD8 vs B'),
         ('D', 'B', '完整4项T细胞 vs B  ★核心问题'), ('D7', 'B', '7项T细胞 vs B（敏感性）')]
say(f"\n{'比较':<34}{'ΔAUPRC (95%CI)':<30}{'ΔAUC (95%CI)':<30}")
say("-" * 94)
rows = []
for a, b, lab in PAIRS:
    dp, pl, ph, du, ul, uh = delta(a, b)
    star = '  ✓' if pl > 0 else ''
    say(f"{lab:<34}{f'{dp:+.4f} ({pl:+.4f}, {ph:+.4f})':<30}"
        f"{f'{du:+.4f} ({ul:+.4f}, {uh:+.4f})':<30}{star}")
    rows.append({'Comparison': f'{a} vs {b}', 'Description': lab,
                 'dAUPRC': dp, 'dAUPRC_lo': pl, 'dAUPRC_hi': ph,
                 'dAUC': du, 'dAUC_lo': ul, 'dAUC_hi': uh})
D = pd.DataFrame(rows)
say("\n✓ = ΔAUPRC 的 95%CI 下界 > 0")
say(f"\n（Δ 与 CI 基于 seed=42 参考重复的 OOF 预测，配对分层 bootstrap {NBOOT} 次；"
    f"表中 mean ± SD 为 5 次重复）")

pickle.dump({'oof': oof, 'res': R, 'delta': D, 'y': y, 'PREV': PREV,
             'SETS': SETS, 'DESC': DESC, 'splits': SPLITS},
            open(CACHE / "s07.pkl", "wb"))
R.to_excel(OUT / "_s07_models_raw.xlsx", index=False)
D.to_excel(OUT / "_s07_delta_raw.xlsx", index=False)
say("\n已保存 -> " + str(CACHE / "s07.pkl"))
LOG.close()
