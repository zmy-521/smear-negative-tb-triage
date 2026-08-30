# -*- coding: utf-8 -*-
"""Figure 2D：特征域逐级叠加

  M1  Age + Sex                                   (2)
  M2  + 血常规                                      (14)
  M3  + 生化                                        (27)  ← 即 Aim 2 的完整候选集
  M4  + 衍生炎症指数                                  (38)

全部复用 Aim 2 的 5×5 nested-CV 分区、同一 XGBoost、同一网格、同一 inner-CV 调参。
M4 额外回答一个问题：文献偏爱的 NLR/SII/PIV/MLR 等衍生指数，
在其原始成分已入模的前提下还能不能再加分。
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
MODEL = "XGBoost"
LOG = open(OUT / "_s13_log.txt", "w", encoding="utf-8")


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s); LOG.write(s + "\n"); LOG.flush()


S1 = pickle.load(open(CACHE / "s01.pkl", "rb"))
S12 = pickle.load(open(CACHE / "s12.pkl", "rb"))
y, SPLITS, FEATS27 = S1['y'], S1['splits'], S1['FEATS']
NR = len(SPLITS)

f = pd.read_excel(FULL, dtype={'Patient_ID': str})
f.columns = [c.strip() for c in f.columns]
keep = pd.read_excel(MOD, dtype={'Patient_ID': str})
f = f[f['Patient_ID'].isin(set(keep['Patient_ID']))].reset_index(drop=True)
assert (f['Group'].values == np.where(y == 1, 1, 0)).all() or len(f) == len(y)

CBC = ['NEUT_abs', 'LYMPH_abs', 'MONO_abs', 'EO_abs', 'BASO_abs', 'PLT', 'HGB',
       'MCV', 'MCHC', 'RDW_CV', 'MPV', 'PDW']
BIO = ['ALB', 'GLB', 'ALT', 'AST', 'ALP', 'GGT', 'TBIL', 'TBA', 'PAB', 'BUN',
       'CREA', 'UA', 'GLU']
DER = ['NLR', 'dNLR', 'PLR', 'MLR', 'SII', 'SIRI', 'PIV', 'PNI', 'HALP', 'NAR', 'PAR']
LEVELS = [("M1", "Age + sex", ['Age', 'Sex']),
          ("M2", "+ Complete blood count", ['Age', 'Sex'] + CBC),
          ("M3", "+ Biochemistry", ['Age', 'Sex'] + CBC + BIO),
          ("M4", "+ Derived inflammatory indices", ['Age', 'Sex'] + CBC + BIO + DER)]
for _, _, cols in LEVELS:
    miss = [c for c in cols if c not in f.columns]
    assert not miss, miss
say(f"n={len(y)} 阳性={y.sum()} 患病率={y.mean():.4f}")
for k, lab, cols in LEVELS:
    say(f"  {k}  {lab:<34} {len(cols):>2} 变量  缺失最高 "
        f"{f[cols].isna().mean().max():.1%}")

# M1 与 M3 已有，直接复用；只跑 M2、M4
have = {'M1': S12['oof_agesex'], 'M3': S1['oof'][MODEL]}
oof = {}
for k, lab, cols in LEVELS:
    if k in have:
        oof[k] = have[k]
        say(f"\n{k} 复用已有结果")
        continue
    t0 = time.time()
    X = f[cols].astype(float).values
    O = np.zeros((NR, len(y)))
    for r in range(NR):
        for tr, te in SPLITS[r]:
            gs = GridSearchCV(build_models()[MODEL], TUNE_GRID[MODEL],
                              scoring='average_precision',
                              cv=StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE),
                              n_jobs=1)
            gs.fit(X[tr], y[tr])
            O[r, te] = gs.best_estimator_.predict_proba(X[te])[:, 1]
    oof[k] = O
    say(f"\n{k} 完成 [{time.time()-t0:.0f}s]")

say(f"\n{'层级':<6}{'变量域':<34}{'n':<5}{'AUPRC':<20}{'AUROC':<20}")
say("-" * 88)
rows = []
M = {}
for k, lab, cols in LEVELS:
    ap = np.array([average_precision_score(y, oof[k][r]) for r in range(NR)])
    au = np.array([roc_auc_score(y, oof[k][r]) for r in range(NR)])
    M[k] = (ap, au)
    say(f"{k:<6}{lab:<34}{len(cols):<5}"
        f"{f'{ap.mean():.4f} ± {ap.std(ddof=1):.4f}':<20}"
        f"{f'{au.mean():.4f} ± {au.std(ddof=1):.4f}':<20}")
    rows.append({'Level': k, 'Domain': lab, 'n_var': len(cols),
                 'AUPRC_mean': ap.mean(), 'AUPRC_sd': ap.std(ddof=1),
                 'AUC_mean': au.mean(), 'AUC_sd': au.std(ddof=1)})

say(f"\n逐级增量（配对相减，mean ± SD，不做显著性检验）")
for a, b in [('M2', 'M1'), ('M3', 'M2'), ('M4', 'M3')]:
    dp = M[a][0] - M[b][0]; du = M[a][1] - M[b][1]
    lab = dict((k, l) for k, l, _ in LEVELS)[a]
    say(f"  {a} − {b}  {lab:<34} ΔAUPRC {dp.mean():+.4f} ± {dp.std(ddof=1):.4f}   "
        f"ΔAUROC {du.mean():+.4f} ± {du.std(ddof=1):.4f}")
dp = M['M4'][0] - M['M1'][0]; du = M['M4'][1] - M['M1'][1]
say(f"  M4 − M1  全部常规候选 vs 人口学        ΔAUPRC {dp.mean():+.4f} ± {dp.std(ddof=1):.4f}   "
    f"ΔAUROC {du.mean():+.4f} ± {du.std(ddof=1):.4f}")
say("\n★ M4 − M3 回答：衍生炎症指数在其原始成分已入模后，是否还能增加判别力。")

pd.DataFrame(rows).to_excel(OUT / "_s13_domain_raw.xlsx", index=False)
pickle.dump({'oof': oof, 'levels': LEVELS, 'metrics': M},
            open(CACHE / "s13.pkl", "wb"))
say("\n已保存 -> " + str(CACHE / "s13.pkl"))
LOG.close()
