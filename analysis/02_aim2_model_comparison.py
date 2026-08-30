# -*- coding: utf-8 -*-
"""Aim 2 · 步骤 1：9 模型比较（嵌套重复交叉验证）

已锁定的统计方案：
  · 外层 5 折分层 CV × 5 次重复（seed 42..46），所有算法用完全相同的切分
  · 内层 5 折 GridSearchCV，优化指标 = AUPRC（average_precision）
  · 主指标 AUPRC，每次重复各算一次，报 mean ± SD（不平均 OOF 概率）
  · 并列竞争集 = 1-SE 规则：mean AUPRC ≥ best − SD_best/√5，超过 4 个取前 4
  · seed=42 那一次的 OOF 预测作为 reference，供阈值/校准/DCA/画图使用
  · 主分析不做类别加权
"""
from public_paths import WORK_ROOT, FONT_PATH
import sys, os, time, json, warnings, pickle
from pathlib import Path
import numpy as np, pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.models import build_models, TUNE_GRID, MODEL_ORDER, RANDOM_STATE
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import average_precision_score, roc_auc_score

DATA = str(WORK_ROOT / "结核数据-25指标-建模集.xlsx")
OUT = WORK_ROOT / "结核-Aim2" / "outputs"
CACHE = OUT / "_cache"
CACHE.mkdir(parents=True, exist_ok=True)
META = ['Group', 'Patient_ID', 'name', 'Age', 'Sex', 'Sex_text', 'diagnosis', 'Admission_diagnosis']
N_REPEAT, N_FOLD = 5, 5

df = pd.read_excel(DATA, dtype={'Patient_ID': str})
labs = [c for c in df.columns if c not in META]
FEATS = ['Age', 'Sex'] + labs
X = df[FEATS].astype(float).values
y = (df['Group'] == 1).astype(int).values
print(f"n={len(y)}  阳性={y.sum()}  对照={(1-y).sum()}  患病率={y.mean():.4f}")
print(f"特征 {len(FEATS)} 个：{', '.join(FEATS)}\n")

# 所有算法共用同一套切分
SPLITS = []
for r in range(N_REPEAT):
    skf = StratifiedKFold(N_FOLD, shuffle=True, random_state=RANDOM_STATE + r)
    SPLITS.append(list(skf.split(X, y)))

oof = {m: np.zeros((N_REPEAT, len(y))) for m in MODEL_ORDER}
best_params = {m: [] for m in MODEL_ORDER}
t_all = time.time()

for name in MODEL_ORDER:
    t0 = time.time()
    grid = TUNE_GRID[name]
    for r in range(N_REPEAT):
        pipe = build_models()[name]
        for tr, te in SPLITS[r]:
            gs = GridSearchCV(pipe, grid, scoring='average_precision',
                              cv=StratifiedKFold(N_FOLD, shuffle=True,
                                                 random_state=RANDOM_STATE),
                              n_jobs=1, refit=True)
            gs.fit(X[tr], y[tr])
            oof[name][r, te] = gs.best_estimator_.predict_proba(X[te])[:, 1]
            best_params[name].append(gs.best_params_)
    ap = [average_precision_score(y, oof[name][r]) for r in range(N_REPEAT)]
    au = [roc_auc_score(y, oof[name][r]) for r in range(N_REPEAT)]
    print(f"{name:<10} AUPRC {np.mean(ap):.4f} ± {np.std(ap, ddof=1):.4f} | "
          f"AUC {np.mean(au):.4f} ± {np.std(au, ddof=1):.4f} | {time.time()-t0:.0f}s")

# ---------- 汇总 ----------
rows = []
for name in MODEL_ORDER:
    ap = np.array([average_precision_score(y, oof[name][r]) for r in range(N_REPEAT)])
    au = np.array([roc_auc_score(y, oof[name][r]) for r in range(N_REPEAT)])
    rows.append({'Model': name, 'AUPRC_mean': ap.mean(), 'AUPRC_sd': ap.std(ddof=1),
                 'AUC_mean': au.mean(), 'AUC_sd': au.std(ddof=1),
                 'AUPRC_reps': list(np.round(ap, 4)), 'AUC_reps': list(np.round(au, 4))})
res = pd.DataFrame(rows).sort_values('AUPRC_mean', ascending=False).reset_index(drop=True)

# ---------- 1-SE 规则 ----------
best = res.iloc[0]
se_best = best['AUPRC_sd'] / np.sqrt(N_REPEAT)
cut = best['AUPRC_mean'] - se_best
comp = res[res['AUPRC_mean'] >= cut]['Model'].tolist()
capped = comp[:4]
print(f"\n最佳模型 {best['Model']}：AUPRC {best['AUPRC_mean']:.4f}，SD {best['AUPRC_sd']:.4f}，"
      f"SE = SD/√5 = {se_best:.4f}")
print(f"1-SE 界 = {cut:.4f}  →  并列竞争集（{len(comp)} 个）：{comp}")
if len(comp) > 4:
    print(f"  超过 4 个，按平均 AUPRC 取前 4：{capped}")
print(f"Figure 2B/2C 用平均 AUPRC 前两名：{res['Model'].tolist()[:2]}")
print(f"Figure 2D 曲线用前 3 名：{res['Model'].tolist()[:3]}")

print("\n=== 9 模型性能（5 次重复 mean ± SD）===")
for _, r in res.iterrows():
    tag = ' ←竞争集' if r['Model'] in capped else ''
    print(f"  {r['Model']:<10} AUPRC {r['AUPRC_mean']:.4f} ± {r['AUPRC_sd']:.4f}   "
          f"AUC {r['AUC_mean']:.4f} ± {r['AUC_sd']:.4f}{tag}")
print(f"\n（AUPRC 随机基线 = 患病率 = {y.mean():.4f}）")
print(f"总耗时 {(time.time()-t_all)/60:.1f} 分钟")

with open(CACHE / "s01.pkl", "wb") as f:
    pickle.dump({'oof': oof, 'res': res, 'competitor_set': capped,
                 'competitor_full': comp, 'se_best': se_best, 'cut': cut,
                 'FEATS': FEATS, 'y': y, 'X': X, 'splits': SPLITS,
                 'best_params': best_params, 'prevalence': float(y.mean()),
                 'top2': res['Model'].tolist()[:2], 'top3': res['Model'].tolist()[:3]}, f)
res.to_excel(OUT / "_s01_model_performance_raw.xlsx", index=False)
print("\n已保存 ->", CACHE / "s01.pkl")
