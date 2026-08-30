# -*- coding: utf-8 -*-
"""Demographic-only XGBoost 基准模型（Age + Sex）

要求（已锁定）：
  · 队列仍为 4177 例，复用 Aim 2 已保存的 5×5 nested-CV 分区，不重新随机
  · 与 Aim 2 完全相同的 XGBoost：同预处理、同不平衡处理（不加权）、同网格、同 inner-CV 调参
  · 特征固定为 Age + Sex，不做 RFE
  · 在完全配对的折上给两个描述性增量，不做显著性检验（重复 CV 结果相关）

三个性能对象必须分开：
  ① 27 特征 XGBoost           —— 完整模型性能上限
  ② 自适应 nested 特征选择流程 —— 特征选择流程的无偏内部性能（各折用自选 k）
  ③ 固定 k=10 曲线            —— 支持"最终精简特征数"与 92.8% 保留率
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
MODEL = "XGBoost"
LOG = open(OUT / "_s12_log.txt", "w", encoding="utf-8")


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s); LOG.write(s + "\n"); LOG.flush()


S1 = pickle.load(open(CACHE / "s01.pkl", "rb"))
S2 = pickle.load(open(CACHE / "s02.pkl", "rb"))
X, y, FEATS, SPLITS = S1['X'], S1['y'], S1['FEATS'], S1['splits']
NR = len(SPLITS)
iA, iS = FEATS.index('Age'), FEATS.index('Sex')
Xd = X[:, [iA, iS]]
say(f"n={len(y)} 阳性={y.sum()} 患病率={y.mean():.4f}")
say(f"基准模型特征: Age, Sex（复用 Aim 2 的 {NR}×{len(SPLITS[0])} 折分区）\n")

t0 = time.time()
oof_d = np.zeros((NR, len(y)))
for r in range(NR):
    for tr, te in SPLITS[r]:
        gs = GridSearchCV(build_models()[MODEL], TUNE_GRID[MODEL],
                          scoring='average_precision',
                          cv=StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE),
                          n_jobs=1)
        gs.fit(Xd[tr], y[tr])
        oof_d[r, te] = gs.best_estimator_.predict_proba(Xd[te])[:, 1]
say(f"完成 [{time.time()-t0:.0f}s]")

OBJ = {
    'Age + Sex (demographic-only comparator)': oof_d,
    '27-feature full XGBoost': S1['oof'][MODEL],
    'Adaptive nested feature-selection pipeline': S2['results'][MODEL]['proc_oof'],
    'Fixed 10-feature model': S2['results'][MODEL]['curve_oof'][10],
}
say(f"\n{'对象':<46}{'AUPRC':<20}{'AUROC':<20}")
say("-" * 86)
M = {}
for k, O in OBJ.items():
    ap = np.array([average_precision_score(y, O[r]) for r in range(NR)])
    au = np.array([roc_auc_score(y, O[r]) for r in range(NR)])
    M[k] = (ap, au)
    say(f"{k:<46}{f'{ap.mean():.4f} ± {ap.std(ddof=1):.4f}':<20}"
        f"{f'{au.mean():.4f} ± {au.std(ddof=1):.4f}':<20}")
say(f"\n（AUPRC 随机基线 = 患病率 = {y.mean():.4f}）")

base = 'Age + Sex (demographic-only comparator)'
say(f"\n配对增量（逐次重复相减后取 mean ± SD，不做显著性检验）")
say(f"{'比较':<58}{'ΔAUPRC':<20}{'ΔAUROC':<20}")
say("-" * 98)
rows = []
for k in ['27-feature full XGBoost', 'Adaptive nested feature-selection pipeline',
          'Fixed 10-feature model']:
    dp = M[k][0] - M[base][0]
    du = M[k][1] - M[base][1]
    say(f"{k + '  −  Age + Sex':<58}"
        f"{f'{dp.mean():+.4f} ± {dp.std(ddof=1):.4f}':<20}"
        f"{f'{du.mean():+.4f} ± {du.std(ddof=1):.4f}':<20}")
    rows.append({'Model': k, 'AUPRC_mean': M[k][0].mean(), 'AUPRC_sd': M[k][0].std(ddof=1),
                 'AUC_mean': M[k][1].mean(), 'AUC_sd': M[k][1].std(ddof=1),
                 'dAUPRC_mean': dp.mean(), 'dAUPRC_sd': dp.std(ddof=1),
                 'dAUC_mean': du.mean(), 'dAUC_sd': du.std(ddof=1)})

say(f"\n对照：Aim 1 中单指标相对 Age+Sex logistic 的最高 ΔAUC = +0.030（HGB）")
say("     —— 两者统计口径不同，Results 中必须分两句写，不可直接相减。")

pd.DataFrame([{'Model': base, 'AUPRC_mean': M[base][0].mean(),
               'AUPRC_sd': M[base][0].std(ddof=1), 'AUC_mean': M[base][1].mean(),
               'AUC_sd': M[base][1].std(ddof=1)}] + rows).to_excel(
    OUT / "_s12_benchmark_raw.xlsx", index=False)
pickle.dump({'oof_agesex': oof_d, 'metrics': {k: (v[0], v[1]) for k, v in M.items()}},
            open(CACHE / "s12.pkl", "wb"))
say("\n已保存 -> " + str(CACHE / "s12.pkl"))
LOG.close()
