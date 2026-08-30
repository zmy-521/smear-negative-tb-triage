# -*- coding: utf-8 -*-
"""Aim 2 · 步骤 2：特征递减（最终锁定版）

锁定流程（每个 outer fold）：
    使用该折自己在 s01 内层 CV 选出的超参
    → 5 个 inner fold 各自【独立跑 RFE】，评价 k = 3,5,8,10,12,15,20,27
    → 对每个 k 得到 5 个内层 AUPRC 的 mean/SD
    → 【AUPRC 1-SE 规则】定该折的 k：k* = argmax(mean)，
       阈值 = mean(k*) − SD(k*)/√5，取 mean ≥ 阈值 的最小 k
    → 回到【完整 outer-training】重新跑一次 RFE，选出 k 个特征，重新拟合
    → outer-test 只做最终评价

其它：
  · 全模型 = 27 特征（25 检验指标 + Age + Sex）
  · Figure 2E 的阈值来自 outer-training 的内层 OOF（Sens ≥ 90% 时特异度最高）
  · RFE 的采用理由是方法学理由（删除特征后重新估计重要性）；
    此前 seed=42 上的三方法比较仅为内部开发探索，不进入论文、不作性能证据
"""
from public_paths import WORK_ROOT, FONT_PATH
import sys, time, pickle, warnings
from collections import Counter
from pathlib import Path
import numpy as np, pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.models import build_models
from common.ranking import rank_features
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score, roc_auc_score

OUT = WORK_ROOT / "结核-Aim2" / "outputs"
CACHE = OUT / "_cache"
KGRID = [3, 5, 8, 10, 12, 15, 20, 27]
SENS_TARGET = 0.90
RUN = ["XGBoost", "LightGBM"]

S = pickle.load(open(CACHE / "s01.pkl", "rb"))
X, y, FEATS, SPLITS, BP = S['X'], S['y'], S['FEATS'], S['splits'], S['best_params']
P = len(FEATS)
N_REPEAT, N_FOLD = len(SPLITS), len(SPLITS[0])
assert P == KGRID[-1], (P, KGRID[-1])
print(f"n={len(y)} 阳性={y.sum()} 患病率={y.mean():.4f} 全特征={P}")
print(f"k 网格 {KGRID} | 选 k 规则：inner-CV AUPRC 的 1-SE | 阈值 Sens≥{SENS_TARGET:.0%}")
print(f"跑 {RUN}（正式 k 由 XGBoost 决定，LightGBM 作对照）\n")


def thr_sens(yt, sc, target=SENS_TARGET):
    ps = np.sort(sc[yt == 1])[::-1]
    if len(ps) == 0:
        return 0.5
    k = int(np.ceil(target * len(ps)))
    return float(ps[min(k, len(ps)) - 1])


def metrics_at(yt, pred):
    tp = int(((pred == 1) & (yt == 1)).sum()); fp = int(((pred == 1) & (yt == 0)).sum())
    fn = int(((pred == 0) & (yt == 1)).sum()); tn = int(((pred == 0) & (yt == 0)).sum())
    sens = tp/max(tp+fn, 1); spec = tn/max(tn+fp, 1)
    ppv = tp/max(tp+fp, 1); npv = tn/max(tn+fn, 1)
    return sens, spec, ppv, npv, 2*ppv*sens/max(ppv+sens, 1e-12)


def rfe_subsets(Xd, yd, mname, params, cols0=None):
    """递归消除：每删一个就重新拟合、重新估计重要性。返回 {k: 列索引列表}。"""
    cur = list(range(Xd.shape[1])) if cols0 is None else list(cols0)
    out = {len(cur): list(cur)}
    while len(cur) > min(KGRID):
        p = build_models()[mname]
        if params:
            p.set_params(**params)
        order, _ = rank_features(p.fit(Xd[:, cur], yd), Xd[:, cur], yd, mname)
        cur = [cur[i] for i in order[:-1]]
        out[len(cur)] = list(cur)
    return {k: out[k] for k in KGRID if k in out}


results = {}
for m in RUN:
    t0 = time.time()
    curve_oof = {k: np.zeros((N_REPEAT, len(y))) for k in KGRID}
    bin_pred = {k: np.zeros((N_REPEAT, len(y)), dtype=int) for k in KGRID}
    proc_oof = np.zeros((N_REPEAT, len(y)))
    k_chosen, feat_hits, final_sets = [], Counter(), []

    for r in range(N_REPEAT):
        for i, (tr, te) in enumerate(SPLITS[r]):
            params = BP[m][r * N_FOLD + i]
            Xtr, ytr = X[tr], y[tr]

            # ---- 内层：每折独立 RFE，只为定 k ----
            inner = StratifiedKFold(5, shuffle=True, random_state=42)
            iap = {k: [] for k in KGRID}
            ioof = {k: np.zeros(len(tr)) for k in KGRID}
            for itr, ite in inner.split(Xtr, ytr):
                subs = rfe_subsets(Xtr[itr], ytr[itr], m, params)
                for k in KGRID:
                    c = subs[k]
                    p = build_models()[m]
                    if params:
                        p.set_params(**params)
                    s = p.fit(Xtr[itr][:, c], ytr[itr]).predict_proba(Xtr[ite][:, c])[:, 1]
                    iap[k].append(average_precision_score(ytr[ite], s))
                    ioof[k][ite] = s
            mean = {k: float(np.mean(iap[k])) for k in KGRID}
            sd = {k: float(np.std(iap[k], ddof=1)) for k in KGRID}
            kbest = max(KGRID, key=lambda k: mean[k])
            thr1se = mean[kbest] - sd[kbest] / np.sqrt(5)
            ksel = min([k for k in KGRID if mean[k] >= thr1se])
            k_chosen.append(ksel)

            # ---- 回到完整 outer-training 重新 RFE ----
            subs_out = rfe_subsets(Xtr, ytr, m, params)
            feat_hits.update([FEATS[j] for j in subs_out[ksel]])
            final_sets.append([FEATS[j] for j in subs_out[ksel]])

            for k in KGRID:
                c = subs_out[k]
                p = build_models()[m]
                if params:
                    p.set_params(**params)
                sc = p.fit(Xtr[:, c], ytr).predict_proba(X[te][:, c])[:, 1]
                curve_oof[k][r, te] = sc
                bin_pred[k][r, te] = (sc >= thr_sens(ytr, ioof[k])).astype(int)
                if k == ksel:
                    proc_oof[r, te] = sc
        print(f"  {m} repeat {r+1}/{N_REPEAT} 完成  ({time.time()-t0:.0f}s)")

    rows = []
    for k in KGRID:
        ap = np.array([average_precision_score(y, curve_oof[k][r]) for r in range(N_REPEAT)])
        au = np.array([roc_auc_score(y, curve_oof[k][r]) for r in range(N_REPEAT)])
        M = np.array([metrics_at(y, bin_pred[k][r]) for r in range(N_REPEAT)])
        rows.append({'k': k, 'AUPRC_mean': ap.mean(), 'AUPRC_sd': ap.std(ddof=1),
                     'AUC_mean': au.mean(), 'AUC_sd': au.std(ddof=1),
                     'Sens': M[:, 0].mean(), 'Spec': M[:, 1].mean(),
                     'PPV': M[:, 2].mean(), 'NPV': M[:, 3].mean(), 'F1': M[:, 4].mean()})
    curve = pd.DataFrame(rows)
    ap_p = np.array([average_precision_score(y, proc_oof[r]) for r in range(N_REPEAT)])
    au_p = np.array([roc_auc_score(y, proc_oof[r]) for r in range(N_REPEAT)])
    kc = Counter(k_chosen)
    results[m] = {'curve': curve, 'curve_oof': curve_oof, 'bin_pred': bin_pred,
                  'proc_oof': proc_oof, 'k_chosen': k_chosen, 'k_counter': kc,
                  'feat_hits': feat_hits, 'final_sets': final_sets,
                  'proc_AUPRC': (ap_p.mean(), ap_p.std(ddof=1)),
                  'proc_AUC': (au_p.mean(), au_p.std(ddof=1))}

    print(f"\n=== {m} ===  {time.time()-t0:.0f}s")
    print(curve[['k', 'AUPRC_mean', 'AUPRC_sd', 'AUC_mean', 'Sens', 'Spec', 'NPV', 'F1']]
          .to_string(index=False, float_format=lambda v: f'{v:.4f}'))
    print(f"  25 折 1-SE 各自选出的 k: {dict(sorted(kc.items()))}")
    print(f"     中位 {int(np.median(k_chosen))} | 众数 {kc.most_common(1)[0][0]} "
          f"({kc.most_common(1)[0][1]}/25 折)")
    print(f"  整套流程性能（各折用自选 k）: AUPRC {ap_p.mean():.4f} ± {ap_p.std(ddof=1):.4f} | "
          f"AUC {au_p.mean():.4f} ± {au_p.std(ddof=1):.4f}")
    print(f"  特征入选频率: {[f'{a}({b})' for a, b in feat_hits.most_common(20)]}\n")

pickle.dump({'results': results, 'KGRID': KGRID, 'SENS_TARGET': SENS_TARGET,
             'RUN': RUN, 'rule': '1-SE on inner-CV AUPRC'}, open(CACHE / "s02.pkl", "wb"))
print("已保存 ->", CACHE / "s02.pkl")
