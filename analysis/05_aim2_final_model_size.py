# -*- coding: utf-8 -*-
"""Aim 2 · 步骤 3：最终模型 + 阈值 / 校准 / DCA

锁定：
  1. 最终 k 在【全部 4177 例】上按同一规则重新确定：
     5 折 CV，每折训练部分独立跑 RFE，评价各 k → AUPRC 的 1-SE 规则 → k_final
     不得人为固定为 nested CV 里的众数 15
  2. 最终特征由【全 4177 例上重新跑 RFE 到 k_final】自然产生
  3. 最终模型仅作为将来第二中心外部验证的冻结模型
  4. 内部性能 / Figure 2F / 校准 / DCA / 阈值 一律使用
     【seed=42 那一次重复、精简流程的 outer-fold OOF 预测】，
     绝不使用最终模型在 4177 例上的表观预测
"""
from public_paths import WORK_ROOT, FONT_PATH
import sys, time, json, pickle, warnings
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
MODEL = "XGBoost"          # 唯一 1-SE 竞争模型
SEED = 42
LOG = open(OUT / "_s03_log.txt", "w", encoding="utf-8")


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s); LOG.write(s + "\n"); LOG.flush()


S1 = pickle.load(open(CACHE / "s01.pkl", "rb"))
S2 = pickle.load(open(CACHE / "s02.pkl", "rb"))
X, y, FEATS, SPLITS, BP = S1['X'], S1['y'], S1['FEATS'], S1['splits'], S1['best_params']
say(f"n={len(y)} 阳性={y.sum()} 患病率={y.mean():.4f} 全特征={len(FEATS)}")


def rfe_subsets(Xd, yd, params):
    cur = list(range(Xd.shape[1]))
    out = {len(cur): list(cur)}
    while len(cur) > min(KGRID):
        p = build_models()[MODEL]
        if params:
            p.set_params(**params)
        order, _ = rank_features(p.fit(Xd[:, cur], yd), Xd[:, cur], yd, MODEL)
        cur = [cur[i] for i in order[:-1]]
        out[len(cur)] = list(cur)
    return {k: out[k] for k in KGRID if k in out}


# ---------- 1. 全 4177 例上确定最终超参 ----------
from sklearn.model_selection import GridSearchCV
from common.models import TUNE_GRID
t0 = time.time()
gs = GridSearchCV(build_models()[MODEL], TUNE_GRID[MODEL], scoring='average_precision',
                  cv=StratifiedKFold(5, shuffle=True, random_state=SEED), n_jobs=1)
gs.fit(X, y)
FINAL_PARAMS = gs.best_params_
say(f"\n① 全数据超参（5 折 CV，优化 AUPRC）: {FINAL_PARAMS}   [{time.time()-t0:.0f}s]")

# ---------- 2. 全 4177 例上按同一规则确定 k_final ----------
t0 = time.time()
skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
ap = {k: [] for k in KGRID}
for tr, te in skf.split(X, y):
    subs = rfe_subsets(X[tr], y[tr], FINAL_PARAMS)      # RFE 只用该折训练部分
    for k in KGRID:
        c = subs[k]
        p = build_models()[MODEL]; p.set_params(**FINAL_PARAMS)
        s = p.fit(X[tr][:, c], y[tr]).predict_proba(X[te][:, c])[:, 1]
        ap[k].append(average_precision_score(y[te], s))
mean = {k: float(np.mean(ap[k])) for k in KGRID}
sd = {k: float(np.std(ap[k], ddof=1)) for k in KGRID}
kbest = max(KGRID, key=lambda k: mean[k])
thr1se = mean[kbest] - sd[kbest] / np.sqrt(5)
K_FINAL = min([k for k in KGRID if mean[k] >= thr1se])
say(f"\n② 全数据 5 折 CV 各 k 的 AUPRC（用于 1-SE 定最终 k）  [{time.time()-t0:.0f}s]")
for k in KGRID:
    mark = ' ←最佳' if k == kbest else (' ←选中' if k == K_FINAL else '')
    say(f"     k={k:>2}  {mean[k]:.4f} ± {sd[k]:.4f}{mark}")
say(f"   最佳 k={kbest}，SE={sd[kbest]/np.sqrt(5):.4f}，1-SE 阈值={thr1se:.4f}")
say(f"   ★ 最终 k = {K_FINAL}")

# ---------- 3. 全数据 RFE → 最终特征 → 冻结模型 ----------
t0 = time.time()
subs_all = rfe_subsets(X, y, FINAL_PARAMS)
FINAL_IDX = subs_all[K_FINAL]
FINAL_FEATS = [FEATS[i] for i in FINAL_IDX]
final_model = build_models()[MODEL]
final_model.set_params(**FINAL_PARAMS)
final_model.fit(X[:, FINAL_IDX], y)
say(f"\n③ 最终特征（全数据 RFE 到 k={K_FINAL}）  [{time.time()-t0:.0f}s]")
say("   " + ", ".join(FINAL_FEATS))

hits = S2['results'][MODEL]['feat_hits']
say("\n   与 nested-CV 入选频率对照（/25 折）:")
for f_ in FINAL_FEATS:
    say(f"     {f_:<12} {hits.get(f_, 0):>2}/25")
drop = [f for f in FEATS if f not in FINAL_FEATS]
say(f"   被剔除 {len(drop)} 项: {', '.join(drop)}")

pickle.dump({'model': final_model, 'features': FINAL_FEATS, 'idx': FINAL_IDX,
             'params': FINAL_PARAMS, 'k': K_FINAL, 'model_name': MODEL,
             'note': 'frozen for external validation; internal performance from nested CV'},
            open(OUT / f"final_{MODEL}_k{K_FINAL}_frozen.pkl", "wb"))
say(f"\n   冻结模型已存 -> final_{MODEL}_k{K_FINAL}_frozen.pkl")

# ---------- 4. 内部性能 / 阈值 / 校准 / DCA：全部来自 seed=42 精简流程 OOF ----------
oof = S2['results'][MODEL]['proc_oof'][0]          # 第 0 次重复 = seed 42
say("\n④ 内部评价（seed=42 那一次重复、精简流程的 outer-fold OOF 预测）")
say(f"   AUPRC = {average_precision_score(y, oof):.4f}   AUC = {roc_auc_score(y, oof):.4f}")


def thr_youden(yt, sc):
    from sklearn.metrics import roc_curve
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
    return dict(threshold=t, TP=tp, FP=fp, FN=fn, TN=tn, Sens=sens, Spec=spec,
                PPV=ppv, NPV=npv, F1=2*ppv*sens/max(ppv+sens, 1e-12),
                Accuracy=(tp+tn)/len(yt), LRneg=(1-sens)/max(spec, 1e-12))


TY, TS = thr_youden(y, oof), thr_sens(y, oof)
MY, MS = mets(y, oof, TY), mets(y, oof, TS)
say(f"\n   Youden 阈值 {TY:.4f}:  Sens {MY['Sens']:.3f} Spec {MY['Spec']:.3f} "
    f"PPV {MY['PPV']:.3f} NPV {MY['NPV']:.3f} F1 {MY['F1']:.3f}")
say(f"   高敏阈值(Sens≥90%) {TS:.4f}:  Sens {MS['Sens']:.3f} Spec {MS['Spec']:.3f} "
    f"PPV {MS['PPV']:.3f} NPV {MS['NPV']:.3f} LR− {MS['LRneg']:.3f}")
say(f"   ★ 冻结用于外部验证的阈值 = {TS:.4f}")

# 5 次重复各自的高敏阈值（稳定性报告，不改选择）
thr_reps = [thr_sens(y, S2['results'][MODEL]['proc_oof'][r]) for r in range(5)]
say(f"   5 次重复各自的高敏阈值: {[f'{t:.4f}' for t in thr_reps]}  "
    f"(range {min(thr_reps):.4f}–{max(thr_reps):.4f})")

pickle.dump({'oof_ref': oof, 'K_FINAL': K_FINAL, 'FINAL_FEATS': FINAL_FEATS,
             'FINAL_PARAMS': FINAL_PARAMS, 'thr_youden': TY, 'thr_sens': TS,
             'mets_youden': MY, 'mets_sens': MS, 'thr_reps': thr_reps,
             'kfinal_table': {'mean': mean, 'sd': sd, 'kbest': kbest, 'thr1se': thr1se}},
            open(CACHE / "s03.pkl", "wb"))
say("\n已保存 -> " + str(CACHE / "s03.pkl"))
LOG.close()
