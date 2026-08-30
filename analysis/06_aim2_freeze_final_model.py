# -*- coding: utf-8 -*-
"""Aim 2 · 步骤 3b：把「内部验证 OOF」与「最终 10 特征模型阈值 OOF」彻底分开

对象 A —— 内部验证（不在本脚本产生，来自 s02 nested CV）
    · 正文性能：AUPRC 0.4991 ± 0.0161，AUC 0.8259 ± 0.0076
    · seed=42 的 nested OOF 用于校准 / DCA / Figure 2F，
      图注一律写 "reference nested-CV out-of-fold predictions"，
      不得称为 "final 10-feature model performance"

对象 B —— 最终冻结模型（本脚本产生）
    1. 在【最终 10 个特征】上，用全部 4177 例 5 折 CV、以 AUPRC 优化，
       在原预定义网格内重新确定最终超参
    2. 全量拟合 → 冻结，供第二中心外部验证
    3. 固定这 10 个特征与超参后，另做一次 5 折 cross-fitting 得到同尺度 OOF 概率，
       据此按 Sens ≥ 90% 定部署阈值并冻结
    4. 本步产生的 AUC/AUPRC 仅用于阈值开发，【不作为内部性能结果报告】
       （特征与超参均在全数据上选定，必然乐观）
"""
from public_paths import WORK_ROOT, FONT_PATH
import sys, pickle, warnings
from pathlib import Path
import numpy as np, pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.models import build_models, TUNE_GRID
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import average_precision_score, roc_auc_score

OUT = WORK_ROOT / "结核-Aim2" / "outputs"
CACHE = OUT / "_cache"
MODEL, SEED = "XGBoost", 42
LOG = open(OUT / "_s03b_log.txt", "w", encoding="utf-8")


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s); LOG.write(s + "\n"); LOG.flush()


S1 = pickle.load(open(CACHE / "s01.pkl", "rb"))
S3 = pickle.load(open(CACHE / "s03.pkl", "rb"))
X, y, FEATS = S1['X'], S1['y'], S1['FEATS']
FF = S3['FINAL_FEATS']
IDX = [FEATS.index(f) for f in FF]
Xf = X[:, IDX]
say(f"最终特征 k={len(FF)}: {', '.join(FF)}")
say(f"n={len(y)} 阳性={y.sum()} 患病率={y.mean():.4f}\n")

# ---------- 1. 在最终 10 特征上重新调参 ----------
gs = GridSearchCV(build_models()[MODEL], TUNE_GRID[MODEL], scoring='average_precision',
                  cv=StratifiedKFold(5, shuffle=True, random_state=SEED), n_jobs=1)
gs.fit(Xf, y)
PAR = gs.best_params_
say(f"① 最终超参（10 特征、全 4177 例、5 折 CV、优化 AUPRC、预定义网格内）:")
say(f"   {PAR}")
say(f"   （此前 27 特征上选出的是 {S3['FINAL_PARAMS']}）")

# ---------- 2. 全量拟合并冻结 ----------
final = build_models()[MODEL]
final.set_params(**PAR)
final.fit(Xf, y)
say("\n② 最终模型已在全部 4177 例上拟合完成")

# ---------- 3. cross-fitting 生成同尺度 OOF → 定部署阈值 ----------
skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
oof_thr = np.zeros(len(y))
for tr, te in skf.split(Xf, y):
    p = build_models()[MODEL]; p.set_params(**PAR)
    oof_thr[te] = p.fit(Xf[tr], y[tr]).predict_proba(Xf[te])[:, 1]


def thr_sens(yt, sc, target=0.90):
    ps = np.sort(sc[yt == 1])[::-1]
    k = int(np.ceil(target * len(ps)))
    return float(ps[min(k, len(ps)) - 1])


def mets(yt, sc, t):
    p = (sc >= t).astype(int)
    tp = int(((p == 1) & (yt == 1)).sum()); fp = int(((p == 1) & (yt == 0)).sum())
    fn = int(((p == 0) & (yt == 1)).sum()); tn = int(((p == 0) & (yt == 0)).sum())
    se = tp/max(tp+fn, 1); sp = tn/max(tn+fp, 1)
    pv = tp/max(tp+fp, 1); nv = tn/max(tn+fn, 1)
    return dict(threshold=t, TP=tp, FP=fp, FN=fn, TN=tn, Sens=se, Spec=sp, PPV=pv,
                NPV=nv, F1=2*pv*se/max(pv+se, 1e-12), LRneg=(1-se)/max(sp, 1e-12))


T = thr_sens(y, oof_thr)
M = mets(y, oof_thr, T)
say("\n③ 阈值开发 cross-fitting（固定 10 特征 + 固定超参，5 折，seed=42）")
say(f"   ★ 冻结部署阈值 = {T:.4f}   (Sens ≥ 90% 时特异度最高)")
say(f"   在该 OOF 上: Sens {M['Sens']:.3f}  Spec {M['Spec']:.3f}  PPV {M['PPV']:.3f}  "
    f"NPV {M['NPV']:.3f}  LR− {M['LRneg']:.3f}")
say(f"   混淆: TP {M['TP']}  FP {M['FP']}  FN {M['FN']}  TN {M['TN']}")
say(f"\n   ⚠ 本步的 AUPRC={average_precision_score(y, oof_thr):.4f} / "
    f"AUC={roc_auc_score(y, oof_thr):.4f} 仅用于阈值开发，")
say("     特征与超参均在全数据上选定，必然乐观，【不作为内部性能结果报告】。")
say("\n   对比：此前误用的、来自变 k 流程 OOF 的阈值 = "
    f"{S3['thr_sens']:.4f}（尺度不同，已作废）")

say("\n④ 内部验证结果（来自 s02 nested CV，本步不改动）")
S2 = pickle.load(open(CACHE / "s02.pkl", "rb"))
ap, au = S2['results'][MODEL]['proc_AUPRC'], S2['results'][MODEL]['proc_AUC']
say(f"   AUPRC {ap[0]:.4f} ± {ap[1]:.4f}   AUC {au[0]:.4f} ± {au[1]:.4f}")
say("   校准 / DCA / Figure 2F 使用 seed=42 的 nested OOF，图注写 "
    "'reference nested-CV out-of-fold predictions'")

bundle = {'model': final, 'features': FF, 'params': PAR, 'k': len(FF),
          'model_name': MODEL, 'threshold_sens90': T, 'threshold_metrics_devOOF': M,
          'prevalence': float(y.mean()),
          'provenance': ('Frozen for external validation. Features selected by RFE on all 4177; '
                         'hyperparameters tuned on the final feature set by 5-fold CV (AUPRC); '
                         'deployment threshold from 5-fold cross-fitting OOF at Sens>=90%. '
                         'Internal performance must be cited from the nested CV '
                         '(AUPRC 0.4991+/-0.0161, AUC 0.8259+/-0.0076), NOT from this object.')}
pickle.dump(bundle, open(OUT / f"final_{MODEL}_k{len(FF)}_frozen.pkl", "wb"))
pickle.dump({'oof_threshold_dev': oof_thr, 'threshold': T, 'params': PAR,
             'features': FF}, open(CACHE / "s03b.pkl", "wb"))
say(f"\n冻结模型已更新 -> final_{MODEL}_k{len(FF)}_frozen.pkl")
LOG.close()
