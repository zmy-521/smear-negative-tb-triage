# -*- coding: utf-8 -*-
"""外部验证 —— 冻结模型原样套用，一个参数不动

纪律（锁定版）：
  · 特征、超参、插补器中位数、阈值 0.0284 全部来自推导集，外部数据不参与任何拟合
  · 不重选特征、不重调超参、不重找阈值
  · AUPRC 基线用外部患病率
  · 截距重校准只作 secondary，原始冻结模型结果先完整报告

分析集：
  Primary          10 变量齐全的完整病例         n = 776
  Sensitivity 1    全队列 + 冻结插补器           n = 1149
  Sensitivity 2    分对照亚组（肺炎 / 肺癌）
"""
from public_paths import WORK_ROOT, FONT_PATH
import sys, pickle, json, warnings
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.metrics import (roc_auc_score, average_precision_score, brier_score_loss,
                             roc_curve, precision_recall_curve, confusion_matrix)
from sklearn.linear_model import LogisticRegression
warnings.filterwarnings("ignore")

DESK = WORK_ROOT
OUT = DESK / "结核-外部验证" / "outputs"
OUT.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(2026)
NBOOT = 2000

FRZ = pickle.load(open(DESK / "结核-Aim2" / "outputs" / "final_XGBoost_k10_frozen.pkl", "rb"))
PIPE, FEATS, THR = FRZ['model'], FRZ['features'], FRZ['threshold_sens90']
ext = pd.read_pickle(OUT / "_external.pkl")
ext['p'] = PIPE.predict_proba(ext[FEATS])[:, 1]          # 冻结插补器已在 pipeline 内
y_all, p_all = ext['Outcome'].values, ext['p'].values


def logit(p, eps=1e-9):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def calib(y, p):
    """CITL（斜率固定为 1 的截距）与校准斜率"""
    lp = logit(p).reshape(-1, 1)
    sl = LogisticRegression(penalty=None, solver='lbfgs', max_iter=1000).fit(lp, y)
    slope = float(sl.coef_[0][0])
    inter_s = float(sl.intercept_[0])
    # CITL: offset = logit(p)，只估截距
    b, lr = 0.0, logit(p)
    for _ in range(200):
        q = 1 / (1 + np.exp(-(lr + b)))
        g = np.sum(y - q); h = np.sum(q * (1 - q))
        if h < 1e-12:
            break
        step = g / h; b += step
        if abs(step) < 1e-10:
            break
    return b, slope, inter_s


def thr_metrics(y, p, t):
    yh = (p >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yh, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if tp + fn else np.nan
    spec = tn / (tn + fp) if tn + fp else np.nan
    return dict(TP=int(tp), FP=int(fp), FN=int(fn), TN=int(tn),
                Sens=sens, Spec=spec,
                PPV=tp / (tp + fp) if tp + fp else np.nan,
                NPV=tn / (tn + fn) if tn + fn else np.nan,
                LRpos=sens / (1 - spec) if spec < 1 else np.inf,
                LRneg=(1 - sens) / spec if spec > 0 else np.inf,
                Flagged=float(yh.mean()))


def boot_ci(y, p, fn, n=NBOOT):
    """按结局分层的 bootstrap 百分位 CI"""
    i1 = np.where(y == 1)[0]; i0 = np.where(y == 0)[0]
    v = []
    for _ in range(n):
        idx = np.concatenate([RNG.choice(i1, len(i1), True), RNG.choice(i0, len(i0), True)])
        try:
            v.append(fn(y[idx], p[idx]))
        except Exception:
            pass
    v = np.array([x for x in v if np.isfinite(x)])
    return (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) if len(v) else (np.nan, np.nan)


def evaluate(y, p, label):
    n, ev = len(y), int(y.sum())
    prev = y.mean()
    au = roc_auc_score(y, p); ap = average_precision_score(y, p); br = brier_score_loss(y, p)
    citl, slope, _ = calib(y, p)
    r = dict(set=label, n=n, events=ev, prevalence=prev,
             AUROC=au, AUROC_lo=None, AUROC_hi=None,
             AUPRC=ap, AUPRC_lo=None, AUPRC_hi=None,
             AUPRC_baseline=prev, Brier=br, CITL=citl, Slope=slope)
    r['AUROC_lo'], r['AUROC_hi'] = boot_ci(y, p, roc_auc_score)
    r['AUPRC_lo'], r['AUPRC_hi'] = boot_ci(y, p, average_precision_score)
    r['Brier_lo'], r['Brier_hi'] = boot_ci(y, p, brier_score_loss)
    r['Slope_lo'], r['Slope_hi'] = boot_ci(y, p, lambda a, b: calib(a, b)[1], n=500)
    r['CITL_lo'], r['CITL_hi'] = boot_ci(y, p, lambda a, b: calib(a, b)[0], n=500)
    tm = thr_metrics(y, p, THR)
    r.update({f"thr_{k}": v for k, v in tm.items()})
    for m in ('Sens', 'Spec', 'PPV', 'NPV'):
        lo, hi = boot_ci(y, p, lambda a, b, m=m: thr_metrics(a, b, THR)[m], n=1000)
        r[f"thr_{m}_lo"], r[f"thr_{m}_hi"] = lo, hi
    return r


SETS = {}
m10 = ext.complete_10 == 1
SETS['Primary: complete cases (10 variables)'] = m10
SETS['Sensitivity 1: full cohort, frozen imputation'] = pd.Series(True, index=ext.index)
SETS['Sensitivity 2a: tuberculosis vs pneumonia'] = m10 & (
    (ext.Outcome == 1) | (ext.Control_subgroup == 'Pneumonia'))
SETS['Sensitivity 2b: tuberculosis vs lung cancer'] = m10 & (
    (ext.Outcome == 1) | (ext.Control_subgroup == 'LungCancer'))

res = []
for lab, m in SETS.items():
    sub = ext[m]
    if sub['Outcome'].nunique() < 2 or sub['Outcome'].sum() < 5:
        print(f"[skip] {lab}: events={int(sub['Outcome'].sum())}")
        res.append(dict(set=lab, n=len(sub), events=int(sub['Outcome'].sum()),
                        note='too few events'))
        continue
    r = evaluate(sub['Outcome'].values, sub['p'].values, lab)
    res.append(r)
    print(f"{lab}\n   n={r['n']} events={r['events']} prev={r['prevalence']:.4f}\n"
          f"   AUROC {r['AUROC']:.3f} ({r['AUROC_lo']:.3f}-{r['AUROC_hi']:.3f})   "
          f"AUPRC {r['AUPRC']:.3f} ({r['AUPRC_lo']:.3f}-{r['AUPRC_hi']:.3f}) "
          f"[baseline {r['AUPRC_baseline']:.3f}]\n"
          f"   Brier {r['Brier']:.4f}  CITL {r['CITL']:+.3f}  slope {r['Slope']:.3f}\n"
          f"   @0.0284  Sens {r['thr_Sens']:.3f}  Spec {r['thr_Spec']:.3f}  "
          f"PPV {r['thr_PPV']:.3f}  NPV {r['thr_NPV']:.3f}  flagged {r['thr_Flagged']*100:.1f}%")

# ---- secondary: 截距重校准（只调截距，斜率与排序完全不变）----
yP, pP = ext.loc[m10, 'Outcome'].values, ext.loc[m10, 'p'].values
citl, slope, _ = calib(yP, pP)
p_rc = 1 / (1 + np.exp(-(logit(pP) + citl)))
rc = dict(set='Secondary: intercept-only recalibration (primary set)',
          n=len(yP), events=int(yP.sum()), prevalence=yP.mean(),
          AUROC=roc_auc_score(yP, p_rc), AUPRC=average_precision_score(yP, p_rc),
          AUPRC_baseline=yP.mean(), Brier=brier_score_loss(yP, p_rc),
          CITL=calib(yP, p_rc)[0], Slope=calib(yP, p_rc)[1], offset=citl)
res.append(rc)
print(f"\nSecondary recalibration: offset {citl:+.3f}  Brier {rc['Brier']:.4f} "
      f"(was {brier_score_loss(yP, pP):.4f})  AUROC unchanged {rc['AUROC']:.3f}")

df = pd.DataFrame(res)
df.to_excel(OUT / "_external_metrics.xlsx", index=False)
pickle.dump(dict(ext=ext, res=res, THR=THR, FEATS=FEATS,
                 masks={k: v.values for k, v in SETS.items()}),
            open(OUT / "_external_results.pkl", "wb"))
print("\n内部参照（嵌套CV）：AUPRC 0.4991±0.0161，AUROC 0.8259±0.0076，患病率 0.0977")
print("saved ->", OUT / "_external_metrics.xlsx")
