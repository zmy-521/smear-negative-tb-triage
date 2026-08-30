# -*- coding: utf-8 -*-
"""Aim 2 · 步骤 5：Figure 2B / 2C —— out-of-fold SHAP beeswarm

锁定：
  · SHAP 走 out-of-fold —— seed=42 那一次重复，每个 outer fold 的模型只解释它没见过的
    测试折患者，5 折拼成全队列 SHAP 矩阵，不存在训练患者解释自己的乐观偏倚
  · 画在【全 27 特征】上 —— B/C 展示的正是驱动 2D/2E 递减的那个重要性排序
  · 版式固定放平均 AUPRC 前两名：XGBoost、LightGBM
  · 另在全数据最终模型上算一遍做一致性检查（只报秩相关，不进正文图）
"""
from public_paths import WORK_ROOT, FONT_PATH
import sys, pickle, warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import shap

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.style import set_paper_style, savefig, FONT
from common.models import build_models
from common.ranking import shap_values_tree
from scipy.stats import spearmanr

OUT = WORK_ROOT / "结核-Aim2" / "outputs"
CACHE = OUT / "_cache"
S1 = pickle.load(open(CACHE / "s01.pkl", "rb"))
X, y, FEATS, SPLITS, BP = S1['X'], S1['y'], S1['FEATS'], S1['splits'], S1['best_params']
TOP2 = S1['top2']
LAB = {'NEUT_abs': 'NEUT#', 'LYMPH_abs': 'LYMPH#', 'MONO_abs': 'MONO#',
       'EO_abs': 'EO#', 'BASO_abs': 'BASO#', 'RDW_CV': 'RDW-CV',
       'PCT_plateletcrit': 'PCT'}
DISP = [LAB.get(f, f) for f in FEATS]
PANEL = {TOP2[0]: 'B', TOP2[1]: 'C'}
print(f"Figure 2B/2C 模型：{TOP2}")

order_store = {}
for m in TOP2:
    SV = np.zeros((len(y), len(FEATS)))
    for i, (tr, te) in enumerate(SPLITS[0]):          # seed=42 那一次重复
        p = build_models()[m]
        pr = BP[m][i]
        if pr:
            p.set_params(**pr)
        p.fit(X[tr], y[tr])
        SV[te] = shap_values_tree(p, X[te])           # 只解释未见过的测试折
    imp = np.abs(SV).mean(axis=0)
    order_store[m] = imp
    print(f"{m} OOF SHAP top10: "
          + ", ".join(f"{DISP[j]}({imp[j]:.3f})" for j in np.argsort(imp)[::-1][:10]))

    set_paper_style(base_fontsize=16)
    plt.figure(figsize=(9, 8))
    shap.summary_plot(SV, X, feature_names=DISP, max_display=20, show=False,
                      plot_size=None)
    fig = plt.gcf(); ax = plt.gca()
    ax.set_xlabel("SHAP value (impact on model output)", fontweight='bold', fontsize=19)
    for t in ax.get_yticklabels():
        t.set_fontweight('bold'); t.set_fontsize(17)
    for t in ax.get_xticklabels():
        t.set_fontweight('bold'); t.set_fontsize(15)
    fig.tight_layout()
    savefig(fig, OUT / f"Figure2{PANEL[m]}_SHAP_beeswarm_{m}")
    print(f"  Figure 2{PANEL[m]} ({m}) 已出")

# ---- 一致性检查：OOF SHAP 排序 vs 全数据最终模型 SHAP 排序 ----
print("\n一致性检查（OOF SHAP 排序 vs 全数据模型 SHAP 排序，Spearman）:")
for m in TOP2:
    p = build_models()[m]
    pr = BP[m][0]
    if pr:
        p.set_params(**pr)
    p.fit(X, y)
    imp_all = np.abs(shap_values_tree(p, X)).mean(axis=0)
    rho = spearmanr(order_store[m], imp_all).statistic
    print(f"  {m:<10} rho = {rho:.3f}")

pickle.dump({'imp': order_store, 'panel': PANEL, 'disp': DISP},
            open(CACHE / "s05.pkl", "wb"))
print("\n完成")
