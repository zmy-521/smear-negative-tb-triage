# -*- coding: utf-8 -*-
"""特征重要性排序：树模型走 SHAP TreeExplainer，其余走置换重要性（同样只用训练折数据）。"""
import numpy as np
import shap
from sklearn.inspection import permutation_importance

TREE_MODELS = {"XGBoost", "LightGBM", "RF", "DT"}


def shap_values_tree(pipe, X):
    """返回 (n_samples, n_features) 的 SHAP 值矩阵（已过 pipeline 的预处理）。"""
    Xt = X
    for nm, step in pipe.steps[:-1]:
        Xt = step.transform(Xt)
    clf = pipe.steps[-1][1]
    ex = shap.TreeExplainer(clf)
    sv = ex.shap_values(Xt)
    if isinstance(sv, list):          # 老版本二分类返回 [neg, pos]
        sv = sv[1]
    sv = np.asarray(sv)
    if sv.ndim == 3:                  # (n, p, 2)
        sv = sv[:, :, 1]
    return sv


def rank_features(pipe, X, y, model_name, rng=42):
    """在给定数据（应为 outer training fold）上算特征重要性，返回按重要性降序的列索引。"""
    if model_name in TREE_MODELS:
        try:
            sv = shap_values_tree(pipe, X)
            imp = np.abs(sv).mean(axis=0)
            return np.argsort(imp)[::-1], imp
        except Exception:
            pass
    r = permutation_importance(pipe, X, y, scoring='average_precision',
                               n_repeats=5, random_state=rng, n_jobs=1)
    imp = r.importances_mean
    return np.argsort(imp)[::-1], imp
