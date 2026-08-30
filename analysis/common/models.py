# -*- coding: utf-8 -*-
"""9 模型面板 + 调参网格（本课题版）。

与 SKILL 原件的差别（按已锁定的统计方案）：
  · 去掉所有 class_weight / scale_pos_weight —— 主分析不做类别加权，
    不平衡留到阈值选择阶段处理；加权版仅作敏感性分析（BALANCED=True）。
  · 插补与标准化留在 Pipeline 内 → 交叉验证时天然折内拟合，不泄漏。
"""
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
import lightgbm as lgb

RANDOM_STATE = 42
MODEL_ORDER = ["LR", "SVM", "KNN", "ANN", "DT", "RF", "AdaBoost", "XGBoost", "LightGBM"]

_imp = lambda: ("imputer", SimpleImputer(strategy="median"))
_scale = lambda: ("scaler", StandardScaler())

# 紧凑网格：每个模型 3–4 组，控制嵌套 CV 的计算量
TUNE_GRID = {
    "LR":       {"clf__C": [0.01, 0.1, 1.0, 10.0]},
    "SVM":      {"clf__C": [0.1, 1.0, 10.0]},
    "KNN":      {"clf__n_neighbors": [5, 15, 31]},
    "ANN":      {"clf__hidden_layer_sizes": [(16, 8), (32, 16)],
                 "clf__alpha": [1e-3, 1e-2]},
    "DT":       {"clf__max_depth": [3, 5, 7]},
    "RF":       {"clf__max_depth": [5, 10, None]},
    "AdaBoost": {"clf__n_estimators": [200, 500], "clf__learning_rate": [0.05, 0.1]},
    "XGBoost":  {"clf__max_depth": [3, 5], "clf__learning_rate": [0.05, 0.1]},
    "LightGBM": {"clf__num_leaves": [15, 31], "clf__learning_rate": [0.05, 0.1]},
}


def build_models(balanced: bool = False, svm_prob: bool = True) -> dict:
    """balanced=True 仅用于敏感性分析；svm_prob=False 时 SVM 用 decision_function（更快）。"""
    cw = "balanced" if balanced else None
    return {
        "LR": Pipeline([_imp(), _scale(), ("clf", LogisticRegression(
            C=1.0, solver="lbfgs", class_weight=cw,          # L2 为默认，不再显式传 penalty
            max_iter=3000, random_state=RANDOM_STATE))]),
        "SVM": Pipeline([_imp(), _scale(), ("clf", SVC(
            C=1.0, kernel="rbf", gamma="scale", probability=svm_prob,
            class_weight=cw, random_state=RANDOM_STATE))]),
        "KNN": Pipeline([_imp(), _scale(), ("clf", KNeighborsClassifier(
            n_neighbors=15, weights="distance", p=1))]),
        "ANN": Pipeline([_imp(), _scale(), ("clf", MLPClassifier(
            hidden_layer_sizes=(16, 8), activation="tanh", alpha=1e-3,
            learning_rate_init=1e-3, max_iter=1000, early_stopping=True,
            random_state=RANDOM_STATE))]),
        "DT": Pipeline([_imp(), ("clf", DecisionTreeClassifier(
            criterion="gini", max_depth=3, class_weight=cw,
            random_state=RANDOM_STATE))]),
        "RF": Pipeline([_imp(), ("clf", RandomForestClassifier(
            n_estimators=300, max_depth=5, min_samples_split=10, min_samples_leaf=2,
            class_weight=cw, random_state=RANDOM_STATE, n_jobs=1))]),
        "AdaBoost": Pipeline([_imp(), ("clf", AdaBoostClassifier(
            n_estimators=300, learning_rate=0.05, random_state=RANDOM_STATE))]),
        "XGBoost": Pipeline([_imp(), ("clf", XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=3, subsample=0.8,
            colsample_bytree=0.8, scale_pos_weight=1.0, eval_metric="aucpr",
            random_state=RANDOM_STATE, n_jobs=1, verbosity=0))]),
        "LightGBM": Pipeline([_imp(), ("clf", lgb.LGBMClassifier(
            n_estimators=300, learning_rate=0.05, num_leaves=31, max_depth=4,
            subsample=0.8, colsample_bytree=0.8, objective="binary",
            class_weight=cw, random_state=RANDOM_STATE, n_jobs=1, verbose=-1))]),
    }


def score_of(fitted_pipe, X):
    """统一取"越大越像阳性"的分数：优先 predict_proba，其次 decision_function。"""
    if hasattr(fitted_pipe, "predict_proba"):
        try:
            return fitted_pipe.predict_proba(X)[:, 1]
        except Exception:
            pass
    return fitted_pipe.decision_function(X)
