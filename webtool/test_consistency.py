# -*- coding: utf-8 -*-
"""确认部署用的模型与冻结对象逐例一致，且阈值未被改动。

    python test_consistency.py
"""
import json, pickle
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent
obj = pickle.load(open(ROOT / "model" / "final_model.pkl", "rb"))
cfg = json.loads((ROOT / "config" / "feature_config.json").read_text(encoding="utf-8"))
feats = [f["key"] for f in cfg["features"]]

assert obj["features"] == feats, "变量顺序与冻结模型不一致"
assert abs(obj["threshold_sens90"] - cfg["threshold_sens90"]) < 1e-12, "阈值被改动"
assert obj["k"] == len(feats) == 10

rng = np.random.default_rng(0)
X = pd.DataFrame(rng.normal(size=(200, len(feats))) * 50 + 60, columns=feats)
X["Sex"] = rng.integers(0, 2, 200)
p1 = obj["model"].predict_proba(X)[:, 1]
p2 = obj["model"].predict_proba(X[feats])[:, 1]
assert np.allclose(p1, p2), "列顺序会改变预测"

X2 = X.copy(); X2.loc[::5, "PAB"] = np.nan
assert np.isfinite(obj["model"].predict_proba(X2)[:, 1]).all(), "缺失值未被管线内插补"

print("OK")
print(f"  变量 {feats}")
print(f"  阈值 {obj['threshold_sens90']:.6f}")
print(f"  200 例随机输入预测范围 {p1.min():.4f} – {p1.max():.4f}")
