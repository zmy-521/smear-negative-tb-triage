from pathlib import Path
import json, pickle
import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]
obj = pickle.load(open(ROOT / "model" / "final_XGBoost_k10_frozen.pkl", "rb"))
features = obj["features"]
assert len(features) == 10
assert abs(obj["threshold_sens90"] - 0.028411144390702248) < 1e-12

rng = np.random.default_rng(0)
X = pd.DataFrame(rng.normal(size=(100, len(features))) * 20 + 60, columns=features)
X["Sex"] = rng.integers(0, 2, len(X))
X.loc[::7, "PAB"] = np.nan
p_pickle = obj["model"].predict_proba(X)[:, 1]
assert np.isfinite(p_pickle).all()

med = json.loads((ROOT / "model" / "imputer_medians.json").read_text(encoding="utf-8"))
Xi = X.copy()
for f in features:
    Xi[f] = Xi[f].fillna(med[f])
booster = xgb.Booster(); booster.load_model(ROOT / "model" / "xgboost_booster.json")
dm = xgb.DMatrix(Xi[features].to_numpy(float), feature_names=features)
p_portable = booster.predict(dm)
assert np.allclose(p_pickle, p_portable, rtol=1e-6, atol=1e-7), np.max(np.abs(p_pickle-p_portable))
print("OK: frozen pickle and portable booster predictions agree")
