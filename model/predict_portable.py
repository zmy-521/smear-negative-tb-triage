"""Prediction using the portable XGBoost booster + frozen median imputer.

This avoids Python pickle for inference while preserving the fitted booster.
"""
from pathlib import Path
import json
import numpy as np
import xgboost as xgb

ROOT = Path(__file__).resolve().parent
CFG = json.loads((ROOT / "final_model_config.json").read_text(encoding="utf-8"))
MED = json.loads((ROOT / "imputer_medians.json").read_text(encoding="utf-8"))
FEATURES = CFG["features"]
BOOSTER = xgb.Booster()
BOOSTER.load_model(ROOT / "xgboost_booster.json")

def predict_proba(values: dict) -> float:
    row = []
    for f in FEATURES:
        v = values.get(f, np.nan)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            v = MED[f]
        row.append(float(v))
    dm = xgb.DMatrix(np.asarray([row], dtype=float), feature_names=FEATURES)
    return float(BOOSTER.predict(dm)[0])

if __name__ == "__main__":
    example = {"Sex": 1, "Age": 34, "UA": 610, "NEUT_abs": 3.10, "CREA": 62,
               "ALT": 15, "RDW_CV": 14.6, "PAB": 118, "EO_abs": 0.18, "ALP": 96}
    print(f"Predicted probability: {predict_proba(example):.6f}")
