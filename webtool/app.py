# -*- coding: utf-8 -*-
"""Smear-negative pulmonary tuberculosis — low-cost triage prediction tool

冻结的 10 变量 XGBoost 模型，原样部署：变量、超参、插补器中位数、阈值全部来自
推导队列，外部验证时一个参数未改。

本地运行：  streamlit run app.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
CFG = json.loads((ROOT / "config" / "feature_config.json").read_text(encoding="utf-8"))
FEATS = [f["key"] for f in CFG["features"]]
SHOW = {"Sex": "Sex", "Age": "Age", "UA": "UA", "NEUT_abs": "NEUT#", "CREA": "CREA",
        "ALT": "ALT", "RDW_CV": "RDW-CV", "PAB": "PAB", "EO_abs": "EO#", "ALP": "ALP"}


def fmt_val(k, v):
    if k == "Sex":
        return "male" if v >= 0.5 else "female"
    return f"{v:.2f}".rstrip("0").rstrip(".")


THR = CFG["threshold_sens90"]

st.set_page_config(page_title="Smear-negative pulmonary tuberculosis",
                   page_icon="🫁", layout="centered")


@st.cache_resource
def load_model():
    import pickle
    obj = pickle.load(open(ROOT / "model" / "final_model.pkl", "rb"))
    assert obj["features"] == FEATS, "配置与冻结模型的变量顺序不一致"
    return obj["model"]


@st.cache_resource
def load_explainer(_pipe):
    import shap
    return shap.TreeExplainer(_pipe.named_steps["clf"])


pipe = load_model()

st.markdown("<style>[data-testid='stHeaderActionElements']{display:none;}</style>",
            unsafe_allow_html=True)
st.title("Smear-negative pulmonary tuberculosis")
st.caption("Low-cost triage prediction tool · 10 routinely available predictors")
st.caption("Intended to support prioritisation for confirmatory testing rather than "
           "replace microbiological or molecular diagnosis.")

st.subheader("Patient values")
vals = {}
cols = st.columns(2)
for i, f in enumerate(CFG["features"]):
    c = cols[i % 2]
    if f.get("type") == "binary":
        choice = c.radio(f["label"], list(f["options"].keys()), horizontal=True,
                         index=list(f["options"].values()).index(f["default"]))
        vals[f["key"]] = f["options"][choice]
    else:
        label = f'{f["label"]} ({f["unit"]})' if f.get("unit") else f["label"]
        vals[f["key"]] = c.number_input(label, min_value=float(f["min"]),
                                        max_value=float(f["max"]),
                                        value=float(f["default"]),
                                        step=float(f["step"]), format="%.2f")

X = pd.DataFrame([[vals[k] for k in FEATS]], columns=FEATS)

if st.button("Predict", type="primary", width="stretch"):
    p = float(pipe.predict_proba(X)[0, 1])
    st.markdown(
        f"### *Based on feature values, predicted possibility of smear-negative "
        f"pulmonary tuberculosis is {p*100:.2f}%*")
    st.caption(f"Prespecified high-sensitivity threshold = {THR*100:.2f}% "
               f"(sensitivity ≥ 90% in the development cohort)")
    st.markdown("**Above the threshold — prioritize confirmatory testing**"
                if p >= THR else
                "**Below the threshold — confirmatory testing is lower priority**")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    expl = load_explainer(pipe)
    Xi = pd.DataFrame(pipe.named_steps["imputer"].transform(X), columns=FEATS)
    sv = np.asarray(expl.shap_values(Xi))[0]
    order = np.argsort(np.abs(sv))[::-1]
    labels = [f"{SHOW.get(FEATS[k], FEATS[k])} = {fmt_val(FEATS[k], Xi.iloc[0][FEATS[k]])}"
              for k in order][::-1]
    vals = sv[order][::-1]

    st.subheader("Why this prediction")
    fig, ax = plt.subplots(figsize=(7.4, 4.0), dpi=200)
    ax.barh(range(len(vals)), vals,
            color=["#c0392b" if v > 0 else "#1f77b4" for v in vals], height=0.68)
    ax.axvline(0, color="0.35", lw=1.2)
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("SHAP value (contribution to log-odds)", fontsize=10.5)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=10)
    fig.tight_layout()
    st.pyplot(fig, width="stretch")
    plt.close("all")
    st.caption("Red bars push the prediction towards tuberculosis, blue bars away from it.")

with st.expander("Model provenance and performance"):
    ip, ep = CFG["internal_performance"], CFG["external_performance"]
    st.markdown(
        f"""
- **Algorithm** {CFG['model_name']}, {CFG['k']} variables, frozen after development.
- **Development cohort** 4177 patients, tuberculosis prevalence
  {CFG['development_prevalence']*100:.2f}%.
  Nested repeated cross-validation: AUPRC {ip['AUPRC']:.3f}, AUROC {ip['AUROC']:.3f}.
- **External validation** independent second centre, {ep['n']} patients with complete data:
  AUROC {ep['AUROC']:.3f}, AUPRC {ep['AUPRC']:.3f}; at the fixed threshold,
  sensitivity {ep['sensitivity']:.3f} and negative predictive value {ep['NPV']:.3f}.
- **Threshold** {THR:.4f}, fixed before external validation and never re-optimized.
- Missing values are filled with the medians of the development cohort, carried inside
  the model pipeline.
"""
    )
