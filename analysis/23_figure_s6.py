# -*- coding: utf-8 -*-
"""Supplementary Figure S6 —— 在线分诊工具（重做后重新截图）

A  实拍界面：输入、预测概率、阈值与判读（截至解释图之前）
B  同一名示例患者的 SHAP 贡献，按论文出图规范另绘（Times New Roman / 300 dpi）

只出 PNG。
"""
from public_paths import WORK_ROOT, FONT_PATH
import sys, pickle
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

Image.MAX_IMAGE_PIXELS = None
DESK = WORK_ROOT
OUT = DESK / "结核-外部验证" / "outputs"
SHOT = OUT / "_shots" / "tool_full.png"

CASE = dict(Sex=1, Age=34, UA=610, NEUT_abs=3.10, CREA=62, ALT=15,
            RDW_CV=14.6, PAB=118, EO_abs=0.18, ALP=96)
SHOW = {"Sex": "Sex", "Age": "Age", "UA": "UA", "NEUT_abs": "NEUT#", "CREA": "CREA",
        "ALT": "ALT", "RDW_CV": "RDW-CV", "PAB": "PAB", "EO_abs": "EO#", "ALP": "ALP"}

frz = pickle.load(open(DESK / "结核-Aim2" / "outputs" / "final_XGBoost_k10_frozen.pkl", "rb"))
pipe, FEATS, THR = frz["model"], frz["features"], frz["threshold_sens90"]
X = pd.DataFrame([[CASE[k] for k in FEATS]], columns=FEATS)
p_hat = float(pipe.predict_proba(X)[0, 1])

import shap
Xi = pd.DataFrame(pipe.named_steps["imputer"].transform(X), columns=FEATS)
sv = np.asarray(shap.TreeExplainer(pipe.named_steps["clf"]).shap_values(Xi))[0]
order = np.argsort(np.abs(sv))[::-1][::-1]


def fv(k, v):
    return "male" if k == "Sex" and v >= .5 else f"{v:g}"


plt.rcParams.update(plt.rcParamsDefault)
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman"],
                     "font.size": 17, "figure.dpi": 300, "axes.linewidth": 2.0,
                     "axes.labelweight": "bold", "axes.labelsize": 19,
                     "xtick.direction": "in", "ytick.direction": "in",
                     "xtick.major.width": 2.0, "ytick.major.width": 2.0,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.unicode_minus": False})
fig, ax = plt.subplots(figsize=(9.6, 6.6))
vals = sv[order]
labs = [f"{SHOW[FEATS[k]]} = {fv(FEATS[k], CASE[FEATS[k]])}" for k in order]
ax.barh(range(len(vals)), vals, height=0.66,
        color=["#c0392b" if v > 0 else "#1f77b4" for v in vals])
ax.axvline(0, color="0.3", lw=2.0)
ax.set_yticks(range(len(vals))); ax.set_yticklabels(labs)
ax.set_xlabel("SHAP value (contribution to log-odds)")
ax.set_title(f"Example patient \u2014 predicted probability {p_hat*100:.1f}%",
             fontsize=20, fontweight="bold", pad=12)
for t in ax.get_xticklabels() + ax.get_yticklabels():
    t.set_fontweight("bold")
fig.tight_layout()
fig.savefig(OUT / "_shots" / "panelB.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# ---------- 拼图 ----------
shot = Image.open(SHOT).convert("RGB")
import json
CUT = json.load(open(OUT / "_shots" / "shot_meta.json"))["cut_y"]
A = shot.crop((0, 0, shot.width, CUT))
xs = np.where((np.array(A.convert("L")) < 245).mean(axis=0) > 0.0005)[0]
ys = np.where((np.array(A.convert("L")) < 245).mean(axis=1) > 0.0005)[0]
A = A.crop((max(0, xs.min() - 30), max(0, ys.min() - 30),
            min(A.width, xs.max() + 30), min(A.height, ys.max() + 30)))
B = Image.open(OUT / "_shots" / "panelB.png").convert("RGB")

h = max(A.height, B.height)
LAB = int(h * 0.075)
A2 = A.resize((int(A.width * (h - LAB) / A.height), h - LAB), Image.LANCZOS)
B2 = B.resize((int(B.width * (h - LAB) / B.height), h - LAB), Image.LANCZOS)
gap = int(h * 0.05)
W = A2.width + gap + B2.width
canvas = Image.new("RGB", (W, h), "white")
canvas.paste(A2, (0, LAB)); canvas.paste(B2, (A2.width + gap, LAB))
d = ImageDraw.Draw(canvas)
f = ImageFont.truetype(FONT_PATH, int(LAB * 0.85))
d.text((0, 0), "A", fill="black", font=f)
d.text((A2.width + gap, 0), "B", fill="black", font=f)
canvas.save(OUT / "FigureS6_online_tool.png", dpi=(300, 300))
print(f"Figure S6 已重做: {canvas.size}")
print(f"  示例患者预测概率 {p_hat*100:.2f}%（阈值 {THR:.4f}）")
print(f"  A {A2.size}（实拍，完整未截断）   B {B2.size}")
