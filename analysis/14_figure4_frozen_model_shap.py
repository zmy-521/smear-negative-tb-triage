# -*- coding: utf-8 -*-
"""Figure 4：最终 10 特征模型的生物学解释

对象：固定 10 特征 + 固定超参的 5 折 cross-fitting，每折模型只解释它没见过的测试折患者，
      5 折拼成全队列 OOF SHAP（不存在训练患者解释自己的乐观偏倚）。
A  mean|SHAP| 条形图
B  SHAP beeswarm
C  按 mean|SHAP| 前 6 的 dependence 图，2×3 网格，整幅只标一个 C
"""
from public_paths import WORK_ROOT, FONT_PATH
import sys, pickle, warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import shap

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.style import set_paper_style, savefig, bold_ticks
from common.models import build_models
from common.ranking import shap_values_tree
from sklearn.model_selection import StratifiedKFold

Image.MAX_IMAGE_PIXELS = None
OUT = WORK_ROOT / "结核-Aim2" / "outputs"
PAN = OUT / "panels" / "Fig4"; PAN.mkdir(parents=True, exist_ok=True)
SEED = 42

S1 = pickle.load(open(OUT / "_cache" / "s01.pkl", "rb"))
B = pickle.load(open(OUT / "final_XGBoost_k10_frozen.pkl", "rb"))
X, y, FEATS = S1['X'], S1['y'], S1['FEATS']
FF, PAR = B['features'], B['params']
IDX = [FEATS.index(f) for f in FF]
Xf = X[:, IDX]
LAB = {'NEUT_abs': 'NEUT#', 'EO_abs': 'EO#', 'RDW_CV': 'RDW-CV'}
DISP = [LAB.get(f, f) for f in FF]
print(f"最终模型 {len(FF)} 特征: {', '.join(DISP)}")

SV = np.zeros((len(y), len(FF)))
oof = np.zeros(len(y))
for tr, te in StratifiedKFold(5, shuffle=True, random_state=SEED).split(Xf, y):
    p = build_models()['XGBoost']; p.set_params(**PAR)
    p.fit(Xf[tr], y[tr])
    SV[te] = shap_values_tree(p, Xf[te])
    oof[te] = p.predict_proba(Xf[te])[:, 1]
imp = np.abs(SV).mean(axis=0)
order = np.argsort(imp)[::-1]
print("\nmean|SHAP| 排序:")
for j in order:
    print(f"   {DISP[j]:<10} {imp[j]:.4f}")
np.save(PAN / "_oof_final10.npy", oof)

# ---------------- A ----------------
set_paper_style(base_fontsize=17)
fig, ax = plt.subplots(figsize=(7.5, 8))
yp = np.arange(len(FF))[::-1]
ax.barh(yp, imp[order], color="#2ca02c", alpha=0.9, height=0.68)
ax.set_yticks(yp); ax.set_yticklabels([DISP[j] for j in order])
ax.set_xlabel('mean |SHAP value|')
ax.set_ylim(-0.7, len(FF) - 0.3)
bold_ticks(ax); fig.tight_layout()
savefig(fig, PAN / "Fig4_A"); print("\nA 已出")

# ---------------- B ----------------
set_paper_style(base_fontsize=17)
plt.figure(figsize=(8.5, 8))
shap.summary_plot(SV, Xf, feature_names=DISP, max_display=10, show=False, plot_size=None)
fig = plt.gcf(); ax = plt.gca()
ax.set_xlabel("SHAP value (impact on model output)", fontweight='bold', fontsize=19)
for t in ax.get_yticklabels():
    t.set_fontweight('bold'); t.set_fontsize(18)
for t in ax.get_xticklabels():
    t.set_fontweight('bold'); t.set_fontsize(15)
fig.tight_layout(); savefig(fig, PAN / "Fig4_B"); print("B 已出")

# ---------------- C：前 6 个 dependence，2×3 ----------------
top6 = order[:6]
set_paper_style(base_fontsize=19)
fig, axes = plt.subplots(2, 3, figsize=(19, 11))
for k, j in enumerate(top6):
    a = axes[k // 3, k % 3]
    v = Xf[:, j]
    lo, hi = np.nanpercentile(v, [0.5, 99.5])
    m = (v >= lo) & (v <= hi)
    sc = a.scatter(v[m], SV[m, j], c=oof[m], cmap='coolwarm', s=14, alpha=0.75,
                   edgecolors='none')
    a.axhline(0, color='0.5', ls='--', lw=1.6)
    a.set_xlabel(DISP[j], fontweight='bold', fontsize=22)
    a.set_ylabel('SHAP value', fontweight='bold', fontsize=20)
    bold_ticks(a)
cb = fig.colorbar(sc, ax=axes.ravel().tolist(), shrink=0.72, pad=0.015)
cb.set_label('Predicted probability', fontweight='bold', fontsize=18)
cb.ax.tick_params(labelsize=15)
savefig(fig, PAN / "Fig4_C"); print("C 已出")


# ---------------- 拼图 ----------------
def crop(im, tol=248):
    a = np.array(im.convert("L")); m = a < tol
    ys, xs = np.where(m)
    return im.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))


A = crop(Image.open(PAN / "Fig4_A.tiff").convert("RGB"))
Bp = crop(Image.open(PAN / "Fig4_B.tiff").convert("RGB"))
C = crop(Image.open(PAN / "Fig4_C.tiff").convert("RGB"))
top_h = max(A.height, Bp.height)
a = A.resize((int(A.width * top_h / A.height), top_h), Image.LANCZOS)
b = Bp.resize((int(Bp.width * top_h / Bp.height), top_h), Image.LANCZOS)
gap = 120
W = max(a.width + b.width + gap, int(C.width * 0.75))
c = C.resize((W, int(C.height * W / C.width)), Image.LANCZOS)
fs = max(52, W // 42); band = int(fs * 1.5)
canvas = Image.new("RGB", (W, band + top_h + band + c.height), "white")
d = ImageDraw.Draw(canvas); font = ImageFont.truetype(FONT_PATH, fs)
canvas.paste(a, (0, band)); d.text((0, 0), "A", fill="black", font=font)
canvas.paste(b, (a.width + gap, band))
d.text((a.width + gap, 0), "B", fill="black", font=font)
canvas.paste(c, (0, band + top_h + band))
d.text((0, band + top_h + int(band * 0.15)), "C", fill="black", font=font)
canvas.save(OUT / "Figure4_final_model_combined.tiff", compression="tiff_lzw", dpi=(300, 300))
canvas.save(OUT / "Figure4_final_model_combined.pdf", "PDF", resolution=300.0)
print("Figure 4 组合图已出:", canvas.size)
