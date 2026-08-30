# -*- coding: utf-8 -*-
"""Figure 6D：加入 T 细胞前后的预测概率散点 + 重拼 Figure 6（A/B/C/D）"""
from public_paths import WORK_ROOT, FONT_PATH
import sys, pickle
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.style import set_paper_style, savefig, bold_ticks

Image.MAX_IMAGE_PIXELS = None
OUT = WORK_ROOT / "结核-Aim2" / "outputs"
PAN = OUT / "panels" / "Fig6"; PAN.mkdir(parents=True, exist_ok=True)

S = pickle.load(open(OUT / "_cache" / "s07.pkl", "rb"))
y, oof = S['y'], S['oof']
b, d = oof['B'][0], oof['D'][0]          # seed=42 参考重复
r = np.corrcoef(b, d)[0, 1]
md = float(np.median(np.abs(d - b)))
print(f"n={len(y)}  Pearson r = {r:.3f}  |Δprobability| 中位 = {md:.4f}")

set_paper_style(base_fontsize=17)
fig, ax = plt.subplots(figsize=(8.2, 7.6))
ax.plot([0, 1], [0, 1], ls='--', color='0.45', lw=2.0, zorder=1)
ax.scatter(b[y == 0], d[y == 0], s=22, c='#7fb3e0', alpha=0.7,
           edgecolors='none', label='Non-tuberculous', zorder=2)
ax.scatter(b[y == 1], d[y == 1], s=26, c='#d62728', alpha=0.75,
           edgecolors='none', label='Tuberculosis', zorder=3)
ax.set_xlabel('Predicted probability — Model B (routine)')
ax.set_ylabel('Predicted probability — Model D (routine + T-cell)')
ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
ax.text(0.97, 0.06, f'r = {r:.3f}\nmedian |Δp| = {md:.3f}', transform=ax.transAxes,
        ha='right', fontsize=15, style='italic')
ax.legend(loc='upper left', fontsize=14)
bold_ticks(ax); fig.tight_layout(); savefig(fig, PAN / "Fig6_D")
print("Panel D 已出")


def crop(im, tol=248):
    a = np.array(im.convert("L")); m = a < tol
    ys, xs = np.where(m)
    return im.crop((xs.min(), ys.min(), xs.max()+1, ys.max()+1))


A = crop(Image.open(PAN / "Figure3A_aim3_model_performance.tiff").convert("RGB"))
B = crop(Image.open(PAN / "Figure3B_aim3_incremental.tiff").convert("RGB"))
C = crop(Image.open(PAN / "Figure3C_aim3_DCA.tiff").convert("RGB"))
D = crop(Image.open(PAN / "Fig6_D.tiff").convert("RGB"))

LEFT = 3800
fs = max(46, LEFT // 36); band = int(fs * 1.5)
fit = lambda im, w: im.resize((w, int(im.height * w / im.width)), Image.LANCZOS)
a, b_ = fit(A, LEFT), fit(B, LEFT)
left_h = a.height + band * 2 + b_.height
# 右列：C 上、D 下，等宽
rw = int(max(C.width, D.width) * (left_h - band * 2) / (C.height + D.height))
c_, d_ = fit(C, rw), fit(D, rw)
CH = max(left_h, c_.height + band * 2 + d_.height)
canvas = Image.new("RGB", (LEFT + rw + 140, CH), "white")
dr = ImageDraw.Draw(canvas); font = ImageFont.truetype(FONT_PATH, fs)
canvas.paste(a, (0, band)); dr.text((0, 0), "A", fill="black", font=font)
canvas.paste(b_, (0, a.height + band*2)); dr.text((0, a.height + band), "B", fill="black", font=font)
x2 = LEFT + 140
canvas.paste(c_, (x2, band)); dr.text((x2, 0), "C", fill="black", font=font)
canvas.paste(d_, (x2, c_.height + band*2))
dr.text((x2, c_.height + band), "D", fill="black", font=font)
canvas.save(OUT / "Figure6_aim3_combined.tiff", compression="tiff_lzw", dpi=(300, 300))
canvas.save(OUT / "Figure6_aim3_combined.pdf", "PDF", resolution=300.0)
print("Figure 6 组合图已出:", canvas.size)
