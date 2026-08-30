# -*- coding: utf-8 -*-
"""Aim 2 · 步骤 6：拼 Figure 2（A–F，2 行 × 3 列），复刻参考图版式。

拼装纪律（取自 SKILL montage）：
  ① 每张先 autocrop 去四周纯白边
  ② 面板等比放大到恰好放满统一格子（contain，居中），面板尽量大
  ③ 面板字母画在每格顶部的留白带里，留白带高度 ≥ 字形高度
"""
from public_paths import WORK_ROOT, FONT_PATH
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np

Image.MAX_IMAGE_PIXELS = None
OUT = WORK_ROOT / "结核-Aim2" / "outputs"
PAN = OUT / "panels" / "Fig3"          # 单格在 panels，组图写根目录
PANELS = [("A", "Figure2A_combined_ROC"),
          ("B", "Figure2B_SHAP_beeswarm_XGBoost"),
          ("C", "Figure2C_SHAP_beeswarm_LightGBM"),
          ("D", "Figure2D_AUPRC_vs_features"),
          ("E", "Figure2E_metrics_vs_features"),
          ("F", "Figure2F_final_5fold_ROC")]
NCOL, NROW = 3, 2


def autocrop(im, tol=248):
    a = np.array(im.convert("L"))
    mask = a < tol
    if not mask.any():
        return im
    ys, xs = np.where(mask)
    return im.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))


imgs = []
for letter, name in PANELS:
    p = PAN / f"{name}.tiff"
    if not p.exists():
        raise SystemExit(f"缺面板: {p}")
    im = autocrop(Image.open(p).convert("RGB"))
    imgs.append((letter, im))
    print(f"{letter}  {name}  {im.size}")

cell_w = max(im.width for _, im in imgs)
cell_h = max(im.height for _, im in imgs)
letter_fs = max(40, cell_h // 14)
band = int(letter_fs * 1.45)
CW, CH = cell_w, cell_h + band
canvas = Image.new("RGB", (CW * NCOL, CH * NROW), "white")
draw = ImageDraw.Draw(canvas)
font = ImageFont.truetype(FONT_PATH, letter_fs)

for i, (letter, im) in enumerate(imgs):
    r, c = divmod(i, NCOL)
    s = min(cell_w / im.width, cell_h / im.height)
    w, h = int(im.width * s), int(im.height * s)
    im2 = im.resize((w, h), Image.LANCZOS)
    x0 = c * CW + (cell_w - w) // 2
    y0 = r * CH + band + (cell_h - h) // 2
    canvas.paste(im2, (x0, y0))
    draw.text((x0, r * CH + int(band * 0.1)), letter, fill="black", font=font)

canvas.save(OUT / "Figure3_ml_combined.tiff", compression="tiff_lzw", dpi=(300, 300))
canvas.save(OUT / "Figure3_ml_combined.pdf", "PDF", resolution=300.0)
print(f"\nFigure 3（ML）组合图已出: {canvas.size}  ->", OUT / "Figure3_ml_combined.tiff")
