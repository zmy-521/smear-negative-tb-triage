# -*- coding: utf-8 -*-
"""Figure 5：最终模型的临床转化与错误画像 + Figure 6D

对象：与 Figure 4 同源 —— 固定 10 特征模型的 5 折 cross-fitting OOF。
切点：该对象自身的预设高敏阈值（Sens≥90%）与 Youden 点，均非事后新增自由度。
A  三层预测概率分布（措辞用 predicted probability，不用 risk）
B  错误画像：假阳性的疾病构成 vs 全体对照构成
C  决策曲线
D  每 1000 例的临床影响
"""
from public_paths import WORK_ROOT, FONT_PATH
import sys, pickle, warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.style import set_paper_style, savefig, bold_ticks
from sklearn.metrics import roc_curve

Image.MAX_IMAGE_PIXELS = None
OUT = WORK_ROOT / "结核-Aim2" / "outputs"
PAN = OUT / "panels" / "Fig5"; PAN.mkdir(parents=True, exist_ok=True)
MOD = str(WORK_ROOT / "结核数据-25指标-建模集.xlsx")

S1 = pickle.load(open(OUT / "_cache" / "s01.pkl", "rb"))
y = S1['y']
oof = np.load(OUT / "panels" / "Fig4" / "_oof_final10.npy")
df = pd.read_excel(MOD, dtype={'Patient_ID': str})
dx = df['diagnosis'].fillna('').astype(str)
acute = dx.str.contains('肺炎|肺部感染|感染|脓肿|流感|新型冠状').values
copd = dx.str.contains('阻塞性肺疾病|慢阻肺|支气管扩张|哮喘|慢性支气管炎').values
tumor = dx.str.contains('恶性肿瘤|肺癌|占位|肿物|结节').values
CAT = np.where(acute, 'Acute infection',
      np.where(copd & ~acute, 'Chronic airway',
      np.where(tumor & ~acute & ~copd, 'Malignancy', 'Other')))
ORDER = ['Acute infection', 'Chronic airway', 'Malignancy', 'Other']
COL = {'Acute infection': '#2ca02c', 'Chronic airway': '#ff7f0e',
       'Malignancy': '#9467bd', 'Other': '#8c8c8c'}


def thr_sens(t=0.90):
    ps = np.sort(oof[y == 1])[::-1]
    return float(ps[min(int(np.ceil(t * len(ps))), len(ps)) - 1])


fpr, tpr, th = roc_curve(y, oof)
T_LOW = thr_sens()
T_HIGH = float(th[np.argmax(tpr - fpr)])
print(f"切点：高敏阈值 {T_LOW:.4f}（Sens≥90%）  Youden 阈值 {T_HIGH:.4f}")
tier = np.where(oof < T_LOW, 0, np.where(oof < T_HIGH, 1, 2))
TN = ['Low\npredicted probability', 'Intermediate\npredicted probability',
      'High\npredicted probability']

# ---------------- A ----------------
set_paper_style(base_fontsize=17)
fig, ax = plt.subplots(figsize=(9, 6.4))
n_t = [int((tier == i).sum()) for i in range(3)]
tb_t = [int(((tier == i) & (y == 1)).sum()) for i in range(3)]
pct = [100 * tb_t[i] / n_t[i] for i in range(3)]
xx = np.arange(3)
ax.bar(xx, [n_t[i] - tb_t[i] for i in range(3)], width=0.6, color='#c7d9ef',
       label='Non-tuberculous', zorder=2)
ax.bar(xx, tb_t, width=0.6, bottom=[n_t[i] - tb_t[i] for i in range(3)],
       color='#d62728', label='Tuberculosis', zorder=2)
for i in range(3):
    ax.text(i, n_t[i] + 60, f'n = {n_t[i]}\nTB {tb_t[i]} ({pct[i]:.1f}%)',
            ha='center', fontsize=15, fontweight='bold')
ax.set_xticks(xx); ax.set_xticklabels(TN)
ax.set_ylabel('Number of patients'); ax.set_ylim(0, max(n_t) * 1.30)
ax.legend(loc='upper right', fontsize=14)
bold_ticks(ax); fig.tight_layout(); savefig(fig, PAN / "Fig5_A")
print(f"A 已出  三层 n={n_t}  TB%={[f'{p:.1f}' for p in pct]}")

# ---------------- B 错误画像 ----------------
pred = (oof >= T_LOW).astype(int)
fp = (pred == 1) & (y == 0)
ctrl = y == 0
comp_ctrl = [100 * ((CAT == c) & ctrl).sum() / ctrl.sum() for c in ORDER]
comp_fp = [100 * ((CAT == c) & fp).sum() / fp.sum() for c in ORDER]
set_paper_style(base_fontsize=17)
fig, ax = plt.subplots(figsize=(9.5, 6.4))
w = 0.36; xx = np.arange(len(ORDER))
ax.bar(xx - w/2, comp_ctrl, w, color='0.72', label=f'All controls (n = {int(ctrl.sum())})')
ax.bar(xx + w/2, comp_fp, w, color=[COL[c] for c in ORDER],
       label=f'False positives (n = {int(fp.sum())})')
for i in range(len(ORDER)):
    r = comp_fp[i] / comp_ctrl[i] if comp_ctrl[i] > 0 else np.nan
    ax.text(xx[i], max(comp_ctrl[i], comp_fp[i]) + 2.5, f'×{r:.2f}',
            ha='center', fontsize=15, fontweight='bold',
            color=('#d62728' if r > 1.15 else ('#1f77b4' if r < 0.85 else '0.4')))
ax.set_xticks(xx); ax.set_xticklabels(ORDER)
ax.set_ylabel('Proportion (%)'); ax.set_ylim(0, max(comp_ctrl + comp_fp) * 1.25)
ax.legend(loc='upper right', fontsize=13.5)
bold_ticks(ax); fig.tight_layout(); savefig(fig, PAN / "Fig5_B")
print("B 已出  对照构成", [f'{v:.1f}' for v in comp_ctrl],
      " 假阳性构成", [f'{v:.1f}' for v in comp_fp])

# ---------------- C DCA ----------------
set_paper_style(base_fontsize=17)
fig, ax = plt.subplots(figsize=(9, 6.4))
ths = np.linspace(0.01, 0.50, 140); n = len(y)
nb = [((oof >= t) & (y == 1)).sum()/n - ((oof >= t) & (y == 0)).sum()/n * (t/(1-t))
      for t in ths]
ax.plot(ths, nb, color='#2ca02c', lw=3.2, label='Final 10-feature model')
ax.plot(ths, [y.mean() - (1-y.mean())*(t/(1-t)) for t in ths], color='#1f77b4',
        lw=2.0, label='Test all')
ax.axhline(0, color='k', lw=2.0, ls='--', label='Test none')
ax.set_xlabel('Threshold probability'); ax.set_ylabel('Net benefit')
ax.set_xlim(0.01, 0.5); ax.set_ylim(-0.02, max(nb) * 1.25)
ax.legend(loc='upper right', fontsize=14)
bold_ticks(ax); fig.tight_layout(); savefig(fig, PAN / "Fig5_C"); print("C 已出")

# ---------------- D 每千人影响 ----------------
set_paper_style(base_fontsize=17)
fig, ax = plt.subplots(figsize=(9.5, 6.4))
grid = [0.02, 0.03, 0.05, 0.08, 0.12, 0.20, 0.30]
rows = []
for t in grid:
    p = (oof >= t)
    tp = int((p & (y == 1)).sum()); fp_ = int((p & (y == 0)).sum())
    fn = int((~p & (y == 1)).sum())
    rows.append((t, 1000*tp/n, 1000*fn/n, 1000*p.sum()/n))
R = np.array(rows)
ax.plot(R[:, 0], R[:, 3], 'o-', color='#1f77b4', lw=2.6, ms=10,
        label='Referred for further testing')
ax.plot(R[:, 0], R[:, 1], 's-', color='#2ca02c', lw=2.6, ms=10, label='TB detected')
ax.plot(R[:, 0], R[:, 2], '^-', color='#d62728', lw=2.6, ms=10, label='TB missed')
ax.axvline(T_LOW, color='0.4', ls='--', lw=2.0)
ax.text(T_LOW*1.06, ax.get_ylim()[1]*0.70,
        'High-sensitivity\nthreshold = ' + f'{T_LOW:.4f}',
        fontsize=13.5, style='italic')
ax.set_xscale('log'); ax.set_xticks(grid)
ax.set_xticklabels([f'{g:g}' for g in grid])
ax.xaxis.set_minor_locator(plt.NullLocator())
ax.set_xlabel('Threshold probability'); ax.set_ylabel('Patients per 1000 screened')
ax.legend(loc='center right', fontsize=14)
bold_ticks(ax); fig.tight_layout(); savefig(fig, PAN / "Fig5_D")
print("D 已出")
for t, tp, fn, ref in rows:
    print(f"   阈值 {t:.2f}: 每千人 检出TB {tp:.1f}  漏诊 {fn:.1f}  需进一步检测 {ref:.0f}")


# ---------------- 拼 Figure 5 ----------------
def crop(im, tol=248):
    a = np.array(im.convert("L")); m = a < tol
    ys, xs = np.where(m)
    return im.crop((xs.min(), ys.min(), xs.max()+1, ys.max()+1))


ims = [crop(Image.open(PAN / f"Fig5_{k}.tiff").convert("RGB")) for k in "ABCD"]
cw = max(i.width for i in ims); ch = max(i.height for i in ims)
fs = max(48, ch // 13); band = int(fs * 1.5)
CW, CH = cw, ch + band
canvas = Image.new("RGB", (CW*2, CH*2), "white")
d = ImageDraw.Draw(canvas); font = ImageFont.truetype(FONT_PATH, fs)
for i, im in enumerate(ims):
    r, c = divmod(i, 2)
    s = min(cw/im.width, ch/im.height)
    w2, h2 = int(im.width*s), int(im.height*s)
    x0 = c*CW + (cw-w2)//2; y0 = r*CH + band + (ch-h2)//2
    canvas.paste(im.resize((w2, h2), Image.LANCZOS), (x0, y0))
    d.text((x0, r*CH + int(band*0.12)), "ABCD"[i], fill="black", font=font)
canvas.save(OUT / "Figure5_clinical_combined.tiff", compression="tiff_lzw", dpi=(300, 300))
canvas.save(OUT / "Figure5_clinical_combined.pdf", "PDF", resolution=300.0)
print("Figure 5 组合图已出:", canvas.size)
