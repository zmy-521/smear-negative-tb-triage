# -*- coding: utf-8 -*-
"""重新截取在线工具界面（Playwright 无头 Chromium，3 倍像素密度）

前置：另一个进程里已经 `streamlit run app.py`，端口 8511。
示例患者取高风险的一例，使面板 A 的 force plot 与面板 B 的条形图对应同一名患者。
"""
from public_paths import WORK_ROOT, FONT_PATH
import sys, time
from pathlib import Path
from playwright.sync_api import sync_playwright

OUT = WORK_ROOT / "结核-外部验证" / "outputs" / "_shots"
OUT.mkdir(parents=True, exist_ok=True)
URL = "http://localhost:8511"

# 示例患者（高风险）：与 Figure S6 面板 B 用同一组取值
CASE = {"Age (years)": 34, "Uric acid (μmol/L)": 610, "Neutrophil count (×10⁹/L)": 3.10,
        "Creatinine (μmol/L)": 62, "Alanine aminotransferase (U/L)": 15,
        "Red cell distribution width (CV) (%)": 14.6, "Prealbumin (mg/L)": 118,
        "Eosinophil count (×10⁹/L)": 0.18, "Alkaline phosphatase (U/L)": 96}

with sync_playwright() as pw:
    b = pw.chromium.launch()
    pg = b.new_page(viewport={"width": 1180, "height": 1400}, device_scale_factor=3)
    pg.goto(URL, wait_until="networkidle", timeout=90000)
    pg.wait_for_selector("text=Patient values", timeout=60000)

    for label, v in CASE.items():
        inp = pg.locator(f'input[aria-label="{label}"]')
        inp.wait_for(timeout=30000)
        inp.fill(str(v))
        inp.press("Tab")
        pg.wait_for_timeout(250)
    pg.wait_for_timeout(1200)

    pg.get_by_role("button", name="Predict").click()
    pg.wait_for_selector("text=Why this prediction", timeout=90000)
    pg.wait_for_timeout(3500)

    # 去掉 Streamlit 自带的工具条 / 页脚，别让开发界面元素进期刊图
    pg.add_style_tag(content="""
      header, [data-testid="stToolbar"], [data-testid="stDecoration"],
      [data-testid="stStatusWidget"], footer, #MainMenu { display: none !important; }
      [data-testid="stAppViewContainer"] > .main { padding-top: 0 !important; }
      .block-container { padding-top: 1.2rem !important; }
    """)
    pg.wait_for_timeout(800)

    txt = pg.inner_text("body")
    line = [x for x in txt.splitlines() if "predicted possibility" in x]
    print("页面读到：", line[0] if line else "(未读到概率行)")

    cont = pg.locator('[data-testid="stAppViewContainer"]')
    cb = cont.bounding_box()
    wb = pg.get_by_text("Why this prediction", exact=True).first.bounding_box()
    import json
    dsf = 3
    json.dump({"cut_y": int((wb["y"] - cb["y"] - 12) * dsf)},
              open(OUT / "shot_meta.json", "w"))
    print(f"  面板 A 裁切位置 y = {int((wb['y']-cb['y']-12)*dsf)} px")
    cont.screenshot(path=str(OUT / "tool_full.png"))
    b.close()

print("截图已保存 ->", OUT / "tool_full.png")
