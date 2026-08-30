# -*- coding: utf-8 -*-
"""外部验证集组装（只做数据整理，不碰模型）

来源：第二中心两份导出
  外部验证_肺结核82例_全部指标.xlsx      —— 全部 PTB
  外部验证_明确菌阴1088例_全部指标.xlsx  —— 有记录涂片阴性的连续病例（含 21 例 TB）

两份文件按 Patient_ID 有 21 例重叠（同一批 TB），取并集去重：
  82 TB + 1067 non-TB = 1149

变量名统一到推导集的英文命名，单位不做任何换算（已核对与推导集一致）。
"""
from public_paths import WORK_ROOT, FONT_PATH
import pandas as pd, numpy as np
from pathlib import Path

DESK = WORK_ROOT
OUT = DESK / "结核-外部验证" / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

# 推导集 55 项候选指标 -> 外部导出的中文列名
MAP = {
    # 血常规 24
    'BASO': '嗜碱性粒细胞百分比', 'BASO_abs': '嗜碱性粒细胞绝对数',
    'EO': '嗜酸性粒细胞百分比', 'EO_abs': '嗜酸性粒细胞绝对数',
    'HCT': '红细胞压积', 'HGB': '血红蛋白',
    'LYMPH': '淋巴细胞百分比', 'LYMPH_abs': '淋巴细胞绝对数',
    'MCH': '平均红细胞血红蛋白量', 'MCHC': '平均红细胞血红蛋白浓度', 'MCV': '平均红细胞体积',
    'MONO': '单核细胞百分比', 'MONO_abs': '单核细胞绝对数',
    'NEUT': '中性粒细胞百分比', 'NEUT_abs': '中性粒细胞绝对数',
    'PLT': '血小板计数', 'RBC': '红细胞计数',
    'RDW_CV': '红细胞分布宽度-CV', 'RDW_SD': '红细胞分布宽度-SD', 'WBC': '白细胞计数',
    'MPV': '平均血小板体积', 'PCT_plateletcrit': '血小板压积',
    'PDW': '血小板分布宽度', 'P_LCR': '大血小板比率',
    # 生化 20
    'ALB': '白蛋白', 'ALP': '碱性磷酸酶', 'ALT': '丙氨酸氨基转移酶',
    'AST': '天门冬氨酸氨基转移酶', 'A_G': '白球比', 'BUN': '尿素', 'CREA': '肌酐',
    'DBIL': '直接胆红素', 'GGT': 'γ-谷氨酰转移酶', 'GLB': '球蛋白', 'IBIL': '间接胆红素',
    'TBIL': '总胆红素', 'TP': '总蛋白｜血｜g/L', 'UA': '尿酸', 'TBA': '总胆汁酸',
    'PAB': '前白蛋白', 'GLU': '葡萄糖｜血｜mmol/L',
    'Cl': '氯｜血｜mmol/L', 'K': '钾', 'Na': '钠',
    # 衍生 11
    'NLR': 'NLR', 'dNLR': 'dNLR', 'PLR': 'PLR', 'MLR': 'MLR', 'SII': 'SII',
    'SIRI': 'SIRI', 'PIV': 'PIV', 'PNI': 'PNI', 'HALP': 'HALP',
    'NAR': 'NAR', 'PAR': 'PAR',
}
# 外部有、推导集没有的额外指标：由列名反推，避免手写中文列名对不上
EXTRA = None   # 见下方按实际列名求补集

tb = pd.read_excel(DESK / "外部验证_肺结核82例_全部指标.xlsx", sheet_name="Patient_data")
ct = pd.read_excel(DESK / "外部验证_明确菌阴1088例_全部指标.xlsx", sheet_name="Patient_data")
dic = pd.read_excel(DESK / "外部验证_肺结核82例_全部指标.xlsx", sheet_name="Variable_dictionary")

META0 = ['Study_ID','Patient_ID','Outcome','Group','Control_subgroup','Index_date',
         'Index_source','Smear_verified','Sex_binary','Age']
EXTRA = [c for c in tb.columns if c not in META0 and c not in set(MAP.values())]

dup = set(tb['Patient_ID']) & set(ct['Patient_ID'])
ctn = ct[ct['Outcome'] == 0].copy()          # 1067 例非结核
assert len(set(tb['Patient_ID']) & set(ctn['Patient_ID'])) == 0
raw = pd.concat([tb, ctn], ignore_index=True)

META = ['Study_ID', 'Patient_ID', 'Outcome', 'Group', 'Control_subgroup',
        'Index_date', 'Index_source', 'Smear_verified']
ext = raw[META].copy()
ext['Sex'] = raw['Sex_binary'].astype(int)
ext['Age'] = pd.to_numeric(raw['Age'], errors='coerce')
for en, cn in MAP.items():
    ext[en] = pd.to_numeric(raw[cn], errors='coerce')
for cn in EXTRA:
    ext['EXT_' + cn] = pd.to_numeric(raw[cn], errors='coerce')
ext = ext.copy()

K10 = ['Sex', 'Age', 'UA', 'NEUT_abs', 'CREA', 'ALT', 'RDW_CV', 'PAB', 'EO_abs', 'ALP']
F27 = ['Age', 'Sex', 'NEUT_abs', 'LYMPH_abs', 'MONO_abs', 'EO_abs', 'BASO_abs', 'PLT',
       'HGB', 'MCV', 'MCHC', 'RDW_CV', 'MPV', 'PDW', 'ALB', 'GLB', 'ALT', 'AST',
       'ALP', 'GGT', 'TBIL', 'TBA', 'PAB', 'BUN', 'CREA', 'UA', 'GLU']
ext['complete_10'] = ext[K10].notna().all(axis=1).astype(int)
ext['complete_27'] = ext[F27].notna().all(axis=1).astype(int)
ext['analysis_set'] = np.where(ext['complete_10'] == 1, 'primary', 'imputed_only')

# ---- 变量说明表 ----
info = dic.set_index('Variable')
vd = []
for en, cn in MAP.items():
    r = info.loc[cn] if cn in info.index else None
    n = int(ext[en].notna().sum())
    vd.append({'Variable': en, 'Chinese_name': cn,
               'Category': (r['Category'] if r is not None else ''),
               'Unit': (r['Unit'] if r is not None and pd.notna(r['Unit']) else ''),
               'Type': (r['Type'] if r is not None else ''),
               'In_candidate_pool': 'Yes',
               'In_final_model': 'Yes' if en in K10 else 'No',
               'Available_n': n, 'Missing_pct': round(100 * (1 - n / len(ext)), 2)})
for cn in EXTRA:
    r = info.loc[cn] if cn in info.index else None
    c = 'EXT_' + cn
    n = int(ext[c].notna().sum())
    vd.append({'Variable': c, 'Chinese_name': cn,
               'Category': (r['Category'] if r is not None else ''),
               'Unit': (r['Unit'] if r is not None and pd.notna(r['Unit']) else ''),
               'Type': (r['Type'] if r is not None else ''),
               'In_candidate_pool': 'No', 'In_final_model': 'No',
               'Available_n': n, 'Missing_pct': round(100 * (1 - n / len(ext)), 2)})
vd = pd.DataFrame(vd)

f = DESK / "外部验证集-整理后.xlsx"
with pd.ExcelWriter(f, engine='openpyxl') as w:
    ext.to_excel(w, sheet_name='外部验证集', index=False)
    vd.to_excel(w, sheet_name='变量说明', index=False)
ext.to_pickle(OUT / "_external.pkl")

print(f"外部验证集已输出 -> {f}")
print(f"  合并前 TB 文件 {len(tb)} 例，菌阴文件 {len(ct)} 例，重叠 {len(dup)} 例（同一批 TB）")
print(f"  合并后 {len(ext)} 例：TB {int(ext.Outcome.sum())}，非结核 {int((ext.Outcome==0).sum())}"
      f"（肺炎 {(ext.Control_subgroup=='Pneumonia').sum()}，肺癌 {(ext.Control_subgroup=='LungCancer').sum()}）")
print(f"  患病率 {ext.Outcome.mean()*100:.2f}%（推导集 9.77%）")
print(f"  10 变量齐全 {ext.complete_10.sum()} 例（TB {int(ext.loc[ext.complete_10==1,'Outcome'].sum())}，"
      f"患病率 {ext.loc[ext.complete_10==1,'Outcome'].mean()*100:.2f}%）")
print(f"  27 变量齐全 {ext.complete_27.sum()} 例")
print(f"  列数 {ext.shape[1]}：元信息 {len(META)+2} + 候选指标 {len(MAP)} + 外部额外指标 {len(EXTRA)} + 标记 3")
