# Private source data

Patient-level source data are **not included** in this public repository. The analysis scripts were sanitised to remove original local absolute paths. To rerun the full analysis, set the environment variable `TB_WORK_ROOT` to a private workspace containing the required source files in the original project layout.

Core private source files used by the archived analysis include:

- `结核数据-82指标-已补全.xlsx` — derivation source with the broader laboratory set.
- `结核数据-25指标-建模集.xlsx` — derivation modelling dataset (row order defines the published split indices).
- `外部验证_肺结核82例_全部指标.xlsx` — external-centre TB export.
- `外部验证_明确菌阴1088例_全部指标.xlsx` — external-centre smear-negative comparison export.

The repository intentionally excludes all patient-level cached predictions, source tables containing row-level data, and intermediate pickle objects that may embed patient-level arrays.

## Confidentiality
Access to patient-level data is subject to institutional ethics, privacy, and data-governance requirements. Public code release does not imply public release of the clinical source data.
