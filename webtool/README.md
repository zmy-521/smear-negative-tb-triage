# Smear-negative pulmonary tuberculosis: research prototype web tool

This Streamlit application deploys the frozen 10-predictor XGBoost pipeline used in the associated manuscript. Predictor identities, order, derivation-fitted median imputer, model parameters, and the fixed research threshold were frozen before external validation.

## Intended use
**Research use only.** This retrospective model has not been validated for real-time clinical decision-making. A low model score does not exclude tuberculosis and must not delay guideline-recommended molecular testing or further clinical evaluation. The tool is not a diagnostic test and does not replace microbiological, molecular, pathological, or clinical diagnosis.

## Predictors
Sex, age, uric acid, absolute neutrophil count, creatinine, alanine aminotransferase, RDW-CV, prealbumin, absolute eosinophil count, and alkaline phosphatase.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Validation summary
- Development cohort: n=4,177; fully nested adaptive feature-selection AUPRC 0.4991, AUROC 0.8259.
- Primary external complete-case analysis: n=776 (61 TB); AUPRC 0.438, AUROC 0.781.
- Fixed research threshold: 0.0284, selected in the derivation cohort and unchanged for external evaluation; external sensitivity 0.934 and NPV 0.981 in the primary complete-case analysis.

Patient-level data are not included in this repository.
