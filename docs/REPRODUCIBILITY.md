# Reproducibility notes

The manuscript uses several distinct performance objects that should not be conflated.

| Object | AUPRC, mean ± SD | AUROC, mean ± SD | Role |
|---|---:|---:|---|
| Age + sex benchmark | 0.2010 ± 0.0059 | 0.6853 ± 0.0064 | Demographic comparator |
| Full 27-predictor XGBoost | 0.5138 ± 0.0128 | 0.8310 ± 0.0045 | Full-model algorithm/domain benchmark |
| Fully nested adaptive feature-selection procedure | **0.4991 ± 0.0161** | **0.8259 ± 0.0076** | Primary unbiased internal estimate for feature selection |
| Nested fixed-k=10 analysis | 0.4770 ± 0.0073 | 0.8135 ± 0.0029 | Performance–parsimony analysis; retained 92.8% of full-model AUPRC |

After internal validation, ten predictor identities were selected in the complete derivation cohort and the final XGBoost pipeline was frozen. Five-fold cross-fitting with this fixed specification generated held-out probabilities and SHAP values for interpretation, threshold development, and clinical-translation analyses. These post-selection outputs are **not** the primary unbiased internal performance estimate.

The high-sensitivity threshold was 0.028411144390702248 (reported as 0.0284), selected as the highest cut-off retaining sensitivity >=90% on the fixed-model cross-fitted probabilities, and frozen before external validation.

## Primary external analysis
- Complete cases for all ten predictors: n=776, including 61 TB.
- AUROC 0.781; AUPRC 0.438.
- At 0.0284: sensitivity 0.934; NPV 0.981.
- Intercept-only recalibration was secondary; threshold-dependent metrics are not interpreted on the recalibrated probability scale.

## Aim 3
The prespecified primary incremental comparison was Model D (fixed ten predictors + CD3#, CD4#, CD8#, CD4/CD8) versus Model B (fixed ten predictors). AUPRC was primary and AUROC secondary.
