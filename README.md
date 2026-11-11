# Routine host-response profiling for smear-negative pulmonary tuberculosis

Public code and frozen-model repository for the manuscript **“Routine host-response profiling for smear-negative pulmonary tuberculosis: model development, external validation, and the incremental value of conventional T-cell subsets.”**

## What is included

- `analysis/`: curated analysis scripts corresponding to the final analysis chain.
- `model/`: frozen 10-predictor XGBoost pipeline, clean configuration JSON, and a portable XGBoost booster export.
- `splits/`: fixed 5-repeat × 5-fold outer split indices for the derivation and T-cell analyses.
- `webtool/`: Streamlit source code for the research-prototype web interface.
- `docs/`: reproducibility, provenance, and private-data notes.
- `tests/`: basic integrity and privacy checks.

## What is not included

No patient-level clinical data are distributed. Intermediate caches containing patient-level predictions or arrays are also intentionally excluded. See `docs/PRIVATE_DATA.md`.

## Analysis design

The derivation cohort contained 4,177 patients (408 TB / 3,769 non-TB). Nine algorithms were compared using five repetitions of stratified five-fold nested cross-validation with AUPRC as the primary metric. XGBoost was selected, followed by fully nested SHAP-guided feature reduction. After internal validation, a ten-predictor model was frozen for exploratory triage analyses and independent external validation.

The independent external cohort contained 1,149 patients (82 TB / 1,067 non-TB); the primary complete-case analysis included 776 patients (61 TB).

See `docs/REPRODUCIBILITY.md` for the distinction between the full model, adaptive nested feature-selection procedure, fixed-k analysis, and post-selection frozen model.

## Running the web tool

```bash
cd webtool
pip install -r requirements.txt
streamlit run app.py
```

## Running the analysis scripts

The full analysis requires the private clinical source files and is **not a one-click public reproduction package**. The code is released to document the complete analytical implementation. To rerun with authorised source data, set:

```bash
# Linux/macOS
export TB_WORK_ROOT=/path/to/private/workspace

# PowerShell
$env:TB_WORK_ROOT = 'D:\path\to\private\workspace'
```

The scripts retain the original project-relative source filenames so that the published fixed split indices remain tied to the original row ordering. See `docs/PRIVATE_DATA.md`.

## Environment

- Python 3.11
- See `requirements-analysis.txt` for analysis dependencies.
- The deployed frozen pickle was created with the versions pinned in `webtool/requirements.txt`. For long-term model portability, `model/xgboost_booster.json` and `model/imputer_medians.json` are also provided.

## Intended use

**Research use only.** This is a research prototype of a retrospectively developed and externally evaluated model. It has not been validated for real-time clinical decision-making. A low model score does not exclude tuberculosis and must not delay guideline-recommended molecular testing or further clinical evaluation. The model does not establish a diagnosis of pulmonary tuberculosis.

## Version note

Commit `7afc3402b99c4a660e50ac00c21a5bd08dff1540` is the archived version corresponding to the original analysis and frozen model cited in the manuscript. Later commits update documentation and web-interface wording only; the frozen model, configuration, threshold and prediction code are unchanged.

## Licence

Code in this repository is released under the MIT License. The clinical data are not covered by this licence and are not distributed.
