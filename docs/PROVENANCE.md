# Code provenance

This public release was prepared from the archived analysis package supplied by the study team. Public-release edits were limited to:

1. removal of local absolute paths and identifying workstation information;
2. exclusion of deprecated/overwritten scripts;
3. exclusion of patient-level and patient-level-derived cache files;
4. replacement of local paths by the configurable `TB_WORK_ROOT`;
5. alignment of the public external-validation script with the final manuscript by removing the deprecated structured-field smear subset and avoiding threshold metrics after recalibration;
6. repair of the final Figure 2 script to use the authoritative recovered Aim 1 result object rather than missing early local caches; and
7. minor terminology cleanup in the web application without changing model parameters or predictions.

No statistical result was recomputed or changed during repository sanitisation.

## Public filename map
- `01_aim1_host_response.py` <- original `external_validation/x08_recover_aim1.py`
- `02_aim2_model_comparison.py` <- original `aim1_2_3/s01_compare.py`
- `03_aim2_curves_and_model_table.py` <- original `aim1_2_3/s01b_fig2A_table.py`
- `04_aim2_nested_feature_reduction.py` <- original `aim1_2_3/s02_reduce.py`
- `05_aim2_final_model_size.py` <- original `aim1_2_3/s03_final.py`
- `06_aim2_freeze_final_model.py` <- original `aim1_2_3/s03b_freeze.py`
- `07_aim2_internal_validation_outputs.py` <- original `aim1_2_3/s04_figs_DEF.py`
- `08_aim2_full_model_shap.py` <- original `aim1_2_3/s05_shap.py`
- `09_aim2_assemble_figure3.py` <- original `aim1_2_3/s06_assemble.py`
- `10_aim3_model_comparison.py` <- original `aim1_2_3/s07_aim3_flow.py`
- `11_aim3_outputs.py` <- original `aim1_2_3/s08_aim3_output.py`
- `12_aim2_age_sex_benchmark.py` <- original `aim1_2_3/s12_agesex_benchmark.py`
- `13_aim2_domain_incremental.py` <- original `aim1_2_3/s13_domain_incremental.py`
- `14_figure4_frozen_model_shap.py` <- original `aim1_2_3/s15_fig4_final_shap.py`
- `15_figure5_clinical_translation.py` <- original `aim1_2_3/s16_fig5_clinical.py`
- `16_figure6_assemble.py` <- original `aim1_2_3/s17_fig6d_assemble.py`
- `17_figure2_final.py` <- original `aim1_2_3/s18_fig2_redesign.py`
- `18_external_build_cohort.py` <- original `external_validation/x01_build_external.py`
- `19_external_validation.py` <- original `external_validation/x04_validate.py`
- `20_external_figure_s5.py` <- original `external_validation/x05_figure7.py`
- `21_final_tables.py` <- original `external_validation/x10_all_tables.py`
- `22_webtool_screenshot.py` <- original `external_validation/x11_shoot_tool.py`
- `23_figure_s6.py` <- original `external_validation/x12_figS6.py`
