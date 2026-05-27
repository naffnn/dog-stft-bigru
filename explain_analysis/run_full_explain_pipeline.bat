@echo off
setlocal

REM ============================================================
REM Full STFT saliency explain analysis pipeline
REM Usage: edit the variables below, then run this script
REM ============================================================

REM -- User: edit these paths --
set "DATA_DIR=E:\newdog_emo"
set "CHECKPOINT=runs\stft_hlfe_bigru_YYYYMMDD_HHMMSS\last_model.pt"
set "OUTPUT_DIR=explain_analysis\outputs"
set "PCA_ICA_OUTPUT_DIR=explain_analysis\saliency_pca_ica\outputs"

REM Python command: use "python" or "py -3"
set "PYTHON=python"

if not exist "%DATA_DIR%" (
    echo [ERROR] DATA_DIR not found. Edit DATA_DIR in this bat file.
    exit /b 1
)
if not exist "%CHECKPOINT%" (
    echo [ERROR] Checkpoint not found. Edit CHECKPOINT in this bat file.
    exit /b 1
)

echo [1/6] Generate STFT saliency heatmaps
"%PYTHON%" explain_analysis\scripts\generate_stft_explain_heatmaps.py --data_dir "%DATA_DIR%" --checkpoint "%CHECKPOINT%" --output_dir "%OUTPUT_DIR%" --split test --max_per_class 50 --device cpu
if errorlevel 1 goto fail_step1

echo [2/6] Convert frequency bins to Hz
"%PYTHON%" explain_analysis\scripts\convert_frequency_bins_to_hz.py --output_dir "%OUTPUT_DIR%\reports"
if errorlevel 1 goto fail_step2

echo [3/6] Unify heatmap colorbar
"%PYTHON%" explain_analysis\scripts\unify_heatmap_colorbar.py --heatmap_dir "%OUTPUT_DIR%\heatmaps_raw" --output_dir "%OUTPUT_DIR%\heatmaps_unified" --frequency_axis_csv "%OUTPUT_DIR%\reports\frequency_axis.csv"
if errorlevel 1 goto fail_step3

echo [4/6] Frequency importance projection
"%PYTHON%" explain_analysis\scripts\frequency_importance_projection.py --index_csv "%OUTPUT_DIR%\reports\heatmap_index.csv" --frequency_axis_csv "%OUTPUT_DIR%\reports\frequency_axis.csv" --output_dir "%OUTPUT_DIR%\frequency_projection"
if errorlevel 1 goto fail_step4

echo [5/6] Frequency band statistics
"%PYTHON%" explain_analysis\scripts\frequency_band_statistics.py --index_csv "%OUTPUT_DIR%\reports\heatmap_index.csv" --frequency_axis_csv "%OUTPUT_DIR%\reports\frequency_axis.csv" --output_dir "%OUTPUT_DIR%\band_statistics"
if errorlevel 1 goto fail_step5

echo [6/6] Saliency frequency PCA / ICA
"%PYTHON%" explain_analysis\saliency_pca_ica\scripts\saliency_frequency_pca_ica.py --heatmap_dir "%OUTPUT_DIR%\heatmaps_raw" --index_csv "%OUTPUT_DIR%\reports\heatmap_index.csv" --frequency_axis_csv "%OUTPUT_DIR%\reports\frequency_axis.csv" --output_dir "%PCA_ICA_OUTPUT_DIR%" --use_only_correct true --standardize true --pca_components 5 --ica_components 5 --random_seed 42
if errorlevel 1 goto fail_step6

echo [DONE] Full explain pipeline completed.
exit /b 0

:fail_step1
echo [ERROR] Step 1 failed: generate_stft_explain_heatmaps.py
exit /b 1

:fail_step2
echo [ERROR] Step 2 failed: convert_frequency_bins_to_hz.py
exit /b 1

:fail_step3
echo [ERROR] Step 3 failed: unify_heatmap_colorbar.py
exit /b 1

:fail_step4
echo [ERROR] Step 4 failed: frequency_importance_projection.py
exit /b 1

:fail_step5
echo [ERROR] Step 5 failed: frequency_band_statistics.py
exit /b 1

:fail_step6
echo [ERROR] Step 6 failed: saliency_frequency_pca_ica.py
exit /b 1