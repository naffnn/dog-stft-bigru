@echo off
REM ============================================================
REM Evaluate and visualize: SST-STFT + HL-FE + BiGRU fallback
REM Edit DATA_DIR before running.
REM ============================================================

set "DATA_DIR=E:\newdog_emo"

python src\evaluate.py --data_dir "%DATA_DIR%" --model mamba --use_hlfe --feature_type sst_stft --use_feature_cache
python src\visualize.py --data_dir "%DATA_DIR%" --feature_type sst_stft
