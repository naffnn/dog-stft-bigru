@echo off
REM ============================================================
REM Train: SST-STFT + HL-FE + BiGRU fallback
REM Edit DATA_DIR and EXP_NAME before running.
REM Requires: ssqueezepy (pip install ssqueezepy)
REM ============================================================

set "DATA_DIR=E:\newdog_emo"
set "EXP_NAME=sst_hlfe_bigru"

python src\train.py --data_dir "%DATA_DIR%" --feature_type sst_stft --use_hlfe --backend bigru_fallback --model mamba --epochs 20 --batch_size 8 --lr 0.001 --loss_type weighted_ce --use_feature_cache --exp_name "%EXP_NAME%"
