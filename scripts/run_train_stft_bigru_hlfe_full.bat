@echo off
REM ============================================================
REM Train: STFT + HL-FE + BiGRU fallback (full training)
REM Edit DATA_DIR and EXP_NAME before running.
REM ============================================================

set "DATA_DIR=E:\newdog_emo"
set "EXP_NAME=stft_hlfe_bigru_full"

python src\train.py --data_dir "%DATA_DIR%" --feature_type stft --use_hlfe --backend bigru_fallback --model mamba --epochs 100 --batch_size 64 --lr 0.001 --loss_type weighted_ce --exp_name "%EXP_NAME%"
