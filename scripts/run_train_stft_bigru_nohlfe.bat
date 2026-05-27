@echo off
REM ============================================================
REM Train: STFT + BiGRU fallback (no HL-FE)
REM Edit DATA_DIR and EXP_NAME before running.
REM ============================================================

set "DATA_DIR=E:\newdog_emo"
set "EXP_NAME=stft_bigru_nohlfe"

python src\train.py --data_dir "%DATA_DIR%" --feature_type stft --backend bigru_fallback --model mamba --epochs 20 --batch_size 8 --lr 0.001 --loss_type weighted_ce --exp_name "%EXP_NAME%"
