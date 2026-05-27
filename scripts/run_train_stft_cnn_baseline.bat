@echo off
REM ============================================================
REM Train: STFT + CNN baseline
REM Edit DATA_DIR and EXP_NAME before running.
REM ============================================================

set "DATA_DIR=E:\newdog_emo"
set "EXP_NAME=stft_cnn_baseline"

python src\train.py --data_dir "%DATA_DIR%" --feature_type stft --model cnn --epochs 30 --batch_size 8 --lr 0.001 --exp_name "%EXP_NAME%"
