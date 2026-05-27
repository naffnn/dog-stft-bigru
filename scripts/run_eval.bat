@echo off
REM ============================================================
REM Evaluate and visualize: STFT + HL-FE + BiGRU fallback
REM Edit DATA_DIR before running.
REM ============================================================

set "DATA_DIR=E:\newdog_emo"

python src\evaluate.py --data_dir "%DATA_DIR%" --model mamba --use_hlfe
python src\visualize.py --data_dir "%DATA_DIR%"
