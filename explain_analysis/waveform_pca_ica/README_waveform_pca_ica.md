# Waveform PCA / ICA 分析

本分析不训练模型，不修改原始音频，只对五类狗叫原始 waveform 做 PCA / ICA 低维可视化，用于观察原始采样点表示下的情绪分布。

## 运行命令

在 Windows PowerShell 中运行：

```powershell
python "E:\newdog_emo\dog_emo_repro\explain_analysis\waveform_pca_ica\scripts\waveform_pca_ica_analysis.py" `
  --data_dir "E:\newdog_emo" `
  --output_dir "E:\newdog_emo\dog_emo_repro\explain_analysis\waveform_pca_ica\outputs" `
  --sample_rate 16000 `
  --duration 4.0 `
  --max_per_class 200 `
  --random_seed 42 `
  --standardize true `
  --pca_components 10 `
  --ica_components 10
```

如果在 `cmd.exe` 中运行，使用 `^` 续行：

```bat
python "E:\newdog_emo\dog_emo_repro\explain_analysis\waveform_pca_ica\scripts\waveform_pca_ica_analysis.py" ^
  --data_dir "E:\newdog_emo" ^
  --output_dir "E:\newdog_emo\dog_emo_repro\explain_analysis\waveform_pca_ica\outputs" ^
  --sample_rate 16000 ^
  --duration 4.0 ^
  --max_per_class 200 ^
  --random_seed 42 ^
  --standardize true ^
  --pca_components 10 ^
  --ica_components 10
```

如果当前机器的 `python` 不在 PATH 中，也可以使用项目虚拟环境：

```powershell
E:\newdog_emo\.venv\Scripts\python.exe "E:\newdog_emo\dog_emo_repro\explain_analysis\waveform_pca_ica\scripts\waveform_pca_ica_analysis.py" --data_dir "E:\newdog_emo" --output_dir "E:\newdog_emo\dog_emo_repro\explain_analysis\waveform_pca_ica\outputs" --sample_rate 16000 --duration 4.0 --max_per_class 200 --random_seed 42 --standardize true --pca_components 10 --ica_components 10
```

如果运行过慢，可以把 `--max_per_class` 改成 `50` 或 `100`。

## 输出

```text
outputs/
  figures/
    waveform_pca_2d.png
    waveform_pca_3d.png
    waveform_pca_explained_variance.png
    waveform_ica_2d.png
    waveform_ica_3d.png
  features/
    waveform_pca_2d.csv
    waveform_pca_3d.csv
    waveform_pca_explained_variance.csv
    waveform_ica_2d.csv
    waveform_ica_3d.csv
  reports/
    skipped_files.csv
    sampled_files.csv
    waveform_pca_ica_separability.json
    waveform_pca_ica_report.md
```
