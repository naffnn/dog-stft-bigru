# 狗叫 4 秒整体人工声学特征分析

本实验用于对狗叫 4 秒样本提取人工声学特征，辅助解释 `angry` / `anxious` / `happy` / `lonely` / `sad` 五类情绪在能量、频谱、MFCC、基频和粗节律上的统计差异。

本实验不是重新训练模型，也不会删除、移动、覆盖或修改任何原始音频文件。脚本只读取原始音频，并把新生成的 CSV、报告和图像保存到独立分析目录中。

## 路径

数据集根目录：

```text
E:\newdog_emo
```

数据类别目录：

```text
E:\newdog_emo\angry
E:\newdog_emo\anxious
E:\newdog_emo\happy
E:\newdog_emo\lonely
E:\newdog_emo\sad
```

输出目录：

```text
E:\newdog_emo\dog_acoustic_analysis
```

目录结构：

```text
E:\newdog_emo\dog_acoustic_analysis
├─ extract_4s_acoustic_features.py
├─ analyze_4s_acoustic_features.py
├─ merge_bigru_predictions.py
├─ README.md
├─ features\
├─ reports\
└─ figures\
```

## 环境依赖

建议在后续高性能电脑上安装以下 Python 包：

```text
numpy
pandas
soundfile
librosa
scipy
matplotlib
seaborn
```

## 运行方式

请在 `E:\newdog_emo\dog_acoustic_analysis` 目录下手动运行脚本。本项目不提供 `.bat` 文件，因为实际运行环境可能在另一台高性能电脑上。

### 1. 提取 4 秒人工声学特征

```powershell
python extract_4s_acoustic_features.py --data_root "E:\newdog_emo" --out_dir "E:\newdog_emo\dog_acoustic_analysis" --sample_rate 16000 --duration 4.0
```

输出：

```text
E:\newdog_emo\dog_acoustic_analysis\features\acoustic_features_4s.csv
```

脚本会遍历 `.wav` / `.mp3` / `.flac` 文件，统一采样率为 16000 Hz，每条音频固定为 4.0 秒。不足 4 秒补零，超过 4 秒从开头截断，原始音频不会被修改。

### 2. 统计分析与可视化

```powershell
python analyze_4s_acoustic_features.py --feature_csv "E:\newdog_emo\dog_acoustic_analysis\features\acoustic_features_4s.csv" --out_dir "E:\newdog_emo\dog_acoustic_analysis"
```

主要输出：

```text
E:\newdog_emo\dog_acoustic_analysis\reports\feature_summary_by_class.csv
E:\newdog_emo\dog_acoustic_analysis\reports\kruskal_test_results.csv
E:\newdog_emo\dog_acoustic_analysis\reports\analysis_summary.txt
E:\newdog_emo\dog_acoustic_analysis\figures\*.png
```

分析包含每类情绪的特征均值、标准差、中位数、最小值、最大值，关键特征的 Kruskal-Wallis 检验，以及箱线图和类别均值热力图。

### 3. 合并 BiGRU 预测结果

```powershell
python merge_bigru_predictions.py --feature_csv "E:\newdog_emo\dog_acoustic_analysis\features\acoustic_features_4s.csv" --pred_csv "你的BiGRU预测结果csv路径" --out_dir "E:\newdog_emo\dog_acoustic_analysis"
```

如果不传入 `--pred_csv`，脚本会尝试在以下目录自动搜索文件名包含 `predictions`、`test_predictions` 或 `pred` 的 CSV：

```text
E:\newdog_emo\dog_emo_repro\runs
E:\newdog_emo\dog_emo_repro\reports
E:\newdog_emo\reports
```

合并结果：

```text
E:\newdog_emo\dog_acoustic_analysis\features\features_with_bigru_predictions.csv
```

正确样本与错误样本的特征对比：

```text
E:\newdog_emo\dog_acoustic_analysis\reports\correct_vs_wrong_feature_summary.csv
E:\newdog_emo\dog_acoustic_analysis\figures\correct_vs_wrong_rms_mean.png
E:\newdog_emo\dog_acoustic_analysis\figures\correct_vs_wrong_f0_mean.png
E:\newdog_emo\dog_acoustic_analysis\figures\correct_vs_wrong_spectral_centroid.png
E:\newdog_emo\dog_acoustic_analysis\figures\correct_vs_wrong_onset_rate.png
```

## 注意事项

- 本实验不会修改原始音频。
- 本实验不是重新训练模型。
- 本实验用于后续 bark / bout 分析和情绪声学特征解释。
- 统计显著性只表示当前数据中的相关差异，不直接等于真实情绪机制。

## Post-hoc and effect size analysis

After `acoustic_features_4s.csv` has already been generated, run the post-hoc script to compute Kruskal-Wallis effect sizes, Dunn pairwise comparisons, feature rankings, and paper/report-ready tables.

```powershell
python posthoc_effect_analysis.py --feature_csv "E:\dog_stft_bigru_github\dog_acoustic_analysis\features\acoustic_features_4s.csv" --out_dir "E:\dog_stft_bigru_github\dog_acoustic_analysis"
```

This script does not modify original audio files and does not re-extract acoustic features. It analyzes the existing 4-second handcrafted feature CSV.

Main outputs:

```text
reports\feature_effect_size_ranking.csv
reports\dunn_posthoc_all_features_long.csv
reports\feature_class_median_ranking.csv
figures\feature_effect_size_ranking.png
figures\dunn_<feature>_heatmap.png
reports\posthoc_effect_analysis_summary.txt
reports\posthoc_effect_failed_features.txt
```

The Dunn post-hoc step requires `scikit-posthocs`. If it is missing, install it with:

```powershell
pip install scikit-posthocs
```
