# Full Explain Pipeline

本流程用于在不重新训练模型的前提下，基于已有 STFT + HL-FE + BiGRU fallback 最优 checkpoint 重新生成解释分析结果：

```text
STFT + HL-FE + BiGRU fallback checkpoint
-> saliency heatmap
-> frequency importance curve
-> PCA / ICA 分解模型关注的频率模式
-> 频段和稳定成分总结
```

流程只读取原始音频和 checkpoint，不会修改原始音频，也不会训练模型。

## 输入依赖

默认数据集路径：

```text
E:\newdog_emo
```

默认 checkpoint：

```text
E:\newdog_emo\dog_emo_repro\runs\stft_hlfe_bigru_20260512_094624（max)）\last_model.pt
```

默认输出目录：

```text
E:\newdog_emo\dog_emo_repro\explain_analysis
```

如果这些路径不存在，请手动修改：

```text
explain_analysis\run_full_explain_pipeline.bat
```

里面的：

```bat
set "PYTHON=..."
set "DATA_DIR=..."
set "CHECKPOINT=..."
set "OUTPUT_DIR=..."
```

## 一键运行

在 Windows 命令行或 PowerShell 中运行：

```bat
cd /d "E:\newdog_emo\dog_emo_repro"
"explain_analysis\run_full_explain_pipeline.bat"
```

如果项目实际在 `D:\newdog_emo`，请先把 `.bat` 中的路径改成对应位置。

## 关键输出

基础解释输出：

```text
explain_analysis\outputs\reports\heatmap_index.csv
explain_analysis\outputs\reports\frequency_axis.csv
explain_analysis\outputs\reports\unified_colorbar_config.json
explain_analysis\outputs\frequency_projection\frequency_importance_curves.png
explain_analysis\outputs\band_statistics\frequency_band_barplot.png
explain_analysis\outputs\band_statistics\frequency_band_heatmap.png
explain_analysis\outputs\reports\explain_summary.md
```

PCA / ICA 输出：

```text
explain_analysis\saliency_pca_ica\outputs\features\saliency_frequency_curves.csv
explain_analysis\saliency_pca_ica\outputs\figures\pca\saliency_frequency_pca_2d.png
explain_analysis\saliency_pca_ica\outputs\figures\ica\saliency_frequency_ica_2d.png
explain_analysis\saliency_pca_ica\outputs\figures\component_loadings\pca_component_loadings.png
explain_analysis\saliency_pca_ica\outputs\figures\component_loadings\ica_component_loadings.png
explain_analysis\saliency_pca_ica\outputs\reports\pca_component_frequency_summary.csv
explain_analysis\saliency_pca_ica\outputs\reports\ica_component_frequency_summary.csv
explain_analysis\saliency_pca_ica\outputs\reports\component_separation_scores.csv
explain_analysis\saliency_pca_ica\outputs\reports\saliency_frequency_pca_ica_report.md
```

## 说明

- saliency 使用 `abs(gradient * input)`。
- 时间维度仅作为局部证据位置。
- 因为音频统一裁剪/补零为 4 秒，不对绝对时间点作强解释。
- 本阶段重点解释频率维度，为后续真实声学参数提取和动物刺激设计准备候选频率区域。
