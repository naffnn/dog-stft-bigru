# 狗叫情绪识别与声学特征分析

## 项目简介

基于 **STFT + HL-FE（高低频特征融合）+ BiGRU fallback** 的狗叫情绪分类模型。

- 数据集类别：angry / anxious / happy / lonely / sad（5 分类）
- 特征类型：STFT（短时傅里叶变换），可选 SST-STFT
- 模型后端：BiGRU fallback（双向 GRU 序列编码器）
- 特色模块：HL-FE（High-Low Frequency Feature Extraction），自适应学习低频/高频融合权重

## 环境要求

### Python 版本

推荐 Python 3.9 或以上。先确认 Python 可用：

```bash
python --version
```

如果 `python` 命令不可用，尝试：

```bash
py -3 --version
```

### 安装依赖

```bash
pip install -r requirements.txt
```

核心依赖：
- torch, torchaudio
- numpy, pandas, scikit-learn
- matplotlib, tqdm, soundfile
- 可选：ssqueezepy（SST-STFT 特征需要）
- 可选：mamba-ssm（如需真 Mamba SSM 后端，当前主线不需要）

## 数据集准备

### ⚠️ 重要说明

**本仓库不包含音频数据集和模型权重。** 你需要自行准备狗叫音频数据。

### 数据集目录结构

将音频文件按情绪类别放入不同文件夹，例如：

```
E:\newdog_emo
├─ angry
├─ anxious
├─ happy
├─ lonely
└─ sad
```

每个文件夹里是 `.wav` 格式的狗叫音频文件。

> `E:\newdog_emo` 只是示例路径。如果数据集在其他位置，请在运行命令时把 `--data_dir` 改成你的实际路径。

## 快速开始

### 1. 扫描数据集

```bash
python src\scan_dataset.py --data_dir "E:\newdog_emo"
```

这会生成 `outputs\reports\dataset_index.csv`，记录所有音频文件的路径、类别、时长等信息。

### 2. 训练模型

STFT + HL-FE + BiGRU fallback（推荐主线）：

```bash
python src\train.py --data_dir "E:\newdog_emo" --feature_type stft --use_hlfe --backend bigru_fallback --epochs 20 --batch_size 8 --lr 0.001 --exp_name stft_hlfe_bigru
```

纯 STFT + BiGRU（不使用 HL-FE）：

```bash
python src\train.py --data_dir "E:\newdog_emo" --feature_type stft --backend bigru_fallback --epochs 20 --batch_size 8 --lr 0.001 --exp_name stft_bigru_nohlfe
```

STFT + CNN 基线：

```bash
python src\train.py --data_dir "E:\newdog_emo" --feature_type stft --model cnn --epochs 30 --batch_size 8 --lr 0.001 --exp_name stft_cnn_baseline
```

如果使用 `py` 启动器：

```bash
py -3 src\train.py --data_dir "E:\newdog_emo" --feature_type stft --use_hlfe --backend bigru_fallback --epochs 20 --batch_size 8 --lr 0.001 --exp_name stft_hlfe_bigru
```

### 3. 评估模型

评估最近训练的模型：

```bash
python src\evaluate.py --data_dir "E:\newdog_emo" --model mamba --use_hlfe
```

指定 checkpoint 评估：

```bash
python evaluate_stft_hlfe.py --data_dir "E:\newdog_emo" --checkpoint "runs\stft_hlfe_bigru_YYYYMMDD_HHMMSS\best_model.pt"
```

### 4. 生成可视化

```bash
python src\visualize.py --data_dir "E:\newdog_emo"
```

### 5. Saliency 解释分析（可选）

```bash
python explain_analysis\scripts\generate_stft_explain_heatmaps.py --data_dir "E:\newdog_emo" --checkpoint "runs\stft_hlfe_bigru_YYYYMMDD_HHMMSS\last_model.pt"
```

完整解释分析流水线（6 步）：

```bash
explain_analysis\run_full_explain_pipeline.bat
```

运行前请编辑 `.bat` 文件中的路径变量。

## 一键运行脚本

`scripts\` 文件夹提供了一键运行的 `.bat` 脚本。使用前请编辑脚本里的 `DATA_DIR` 变量：

| 脚本 | 说明 |
|------|------|
| `run_train_stft_bigru_hlfe.bat` | STFT + HL-FE + BiGRU 训练（20 epochs） |
| `run_train_stft_bigru_nohlfe.bat` | STFT + BiGRU 训练（无 HL-FE） |
| `run_train_stft_cnn_baseline.bat` | STFT + CNN 基线训练 |
| `run_train_stft_bigru_hlfe_full.bat` | STFT + HL-FE + BiGRU 完整训练（100 epochs, batch 64） |
| `run_train_stft_bigru_hlfe_quick.bat` | STFT + HL-FE + BiGRU 快速测试（5 epochs） |
| `run_train_sst_bigru_hlfe.bat` | SST-STFT + HL-FE + BiGRU 训练 |
| `run_eval.bat` | 评估 + 可视化 |
| `run_eval_sst_bigru_hlfe.bat` | SST-STFT 评估 + 可视化 |

## 命令行参数说明

### train.py 主要参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--data_dir` | 数据集根目录（**必填**） | 无 |
| `--project_dir` | 项目输出目录 | 当前目录 |
| `--exp_name` | 实验名称 | 自动生成 |
| `--feature_type` | 特征类型：stft 或 sst_stft | stft |
| `--model` | 模型类型：mamba 或 cnn | mamba |
| `--backend` | 序列后端：bigru_fallback / real_mamba_ssm / auto | bigru_fallback |
| `--use_hlfe` | 启用 HL-FE 模块 | 关闭 |
| `--epochs` | 训练轮数 | 20 |
| `--batch_size` | 批次大小 | 64 |
| `--lr` | 学习率 | 0.001 |
| `--loss_type` | 损失函数：ce / weighted_ce / focal | weighted_ce |
| `--seed` | 随机种子 | 42 |

## 输出目录说明

训练输出保存在 `runs\<exp_name>_<timestamp>\`：

```
runs\stft_hlfe_bigru_20260512_104102\
├── best_model.pt          # 最佳模型权重
├── last_model.pt          # 最后一轮模型权重
├── config.json            # 训练配置
├── metrics.json           # 评估指标
├── train_log.csv          # 训练日志
├── classification_report.txt  # 分类报告
├── confusion_matrix.csv   # 混淆矩阵
├── loss_curve.png         # 损失曲线
├── acc_curve.png          # 准确率曲线
├── confusion_matrix.png   # 混淆矩阵图
└── eval\                  # 独立评估输出（可选）
```

项目级输出在 `outputs\`：

```
outputs\
├── reports\               # 数据集索引、实验汇总
├── figures\               # 可视化图片
├── checkpoints\           # 旧版 checkpoint
├── features_cache\        # STFT 特征缓存
└── logs\                  # 旧版训练日志
```

## 代码结构

```
dog_stft_bigru_github\
├── src\
│   ├── train.py           # 主训练入口
│   ├── dataset.py         # 数据读取（soundfile + torchaudio fallback）
│   ├── features.py        # STFT / SST-STFT 特征提取
│   ├── models.py          # HLFE 模块 + BiGRU fallback + CNN baseline
│   ├── evaluate.py        # 评估脚本
│   ├── visualize.py       # 可视化脚本（分布、曲线、混淆矩阵）
│   ├── scan_dataset.py    # 数据集扫描
│   └── utils.py           # 工具函数
├── evaluate_stft_hlfe.py  # 独立评估入口（支持指定 checkpoint）
├── explain_analysis\      # Saliency 解释分析
│   ├── scripts\           # heatmap、频率投影、频带统计
│   ├── saliency_pca_ica\  # 显著性频率 PCA/ICA
│   └── waveform_pca_ica\  # 波形 PCA/ICA
├── scripts\               # 一键运行 bat 脚本
├── requirements.txt
├── .gitignore
└── README.md
```

## 常见问题

### Q: 提示找不到 `mamba_ssm`？

A: 不需要安装 `mamba-ssm`。当前主线使用 `--backend bigru_fallback`，会自动使用内置的 BiGRU 编码器。

### Q: 提示 `No valid wav files found`？

A: 请确认 `--data_dir` 路径正确，且子文件夹中包含 `.wav` 文件。

### Q: 如何切换 GPU / CPU？

A: 代码会自动检测 CUDA。如需强制 CPU，可设置环境变量：
```bash
set CUDA_VISIBLE_DEVICES=
```

### Q: 训练时出现 encoding 错误？

A: 请确保所有 `.bat` 和 `.py` 文件使用 UTF-8 编码。如果遇到 CSV 乱码，Excel 请用 UTF-8-BOM 模式打开。

## 项目说明

- 类名 `MambaEmotionModel` 是历史命名，在 `--backend bigru_fallback` 时实际使用 BiGRU 序列编码器。
- 本仓库只包含代码、配置和文档，不包含任何音频数据或模型权重。
- `E:\newdog_emo` 只作为示例路径出现在文档中，代码本身不依赖任何固定盘符。