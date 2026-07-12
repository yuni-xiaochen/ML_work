# 家庭电力消耗预测 - 多变量时间序列

基于 **LSTM**、**Transformer** 和 **CNN-Transformer** 三种模型，对家庭电力消耗进行多变量时间序列预测。本项目为机器学习课程期末项目。

## 项目概述

根据过去 90 天的电力消耗数据和天气数据，预测未来 90 天（短期）和 365 天（长期）的家庭总有功功率（`global_active_power`）。

### 数据集

1. **家庭电力数据** - [UCI Individual household electric power consumption](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption)
   - 法国一户家庭 2006年12月 ~ 2010年11月 的分钟级用电记录
   - 包含有功功率、无功功率、电压、电流、各子表能耗

2. **气象数据** - [Météo-France 月度气候数据](https://www.data.gouv.fr/fr/datasets/donnees-climatologiques-de-base-mensuelles)
   - 选用 BAGNEUX 站（站号92007001）和 MEUDON 站（站号92048001）
   - 包含月降水量（RR）、降水天数（NBJRR1/5/10）、雾天数（NBJBROU）

### 特征工程

| 类别 | 特征 | 数量 |
|------|------|:---:|
| 电力变量 | Global\_reactive\_power, Voltage, Global\_intensity, Sub\_metering\_1/2/3, Sub\_metering\_remainder | 7 |
| 气象变量 | RR, NBJRR1, NBJRR5, NBJRR10, NBJBROU | 5 |
| 时间编码 | sin/cos(day\_of\_year), sin/cos(day\_of\_week) | 4 |
| 目标滞后 | Global\_active\_power lag\_1d/2d/3d/7d/14d/30d | 6 |
| 滚动统计 | roll\_mean\_7d/14d/30d, roll\_std\_7d/14d/30d | 6 |
| **合计** | | **28** |

### 预测任务

| 类型 | 输入窗口 | 输出窗口 |
|------|---------|---------|
| 短期预测 | 90 天 | 90 天 |
| 长期预测 | 90 天 | 365 天 |

## 模型

| 模型 | 参数量 | 编码器 | 解码器 |
|------|:------:|--------|--------|
| **LSTM Seq2Seq + Attention** | ~3.06M | 3层单向 LSTM（hidden=256） | LSTM 自回归 + Luong Attention |
| **Transformer** | ~2.62M | 4层 TransformerEncoder（d\_model=256, 8head） | MLP 一次性输出（非自回归） |
| **CNN-Transformer** ✨ | ~2.67M | 多尺度 1D-CNN → 4层 TransformerEncoder | MLP 一次性输出（非自回归） |

### 训练策略

| 配置项 | 值 |
|------|-----|
| 优化器 | AdamW（lr=5e-4, weight\_decay=1e-5） |
| 学习率调度 | CosineAnnealingWarmRestarts（T₀=30, T\_mult=2） |
| 损失函数 | Huber Loss（δ=1.0） |
| 批量大小 | 64 |
| 最大轮数 / 早停 | 300 / patience=50 |
| 混合精度 | AMP（GradScaler） |
| 设备 | NVIDIA RTX 5090（31.4 GB） |

## 项目结构

```
├── data_processing.py     # 数据加载、清洗、特征工程（日聚合/天气融合/滞后+滚动特征）
├── models.py              # LSTM / Transformer / CNN-Transformer 模型定义
├── train_evaluate.py      # 训练循环（AMP混合精度）与评估
├── main.py                # 主入口
├── report.tex             # 实验报告（LaTeX）
├── requirements.txt       # 依赖
└── results/               # 实验结果输出目录
    └── <timestamp>/       # 每次运行的结果
        ├── report.txt     # 文字报告（MSE/MAE/退化分析）
        ├── results_*.csv  # 汇总结果
        ├── predictions_*  # 预测值 CSV + 曲线图
        ├── comparison_*   # 模型对比图
        └── train.csv / test.csv
```

## 使用方法

### 1. 安装依赖

```bash
# 安装 PyTorch（根据 CUDA 版本选择）
pip install torch --index-url https://download.pytorch.org/whl/cu121

# 安装其他依赖
pip install -r requirements.txt
```

### 2. 运行实验

```bash
# 完整实验（所有模型 × 短期/长期，5轮）
python main.py

# 仅短期预测
python main.py --mode short

# 仅长期预测
python main.py --mode long

# 快速测试（LSTM短期，1轮）
python main.py --mode quick

# 单模型测试
python main.py --mode single --model lstm --output_len 90
```

### 3. 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--mode` | full | 运行模式：full/quick/short/long/single |
| `--model` | lstm | 模型名称：lstm/transformer/cnn_transformer |
| `--output_len` | 90 | 输出序列长度 |
| `--epochs` | 300 | 最大训练轮数 |
| `--batch_size` | 64 | 批量大小 |

## 评估指标

- **MSE** (均方误差)
- **MAE** (平均绝对误差)
- 每轮实验报告 **均值 ± 标准差**（至少 5 轮）

## 结果

实验结果（预测值 vs 真实值曲线对比图）和数值指标将输出到 `results/` 目录下。

## 参考文献

1. [LSTM 时间序列预测参考](https://blog.csdn.net/qq_47885795/article/details/143462299)
2. [Transformer 时间序列预测参考](https://blog.csdn.net/weixin_39653948/article/details/105431099)
3. [数据处理参考](https://datac.blog.csdn.net/article/details/105928752)
