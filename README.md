# 家庭电力消耗预测 - 多变量时间序列

基于 **LSTM**、**Transformer** 和 **CNN-Transformer** 三种模型，对家庭电力消耗进行多变量时间序列预测。本项目为机器学习课程期末项目。

## 项目概述

根据过去 90 天的电力消耗数据和天气数据，预测未来 90 天（短期）和 365 天（长期）的家庭总有功功率（`global_active_power`）。

### 数据集

1. **家庭电力数据** - [UCI Individual household electric power consumption](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption)
   - 法国一户家庭 2006年12月 ~ 2010年11月 的分钟级用电记录
   - 包含有功功率、无功功率、电压、电流、各子表能耗

2. **气象数据** - [Météo-France 月度气候数据](https://www.data.gouv.fr/fr/datasets/donnees-climatologiques-de-base-mensuelles)
   - 选用 BAGNEUX 站（距 Sceaux 最近）
   - 包含月降水量、降水天数、雾天数等特征

### 预测任务

| 类型 | 输入窗口 | 输出窗口 |
|------|---------|---------|
| 短期预测 | 90 天 | 90 天 |
| 长期预测 | 90 天 | 365 天 |

## 模型

| 模型 | 参数量 | 说明 |
|------|--------|------|
| **LSTM Seq2Seq** | ~505K | 编码器-解码器 LSTM + Luong Attention |
| **Transformer** | ~996K | 标准 Transformer 编码器-解码器 |
| **CNN-Transformer** | ~550K | 多尺度 1D-CNN + Transformer 编码器 + MLP 解码 |

### 改进模型（CNN-Transformer）创新点

- 使用三个平行 1D 卷积（kernel_size=3, 5, 7）提取多尺度局部时序特征
- Transformer 编码器建模长程依赖关系
- 直接 MLP 解码（非自回归），减少误差累积

## 项目结构

```
├── data_processing.py    # 数据加载、清洗、特征工程
├── models.py             # LSTM / Transformer / CNN-Transformer 模型
├── train_evaluate.py     # 训练循环与评估
├── main.py               # 主入口
├── requirements.txt      # 依赖
├── README.md             # 本文件
└── results/              # 实验结果输出目录
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
| `--epochs` | 200 | 最大训练轮数 |
| `--batch_size` | 32 | 批量大小 |

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
