"""
main.py - 主入口
运行完整的家庭电力消耗预测实验
使用方法：
  python main.py                    # 运行完整实验
  python main.py --mode quick       # 快速测试（仅LSTM短期，1轮）
  python main.py --mode short       # 仅短期预测
  python main.py --mode long         # 仅长期预测
  python main.py --mode single       # 单模型测试
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
from datetime import datetime

# 添加项目路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

from data_processing import process_all
from train_evaluate import run_experiment, set_seed

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def get_paths():
    """获取数据文件路径"""
    power_path = os.path.join(BASE_DIR, 'household_power_consumption.txt.gz')
    weather_path = os.path.join(BASE_DIR, 'weather_data.csv')
    return power_path, weather_path


def plot_predictions(targets, predictions, title, save_path, n_samples=3):
    """绘制预测vs真实值对比图"""
    fig, axes = plt.subplots(n_samples, 1, figsize=(12, 4 * n_samples))
    if n_samples == 1:
        axes = [axes]

    indices = np.linspace(0, len(targets) - 1, n_samples, dtype=int)

    for i, idx in enumerate(indices):
        ax = axes[i]
        out_len = len(targets[idx])
        days = np.arange(out_len)

        ax.plot(days, targets[idx], 'b-', label='Ground Truth', linewidth=1.5)
        ax.plot(days, predictions[idx], 'r--', label='Prediction', linewidth=1.5)
        ax.fill_between(days, targets[idx], predictions[idx], alpha=0.2, color='gray')

        mse = np.mean((targets[idx] - predictions[idx]) ** 2)
        mae = np.mean(np.abs(targets[idx] - predictions[idx]))
        ax.set_title(f'Sample {idx+1} | MSE={mse:.4f}, MAE={mae:.4f}')
        ax.set_xlabel('Days')
        ax.set_ylabel('Global Active Power (kWh)')
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  图表已保存: {save_path}")


def plot_comparison(summaries, save_path):
    """绘制模型对比图（MSE/MAE bar chart）"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    model_names = [s['model_name'].upper() for s in summaries]
    mse_means = [s['mse_mean'] for s in summaries]
    mse_stds = [s['mse_std'] for s in summaries]
    mae_means = [s['mae_mean'] for s in summaries]
    mae_stds = [s['mae_std'] for s in summaries]

    colors = ['#3498db', '#e74c3c', '#2ecc71']

    # MSE
    ax = axes[0]
    bars = ax.bar(model_names, mse_means, yerr=mse_stds, color=colors,
                  capsize=10, alpha=0.8, edgecolor='black')
    ax.set_title('MSE Comparison', fontsize=13, fontweight='bold')
    ax.set_ylabel('Mean Squared Error')
    for bar, val in zip(bars, mse_means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + mse_stds[0]*0.1,
                f'{val:.4f}', ha='center', va='bottom', fontsize=10)

    # MAE
    ax = axes[1]
    bars = ax.bar(model_names, mae_means, yerr=mae_stds, color=colors,
                  capsize=10, alpha=0.8, edgecolor='black')
    ax.set_title('MAE Comparison', fontsize=13, fontweight='bold')
    ax.set_ylabel('Mean Absolute Error')
    for bar, val in zip(bars, mae_means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + mae_stds[0]*0.1,
                f'{val:.4f}', ha='center', va='bottom', fontsize=10)

    plt.suptitle('Model Performance Comparison', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  对比图已保存: {save_path}")


def save_results(summaries, output_dir, mode_name):
    """保存结果到CSV"""
    rows = []
    for s in summaries:
        rows.append({
            'Model': s['model_name'].upper(),
            'Output_Length': s['output_len'],
            'MSE_Mean': s['mse_mean'],
            'MSE_Std': s['mse_std'],
            'MAE_Mean': s['mae_mean'],
            'MAE_Std': s['mae_std'],
            'Rounds': s['n_rounds'],
            'Individual_MSE': str([f'{v:.4f}' for v in s['mse_values']]),
            'Individual_MAE': str([f'{v:.4f}' for v in s['mae_values']]),
        })

    df = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, f'results_{mode_name}.csv')
    df.to_csv(csv_path, index=False)
    print(f"  结果已保存: {csv_path}")


def main():
    parser = argparse.ArgumentParser(description='家庭电力消耗预测')
    parser.add_argument('--mode', type=str, default='full',
                        choices=['full', 'quick', 'short', 'long', 'single'],
                        help='运行模式')
    parser.add_argument('--model', type=str, default='lstm',
                        choices=['lstm', 'transformer', 'cnn_transformer', 'mlp'],
                        help='单模型测试时指定模型')
    parser.add_argument('--output_len', type=int, default=90,
                        help='输出序列长度（单模型测试时使用）')
    parser.add_argument('--epochs', type=int, default=200,
                        help='最大训练轮数')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批量大小')
    args = parser.parse_args()

    # 创建输出目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(BASE_DIR, 'results', timestamp)
    os.makedirs(output_dir, exist_ok=True)

    # 设置全局随机种子（数据处理可复现）
    set_seed(42)

    # ==========================================
    # 数据处理
    # ==========================================
    print("\n" + "="*60)
    print("家庭电力消耗预测 - 多变量时间序列")
    print("="*60)

    power_path, weather_path = get_paths()
    data = process_all(power_path, weather_path)

    # 保存 train.csv 和 test.csv
    train_csv_path = os.path.join(output_dir, 'train.csv')
    test_csv_path = os.path.join(output_dir, 'test.csv')
    data['train_csv'].to_csv(train_csv_path, index=False)
    data['test_csv'].to_csv(test_csv_path, index=False)
    print(f"\ntrain.csv 已保存: {train_csv_path}")
    print(f"test.csv 已保存: {test_csv_path}")

    scaler_y = data['scalers']['y']

    # ==========================================
    # 确定运行配置
    # ==========================================
    if args.mode == 'quick':
        configs = [
            ('lstm', 'short', 42),
        ]
        seeds = [42]
        train_epochs = 50
        print("\n快速测试模式：仅LSTM短期，1轮")
    elif args.mode == 'short':
        configs = [
            ('lstm', 'short', None),
            ('transformer', 'short', None),
            ('cnn_transformer', 'short', None),
        ]
        seeds = [42, 123, 456, 789, 1024]
        train_epochs = args.epochs
        print("\n短期预测模式：所有模型，90→90")
    elif args.mode == 'long':
        configs = [
            ('lstm', 'long', None),
            ('transformer', 'long', None),
            ('cnn_transformer', 'long', None),
        ]
        seeds = [42, 123, 456, 789, 1024]
        train_epochs = args.epochs
        print("\n长期预测模式：所有模型，90→365")
    elif args.mode == 'single':
        configs = [(args.model, 'short' if args.output_len == 90 else 'long', None)]
        seeds = [42]
        train_epochs = args.epochs
        print(f"\n单模型测试：{args.model}, output_len={args.output_len}")
    else:  # full
        configs = [
            ('lstm', 'short', None),
            ('lstm', 'long', None),
            ('transformer', 'short', None),
            ('transformer', 'long', None),
            ('cnn_transformer', 'short', None),
            ('cnn_transformer', 'long', None),
        ]
        seeds = [42, 123, 456, 789, 1024]
        train_epochs = args.epochs
        print("\n完整实验模式：所有模型 × 短期/长期")

    # ==========================================
    # 运行实验
    # ==========================================
    all_summaries = []

    for model_name, horizon, _ in configs:
        output_len = 90 if horizon == 'short' else 365
        data_key = 'short' if horizon == 'short' else 'long'

        X_train = data[data_key]['X_train']
        y_train = data[data_key]['y_train']
        X_test = data[data_key]['X_test']
        y_test = data[data_key]['y_test']

        if len(X_test) == 0:
            print(f"\n警告: {model_name} {horizon} 测试集为空，跳过")
            continue

        # 运行多轮实验
        results, summary = run_experiment(
            model_name, X_train, y_train, X_test, y_test, scaler_y,
            output_len=output_len,
            seeds=seeds,
            epochs=train_epochs,
            batch_size=args.batch_size,
            lr=0.001,
            teacher_forcing_ratio=0.5,
            patience=20)

        all_summaries.append(summary)

        # 绘制最佳模型的预测曲线
        best_result = summary['best_result']
        plot_predictions(
            best_result['targets'],
            best_result['predictions'],
            f"{model_name.upper()} - {horizon} Prediction ({output_len} days)",
            os.path.join(output_dir, f'predictions_{model_name}_{horizon}.png')
        )

    # ==========================================
    # 汇总与对比
    # ==========================================
    if len(all_summaries) >= 2:
        # 分别绘制短期和长期对比
        short_summaries = [s for s in all_summaries if s['output_len'] == 90]
        long_summaries = [s for s in all_summaries if s['output_len'] == 365]

        if short_summaries:
            save_results(short_summaries, output_dir, 'short')
            plot_comparison(short_summaries,
                            os.path.join(output_dir, 'comparison_short.png'))

        if long_summaries:
            save_results(long_summaries, output_dir, 'long')
            plot_comparison(long_summaries,
                            os.path.join(output_dir, 'comparison_long.png'))

    # 保存所有结果
    save_results(all_summaries, output_dir, 'all')

    # 打印最终汇总
    print("\n" + "="*60)
    print("最终结果汇总")
    print("="*60)
    for s in all_summaries:
        print(f"\n{s['model_name'].upper()} (output_len={s['output_len']}):")
        print(f"  MSE: {s['mse_mean']:.4f} ± {s['mse_std']:.4f}")
        print(f"  MAE: {s['mae_mean']:.4f} ± {s['mae_std']:.4f}")
        print(f"  各轮MSE: {[f'{v:.4f}' for v in s['mse_values']]}")
        print(f"  各轮MAE: {[f'{v:.4f}' for v in s['mae_values']]}")

    print(f"\n所有结果已保存到: {output_dir}")
    return all_summaries, data


if __name__ == '__main__':
    main()
