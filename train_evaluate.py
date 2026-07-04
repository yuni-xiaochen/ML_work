"""
train_evaluate.py - 训练循环与评估
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from models import create_model
from tqdm import tqdm
import time


def set_seed(seed):
    """设置随机种子"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_epoch(model, dataloader, optimizer, criterion, device,
                teacher_forcing_ratio, output_len, pbar=None):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch_X, batch_y in dataloader:
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()

        predictions = model(batch_X, target_seq=batch_y,
                            teacher_forcing_ratio=teacher_forcing_ratio,
                            output_len=output_len)

        loss = criterion(predictions, batch_y)
        loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

        if pbar is not None:
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    return total_loss / n_batches if n_batches > 0 else float('inf')


@torch.no_grad()
def evaluate(model, dataloader, criterion, device, output_len, scaler_y=None):
    """评估模型"""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_predictions = []
    all_targets = []

    for batch_X, batch_y in dataloader:
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)

        # 推理时不使用teacher forcing
        predictions = model(batch_X, target_seq=None,
                            teacher_forcing_ratio=0.0,
                            output_len=output_len)

        loss = criterion(predictions, batch_y)
        total_loss += loss.item()
        n_batches += 1

        all_predictions.append(predictions.cpu().numpy())
        all_targets.append(batch_y.cpu().numpy())

    avg_loss = total_loss / n_batches if n_batches > 0 else float('inf')

    # 合并所有预测和目标
    preds = np.concatenate(all_predictions, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    # 如果提供了scaler，反标准化后计算指标
    if scaler_y is not None:
        n_samples, out_len = preds.shape
        preds_flat = preds.reshape(-1, 1)
        targets_flat = targets.reshape(-1, 1)
        preds_orig = scaler_y.inverse_transform(preds_flat).reshape(n_samples, out_len)
        targets_orig = scaler_y.inverse_transform(targets_flat).reshape(n_samples, out_len)
    else:
        preds_orig = preds
        targets_orig = targets

    # 计算指标
    mse = mean_squared_error(targets_orig.flatten(), preds_orig.flatten())
    mae = mean_absolute_error(targets_orig.flatten(), preds_orig.flatten())

    return {
        'loss': avg_loss,
        'mse': mse,
        'mae': mae,
        'predictions': preds_orig,
        'targets': targets_orig
    }


def train_model(model, X_train, y_train, X_test, y_test, scaler_y,
                output_len=90, batch_size=32, epochs=200,
                lr=0.001, teacher_forcing_ratio=0.5,
                patience=20, device='cpu', verbose=True):
    """
    完整训练流程

    参数:
        model: 模型实例
        X_train, y_train: 训练数据（numpy数组）
        X_test, y_test: 测试数据（numpy数组）
        scaler_y: 目标变量的StandardScaler
        output_len: 输出序列长度
        batch_size: 批量大小
        epochs: 最大训练轮数
        lr: 学习率
        teacher_forcing_ratio: teacher forcing比例
        patience: 早停耐心值
        device: 计算设备
        verbose: 是否打印进度

    返回:
        results: 包含训练历史和测试结果的字典
    """
    model = model.to(device)

    # 创建数据加载器
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test), torch.FloatTensor(y_test))

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                               shuffle=True, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                              shuffle=False, drop_last=False)

    # 优化器和损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10)
    criterion = nn.MSELoss()

    # 训练
    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': []}

    start_time = time.time()

    # 使用 tqdm 进度条
    epoch_pbar = tqdm(range(epochs), desc='Training', unit='epoch',
                        ncols=100, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

    for epoch in epoch_pbar:
        # 训练
        train_pbar = epoch_pbar if hasattr(epoch_pbar, 'set_postfix') else None
        train_loss = train_epoch(model, train_loader, optimizer, criterion,
                                 device, teacher_forcing_ratio, output_len,
                                 pbar=train_pbar)
        history['train_loss'].append(train_loss)

        # 验证
        val_results = evaluate(model, test_loader, criterion, device,
                               output_len, scaler_y=None)
        val_loss = val_results['loss']
        history['val_loss'].append(val_loss)

        # 学习率调度
        scheduler.step(val_loss)

        # 更新进度条描述
        if verbose:
            epoch_pbar.set_postfix({
                'train': f'{train_loss:.4f}',
                'val': f'{val_loss:.4f}',
                'best': f'{best_val_loss:.4f}',
                'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
            })

        # 早停
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            if verbose:
                tqdm.write(f"  早停于 epoch {epoch + 1}")
            break

    if verbose and epoch_pbar is not None and hasattr(epoch_pbar, 'close'):
        epoch_pbar.close()

    # 恢复最佳模型
    if best_state is not None:
        model.load_state_dict(best_state)

    # 最终评估（使用原始尺度）
    train_eval = evaluate(model, train_loader, criterion, device,
                          output_len, scaler_y=scaler_y)
    test_eval = evaluate(model, test_loader, criterion, device,
                         output_len, scaler_y=scaler_y)

    elapsed = time.time() - start_time

    if verbose:
        print(f"  训练完成 | 用时: {elapsed:.0f}s | "
              f"Best Val Loss: {best_val_loss:.6f}")
        print(f"  Train MSE: {train_eval['mse']:.4f} | MAE: {train_eval['mae']:.4f}")
        print(f"  Test  MSE: {test_eval['mse']:.4f} | MAE: {test_eval['mae']:.4f}")

    return {
        'model': model,
        'history': history,
        'train_mse': train_eval['mse'],
        'train_mae': train_eval['mae'],
        'test_mse': test_eval['mse'],
        'test_mae': test_eval['mae'],
        'predictions': test_eval['predictions'],
        'targets': test_eval['targets'],
        'best_val_loss': best_val_loss,
        'epochs_trained': len(history['train_loss']),
        'time': elapsed
    }


def run_experiment(model_name, X_train, y_train, X_test, y_test, scaler_y,
                   output_len=90, seeds=None, **kwargs):
    """
    运行多轮实验

    参数:
        model_name: 模型名称
        X_train, y_train: 训练数据
        X_test, y_test: 测试数据
        scaler_y: 目标变量scaler
        output_len: 输出序列长度
        seeds: 随机种子列表
        **kwargs: 传递给create_model的额外参数

    返回:
        all_results: 每轮实验的结果列表
        summary: 汇总统计
    """
    if seeds is None:
        seeds = [42, 123, 456, 789, 1024]

    input_dim = X_train.shape[2]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    all_results = []

    print(f"\n{'='*60}")
    print(f"实验: {model_name.upper()} | 输出长度: {output_len} | 设备: {device}")
    print(f"{'='*60}")

    for i, seed in enumerate(seeds):
        print(f"\n--- Round {i+1}/{len(seeds)} (seed={seed}) ---")

        set_seed(seed)

        # 创建新模型
        model_kwargs = kwargs.copy()
        model_kwargs.setdefault('output_len', output_len)
        model = create_model(model_name, input_dim, **model_kwargs)

        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  参数量: {n_params:,}")

        # 训练
        result = train_model(
            model, X_train, y_train, X_test, y_test, scaler_y,
            output_len=output_len,
            batch_size=kwargs.get('batch_size', 32),
            epochs=kwargs.get('epochs', 200),
            lr=kwargs.get('lr', 0.001),
            teacher_forcing_ratio=kwargs.get('teacher_forcing_ratio', 0.5),
            patience=kwargs.get('patience', 20),
            device=device,
            verbose=True)

        all_results.append(result)

    # 汇总统计
    test_mses = [r['test_mse'] for r in all_results]
    test_maes = [r['test_mae'] for r in all_results]

    summary = {
        'model_name': model_name,
        'output_len': output_len,
        'n_rounds': len(seeds),
        'mse_mean': np.mean(test_mses),
        'mse_std': np.std(test_mses),
        'mae_mean': np.mean(test_maes),
        'mae_std': np.std(test_maes),
        'mse_values': test_mses,
        'mae_values': test_maes,
        'best_result': all_results[np.argmin(test_mses)],
        'all_results': all_results
    }

    print(f"\n{'='*60}")
    print(f"汇总: {model_name.upper()} (output_len={output_len})")
    print(f"  MSE: {summary['mse_mean']:.4f} ± {summary['mse_std']:.4f}")
    print(f"  MAE: {summary['mae_mean']:.4f} ± {summary['mae_std']:.4f}")
    print(f"{'='*60}")

    return all_results, summary
