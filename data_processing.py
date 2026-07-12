"""
data_processing.py - 家庭电力消耗预测数据处理模块
功能：加载、清洗、聚合电力数据与天气数据，构建时间序列样本
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# 1. 电力数据处理
# ============================================================

def load_power_data(filepath):
    """加载原始电力数据（支持 .gz 压缩）"""
    compression = 'gzip' if filepath.endswith('.gz') else None
    df = pd.read_csv(filepath, sep=';', low_memory=False, compression=compression)

    # 解析日期时间
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'],
                                     format='%d/%m/%Y %H:%M:%S', errors='coerce')
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', errors='coerce')

    # 删除无法解析的行
    df = df.dropna(subset=['Date']).copy()

    # 将 '?' 替换为 NaN 并转换为数值类型
    numeric_cols = ['Global_active_power', 'Global_reactive_power', 'Voltage',
                    'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # 按时间排序
    df = df.sort_values('datetime').reset_index(drop=True)

    return df


def clean_power_data(df):
    """处理电力数据中的缺失值"""
    # 前向填充 + 后向填充（时间序列场景）
    numeric_cols = ['Global_active_power', 'Global_reactive_power', 'Voltage',
                    'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']
    for col in numeric_cols:
        df[col] = df[col].ffill().bfill()

    return df


def aggregate_daily(df):
    """
    按天聚合电力数据
    - 功率类变量：求和（kW·min → 日总kWh = sum / 60）
    - 电压/电流：取平均
    """
    daily = df.groupby('Date').agg({
        'Global_active_power': 'sum',      # kW·min → /60 = kWh
        'Global_reactive_power': 'sum',
        'Voltage': 'mean',
        'Global_intensity': 'mean',
        'Sub_metering_1': 'sum',           # Wh·min → 保持 Wh
        'Sub_metering_2': 'sum',
        'Sub_metering_3': 'sum'
    }).reset_index()

    # 转换单位
    # 每分钟记录一次，功率kW × 1分钟 = kW·min，除以60得kWh
    daily['Global_active_power'] = daily['Global_active_power'] / 60.0
    daily['Global_reactive_power'] = daily['Global_reactive_power'] / 60.0

    # Sub_metering 单位是Wh（有功电能），每分钟累加，直接求和即可（单位仍为Wh）
    # 转换为kWh
    daily['Sub_metering_1'] = daily['Sub_metering_1'] / 1000.0
    daily['Sub_metering_2'] = daily['Sub_metering_2'] / 1000.0
    daily['Sub_metering_3'] = daily['Sub_metering_3'] / 1000.0

    # 计算剩余能耗 (sub_metering_remainder)
    # global_active_power(kWh) * 1000 → Wh, /60 回到每分钟, 但这个公式是针对分钟级数据的
    # 对于日聚合数据: sub_metering_remainder = global_active_power(kWh) - (sub1+sub2+sub3)(kWh)
    daily['Sub_metering_remainder'] = daily['Global_active_power'] - (
        daily['Sub_metering_1'] + daily['Sub_metering_2'] + daily['Sub_metering_3'])
    daily['Sub_metering_remainder'] = daily['Sub_metering_remainder'].clip(lower=0)

    return daily


# ============================================================
# 2. 天气数据处理
# ============================================================

def load_weather_data(filepath, station_id=92007001):
    """加载并处理天气数据（BAGNEUX站）"""
    w = pd.read_csv(filepath, sep=';', low_memory=False)

    # 过滤指定站点
    station = w[w['NUM_POSTE'] == station_id].copy()

    # 提取需要的列
    weather_cols = ['AAAAMM', 'RR', 'NBJRR1', 'NBJRR5', 'NBJRR10']
    weather = station[weather_cols].copy()

    # 转换为数值
    for col in ['RR', 'NBJRR1', 'NBJRR5', 'NBJRR10']:
        weather[col] = pd.to_numeric(weather[col], errors='coerce')

    # RR 单位转换：十分之一mm → mm
    weather['RR'] = weather['RR'] / 10.0

    # 确保 AAAAMM 为整数
    weather['AAAAMM'] = weather['AAAAMM'].astype(int)

    return weather


def load_fog_data(filepath, station_id=92048001):
    """从 MEUDON 站加载雾天数据"""
    w = pd.read_csv(filepath, sep=';', low_memory=False)
    station = w[w['NUM_POSTE'] == station_id].copy()
    fog = station[['AAAAMM', 'NBJBROU']].copy()
    fog['NBJBROU'] = pd.to_numeric(fog['NBJBROU'], errors='coerce')
    fog['AAAAMM'] = fog['AAAAMM'].astype(int)
    fog = fog.dropna(subset=['NBJBROU'])
    return fog


def build_weather_mapping(power_dates, weather_df, fog_df):
    """
    构建每日→天气数据的映射
    - 对电力数据覆盖的每个月份，填入对应的天气数据
    - 缺失月份进行插值
    """
    # 确定需要的月份范围
    min_date = power_dates.min()
    max_date = power_dates.max()
    months_needed = pd.date_range(
        min_date.replace(day=1), max_date.replace(day=28), freq='MS')

    # 创建完整月份列表
    month_list = []
    for m in months_needed:
        month_list.append(int(m.strftime('%Y%m')))

    full_weather = pd.DataFrame({'AAAAMM': month_list})

    # 合并降水数据
    full_weather = full_weather.merge(weather_df, on='AAAAMM', how='left')

    # 合并雾天数据
    full_weather = full_weather.merge(fog_df, on='AAAAMM', how='left')

    # 线性插值填充缺失
    cols_to_interp = ['RR', 'NBJRR1', 'NBJRR5', 'NBJRR10', 'NBJBROU']
    for col in cols_to_interp:
        if col in full_weather.columns:
            full_weather[col] = full_weather[col].interpolate(
                method='linear', limit_direction='both')
        else:
            full_weather[col] = 0.0

    # 剩余NaN填充为0（NBJBROU在开始或结束时可能全是NaN）
    full_weather[cols_to_interp] = full_weather[cols_to_interp].fillna(0)

    return full_weather


def merge_weather_to_daily(daily_df, weather_monthly):
    """
    将月度天气数据合并到每日数据
    每月所有天使用相同的天气值
    """
    # 创建年月列用于合并
    daily_df = daily_df.copy()
    daily_df['AAAAMM'] = daily_df['Date'].dt.year * 100 + daily_df['Date'].dt.month

    # 合并
    weather_map = weather_monthly.set_index('AAAAMM')
    weather_cols = ['RR', 'NBJRR1', 'NBJRR5', 'NBJRR10', 'NBJBROU']

    for col in weather_cols:
        daily_df[col] = daily_df['AAAAMM'].map(weather_map[col])

    daily_df = daily_df.drop(columns=['AAAAMM'])

    return daily_df


# ============================================================
# 3. 特征工程
# ============================================================

def add_time_features(df):
    """添加时间周期特征（sin/cos编码）"""
    date_series = df['Date']

    # Day of year (1-365/366)
    day_of_year = date_series.dt.dayofyear
    max_doy = 365.25
    df['sin_dayofyear'] = np.sin(2 * np.pi * day_of_year / max_doy)
    df['cos_dayofyear'] = np.cos(2 * np.pi * day_of_year / max_doy)

    # Day of week (0-6)
    day_of_week = date_series.dt.dayofweek
    df['sin_dayofweek'] = np.sin(2 * np.pi * day_of_week / 7)
    df['cos_dayofweek'] = np.cos(2 * np.pi * day_of_week / 7)

    return df


def add_target_lag_features(df, target_col, lag_days=(1, 2, 3, 7, 14, 30)):
    """添加目标变量的滞后特征 — 最关键的特征工程"""
    for lag in lag_days:
        df[f'{target_col}_lag_{lag}d'] = df[target_col].shift(lag)
    return df


def add_rolling_features(df, target_col, windows=(7, 14, 30)):
    """添加目标变量的滚动统计量"""
    for w in windows:
        df[f'{target_col}_roll_mean_{w}d'] = (
            df[target_col].shift(1).rolling(w).mean())
        df[f'{target_col}_roll_std_{w}d'] = (
            df[target_col].shift(1).rolling(w).std())
    return df


# ============================================================
# 4. 序列构建
# ============================================================

def create_sequences(data, target_col, feature_cols, input_len=90, output_len=90):
    """
    从时间序列数据构建输入-输出序列样本

    参数:
        data: DataFrame，包含特征和目标
        target_col: 目标列名
        feature_cols: 特征列名列表
        input_len: 输入序列长度（天）
        output_len: 输出序列长度（天）

    返回:
        X: shape (n_samples, input_len, n_features)
        y: shape (n_samples, output_len)
    """
    X_list, y_list = [], []

    feature_values = data[feature_cols].values.astype(np.float32)
    target_values = data[target_col].values.astype(np.float32)

    total_len = input_len + output_len
    n_samples = len(data) - total_len + 1

    for i in range(n_samples):
        X_list.append(feature_values[i:i + input_len])
        y_list.append(target_values[i + input_len:i + total_len])

    if len(X_list) == 0:
        return np.array([]), np.array([])

    X = np.stack(X_list)
    y = np.stack(y_list)

    return X, y


# ============================================================
# 5. 完整数据处理流水线
# ============================================================

def process_all(power_path, weather_path, input_len=90,
                short_output=90, long_output=365, test_ratio=0.3):
    """
    完整数据处理流水线

    返回:
        data_dict: 包含所有处理后数据的字典
    """
    print("=" * 60)
    print("数据处理流水线")
    print("=" * 60)

    # 1. 加载电力数据
    print("\n[1/6] 加载电力数据...")
    df_power = load_power_data(power_path)
    print(f"  原始数据: {len(df_power)} 行")

    # 2. 清洗
    print("\n[2/6] 清洗缺失值...")
    df_power = clean_power_data(df_power)

    # 3. 按天聚合
    print("\n[3/6] 按天聚合...")
    daily = aggregate_daily(df_power)
    print(f"  日聚合后: {len(daily)} 天")
    print(f"  日期范围: {daily['Date'].min().date()} ~ {daily['Date'].max().date()}")

    # 4. 加载天气数据并合并
    print("\n[4/6] 处理天气数据...")
    weather = load_weather_data(weather_path)
    fog = load_fog_data(weather_path)
    weather_monthly = build_weather_mapping(daily['Date'], weather, fog)
    daily = merge_weather_to_daily(daily, weather_monthly)
    print(f"  天气数据月份数: {len(weather_monthly)}")

    # 5. 特征工程
    print("\n[5/7] 特征工程...")
    daily = add_time_features(daily)

    # 定义特征和目标
    target_col = 'Global_active_power'
    base_feature_cols = [
        'Global_reactive_power', 'Voltage', 'Global_intensity',
        'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3',
        'Sub_metering_remainder',
        'RR', 'NBJRR1', 'NBJRR5', 'NBJRR10', 'NBJBROU',
        'sin_dayofyear', 'cos_dayofyear', 'sin_dayofweek', 'cos_dayofweek'
    ]

    # 5b. 目标滞后 + 滚动统计（最关键的特征！）
    print("\n[6/7] 添加目标滞后与滚动特征...")
    daily = add_target_lag_features(daily, target_col)
    daily = add_rolling_features(daily, target_col)

    # 构建完整特征列表
    lag_cols = [f'{target_col}_lag_{d}d' for d in (1, 2, 3, 7, 14, 30)]
    roll_cols = []
    for w in (7, 14, 30):
        roll_cols.append(f'{target_col}_roll_mean_{w}d')
        roll_cols.append(f'{target_col}_roll_std_{w}d')
    feature_cols = base_feature_cols + lag_cols + roll_cols

    # 删除滞后产生的 NaN 行（前30天）
    daily = daily.dropna(subset=lag_cols + roll_cols).reset_index(drop=True)
    print(f"  添加滞后+滚动特征后: {len(daily)} 天 (删除了前30天NaN)")
    print(f"  特征总数: {len(feature_cols)} (基础{len(base_feature_cols)} + 滞后{len(lag_cols)} + 滚动{len(roll_cols)})")
    print(f"  目标: {target_col}")

    # 检查缺失值
    all_cols = feature_cols + [target_col]
    n_missing = daily[all_cols].isnull().sum().sum()
    if n_missing > 0:
        print(f"  警告: {n_missing} 个缺失值，进行填充...")
        daily[all_cols] = daily[all_cols].ffill().bfill().fillna(0)

    # 7. 数据划分与序列构建
    print("\n[7/7] 数据划分与序列构建...")

    # 按时间顺序划分，确保测试集至少有 max(input_len + output_len) 天
    n_total = len(daily)
    min_test_days = input_len + max(short_output, long_output)  # 455天
    n_test = max(int(n_total * test_ratio), min_test_days)

    # 如果测试集占比过大，适当减少
    max_test = n_total - input_len - short_output  # 至少保留一些训练样本
    if n_test > max_test:
        n_test = max_test

    n_train = n_total - n_test

    train_data = daily.iloc[:n_train].reset_index(drop=True)
    test_data = daily.iloc[n_train:].reset_index(drop=True)

    print(f"  训练集: {len(train_data)} 天 ({train_data['Date'].min().date()} ~ {train_data['Date'].max().date()})")
    print(f"  测试集: {len(test_data)} 天 ({test_data['Date'].min().date()} ~ {test_data['Date'].max().date()})")

    # 标准化
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    scaler_X.fit(train_data[feature_cols].values)
    scaler_y.fit(train_data[[target_col]].values)

    # 标准化训练集和测试集
    train_scaled = train_data.copy()
    test_scaled = test_data.copy()
    train_scaled[feature_cols] = scaler_X.transform(train_data[feature_cols].values)
    test_scaled[feature_cols] = scaler_X.transform(test_data[feature_cols].values)
    train_scaled[target_col] = scaler_y.transform(train_data[[target_col]].values)
    test_scaled[target_col] = scaler_y.transform(test_data[[target_col]].values)

    # 构建序列
    # 短期预测 (90→90)
    X_train_short, y_train_short = create_sequences(
        train_scaled, target_col, feature_cols,
        input_len=input_len, output_len=short_output)
    X_test_short, y_test_short = create_sequences(
        test_scaled, target_col, feature_cols,
        input_len=input_len, output_len=short_output)

    # 长期预测 (90→365)
    X_train_long, y_train_long = create_sequences(
        train_scaled, target_col, feature_cols,
        input_len=input_len, output_len=long_output)
    X_test_long, y_test_long = create_sequences(
        test_scaled, target_col, feature_cols,
        input_len=input_len, output_len=long_output)

    print(f"\n  短期预测 (90→90):")
    print(f"    训练集 X: {X_train_short.shape}, y: {y_train_short.shape}")
    print(f"    测试集 X: {X_test_short.shape}, y: {y_test_short.shape}")

    print(f"\n  长期预测 (90→365):")
    print(f"    训练集 X: {X_train_long.shape}, y: {y_train_long.shape}")
    print(f"    测试集 X: {X_test_long.shape}, y: {y_test_long.shape}")

    # 保存 train.csv 和 test.csv（原始尺度，用于提交）
    train_csv = train_data[['Date'] + all_cols].copy()
    test_csv = test_data[['Date'] + all_cols].copy()

    # 打包返回
    data_dict = {
        'short': {
            'X_train': X_train_short, 'y_train': y_train_short,
            'X_test': X_test_short, 'y_test': y_test_short,
        },
        'long': {
            'X_train': X_train_long, 'y_train': y_train_long,
            'X_test': X_test_long, 'y_test': y_test_long,
        },
        'scalers': {'X': scaler_X, 'y': scaler_y},
        'feature_cols': feature_cols,
        'target_col': target_col,
        'train_csv': train_csv,
        'test_csv': test_csv,
        'daily_raw': daily
    }

    print("\n数据处理完成!")
    return data_dict


if __name__ == '__main__':
    # 测试数据处理
    import os
    # WSL环境下使用 /mnt/ 路径
    base = '/mnt/c/Users/19023/Desktop/文档/研究生课程/机器学习期末大作业'
    power_path = os.path.join(base, 'household_power_consumption.txt.gz')
    weather_path = os.path.join(base, 'weather_data.csv')

    data = process_all(power_path, weather_path)
