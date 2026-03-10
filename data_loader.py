# data_loader.py
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from config import *


def load_cv_folds(cv_file_path):
    """
    从JSON文件加载交叉验证划分
    """
    with open(cv_file_path, 'r') as f:
        folds_data = json.load(f)
    return folds_data


def load_exp_data(phenotype, num_features=3000):
    """
    加载基因表达数据，处理缺失值和异常值，并筛选出最重要的特征

    Args:
        phenotype (str): 表型列名
        num_features (int): 筛选后保留的特征数量

    Returns:
        exp_X_standardized (np.array): 标准化的表达数据
        y_scaled (np.array): 标准化的表型数据
        sample_ids (list): 样本ID列表
        selected_feature_names (list): 筛选后的特征名称列表
    """
    # 加载表型数据并过滤缺失值
    phenotypes_df = pd.read_csv(PHENOTYPES_FILE)
    phenotypes_df = phenotypes_df[phenotypes_df[phenotype].notna()]

    # 加载表达数据
    data_frame = pd.read_csv(EXP_FILE)

    # 保留同时存在表型和表达数据的样本
    data_frame = data_frame[data_frame['ID'].isin(phenotypes_df['ID'])]

    # 获取数值列（排除 'ID' 列）
    numeric_columns = data_frame.select_dtypes(include=[np.number]).columns
    if 'ID' in numeric_columns:
        numeric_columns = numeric_columns.drop('ID')

    # 替换无穷大值
    data_frame[numeric_columns] = data_frame[numeric_columns].replace([np.inf, -np.inf], np.nan)

    # 移除全为NaN的列
    data_frame = data_frame.dropna(axis=1, how='all')

    # 使用列均值填充剩余的NaN
    data_frame[numeric_columns] = data_frame[numeric_columns].fillna(data_frame[numeric_columns].mean())

    # 提取样本ID和特征
    sample_ids = data_frame['ID'].tolist()
    exp_X = data_frame[numeric_columns].values

    # 获取表型值
    y = np.array([phenotypes_df.loc[phenotypes_df['ID'] == item, phenotype].values[0] for item in sample_ids])

    # 处理表型数据中的异常值
    y = np.nan_to_num(y, nan=np.nanmean(y))

    # 特征选择：保留最重要的 num_features 个特征
    selector = SelectKBest(score_func=f_regression, k=num_features)
    exp_X_selected = selector.fit_transform(exp_X, y)

    # 获取被选中的特征名称
    selected_feature_mask = selector.get_support()
    selected_feature_names = numeric_columns[selected_feature_mask].tolist()

    # 标准化
    X_scaler = StandardScaler()
    y_scaler = StandardScaler()

    exp_X_standardized = X_scaler.fit_transform(exp_X_selected).astype(np.float32)
    y_scaled = y_scaler.fit_transform(y.reshape(-1, 1)).flatten().astype(np.float32)

    return exp_X_standardized, y_scaled, sample_ids, selected_feature_names


def load_snp_data(phenotype):
    """
    加载SNP数据，处理缺失值和异常值

    Returns:
        snp_X_standardized (np.array): 标准化的SNP数据
        y_scaled (np.array): 标准化的表型数据
        sample_ids (list): 样本ID列表
        snp_feature_names (list): SNP特征名称列表
    """
    # 加载表型数据并过滤缺失值
    SNP_FILE = f'/data1/wangchengrui/final_results/eqtl/rice4k_219/{phenotype}.csv'
    phenotypes_df = pd.read_csv(PHENOTYPES_FILE)
    phenotypes_df = phenotypes_df[phenotypes_df[phenotype].notna()]

    # 直接加载SNP数据
    snp_df = pd.read_csv(SNP_FILE)

    # 删除非数值列
    non_numeric_columns = ['FID', 'PAT', 'MAT', 'SEX', 'PHENOTYPE']
    snp_df = snp_df.drop(columns=[col for col in non_numeric_columns if col in snp_df.columns], errors='ignore')

    # 获取特征名称（去掉ID列）
    snp_feature_names = list(snp_df.columns.drop('ID'))

    # 保留同时存在表型和SNP数据的样本
    snp_df = snp_df[snp_df['ID'].isin(phenotypes_df['ID'])]

    # 处理数据中的异常值
    numeric_columns = snp_df.select_dtypes(include=[np.number]).columns

    # 替换无穷大值
    snp_df[numeric_columns] = snp_df[numeric_columns].replace([np.inf, -np.inf], np.nan)

    # 移除全为NaN的列
    snp_df = snp_df.dropna(axis=1, how='all')

    # 使用列均值填充剩余的NaN
    snp_df[numeric_columns] = snp_df[numeric_columns].fillna(snp_df[numeric_columns].mean())

    # 提取样本ID和特征
    sample_ids = snp_df['ID'].tolist()
    snp_X = snp_df.drop(['ID'], axis=1).astype(np.float32).values

    # 获取表型值
    y = np.array([phenotypes_df.loc[phenotypes_df['ID'] == item, phenotype].values[0] for item in sample_ids])

    # 处理表型数据中的异常值
    y = np.nan_to_num(y, nan=np.nanmean(y))

    # 标准化
    X_scaler = StandardScaler()
    y_scaler = StandardScaler()

    snp_X_standardized = X_scaler.fit_transform(snp_X).astype(np.float32)
    y_scaled = y_scaler.fit_transform(y.reshape(-1, 1)).flatten().astype(np.float32)

    return snp_X_standardized, y_scaled, sample_ids, snp_feature_names


def load_combined_data(phenotype, num_exp_features=3000, num_snp_features=5000):
    """
    加载组合数据，处理缺失值和异常值，并筛选特征

    Args:
        phenotype (str): 表型列名
        num_exp_features (int): 筛选后保留的表达数据特征数量
        num_snp_features (int): 筛选后保留的SNP数据特征数量

    Returns:
        exp_X_standardized (np.array): 标准化的表达数据
        snp_X_standardized (np.array): 标准化的SNP数据
        y_scaled (np.array): 标准化的表型数据
        sample_ids (list): 样本ID列表
        selected_exp_feature_names (list): 筛选后的表达特征名称列表
        selected_snp_feature_names (list): 筛选后的SNP特征名称列表
    """
    # 加载表型数据并过滤缺失值
    SNP_FILE = f'/data1/wangchengrui/final_results/eqtl/rice4k_219/{phenotype}.csv'
    phenotypes_df = pd.read_csv(PHENOTYPES_FILE)
    phenotypes_df = phenotypes_df[phenotypes_df[phenotype].notna()]

    # 加载基因表达和SNP数据
    exp_data = pd.read_csv(EXP_FILE)
    snp_data = pd.read_csv(SNP_FILE, sep=' ')

    # 删除SNP数据中的非数值列（但保留 ID 列）
    non_numeric_columns = ['FID', 'PAT', 'MAT', 'SEX', 'PHENOTYPE']
    snp_data = snp_data.drop(columns=[col for col in non_numeric_columns if col in snp_data.columns], errors='ignore')

    # 确保 'ID' 列存在
    if 'ID' not in exp_data.columns or 'ID' not in snp_data.columns:
        raise ValueError("表达数据或SNP数据中缺少 'ID' 列，请检查输入文件的列名")

    # 获取特征名称
    exp_feature_names = list(exp_data.columns.drop('ID', errors='ignore'))
    snp_feature_names = list(snp_data.columns.drop('ID', errors='ignore'))

    # 找出共同的样本
    common_ids = set(exp_data['ID']) & set(snp_data['ID']) & set(phenotypes_df['ID'])

    if len(common_ids) == 0:
        raise ValueError("没有同时具有SNP和表达数据的样本")

    # 过滤数据
    exp_data = exp_data[exp_data['ID'].isin(common_ids)]
    snp_data = snp_data[snp_data['ID'].isin(common_ids)]
    phenotypes_df = phenotypes_df[phenotypes_df['ID'].isin(common_ids)]

    # 处理异常值
    exp_numeric_cols = exp_data.select_dtypes(include=[np.number]).columns.drop('ID', errors='ignore')
    snp_numeric_cols = snp_data.select_dtypes(include=[np.number]).columns.drop('ID', errors='ignore')

    # 替换无穷大值并填充NaN
    exp_data[exp_numeric_cols] = exp_data[exp_numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(
        exp_data[exp_numeric_cols].mean())
    snp_data[snp_numeric_cols] = snp_data[snp_numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(
        snp_data[snp_numeric_cols].mean())

    # 提取数据
    sample_ids = list(common_ids)
    exp_X = exp_data.set_index('ID').loc[sample_ids][exp_numeric_cols].values
    snp_X = snp_data.set_index('ID').loc[sample_ids][snp_numeric_cols].values

    # 获取表型值
    y = np.array([phenotypes_df.loc[phenotypes_df['ID'] == item, phenotype].values[0] for item in sample_ids])

    # 处理表型数据中的异常值
    y = np.nan_to_num(y, nan=np.nanmean(y))

    # 特征选择：使用 SelectKBest 对 SNP 和 EXP 数据分别筛选特征
    num_exp_features = min(num_exp_features, len(exp_numeric_cols))
    num_snp_features = min(num_snp_features, len(snp_numeric_cols))

    exp_selector = SelectKBest(score_func=f_regression, k=num_exp_features)
    snp_selector = SelectKBest(score_func=f_regression, k=num_snp_features)

    exp_X_selected = exp_selector.fit_transform(exp_X, y)
    snp_X_selected = snp_selector.fit_transform(snp_X, y)

    # 获取被选中的特征名称
    selected_exp_feature_mask = exp_selector.get_support()
    selected_snp_feature_mask = snp_selector.get_support()

    selected_exp_feature_names = list(exp_numeric_cols[selected_exp_feature_mask])
    selected_snp_feature_names = list(snp_numeric_cols[selected_snp_feature_mask])

    # 标准化
    exp_scaler = StandardScaler()
    snp_scaler = StandardScaler()
    y_scaler = StandardScaler()

    exp_X_standardized = exp_scaler.fit_transform(exp_X_selected).astype(np.float32)
    snp_X_standardized = snp_scaler.fit_transform(snp_X_selected).astype(np.float32)
    y_scaled = y_scaler.fit_transform(y.reshape(-1, 1)).flatten().astype(np.float32)

    return exp_X_standardized, snp_X_standardized, y_scaled, sample_ids, selected_exp_feature_names, selected_snp_feature_names


def create_data_loaders(X_train, y_train, X_test, y_test, snp_X_train=None, snp_X_test=None):
    """
    创建数据加载器
    """
    batch_size = TRAINING_PARAMS['batch_size']

    if snp_X_train is not None and snp_X_test is not None:
        # 组合模型数据加载器
        train_loader = DataLoader(
            TensorDataset(
                torch.tensor(X_train, dtype=torch.float32),
                torch.tensor(snp_X_train, dtype=torch.float32),
                torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
            ),
            batch_size=batch_size,
            shuffle=True
        )

        test_loader = DataLoader(
            TensorDataset(
                torch.tensor(X_test, dtype=torch.float32),
                torch.tensor(snp_X_test, dtype=torch.float32),
                torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)
            ),
            batch_size=batch_size,
            shuffle=False
        )
    else:
        # 单模态模型数据加载器
        train_loader = DataLoader(
            TensorDataset(
                torch.tensor(X_train, dtype=torch.float32),
                torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
            ),
            batch_size=batch_size,
            shuffle=True
        )

        test_loader = DataLoader(
            TensorDataset(
                torch.tensor(X_test, dtype=torch.float32),
                torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)
            ),
            batch_size=batch_size,
            shuffle=False
        )

    return train_loader, test_loader


def create_prediction_loader(X, snp_X=None):
    """
    创建用于预测的数据加载器
    """
    batch_size = TRAINING_PARAMS['batch_size']

    if snp_X is not None:
        # 组合模型预测加载器
        pred_loader = DataLoader(
            TensorDataset(
                torch.tensor(X, dtype=torch.float32),
                torch.tensor(snp_X, dtype=torch.float32)
            ),
            batch_size=batch_size,
            shuffle=False
        )
    else:
        # 单模态模型预测加载器
        pred_loader = DataLoader(
            TensorDataset(torch.tensor(X, dtype=torch.float32)),
            batch_size=batch_size,
            shuffle=False
        )

    return pred_loader