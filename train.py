import torch
import copy
from scipy.stats import pearsonr
import numpy as np
import pandas as pd
from config import *
import random


def set_random_seed(seed=42):
    # 设置 Python 的 random 模块的随机种子
    random.seed(seed)

    # 设置 NumPy 的随机种子
    np.random.seed(seed)

    # 设置 PyTorch 的随机种子
    torch.manual_seed(seed)

    # 如果使用 CUDA（GPU），还需要设置以下内容
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # 为当前 GPU 设置种子
        torch.cuda.manual_seed_all(seed)  # 为所有 GPU 设置种子
        # 确保每次返回的卷积算法是确定的
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, combined=False):
    """
    训练模型并返回最佳模型、最佳R²和最佳epoch
    """
    best_r2 = float('-inf')
    best_model_wts = None
    epochs_no_improve = 0
    best_epoch = 0

    # 检查GPU可用性
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 存储每个epoch的R²值，用于后续恢复最佳模型
    epoch_r2_values = []

    for epoch in range(TRAINING_PARAMS['num_epochs']):
        # 训练阶段
        model.train()
        running_loss = 0.0

        for batch in train_loader:
            optimizer.zero_grad()

            # 根据模型类型处理输入
            if combined:
                exp_inputs, snp_inputs, targets = [data.to(device) for data in batch]
                outputs = model(exp_inputs, snp_inputs)
            else:
                inputs, targets = [data.to(device) for data in batch]
                outputs = model(inputs)

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        scheduler.step(train_loss)

        # 评估阶段
        model.eval()
        test_loss = 0.0
        all_test_outputs = []
        all_test_targets = []

        with torch.no_grad():
            for batch in test_loader:
                if combined:
                    exp_test_inputs, snp_test_inputs, test_targets = [data.to(device) for data in batch]
                    test_outputs = model(exp_test_inputs, snp_test_inputs)
                else:
                    test_inputs, test_targets = [data.to(device) for data in batch]
                    test_outputs = model(test_inputs)

                test_loss += criterion(test_outputs, test_targets).item()
                all_test_outputs.append(test_outputs.cpu())
                all_test_targets.append(test_targets.cpu())

        test_loss /= len(test_loader)

        # 合并所有测试结果
        all_test_outputs = torch.cat(all_test_outputs, dim=0)
        all_test_targets = torch.cat(all_test_targets, dim=0)

        # 计算评估指标
        pearson_corr, p_value, mse_loss, r2_score = calculate_metrics(all_test_outputs, all_test_targets)

        print(
            f'Epoch {epoch + 1}/{TRAINING_PARAMS["num_epochs"]}, '  
            f'Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, '  
            f'Test Pearson Correlation: {pearson_corr:.4f}, p值: {p_value}, '  
            f'MSE损失: {mse_loss:.4f}, R²: {r2_score:.4f}'
        )

        # 存储当前epoch的R²值
        epoch_r2_values.append(r2_score)

        # 更新最佳模型
        if r2_score > best_r2:
            best_r2 = r2_score
            best_model_wts = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # 早停条件
        if epochs_no_improve >= TRAINING_PARAMS['patience']:
            print(f'Early stopping at epoch {epoch + 1}')
            break

    # 加载最佳模型权重
    if best_model_wts is not None:
        model.load_state_dict(best_model_wts)
        print(f'Best R² achieved: {best_r2:.4f} at epoch {best_epoch + 1}')

    return model, best_r2, best_epoch


def train_final_model(model, train_loader, criterion, optimizer, scheduler, combined=False):
    """
    使用所有数据训练最终模型
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(TRAINING_PARAMS['num_epochs']):
        # 训练阶段
        model.train()
        running_loss = 0.0

        for batch in train_loader:
            optimizer.zero_grad()

            # 根据模型类型处理输入
            if combined:
                exp_inputs, snp_inputs, targets = [data.to(device) for data in batch]
                outputs = model(exp_inputs, snp_inputs)
            else:
                inputs, targets = [data.to(device) for data in batch]
                outputs = model(inputs)

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        scheduler.step(train_loss)

        print(f'Epoch {epoch + 1}/{TRAINING_PARAMS["num_epochs"]}, Train Loss: {train_loss:.4f}')

    return model


def evaluate_model(model, X_tensor, y_tensor, combined=False, snp_tensor=None):
    """
    评估模型性能
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    with torch.no_grad():
        if combined:
            predictions = model(X_tensor.to(device), snp_tensor.to(device))
        else:
            predictions = model(X_tensor.to(device))

    return predictions.cpu()


def predict_with_model(model, pred_loader, combined=False):
    """
    使用保存的模型进行预测
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    all_predictions = []

    with torch.no_grad():
        for batch in pred_loader:
            if combined:
                exp_inputs, snp_inputs = [data.to(device) for data in batch]
                outputs = model(exp_inputs, snp_inputs)
            else:
                inputs = batch[0].to(device)
                outputs = model(inputs)

            all_predictions.append(outputs.cpu())

            # 合并所有预测结果
    return torch.cat(all_predictions, dim=0)


def calculate_metrics(predictions, targets):
    """
    计算各种评估指标
    """
    preds = predictions.detach().squeeze().cpu().numpy()
    targets = targets.detach().squeeze().cpu().numpy()

    pearson_corr, p_value = pearsonr(preds, targets)
    mse_loss = np.mean((preds - targets) ** 2)
    r2_score = 1 - np.sum((targets - preds) ** 2) / np.sum((targets - np.mean(targets)) ** 2)

    return pearson_corr, p_value, mse_loss, r2_score


def save_model(model, phenotype, model_type, scalers=None):
    """
    保存模型和标准化器
    """
    # 确保模型目录存在
    models_dir = os.path.join(RESULTS_DIR, "models")
    os.makedirs(models_dir, exist_ok=True)

    # 保存模型
    model_path = os.path.join(models_dir, f"{model_type}_{phenotype}_model.pt")
    torch.save(model.state_dict(), model_path)

    # 如果有标准化器，也保存它们
    if scalers is not None:
        scalers_path = os.path.join(models_dir, f"{model_type}_{phenotype}_scalers.pt")
        torch.save(scalers, scalers_path)

    return model_path


def load_model(model, model_path):
    """
    加载保存的模型
    """
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def save_shap_values(phenotype, model_type, shap_values, data_type, feature_names=None):
    """
    保存SHAP值到CSV文件

    Args:
        phenotype (str): 表型名称
        model_type (str): 模型类型
        shap_values (list): SHAP值列表
        data_type (str): 数据类型 ('exp' 或 'snp')
        feature_names (list, optional): 原始特征名称列表
    """
    if shap_values is None or len(shap_values) == 0:
        print(f"没有{data_type}数据的SHAP值")
        return

        # 确保结果目录存在
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 生成输出文件路径
    output_file = os.path.join(
        RESULTS_DIR,
        f'{model_type}.{phenotype}_shap_values_{data_type}.csv'
    )

    try:
        # 收集所有fold的SHAP值
        all_shap_data = []

        # 如果没有提供特征名称，使用默认名称
        if feature_names is None:
            feature_names = [f'{data_type}_feature_{i}' for i in range(shap_values[0].shape[1])]

        for fold_idx, fold_shap in enumerate(shap_values):
            # 将每个fold的SHAP值转换为numpy数组
            fold_shap_array = np.array(fold_shap)

            # 确保特征名称数量与SHAP值列数匹配
            if len(feature_names) != fold_shap_array.shape[1]:
                print(f"警告：特征名称数量({len(feature_names)})与SHAP值列数({fold_shap_array.shape[1]})不匹配")
                # 截取或扩展特征名称列表
                feature_names = feature_names[:fold_shap_array.shape[1]] + \
                                [f'{data_type}_feature_{i}' for i in
                                 range(len(feature_names), fold_shap_array.shape[1])]

                # 为每个特征创建行
            for feature_idx, feature_name in enumerate(feature_names):
                # 获取该特征在所有样本中的SHAP值
                feature_shap_values = fold_shap_array[:, feature_idx]

                # 为每个样本创建一行数据
                for sample_idx, shap_value in enumerate(feature_shap_values):
                    all_shap_data.append({
                        'Fold': fold_idx + 1,
                        'Feature': feature_name,
                        'Sample': sample_idx + 1,
                        'SHAP_Value': shap_value
                    })

                    # 创建DataFrame并保存
        df_shap = pd.DataFrame(all_shap_data)
        # df_shap.to_csv(output_file, index=False)
        # print(f"{data_type.upper()}数据SHAP值已保存到 {output_file}")

        # 打印基本统计信息
        print("\nSHAP值基本统计:")
        feature_stats = df_shap.groupby('Feature')['SHAP_Value'].agg(['mean', 'std'])
        feature_stats['abs_mean'] = np.abs(feature_stats['mean'])
        print(feature_stats.sort_values('abs_mean', ascending=False))
        feature_stats.to_csv(output_file, index=True)
        print(f"\nSHAP值基本统计已保存为CSV文件: {output_file}")

    except Exception as e:
        print(f"保存{data_type} SHAP值时出错: {e}")


def save_results(phenotype, model_type, fold_results,
                 shap_values_exp=None, shap_values_snp=None,
                 exp_feature_names=None, snp_feature_names=None):
    """
    保存实验结果和SHAP值

    Args:
        phenotype (str): 表型名称
        model_type (str): 模型类型
        fold_results (list): 交叉验证结果
        shap_values_exp (list, optional): 表达数据SHAP值
        shap_values_snp (list, optional): SNP数据SHAP值
        exp_feature_names (list, optional): 表达数据特征名称
        snp_feature_names (list, optional): SNP数据特征名称
    """
    # 确保结果目录存在
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 保存评估指标结果
    avg_results = np.mean(fold_results, axis=0)
    results_file = os.path.join(RESULTS_DIR, f'{model_type}.results.csv')

    with open(results_file, 'a') as f:
        results_line = (
            f'{phenotype}\t'
            f'{avg_results[0]:.4f}\t'  # Pearson相关系数  
            f'{avg_results[1]:.4f}\t'  # p值  
            f'{avg_results[2]:.4f}\t'  # MSE损失  
            f'{avg_results[3]:.4f}\n'  # R²
        )
        f.write(results_line)

        # 保存SHAP值
    if shap_values_exp is not None:
        save_shap_values(phenotype, model_type, shap_values_exp, 'exp', exp_feature_names)

    if shap_values_snp is not None:
        save_shap_values(phenotype, model_type, shap_values_snp, 'snp', snp_feature_names)