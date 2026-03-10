import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import shap
import logging
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from model import ExpTraitPredictionModel, SnpTraitPredictionModel, CombinedTraitPredictionModel
from data_loader import (
    load_cv_folds, load_exp_data, load_snp_data, load_combined_data,
    create_data_loaders, create_prediction_loader
)
from train import (
    train_model, evaluate_model, calculate_metrics, save_results,
    train_final_model, save_model, load_model, predict_with_model, set_random_seed
)
from config import *


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(RESULTS_DIR, "training.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def init_model(model_type, input_dims, output_dim=1):
    """
    初始化模型
    """
    if model_type == "exp":
        return ExpTraitPredictionModel(
            input_dim=input_dims[0],
            hidden_dim=MODEL_PARAMS['exp']['hidden_dim'],
            output_dim=output_dim,
            hidden_size=MODEL_PARAMS['exp']['hidden_size'],
            num_heads=MODEL_PARAMS['exp']['num_heads'],
            num_layers=MODEL_PARAMS['exp']['num_layers'],
            dropout_rate=MODEL_PARAMS['exp']['dropout_rate']
        )
    elif model_type == "snp":
        return SnpTraitPredictionModel(
            input_dim=input_dims[1],
            hidden_dim=MODEL_PARAMS['snp']['hidden_dim'],
            output_dim=output_dim,
            hidden_size=MODEL_PARAMS['snp']['hidden_size'],
            num_heads=MODEL_PARAMS['snp']['num_heads'],
            num_layers=MODEL_PARAMS['snp']['num_layers']
        )
    elif model_type == "combined":
        return CombinedTraitPredictionModel(
            exp_input_dim=input_dims[0],
            snp_input_dim=input_dims[1],
            hidden_dim=MODEL_PARAMS['exp']['hidden_dim'],
            output_dim=output_dim,
            # hidden_size=MODEL_PARAMS['combined']['hidden_size'],
            num_heads=MODEL_PARAMS['combined']['num_heads'],
            num_layers=MODEL_PARAMS['combined']['num_layers'],
            dropout_rate=MODEL_PARAMS['combined']['dropout_rate']
        )
    else:
        raise ValueError(f"未知模型类型: {model_type}")


def run_cv_from_file(phenotype, model_type):
    """
    使用预定义的交叉验证文件进行训练
    """
    logger.info(f"开始处理表型: {phenotype}, 模型类型: {model_type}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载交叉验证划分
    cv_folds = load_cv_folds(CROSS_VALIDATION_FILE)

    # 根据模型类型加载数据
    if model_type == "exp":
        X, y, sample_ids, exp_feature_names = load_exp_data(phenotype)
        input_dims = [X.shape[1], None]
        is_combined = False
    elif model_type == "snp":
        X, y, sample_ids, snp_feature_names = load_snp_data(phenotype)
        input_dims = [None, X.shape[1]]
        is_combined = False
    elif model_type == "combined":
        exp_X, snp_X, y, sample_ids, exp_feature_names, snp_feature_names = load_combined_data(phenotype)
        input_dims = [exp_X.shape[1], snp_X.shape[1]]
        is_combined = True
    else:
        raise ValueError(f"未知模型类型: {model_type}")

        # 创建样本ID到索引的映射
    id_to_index = {id: i for i, id in enumerate(sample_ids)}

    # 初始化结果存储
    fold_results = []
    fold_shap_values_exp = []
    fold_shap_values_snp = []

    for fold_data in cv_folds:
        fold_num = fold_data["fold"]
        # if fold_num == 1:
        #     continue
        logger.info(f"开始处理折 {fold_num}")

        # 获取训练和测试样本索引
        train_indices = [id_to_index[id] for id in fold_data["train_samples"] if id in id_to_index]
        test_indices = [id_to_index[id] for id in fold_data["test_samples"] if id in id_to_index]

        if not train_indices or not test_indices:
            logger.warning(f"折 {fold_num} 缺少训练或测试样本，跳过")
            continue

        # 准备数据
        if is_combined:
            # 提取训练和测试数据
            exp_X_train, exp_X_test = exp_X[train_indices], exp_X[test_indices]
            snp_X_train, snp_X_test = snp_X[train_indices], snp_X[test_indices]
            y_train, y_test = y[train_indices], y[test_indices]

            # 创建数据加载器
            train_loader, test_loader = create_data_loaders(
                exp_X_train, y_train, exp_X_test, y_test,
                snp_X_train=snp_X_train, snp_X_test=snp_X_test
            )
        else:
            # 提取训练和测试数据
            X_train, X_test = X[train_indices], X[test_indices]
            y_train, y_test = y[train_indices], y[test_indices]

            # 创建数据加载器
            train_loader, test_loader = create_data_loaders(X_train, y_train, X_test, y_test)

        # 初始化模型
        model = init_model(model_type, input_dims)
        model = model.to(device)

        # 初始化优化器和调度器
        optimizer = optim.Adam(
            model.parameters(),
            lr=TRAINING_PARAMS['learning_rate'],
            weight_decay=TRAINING_PARAMS['weight_decay']
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, verbose=True)
        criterion = nn.MSELoss()

        # 训练模型并获取最佳模型
        best_model, best_r2, best_epoch = train_model(
            model, train_loader, test_loader, criterion, optimizer, scheduler,
            combined=is_combined
        )

        # 重新评估最佳模型
        best_model.eval()
        with torch.no_grad():
            if is_combined:
                exp_X_test_tensor = torch.tensor(exp_X_test, dtype=torch.float32).to(device)
                snp_X_test_tensor = torch.tensor(snp_X_test, dtype=torch.float32).to(device)
                y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).to(device)

                predictions = best_model(exp_X_test_tensor, snp_X_test_tensor)
            else:
                X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
                y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).to(device)

                predictions = best_model(X_test_tensor)

        # 计算评估指标
        pearson_corr, p_value, mse_loss, r2_score = calculate_metrics(predictions, y_test_tensor)

        logger.info(
            f'折 {fold_num} 测试结果 (最佳epoch {best_epoch + 1}): '
            f'Pearson相关系数: {pearson_corr:.4f}, p值: {p_value}, '
            f'MSE损失: {mse_loss:.4f}, R²: {r2_score:.4f} (最佳R²: {best_r2:.4f})'
        )

        fold_results.append((pearson_corr, p_value, mse_loss, r2_score))

        # # 计算SHAP值
        # try:
        #     # 将模型移到CPU进行SHAP解释
        #     # best_model_cpu = best_model.to('cpu')
        #     # best_model_cpu = best_model
        #
        #     # 准备数据计算SHAP值
        #     if is_combined:
        #         background = [
        #             torch.tensor(exp_X_train[:100], dtype=torch.float32).to(device),
        #             torch.tensor(snp_X_train[:100], dtype=torch.float32).to(device)
        #         ]
        #         test_data = [
        #             torch.tensor(exp_X_test, dtype=torch.float32).to(device),
        #             torch.tensor(snp_X_test, dtype=torch.float32).to(device)
        #         ]
        #     else:
        #         background = torch.tensor(X_train[:100], dtype=torch.float32).to(device)
        #         test_data = torch.tensor(X_test, dtype=torch.float32).to(device)
        #
        #     # 使用DeepExplainer计算SHAP值
        #     explainer = shap.DeepExplainer(best_model, background)
        #     shap_values = explainer.shap_values(test_data, check_additivity=False)
        #
        #     # 保存SHAP值
        #     if is_combined:
        #         fold_shap_values_exp.append(shap_values[0])
        #         fold_shap_values_snp.append(shap_values[1])
        #     elif model_type == 'exp':
        #         fold_shap_values_exp.append(shap_values)
        #     else:
        #         fold_shap_values_snp.append(shap_values)
        #
        # except Exception as shap_error:
        #     logger.warning(f"折 {fold_num} SHAP解释错误: {shap_error}")

    # 保存汇总结果
    save_results(
        phenotype, model_type, fold_results,
        shap_values_exp=fold_shap_values_exp if is_combined or model_type == 'exp' else None,
        shap_values_snp=fold_shap_values_snp if is_combined or model_type == 'snp' else None,
        exp_feature_names=exp_feature_names if is_combined or model_type == 'exp' else None,
        snp_feature_names=snp_feature_names if is_combined or model_type == 'snp' else None
    )

    # 计算并输出平均结果
    avg_results = np.mean(fold_results, axis=0)
    logger.info(
        f'表型 {phenotype} 平均测试结果: Pearson相关系数: {avg_results[0]:.4f}, '
        f'p值: {avg_results[1]:.4f}, MSE损失: {avg_results[2]:.4f}, R²: {avg_results[3]:.4f}'
    )

    return avg_results


def predict_from_saved_model(phenotype, model_type, new_data_path, output_path=None):
    """
    加载保存的模型并进行预测
    """
    logger.info(f"使用保存的模型预测 {phenotype} 的 {model_type} 值")

    # 加载数据
    if model_type == "exp":
        # 假设新数据的格式与训练数据相同，但没有表型列
        new_data = pd.read_csv(new_data_path, sep='\t')
        sample_ids = new_data['id'].tolist()
        X = new_data.drop('id', axis=1).values

        # 加载模型和标准化器
        models_dir = os.path.join(RESULTS_DIR, "models")
        model_path = os.path.join(models_dir, f"{model_type}_{phenotype}_model.pt")
        scalers_path = os.path.join(models_dir, f"{model_type}_{phenotype}_scalers.pt")

        input_dims = [X.shape[1], None]
        model = init_model(model_type, input_dims)
        model = load_model(model, model_path)

        scalers = torch.load(scalers_path)
        X_scaled = scalers['x_scaler'].transform(X).astype(np.float32)

        # 创建预测数据加载器
        pred_loader = create_prediction_loader(X_scaled)
        is_combined = False

    elif model_type == "snp":
        # 假设新数据格式与训练数据相同
        new_data = pd.read_csv(new_data_path)
        sample_ids = new_data['IID'].tolist()
        X = new_data.drop(['IID'], axis=1).values

        # 加载模型和标准化器
        models_dir = os.path.join(RESULTS_DIR, "models")
        model_path = os.path.join(models_dir, f"{model_type}_{phenotype}_model.pt")
        scalers_path = os.path.join(models_dir, f"{model_type}_{phenotype}_scalers.pt")

        input_dims = [None, X.shape[1]]
        model = init_model(model_type, input_dims)
        model = load_model(model, model_path)

        scalers = torch.load(scalers_path)
        X_scaled = scalers['x_scaler'].transform(X).astype(np.float32)

        # 创建预测数据加载器
        pred_loader = create_prediction_loader(X_scaled)
        is_combined = False

    elif model_type == "combined":
        # 需要两个数据文件
        exp_data_path = new_data_path  # 假设传入的是exp数据路径
        snp_data_path = new_data_path.replace("exp", "snp")  # 假设有一个对应的snp数据文件

        exp_data = pd.read_csv(exp_data_path, sep='\t')
        snp_data = pd.read_csv(snp_data_path)

        # 找出共有的样本ID
        exp_ids = set(exp_data['ID'].tolist())
        snp_ids = set(snp_data['ID'].tolist())
        common_ids = list(exp_ids & snp_ids)

        if not common_ids:
            raise ValueError("没有共同的样本ID")

            # 过滤数据
        exp_data_filtered = exp_data[exp_data['ID'].isin(common_ids)]
        snp_data_filtered = snp_data[snp_data['ID'].isin(common_ids)]

        # 确保顺序一致
        sample_ids = common_ids
        exp_data_filtered = exp_data_filtered.set_index('ID').loc[sample_ids].reset_index()
        snp_data_filtered = snp_data_filtered.set_index('ID').loc[sample_ids].reset_index()

        exp_X = exp_data_filtered.drop('ID', axis=1).values
        snp_X = snp_data_filtered.drop(['ID'], axis=1).values

        # 加载模型和标准化器
        models_dir = os.path.join(RESULTS_DIR, "models")
        model_path = os.path.join(models_dir, f"{model_type}_{phenotype}_model.pt")
        scalers_path = os.path.join(models_dir, f"{model_type}_{phenotype}_scalers.pt")

        input_dims = [exp_X.shape[1], snp_X.shape[1]]
        model = init_model(model_type, input_dims)
        model = load_model(model, model_path)

        scalers = torch.load(scalers_path)
        exp_X_scaled = scalers['exp_scaler'].transform(exp_X).astype(np.float32)
        snp_X_scaled = scalers['snp_scaler'].transform(snp_X).astype(np.float32)

        # 创建预测数据加载器
        pred_loader = create_prediction_loader(exp_X_scaled, snp_X=snp_X_scaled)
        is_combined = True
    else:
        raise ValueError(f"未知模型类型: {model_type}")

    # 进行预测
    predictions = predict_with_model(model, pred_loader, combined=is_combined)

    # 反转标准化
    if model_type == "combined":
        y_pred = scalers['y_scaler'].inverse_transform(predictions.numpy())
    else:
        y_pred = scalers['y_scaler'].inverse_transform(predictions.numpy())

        # 创建结果数据框
    results_df = pd.DataFrame({
        'ID': sample_ids,
        f'Predicted_{phenotype}': y_pred.flatten()
    })

    # 保存预测结果
    if output_path is None:
        output_path = os.path.join(RESULTS_DIR, f"{model_type}_{phenotype}_predictions.csv")

    results_df.to_csv(output_path, index=False)
    logger.info(f"预测结果已保存到: {output_path}")

    return results_df


def run_experiment(model_type, seed=None):
    """
    运行一个完整的实验，包括交叉验证和训练最终模型
    """
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        logger.info(f"设置随机种子: {seed}")

    # 记录实验开始
    with open(os.path.join(RESULTS_DIR, f'{model_type}.results.csv'), 'a') as f:
        f.write(f'\n所有样本, 随机种子: {seed if seed else TRAINING_PARAMS["random_seed"]}\n')
        f.write('表型\tPearson相关系数\tp值\tMSE损失\tR²\n')

    # 对每个表型运行交叉验证实验
    cv_results = {}
    for phenotype in PHENOTYPES:
        cv_results[phenotype] = run_cv_from_file(phenotype, model_type)

    return cv_results


def main():
    """
    主程序入口，直接执行所有表型的训练
    """
    # 设置随机种子
    np.random.seed(TRAINING_PARAMS['random_seed'])
    torch.manual_seed(TRAINING_PARAMS['random_seed'])

    # 定义要训练的模型类型
    model_types = ['snp', 'exp']

    # 遍历所有模型类型
    for model_type in model_types:
        logger.info(f"开始训练 {model_type} 模型")

        # 创建结果文件
        with open(os.path.join(RESULTS_DIR, f'{model_type}.results.csv'), 'w') as f:
            f.write(f'随机种子: {TRAINING_PARAMS["random_seed"]}\n')
            f.write('表型\tPearson相关系数\tp值\tMSE损失\tR²\n')

            # 对每个表型进行交叉验证和最终模型训练
        for phenotype in PHENOTYPES:
            logger.info(f"处理表型: {phenotype}")

            # try:
            # 运行交叉验证
            cv_results = run_cv_from_file(phenotype, model_type)


if __name__ == '__main__':
    set_random_seed(random_seed)
    # import argparse
    #
    # parser = argparse.ArgumentParser(description='运行基因表达或SNP预测模型')
    # parser.add_argument('--model', type=str, choices=['exp', 'snp', 'combined'], default='exp',
    #                     help='要运行的模型类型 (exp, snp, combined)')
    # parser.add_argument('--seed', type=int, default=None, help='随机种子')
    # parser.add_argument('--phenotype', type=str, default=None, help='单独处理一个表型')
    # parser.add_argument('--mode', type=str, choices=['train', 'predict'], default='train',
    #                     help='训练模式或预测模式')
    # parser.add_argument('--data_path', type=str, default=None, help='预测模式下的输入数据路径')
    # parser.add_argument('--output_path', type=str, default=None, help='预测模式下的输出结果路径')
    #
    # args = parser.parse_args()
    #
    # if args.mode == 'train':
    #     if args.phenotype:
    #         if args.phenotype not in PHENOTYPES:
    #             logger.error(f"未知的表型: {args.phenotype}")
    #         else:
    #             # 运行交叉验证
    #             run_cv_from_file(args.phenotype, args.model)
    #     else:
    #         # 运行完整实验（所有表型）
    #         run_experiment(args.model, args.seed)
    # elif args.mode == 'predict':
    #     if not args.phenotype:
    #         logger.error("预测模式必须指定表型")
    #     elif not args.data_path:
    #         logger.error("预测模式必须提供数据路径")
    #     else:
    #         predict_from_saved_model(
    #             args.phenotype, args.model,
    #             args.data_path, args.output_path
    #         )
    main()