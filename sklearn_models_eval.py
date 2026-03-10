import json
import os

import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression

# 导入模型
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor


def get_model_and_params(model_name):
    """返回模型及其对应的网格搜索参数"""
    models_and_params = {
        'catboost': {
            'model': CatBoostRegressor(verbose=False, random_state=123),
            'params': {
                'iterations': [500, 1000],
                'learning_rate': [0.01, 0.05, 0.1],
                'depth': [6, 8, 10],
                'l2_leaf_reg': [1, 3, 5, 7]
            }
        },
        'gb': {
            'model': GradientBoostingRegressor(random_state=123),
            'params': {
                'n_estimators': [100, 200, 500],
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': [3, 5, 7],
                'min_samples_split': [2, 5, 10]
            }
        },
        'br': {
            'model': BayesianRidge(),
            'params': {
                'alpha_1': [1e-6, 1e-5, 1e-4],
                'alpha_2': [1e-6, 1e-5, 1e-4],
                'lambda_1': [1e-6, 1e-5, 1e-4],
                'lambda_2': [1e-6, 1e-5, 1e-4]
            }
        },
        'svm': {
            'model': SVR(),
            'params': {
                'C': [0.1, 1, 10],
                'epsilon': [0.1, 0.2, 0.5],
                'kernel': ['linear', 'rbf'],
                'gamma': ['scale', 'auto', 0.1, 1]
            }
        },
        'rf': {
            'model': RandomForestRegressor(random_state=123),
            'params': {
                'n_estimators': [100, 200, 500],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'max_features': ['auto', 'sqrt']
            }
        },
        'lightgbm': {
            'model': LGBMRegressor(random_state=123),
            'params': {
                'n_estimators': [100, 200, 500],
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': [3, 5, 7],
                'num_leaves': [31, 50, 70],
                'min_child_samples': [20, 30, 50]
            }
        },
        'xgboost': {
            'model': XGBRegressor(random_state=123),
            'params': {
                'n_estimators': [100, 200, 500],
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': [3, 5, 7],
                'min_child_weight': [1, 3, 5],
                'subsample': [0.8, 0.9, 1.0]
            }
        },
        'knn': {
            'model': KNeighborsRegressor(),
            'params': {
                'n_neighbors': [3, 5, 7, 9],
                'weights': ['uniform', 'distance'],
                'p': [1, 2]
            }
        },
        'lasso': {
            'model': Lasso(random_state=123),
            'params': {
                'alpha': [0.0001, 0.001, 0.01, 0.1, 1.0],
                'max_iter': [1000, 2000, 3000]
            }
        },
        'en': {
            'model': ElasticNet(random_state=123),
            'params': {
                'alpha': [0.0001, 0.001, 0.01, 0.1, 1.0],
                'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],
                'max_iter': [1000, 2000, 3000]
            }
        },
        'lr': {
            'model': LinearRegression(),
            'params': {}
        },
        'ridge': {
            'model': Ridge(random_state=123),
            'params': {
                'alpha': [0.0001, 0.001, 0.01, 0.1, 1.0],
                'max_iter': [1000, 2000, 3000]
            }
        }
    }
    if model_name in models_and_params:
        return models_and_params[model_name]
    else:
        raise ValueError(f"Model '{model_name}' not found in model list.")

def get_or_create_cv_splits(cv_dir, valid_samples, first_phenotype, phenotypes_df, n_splits=5):
    """获取或创建交叉验证划分"""
    cv_splits_file = os.path.join(cv_dir, 'global_cv_splits.json')

    if os.path.exists(cv_splits_file):
        print("Loading existing CV splits...")
        with open(cv_splits_file, 'r') as f:
            fold_info = json.load(f)
        is_valid = True
        for fold in fold_info:
            if not all(sample in valid_samples for sample in fold['train_samples'] + fold['test_samples']):
                is_valid = False
                break
        if is_valid:
            print("Loaded CV splits are valid.")
            return fold_info
        else:
            print("Existing CV splits are invalid. Creating new splits...")

    print("Creating new CV splits...")
    y_values = phenotypes_df.loc[valid_samples, first_phenotype]
    y_binned = pd.qcut(y_values, q=5, labels=False)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)
    cv_splits = list(skf.split(y_values, y_binned))

    fold_info = []
    for fold_idx, (train_idx, test_idx) in enumerate(cv_splits):
        train_samples = y_values.index[train_idx].tolist()
        test_samples = y_values.index[test_idx].tolist()
        fold_info.append({
            'fold': fold_idx + 1,
            'train_samples': train_samples,
            'test_samples': test_samples
        })

    os.makedirs(cv_dir, exist_ok=True)
    with open(cv_splits_file, 'w') as f:
        json.dump(fold_info, f, indent=4)
    print("New CV splits created and saved.")
    return fold_info


def save_grid_search_results(grid_search, model_name, phenotype, fold_idx, result_dir):
    """保存网格搜索的结果"""
    results = {
        'best_params': grid_search.best_params_,
        'best_score': grid_search.best_score_,
        'cv_results': {
            'params': [str(params) for params in grid_search.cv_results_['params']],
            'mean_test_score': grid_search.cv_results_['mean_test_score'].tolist(),
            'std_test_score': grid_search.cv_results_['std_test_score'].tolist()
        }
    }

    save_dir = os.path.join(result_dir, 'grid_search_results', phenotype)
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, f'{model_name}_fold{fold_idx + 1}.json'), 'w') as f:
        json.dump(results, f, indent=4)


def train_and_compare_methods(train_file, phenotypes_file, phenotypes, output_dir, cv_dir):
    # 加载表型数据
    phenotypes_df = pd.read_csv(phenotypes_file).rename(columns={'Accession': 'id'}).set_index('id')

    # 加载SNP数据
    print(f"Loading SNP data from {train_file}...")
    snp_X = pd.read_csv(train_file, index_col='id', sep='\t').drop(columns=['Oringe'])
    snp_X = snp_X.replace({-1: np.nan})
    valid_snp_samples = snp_X.index

    # 获取所有表型共有的有效样本
    valid_samples = phenotypes_df.index[phenotypes_df[phenotypes].notna().all(axis=1)].intersection(valid_snp_samples)
    print(f"Number of valid samples: {len(valid_samples)}")

    # 获取或创建交叉验证划分
    fold_info = get_or_create_cv_splits(cv_dir, valid_samples, phenotypes[0], phenotypes_df)

    # 定义要比较的模型
    models_to_compare = [
        'catboost', 'br', 'rf', 'lightgbm', 'xgboost',
        'knn', 'lasso', 'en', 'lr', 'ridge'
    ]

    # 循环处理每一个表型
    for phenotype in phenotypes:
        print(f"\nProcessing phenotype: {phenotype}")

        # 合并SNP数据和表型数据
        merged_df = snp_X.loc[valid_samples].join(phenotypes_df[[phenotype]], how='inner')

        # 标准化表型数据
        scaler = StandardScaler()
        merged_df[phenotype] = scaler.fit_transform(merged_df[[phenotype]]).flatten()

        # 特征选择
        print(f"Performing feature selection for {phenotype}...")
        X = merged_df.drop(columns=[phenotype])
        y = merged_df[phenotype]

        selector = SelectKBest(score_func=f_regression, k=5000)
        X_selected = selector.fit_transform(X, y)

        selected_features = X.columns[selector.get_support()].tolist()
        X_selected = pd.DataFrame(X_selected, index=X.index, columns=selected_features)

        merged_df = X_selected.join(merged_df[[phenotype]])

        # 存储该表型所有fold的结果
        all_fold_results = []

        # 循环每个fold
        for fold_idx, fold in enumerate(fold_info):
            print(f"Processing fold {fold_idx + 1}")

            # 获取训练集和测试集数据
            X_train = merged_df.loc[fold['train_samples']].drop(columns=[phenotype])
            y_train = merged_df.loc[fold['train_samples'], phenotype]
            X_test = merged_df.loc[fold['test_samples']].drop(columns=[phenotype])
            y_test = merged_df.loc[fold['test_samples'], phenotype]

            fold_results = []

            # 对每个模型进行训练和评估
            for model_name in models_to_compare:
                print(f"Training {model_name} in fold {fold_idx + 1}")
                try:
                    model_info = get_model_and_params(model_name)
                    model = model_info['model']
                    params = model_info['params']

                    if params:
                        grid_search_file = os.path.join(
                            output_dir, 'grid_search_results', phenotype,
                            f'{model_name}_fold{fold_idx + 1}.json'
                        )

                        if os.path.exists(grid_search_file):
                            print(f"Loading existing grid search results for {model_name}")
                            with open(grid_search_file, 'r') as f:
                                grid_results = json.load(f)
                            best_params = grid_results['best_params']
                            model.set_params(**best_params)
                            model.fit(X_train, y_train)
                        else:
                            print(f"Performing grid search for {model_name}")
                            grid_search = GridSearchCV(
                                model,
                                params,
                                cv=2,
                                scoring='r2',
                                n_jobs=2,
                                verbose=1
                            )
                            grid_search.fit(X_train, y_train)
                            save_grid_search_results(grid_search, model_name, phenotype, fold_idx, result_dir)
                            model = grid_search.best_estimator_

                        predictions = model.predict(X_test)
                    else:
                        model.fit(X_train, y_train)
                        predictions = model.predict(X_test)

                    pearson_corr = np.corrcoef(y_test, predictions)[0, 1]
                    r2 = r2_score(y_test, predictions)

                    fold_results.append({
                        'Model': model_name,
                        'Fold': fold_idx + 1,
                        'Pearson': pearson_corr,
                        'R2': r2
                    })
                except Exception as e:
                    print(f"Error in {model_name} training: {str(e)}")
                    fold_results.append({
                        'Model': model_name,
                        'Fold': fold_idx + 1,
                        'Pearson': np.nan,
                        'R2': np.nan
                    })

            all_fold_results.extend(fold_results)

        # 保存所有fold的结果
        all_fold_df = pd.DataFrame(all_fold_results)
        os.makedirs(os.path.join(output_dir, 'cv_results'), exist_ok=True)
        all_fold_df.to_csv(
            os.path.join(output_dir, 'cv_results', f'merge_{phenotype}_all_folds.csv'),
            index=False
        )

        # 计算每个模型的平均指标
        avg_metrics = []
        for model_name in models_to_compare:
            model_results = all_fold_df[all_fold_df['Model'] == model_name]
            avg_metrics.append({
                'Model': model_name,
                'Pearson_Mean': model_results['Pearson'].mean(),
                'R²_Mean': model_results['R2'].mean(),
                'Pearson_Std': model_results['Pearson'].std(),
                'R²_Std': model_results['R2'].std()
            })

        # 保存平均结果
        results_df = pd.DataFrame(avg_metrics)
        results_df.to_csv(
            os.path.join(output_dir, f'merge_{phenotype}_metrics.csv'),
            index=False
        )


if __name__ == "__main__":
    # 参数设置
    snp_path = ''
    phenotype_file = ''
    out_path = ''
    cv_dir = os.path.join(out_path, 'cv_splits')

    phenotypes = [
        'grain_thickness', '1000-grains_weigh', 'length_weidth_ritio', 'grain_length', 'grain_weight',
        'brown_rice_ratio', 'polished_rice_ritio', 'complete_polished_rice_ritio', 'transparence',
        'LSvsT', 'chalkiness_degress', 'AC', 'gel_consistency', 'alkali_spreading'
    ]

    train_and_compare_methods(snp_path, phenotype_file, phenotypes, out_path, cv_dir)