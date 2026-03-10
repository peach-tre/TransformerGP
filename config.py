# config.py
import os

# 路径配置
DATA_DIR = '/data2/users/zhangyue/TransformerGP-main/data'
# CROSS_VALIDATION_FILE = os.path.join(DATA_DIR, 'rice4k_219/new_rice4k_cv_splits2.json')
CROSS_VALIDATION_FILE = os.path.join(DATA_DIR, 'rice4k_219/rice4k_fold8.json')
# EXP_FILE = os.path.join(DATA_DIR, 'rice4k_219/rice4k_exp.csv')
PHENOTYPES_FILE = os.path.join(DATA_DIR, 'rice4k_219/rice4k_ph.csv')
SNP_FILE = os.path.join(DATA_DIR, 'rice4k_219/rice18k_sw.csv')
RESULTS_DIR = os.path.join(DATA_DIR, 'results')

# 模型参数
MODEL_PARAMS = {
    'exp': {
        'hidden_size': 64,
        'num_heads': 2,
        'num_layers': 1,
        'dropout_rate': 0.3,
        'input_dim': 5000,
        'hidden_dim': 1024
    },
    'snp': {
        'hidden_size': 128,
        'num_heads': 2,
        'num_layers': 1,
        'input_dim': 10000,
        'hidden_dim': 2000
    },
    'combined': {
        'hidden_size': 64,
        'num_heads': 2,
        'num_layers': 1,
        'dropout_rate': 0.5
    }
}

# 训练参数
TRAINING_PARAMS = {
    'batch_size': 16,
    'learning_rate': 0.0001,
    'weight_decay': 1e-5,
    'num_epochs': 20,
    'patience': 5,
    'feature_selection_k': 2500,
    'random_seed': 55
}

# 随机种子
random_seed = 55

# # 交叉验证参数
# CV_PARAMS = {
#     'n_splits': 10,
#     'shuffle': True,
#     'random_seed': 42
# }

# 待分析的表型列表
# rice4k
PHENOTYPES = [
    'Heading_date', 'Plant_height', 'Num_panicles',
    'Num_effective_panicles', 'Yield', 'Grain_weight',
    'Spikelet_length', 'Grain_length', 'Grain_width',
    'Grain_thickness'
]

# rice18k
# PHENOTYPES = ['Heading_date', 'Plant_height', 'Culm_length',
#               'Panicle_length', 'Leaf_length', 'Leaf_width',
#               'Leaf_angle', 'Grain_yield', 'Grain_length',
#               'Grain_width', 'Grain_protein_content']

# PHENOTYPES = ['Grain_yield']