import os
import json
import math
import random
import warnings
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from scipy.stats import pearsonr
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.svm import SVR
from sklearn.linear_model import Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from config import CROSS_VALIDATION_FILE, PHENOTYPES_FILE, RESULTS_DIR, TRAINING_PARAMS, SNP_FILE
try:
    from config import PHENOTYPES as CONFIG_PHENOTYPES
except Exception:
    CONFIG_PHENOTYPES = None

warnings.filterwarnings("ignore")

DEFAULT_PHENOTYPES = ['Heading_date', 'Plant_height', 'Num_panicles', 'Num_effective_panicles', 'Yield',
                      'Grain_weight', 'Spikelet_length', 'Grain_length', 'Grain_width', 'Grain_thickness']
SKLEARN_MODEL_NAMES = ['SVR', 'Lasso', 'MLP']
ALL_MODEL_NAMES = SKLEARN_MODEL_NAMES + ['Transformer']


def set_random_seed(seed: int = 55) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_cv_folds(cv_file_path: str) -> List[dict]:
    with open(cv_file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def select_cv_folds(cv_folds: List[dict], max_folds: Optional[int], seed: int, mode: str) -> List[dict]:
    if max_folds is None or max_folds <= 0 or max_folds >= len(cv_folds):
        return cv_folds
    if mode == 'random':
        rng = random.Random(seed)
        selected_idx = sorted(rng.sample(range(len(cv_folds)), k=max_folds))
        return [cv_folds[i] for i in selected_idx]
    return cv_folds[:max_folds]


def normalize_model_selection(models_arg: Optional[List[str]], skip_transformer: bool, only_transformer: bool) -> List[str]:
    if only_transformer:
        return ['Transformer']

    selected: List[str]
    if not models_arg:
        selected = ALL_MODEL_NAMES.copy()
    else:
        normalized = []
        alias_map = {
            'svr': 'SVR',
            'lasso': 'Lasso',
            'mlp': 'MLP',
            'transformer': 'Transformer',
            'all': 'ALL',
            'sklearn': 'SKLEARN',
        }
        for item in models_arg:
            key = str(item).strip().lower()
            if key not in alias_map:
                raise ValueError(f"不支持的模型名称: {item}，可选: {ALL_MODEL_NAMES}，以及 all / sklearn")
            normalized.append(alias_map[key])

        selected = []
        if 'ALL' in normalized:
            selected.extend(ALL_MODEL_NAMES)
        if 'SKLEARN' in normalized:
            selected.extend(SKLEARN_MODEL_NAMES)
        for name in normalized:
            if name in ALL_MODEL_NAMES:
                selected.append(name)
        selected = list(dict.fromkeys(selected))

    if skip_transformer:
        selected = [m for m in selected if m != 'Transformer']

    if not selected:
        raise ValueError('最终没有可运行的模型，请检查 --models / --skip-transformer / --only-transformer 参数。')
    return selected


def safe_pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    if len(y_true) < 2:
        return np.nan
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return np.nan
    return float(pearsonr(y_true, y_pred)[0])


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    return {
        'pearson': safe_pearson(y_true, y_pred),
        'mse': float(mean_squared_error(y_true, y_pred)),
        'r2': float(r2_score(y_true, y_pred)),
    }


ID_CANDIDATES = ['ID', 'id', 'IID', 'Accession']
DROP_COLS = {'FID', 'PAT', 'MAT', 'SEX', 'PHENOTYPE'}


@dataclass
class CachedDataset:
    X: np.ndarray
    feature_names: List[str]
    sample_ids: List[str]
    phenotype_df: pd.DataFrame
    id_to_index: Dict[str, int]


def _find_id_col(columns: List[str], candidates: List[str]) -> Optional[str]:
    return next((c for c in candidates if c in columns), None)


def load_dataset_once() -> CachedDataset:
    phenotypes_df = pd.read_csv(PHENOTYPES_FILE)
    snp_df = pd.read_csv(SNP_FILE)

    ph_id_col = _find_id_col(phenotypes_df.columns.tolist(), ID_CANDIDATES)
    snp_id_col = _find_id_col(snp_df.columns.tolist(), ID_CANDIDATES)
    if ph_id_col is None:
        raise ValueError(f"PHENOTYPES_FILE 中未找到样本 ID 列，候选列: {ID_CANDIDATES}")
    if snp_id_col is None:
        raise ValueError(f"SNP_FILE 中未找到样本 ID 列，候选列: {ID_CANDIDATES}")

    phenotypes_df = phenotypes_df.copy()
    snp_df = snp_df.copy()
    phenotypes_df[ph_id_col] = phenotypes_df[ph_id_col].astype(str)
    snp_df[snp_id_col] = snp_df[snp_id_col].astype(str)

    drop_existing = [c for c in snp_df.columns if c in DROP_COLS]
    if drop_existing:
        snp_df = snp_df.drop(columns=drop_existing)

    common_ids = set(phenotypes_df[ph_id_col]) & set(snp_df[snp_id_col])
    if not common_ids:
        raise ValueError("PHENOTYPES_FILE 与 SNP_FILE 没有对齐到任何共同样本。")

    phenotypes_df = phenotypes_df[phenotypes_df[ph_id_col].isin(common_ids)].copy()
    phenotypes_df = phenotypes_df.sort_values(by=ph_id_col).reset_index(drop=True)
    sample_ids = phenotypes_df[ph_id_col].tolist()

    snp_df = snp_df[snp_df[snp_id_col].isin(common_ids)].copy()
    snp_df = snp_df.set_index(snp_id_col).loc[sample_ids].reset_index()

    feature_cols = [c for c in snp_df.columns if c != snp_id_col]
    numeric_df = snp_df[feature_cols].apply(pd.to_numeric, errors='coerce')
    numeric_df = numeric_df.replace([np.inf, -np.inf], np.nan)
    numeric_df = numeric_df.dropna(axis=1, how='all')

    feature_names = numeric_df.columns.tolist()
    X = numeric_df.values.astype(np.float32)
    phenotype_df = phenotypes_df.set_index(ph_id_col)
    phenotype_df.index = phenotype_df.index.astype(str)

    return CachedDataset(
        X=X,
        feature_names=feature_names,
        sample_ids=sample_ids,
        phenotype_df=phenotype_df,
        id_to_index={sid: i for i, sid in enumerate(sample_ids)},
    )


def get_phenotype_vector(dataset: CachedDataset, phenotype: str) -> Tuple[np.ndarray, List[str], np.ndarray]:
    if phenotype not in dataset.phenotype_df.columns:
        raise ValueError(f"表型文件中不存在列: {phenotype}")
    y_series = pd.to_numeric(dataset.phenotype_df[phenotype], errors='coerce')
    mask = y_series.notna().values
    if mask.sum() == 0:
        raise ValueError(f"表型 {phenotype} 全部为空，无法建模。")
    X = dataset.X[mask]
    y = y_series.values[mask].astype(np.float32)
    sample_ids = [sid for sid, keep in zip(dataset.sample_ids, mask) if keep]
    return X, sample_ids, y


@dataclass
class XFoldState:
    imputer: SimpleImputer
    selector: Optional[SelectKBest]
    x_scaler: StandardScaler
    selected_feature_names: List[str]


class XPreprocessor:
    def __init__(self, feature_k: int):
        self.feature_k = feature_k
        self.state: Optional[XFoldState] = None

    def fit_transform(self, X_train: np.ndarray, y_train: np.ndarray, feature_names: List[str]) -> np.ndarray:
        imputer = SimpleImputer(strategy='mean')
        X_train_imp = imputer.fit_transform(X_train)

        k = min(self.feature_k, X_train_imp.shape[1]) if self.feature_k and self.feature_k > 0 else 'all'
        if k == 'all' or X_train_imp.shape[1] == 0:
            selector = None
            X_train_sel = X_train_imp
            selected_feature_names = feature_names.copy()
        else:
            selector = SelectKBest(score_func=f_regression, k=k)
            X_train_sel = selector.fit_transform(X_train_imp, y_train)
            mask = selector.get_support()
            selected_feature_names = [f for f, keep in zip(feature_names, mask) if keep]

        x_scaler = StandardScaler()
        X_train_proc = x_scaler.fit_transform(X_train_sel).astype(np.float32)
        self.state = XFoldState(imputer=imputer, selector=selector, x_scaler=x_scaler, selected_feature_names=selected_feature_names)
        return X_train_proc

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.state is None:
            raise RuntimeError("XPreprocessor 尚未 fit。")
        X_proc = self.state.imputer.transform(X)
        if self.state.selector is not None:
            X_proc = self.state.selector.transform(X_proc)
        return self.state.x_scaler.transform(X_proc).astype(np.float32)


@dataclass
class YProcessor:
    scale_y: bool
    scaler: Optional[StandardScaler]

    @classmethod
    def fit(cls, y_train: np.ndarray, scale_y: bool) -> "YProcessor":
        if scale_y:
            scaler = StandardScaler()
            scaler.fit(y_train.reshape(-1, 1))
            return cls(scale_y=True, scaler=scaler)
        return cls(scale_y=False, scaler=None)

    def transform(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=np.float32).reshape(-1)
        if self.scaler is None:
            return y
        return self.scaler.transform(y.reshape(-1, 1)).reshape(-1).astype(np.float32)

    def inverse_transform(self, y_pred: np.ndarray) -> np.ndarray:
        y_pred = np.asarray(y_pred).reshape(-1, 1)
        if self.scaler is None:
            return y_pred.reshape(-1)
        return self.scaler.inverse_transform(y_pred).reshape(-1)


class SimpleTransformerRegressor(nn.Module):
    def __init__(self, input_dim: int, d_model: int = 32, nhead: int = 2, num_layers: int = 1, dim_feedforward: int = 64, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(1, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='gelu',
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, max(16, d_model // 2)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(max(16, d_model // 2), 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(-1)
        x = self.input_proj(x)
        x = self.encoder(x)
        x = self.norm(x)
        x = x.mean(dim=1)
        return self.head(x).squeeze(-1)


class TorchTransformerWrapper:
    def __init__(self, input_dim: int, device: str = 'cpu', epochs: int = 8, batch_size: int = 8, lr: float = 1e-3,
                 weight_decay: float = 1e-4, patience: int = 2, d_model: int = 32, nhead: int = 2,
                 num_layers: int = 1, dim_feedforward: int = 64, dropout: float = 0.1, seed: int = 55):
        self.input_dim = input_dim
        self.device = torch.device(device)
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.patience = patience
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.seed = seed
        self.model: Optional[SimpleTransformerRegressor] = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "TorchTransformerWrapper":
        set_random_seed(self.seed)
        X_train = np.asarray(X_train, dtype=np.float32)
        y_train = np.asarray(y_train, dtype=np.float32)
        dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
        use_pin = self.device.type == 'cuda'
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, pin_memory=use_pin)

        self.model = SimpleTransformerRegressor(
            input_dim=self.input_dim,
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
        ).to(self.device)

        criterion = nn.MSELoss()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        best_loss = math.inf
        best_state = None
        no_improve = 0

        self.model.train()
        for _ in range(self.epochs):
            running_loss = 0.0
            n_seen = 0
            for xb, yb in loader:
                xb = xb.to(self.device, non_blocking=use_pin)
                yb = yb.to(self.device, non_blocking=use_pin)
                optimizer.zero_grad(set_to_none=True)
                pred = self.model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * xb.size(0)
                n_seen += xb.size(0)
            epoch_loss = running_loss / max(n_seen, 1)
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.model.eval()
        return self

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Transformer 模型尚未训练。")
        X_test = np.asarray(X_test, dtype=np.float32)
        dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32))
        use_pin = self.device.type == 'cuda'
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, pin_memory=use_pin)
        preds = []
        with torch.no_grad():
            for (xb,) in loader:
                xb = xb.to(self.device, non_blocking=use_pin)
                preds.append(self.model(xb).detach().cpu().numpy())
        return np.concatenate(preds, axis=0).reshape(-1)


def build_sklearn_models(args) -> Dict[str, object]:
    return {
        'SVR': SVR(kernel='rbf', C=1.0, epsilon=0.1, gamma='scale'),
        'Lasso': Lasso(alpha=0.01, max_iter=5000, random_state=args.seed),
        'MLP': MLPRegressor(
            hidden_layer_sizes=(256, 128),
            activation='relu',
            solver='adam',
            alpha=1e-4,
            batch_size=min(args.batch_size, 64),
            learning_rate_init=1e-3,
            max_iter=max(args.epochs, 50),
            early_stopping=True,
            n_iter_no_change=args.patience,
            random_state=args.seed,
        ),
    }


def build_transformer(input_dim: int, device: str, args) -> TorchTransformerWrapper:
    return TorchTransformerWrapper(
        input_dim=input_dim,
        device=device,
        epochs=args.transformer_epochs,
        batch_size=args.transformer_batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.transformer_patience,
        d_model=args.transformer_d_model,
        nhead=args.transformer_nhead,
        num_layers=args.transformer_num_layers,
        dim_feedforward=args.transformer_ffn_dim,
        dropout=args.transformer_dropout,
        seed=args.seed,
    )


def evaluate_model(model_name: str, model_obj, X_train: np.ndarray, X_test: np.ndarray, y_train_proc: np.ndarray,
                   y_test_raw: np.ndarray, y_processor: YProcessor) -> Dict[str, float]:
    model = clone(model_obj) if model_name in SKLEARN_MODEL_NAMES else model_obj
    model.fit(X_train, y_train_proc)
    y_pred_model_scale = model.predict(X_test)
    y_pred_raw = y_processor.inverse_transform(y_pred_model_scale)
    return compute_metrics(y_test_raw, y_pred_raw)


def cleanup_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def evaluate_transformer_with_fallback(X_train_tr: np.ndarray, X_test_tr: np.ndarray, y_train_proc: np.ndarray,
                                       y_test_raw: np.ndarray, y_processor: YProcessor, args):
    tried = []
    run_plan = [(args.device, X_train_tr.shape[1])]
    if args.transformer_cpu_fallback and args.device != 'cpu':
        run_plan.append(('cpu', min(args.cpu_fallback_feature_k, X_train_tr.shape[1])))

    for device, feature_k in run_plan:
        tried.append(f"{device}:{feature_k}")
        try:
            Xtr = X_train_tr[:, :feature_k]
            Xte = X_test_tr[:, :feature_k]
            model = build_transformer(input_dim=feature_k, device=device, args=args)
            metrics = evaluate_model(
                model_name='Transformer',
                model_obj=model,
                X_train=Xtr,
                X_test=Xte,
                y_train_proc=y_train_proc,
                y_test_raw=y_test_raw,
                y_processor=y_processor,
            )
            del model
            cleanup_cuda()
            return metrics, device, feature_k, None
        except Exception:
            cleanup_cuda()
            continue
    return None, None, None, f"all transformer attempts failed: {tried}"


def run_experiment(args):
    ensure_dir(args.output_dir)
    cv_folds = load_cv_folds(CROSS_VALIDATION_FILE)
    cv_folds = select_cv_folds(cv_folds=cv_folds, max_folds=args.max_folds, seed=args.seed, mode=args.fold_sampling)
    dataset = load_dataset_once()
    print(f"[INFO] Loaded SNP matrix once: samples={dataset.X.shape[0]}, features={dataset.X.shape[1]}")
    print(f"[INFO] Using {len(cv_folds)} folds")
    print(f"[INFO] Selected models: {', '.join(args.selected_models)}")

    results_rows = []
    feature_rows = []
    phenotypes = args.phenotypes if args.phenotypes else (CONFIG_PHENOTYPES or DEFAULT_PHENOTYPES)
    sk_models_all = build_sklearn_models(args)
    sk_models = {k: v for k, v in sk_models_all.items() if k in args.selected_models}
    run_transformer = 'Transformer' in args.selected_models

    for phenotype in phenotypes:
        print(f"\n=== Processing phenotype: {phenotype} ===")
        X, sample_ids, y = get_phenotype_vector(dataset, phenotype)
        id_to_index = {sid: i for i, sid in enumerate(sample_ids)}

        for fold in cv_folds:
            fold_num = fold['fold']
            train_idx = [id_to_index[str(sid)] for sid in fold['train_samples'] if str(sid) in id_to_index]
            test_idx = [id_to_index[str(sid)] for sid in fold['test_samples'] if str(sid) in id_to_index]
            if len(train_idx) == 0 or len(test_idx) == 0:
                print(f"[WARN] phenotype={phenotype}, fold={fold_num}: 无有效训练/测试样本，跳过")
                continue

            X_train_raw, X_test_raw = X[train_idx], X[test_idx]
            y_train_raw, y_test_raw = y[train_idx], y[test_idx]

            print(f"  Fold {fold_num} | preprocessing once")
            x_preprocessor = XPreprocessor(feature_k=args.feature_k)
            X_train_proc = x_preprocessor.fit_transform(X_train_raw, y_train_raw, dataset.feature_names)
            X_test_proc = x_preprocessor.transform(X_test_raw)
            n_features_after = X_train_proc.shape[1]

            transformer_k = min(args.transformer_feature_k, n_features_after) if args.transformer_feature_k > 0 else n_features_after
            X_train_tr = X_train_proc[:, :transformer_k]
            X_test_tr = X_test_proc[:, :transformer_k]

            y_processors = {
                'y_raw': YProcessor.fit(y_train_raw, scale_y=False),
                'y_standardized': YProcessor.fit(y_train_raw, scale_y=True),
            }

            for scale_tag, y_processor in y_processors.items():
                y_train_proc = y_processor.transform(y_train_raw)

                for model_name, model_obj in sk_models.items():
                    print(f"    Fold {fold_num} | {model_name} | {scale_tag}")
                    try:
                        metrics = evaluate_model(model_name, model_obj, X_train_proc, X_test_proc, y_train_proc, y_test_raw, y_processor)
                    except Exception as e:
                        print(f"[ERROR] phenotype={phenotype}, fold={fold_num}, model={model_name}, y_processing={scale_tag}: {e}")
                        metrics = {'pearson': np.nan, 'mse': np.nan, 'r2': np.nan}
                    results_rows.append({
                        'phenotype': phenotype, 'fold': fold_num, 'model': model_name, 'y_processing': scale_tag,
                        **metrics, 'n_train': len(train_idx), 'n_test': len(test_idx),
                        'n_features_before': X.shape[1], 'n_features_after': n_features_after,
                        'run_device': 'cpu', 'notes': ''
                    })

                if run_transformer:
                    print(f"    Fold {fold_num} | Transformer | {scale_tag}")
                    metrics, used_device, used_k, err = evaluate_transformer_with_fallback(
                        X_train_tr, X_test_tr, y_train_proc, y_test_raw, y_processor, args
                    )
                    if metrics is None:
                        print(f"[ERROR] phenotype={phenotype}, fold={fold_num}, model=Transformer, y_processing={scale_tag}: {err}")
                        metrics = {'pearson': np.nan, 'mse': np.nan, 'r2': np.nan}
                        used_device = 'failed'
                        used_k = transformer_k
                    elif used_device != args.device or used_k != transformer_k:
                        print(f"[INFO] Transformer fallback used: device={used_device}, features={used_k}")
                    results_rows.append({
                        'phenotype': phenotype, 'fold': fold_num, 'model': 'Transformer', 'y_processing': scale_tag,
                        **metrics, 'n_train': len(train_idx), 'n_test': len(test_idx),
                        'n_features_before': X.shape[1], 'n_features_after': used_k,
                        'run_device': used_device, 'notes': err or ''
                    })
                    cleanup_cuda()

            feature_rows.append({
                'phenotype': phenotype,
                'fold': fold_num,
                'n_features_before': X.shape[1],
                'n_features_after': n_features_after,
                'transformer_n_features': transformer_k if run_transformer else np.nan,
            })

    per_fold_df = pd.DataFrame(results_rows)
    features_df = pd.DataFrame(feature_rows)
    summary_by_phenotype_model = (
        per_fold_df.groupby(['phenotype', 'model', 'y_processing'], dropna=False)
        .agg(pearson_mean=('pearson', 'mean'), pearson_std=('pearson', 'std'), mse_mean=('mse', 'mean'), mse_std=('mse', 'std'),
             r2_mean=('r2', 'mean'), r2_std=('r2', 'std'), folds=('fold', 'nunique'), n_train_mean=('n_train', 'mean'),
             n_test_mean=('n_test', 'mean'), n_features_before_mean=('n_features_before', 'mean'), n_features_after_mean=('n_features_after', 'mean'))
        .reset_index().sort_values(['phenotype', 'model', 'y_processing'])
    )
    summary_overall = (
        per_fold_df.groupby(['model', 'y_processing'], dropna=False)
        .agg(pearson_mean=('pearson', 'mean'), pearson_std=('pearson', 'std'), mse_mean=('mse', 'mean'), mse_std=('mse', 'std'),
             r2_mean=('r2', 'mean'), r2_std=('r2', 'std'), n_results=('r2', 'count'))
        .reset_index().sort_values(['model', 'y_processing'])
    )

    per_fold_path = os.path.join(args.output_dir, 'per_fold_results.csv')
    feature_path = os.path.join(args.output_dir, 'feature_summary.csv')
    summary_pm_path = os.path.join(args.output_dir, 'summary_by_phenotype_model.csv')
    summary_overall_path = os.path.join(args.output_dir, 'summary_overall.csv')
    per_fold_df.to_csv(per_fold_path, index=False)
    features_df.to_csv(feature_path, index=False)
    summary_by_phenotype_model.to_csv(summary_pm_path, index=False)
    summary_overall.to_csv(summary_overall_path, index=False)

    print("\nSaved files:")
    print(f"- {per_fold_path}")
    print(f"- {feature_path}")
    print(f"- {summary_pm_path}")
    print(f"- {summary_overall_path}")


def parse_args():
    parser = argparse.ArgumentParser(description='Fast comparison of phenotype standardization effect using SNP_FILE input.')
    parser.add_argument('--phenotypes', nargs='*', default=None)
    parser.add_argument('--feature-k', type=int, default=TRAINING_PARAMS.get('feature_selection_k', 2500))
    parser.add_argument('--epochs', type=int, default=TRAINING_PARAMS.get('num_epochs', 20))
    parser.add_argument('--batch-size', type=int, default=TRAINING_PARAMS.get('batch_size', 16))
    parser.add_argument('--lr', type=float, default=TRAINING_PARAMS.get('learning_rate', 1e-4))
    parser.add_argument('--weight-decay', type=float, default=TRAINING_PARAMS.get('weight_decay', 1e-5))
    parser.add_argument('--patience', type=int, default=TRAINING_PARAMS.get('patience', 5))
    parser.add_argument('--seed', type=int, default=TRAINING_PARAMS.get('random_seed', 55))
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cuda:1')
    parser.add_argument('--output-dir', type=str, default=os.path.join(RESULTS_DIR, 'compare_standardization'))

    parser.add_argument('--max-folds', type=int, default=None, help='限制实际参与运行的 fold 数；默认 None 表示使用全部 fold。')
    parser.add_argument('--fold-sampling', type=str, default='first', choices=['first', 'random'], help='fold 截取方式：first=取前 N 个，random=按 seed 随机抽取 N 个。')

    parser.add_argument('--models', nargs='+', default=None,
                        help='指定参与运行的模型，可选: SVR Lasso MLP Transformer；也支持 all 或 sklearn。')
    parser.add_argument('--skip-transformer', action='store_true')
    parser.add_argument('--only-transformer', action='store_true')

    parser.add_argument('--transformer-feature-k', type=int, default=128)
    parser.add_argument('--transformer-batch-size', type=int, default=8)
    parser.add_argument('--transformer-epochs', type=int, default=min(TRAINING_PARAMS.get('num_epochs', 20), 8))
    parser.add_argument('--transformer-patience', type=int, default=min(TRAINING_PARAMS.get('patience', 5), 2))
    parser.add_argument('--transformer-d-model', type=int, default=32)
    parser.add_argument('--transformer-nhead', type=int, default=2)
    parser.add_argument('--transformer-num-layers', type=int, default=1)
    parser.add_argument('--transformer-ffn-dim', type=int, default=64)
    parser.add_argument('--transformer-dropout', type=float, default=0.1)
    parser.add_argument('--transformer-cpu-fallback', action='store_true', default=True)
    parser.add_argument('--cpu-fallback-feature-k', type=int, default=64)

    args = parser.parse_args()
    args.selected_models = normalize_model_selection(
        models_arg=args.models,
        skip_transformer=args.skip_transformer,
        only_transformer=args.only_transformer,
    )
    return args


if __name__ == '__main__':
    args = parse_args()
    set_random_seed(args.seed)
    ensure_dir(args.output_dir)
    run_experiment(args)
