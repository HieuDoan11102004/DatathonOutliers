"""
Configuration for Revenue Forecasting Pipeline.
Seed, paths, hyperparameters.
"""
import os
import random
import numpy as np
import torch

# ============================================================
# Reproducibility
# ============================================================
SEED = 42

def set_seed(seed=SEED):
    """Set random seed everywhere for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ============================================================
# Paths
# ============================================================
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SALES_FILE = os.path.join(ROOT_DIR, 'sales.csv')
SUBMISSION_FILE = os.path.join(ROOT_DIR, 'sample_submission.csv')
ORDERS_FILE = os.path.join(ROOT_DIR, 'orders.csv')
WEB_TRAFFIC_FILE = os.path.join(ROOT_DIR, 'web_traffic.csv')
INVENTORY_FILE = os.path.join(ROOT_DIR, 'inventory.csv')
RETURNS_FILE = os.path.join(ROOT_DIR, 'returns.csv')
PROMOTIONS_FILE = os.path.join(ROOT_DIR, 'promotions.csv')
CUSTOMERS_FILE = os.path.join(ROOT_DIR, 'customers.csv')

OUTPUT_DIR = os.path.join(ROOT_DIR, 'models')
PLOTS_DIR = os.path.join(ROOT_DIR, 'plots')

# ============================================================
# Data Split Dates (Internal split from sales.csv)
# Ratio 7:2:1 → Train ~7.5yr, Val 2yr, Test 1yr
# ============================================================
TRAIN_END = '2019-12-31'
VAL_END = '2021-12-31'
# Test = everything after VAL_END (2022-01-01 → 2022-12-31)

# ============================================================
# LightGBM Hyperparameters (Residual Predictor)
# ============================================================
LGBM_PARAMS = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'num_leaves': 127,
    'learning_rate': 0.01,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'min_child_samples': 20,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'verbose': -1,
    'seed': SEED,
    'n_jobs': -1,
}
LGBM_N_ESTIMATORS = 3000
LGBM_EARLY_STOPPING = 200

# ============================================================
# PatchTST Hyperparameters (Multivariate)
# ============================================================
PATCHTST_CONTEXT_LEN = 360   # ~1 year lookback (captures yearly seasonality)
PATCHTST_PATCH_LEN = 15      # 15 days per patch
PATCHTST_STRIDE = 15          # non-overlapping → 24 patches
PATCHTST_D_MODEL = 128       # Increased capacity for multivariate
PATCHTST_N_HEADS = 4
PATCHTST_N_LAYERS = 2
PATCHTST_D_FF = 256           # Increased feed-forward
PATCHTST_DROPOUT = 0.3
PATCHTST_LR = 1e-4
PATCHTST_WEIGHT_DECAY = 1e-4
PATCHTST_EPOCHS = 100
PATCHTST_PATIENCE = 15
PATCHTST_BATCH_SIZE = 64

# Features fed into PatchTST (channel 0 = Revenue, always first)
PATCHTST_FEATURES = [
    'log_revenue',       # Channel 0: log1p(Revenue) — main series
    'month_sin',         # Monthly seasonality
    'month_cos',
    'dow_sin',           # Weekly seasonality
    'dow_cos',
    'doy_sin',           # Yearly seasonality
    'doy_cos',
    'is_weekend',        # Weekend flag
    'is_tet',            # Tet holiday
    'is_any_holiday',    # Any holiday
    'quarter',           # Quarter of the year
    'season',            # Season mapping (1=Spring..4=Winter)
]

# ============================================================
# Cross-Validation
# ============================================================
N_FOLDS = 5
TOP_K_FEATURES = 50
