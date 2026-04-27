"""
Configuration for Revenue Forecasting Pipeline v3 (Tree Ensemble).
Seed, paths, hyperparameters for LGBM, XGBoost, and CatBoost.
"""
import os
import random
import numpy as np

# ============================================================
# Reproducibility
# ============================================================
SEED = 42

def set_seed(seed=SEED):
    """Set random seed everywhere for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)

# ============================================================
# Paths
# ============================================================
# Pointing back to Datathon root for raw data
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT_DIR = os.path.dirname(PROJECT_DIR)

SALES_FILE = os.path.join(ROOT_DIR, 'sales.csv')
SUBMISSION_FILE = os.path.join(ROOT_DIR, 'sample_submission.csv')
ORDERS_FILE = os.path.join(ROOT_DIR, 'orders.csv')
WEB_TRAFFIC_FILE = os.path.join(ROOT_DIR, 'web_traffic.csv')
INVENTORY_FILE = os.path.join(ROOT_DIR, 'inventory.csv')
RETURNS_FILE = os.path.join(ROOT_DIR, 'returns.csv')
PROMOTIONS_FILE = os.path.join(ROOT_DIR, 'promotions.csv')
CUSTOMERS_FILE = os.path.join(ROOT_DIR, 'customers.csv')

OUTPUT_DIR = os.path.join(PROJECT_DIR, 'models')
PLOTS_DIR = os.path.join(PROJECT_DIR, 'plots')

# ============================================================
# Data Split Dates (Internal split from sales.csv)
# ============================================================
TRAIN_END = '2019-12-31'
VAL_END = '2021-12-31'
# Test = 2022-01-01 → 2022-12-31

# ============================================================
# Hyperparameters
# Target is log1p(Revenue). Metric is RMSE.
# ============================================================
N_ESTIMATORS = 3000
EARLY_STOPPING = 200

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

XGB_PARAMS = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'max_depth': 8,
    'learning_rate': 0.01,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 20,
    'alpha': 0.1,
    'lambda': 0.1,
    'seed': SEED,
    'n_jobs': -1,
}

CATBOOST_PARAMS = {
    'loss_function': 'RMSE',
    'eval_metric': 'RMSE',
    'iterations': N_ESTIMATORS,
    'depth': 8,
    'learning_rate': 0.01,
    'l2_leaf_reg': 3,
    'rsm': 0.8,  # feature_fraction equivalent
    'random_seed': SEED,
    'verbose': False,
    'thread_count': -1,
    'early_stopping_rounds': EARLY_STOPPING,
}

# ============================================================
# Cross-Validation
# ============================================================
N_FOLDS = 5
