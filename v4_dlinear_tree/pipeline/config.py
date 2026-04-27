"""
Configuration for Revenue Forecasting Pipeline v6 (LSTF-Linear + Tree Ensemble).
Contains paths, Linear model config, and Tree hyperparameters.
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
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ============================================================
# Paths
# ============================================================
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
REVIEWS_FILE = os.path.join(ROOT_DIR, 'reviews.csv')
SHIPMENTS_FILE = os.path.join(ROOT_DIR, 'shipments.csv')
PAYMENTS_FILE = os.path.join(ROOT_DIR, 'payments.csv')
PRODUCTS_FILE = os.path.join(ROOT_DIR, 'products.csv')

OUTPUT_DIR = os.path.join(PROJECT_DIR, 'models')
PLOTS_DIR = os.path.join(PROJECT_DIR, 'plots')

# ============================================================
# Data Split Dates
# ============================================================
# sales.csv holds data up to 2022-12-31.
# test.csv covers 2023-01-01 to 2024-07-01 (548 days).
TRAIN_END = '2019-12-31'
VAL_END = '2021-12-31'

# ============================================================
# LSTF-Linear Parameters
# ============================================================
LINEAR_MODEL = 'NLinear'  # Options: 'NLinear', 'DLinear', 'Linear'
SEQ_LEN = 730             # Look back 2 years
PRED_LEN = 548            # Predict exactly the test horizon
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
EPOCHS = 100
PATIENCE = 10

# ============================================================
# Tree Hyperparameters
# ============================================================
N_ESTIMATORS = 3000
EARLY_STOPPING = 200
N_FOLDS = 5

LGBM_PARAMS = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'num_leaves': 15,
    'learning_rate': 0.01,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'min_child_samples': 30,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'verbose': -1,
    'seed': SEED,
    'n_jobs': -1,
}

XGB_PARAMS = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'max_depth': 4,
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
    'depth': 4,
    'learning_rate': 0.01,
    'l2_leaf_reg': 3,
    'rsm': 0.8,
    'random_seed': SEED,
    'verbose': False,
    'thread_count': -1,
    'early_stopping_rounds': EARLY_STOPPING,
}
