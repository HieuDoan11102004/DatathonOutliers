import os
import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

import torch
from pipeline.config import *
from pipeline.feature_engineering import build_features, load_sales
from pipeline.lru_model import train_lru, build_lru_feature
from pipeline.evaluate import print_metrics

def get_feature_cols(df):
    exclude = {'Date', 'Revenue', 'COGS', 'log_revenue', 'Residual', 'lru_pred'}
    cols = [c for c in df.columns if c not in exclude]
    return [c for c in cols if df[c].dtype in [np.float64, np.float32, np.int64, np.int32]] + ['lru_pred']

def main():
    set_seed(SEED)
    print("=" * 60)
    print("🌲 V6 ENSEMBLE TREE ABLATION STUDY 🌲")
    print("=" * 60)
    
    # 1. Prepare Data
    sales_df = load_sales()
    sub_raw = pd.read_csv(SUBMISSION_FILE, parse_dates=['Date'])
    sub_dummy = sub_raw.copy()
    sub_dummy['Revenue'] = np.nan
    full_df = pd.concat([sales_df, sub_dummy], ignore_index=True)
    full_df = build_features(full_df)

    internal_df = full_df[full_df['Revenue'].notna()].copy()
    internal_df['log_revenue'] = np.log1p(internal_df['Revenue'])
    top_features = get_feature_cols(internal_df)

    # 2. RUN LRU (Only <= 2021 for Strict Evaluation)
    print("\n>> Running LRU base extractor...")
    eval_mask = internal_df['Date'] <= VAL_END
    y_eval_train = internal_df[eval_mask]['log_revenue'].values
    
    model_lru_eval = train_lru(
        y_train=y_eval_train, seq_len=SEQ_LEN, pred_len=PRED_LEN, hidden_size=HIDDEN_UNITS,
        num_blocks=NUM_BLOCKS, attn_dropout=ATTN_DROPOUT, ff_dropout=FF_DROPOUT,
        batch_size=BATCH_SIZE, epochs=EPOCHS, lr=LEARNING_RATE, patience=PATIENCE
    )
    
    df_eval = full_df.copy()
    df_eval['lru_pred'] = build_lru_feature(
        model_lru_eval, y_eval_train, len(df_eval), SEQ_LEN, PRED_LEN
    )
    
    # 3. Train Trees
    print("\n>> Training Trees (LightGBM, XGBoost, CatBoost)...")
    internal_eval = df_eval[df_eval['Revenue'].notna()].copy()
    internal_eval['log_revenue'] = np.log1p(internal_eval['Revenue'])
    
    tr_df = internal_eval[internal_eval['Date'] <= TRAIN_END]
    v_df = internal_eval[(internal_eval['Date'] > TRAIN_END) & (internal_eval['Date'] <= VAL_END)]
    te_df = internal_eval[internal_eval['Date'] > VAL_END] # 2022
    
    X_tr, y_tr = tr_df[top_features].values, tr_df['log_revenue'].values
    X_v, y_v = v_df[top_features].values, v_df['log_revenue'].values
    X_te, y_te_true = te_df[top_features].values, te_df['Revenue'].values
    
    # LightGBM
    model_lgb = lgb.LGBMRegressor(n_estimators=N_ESTIMATORS, **LGBM_PARAMS)
    model_lgb.fit(X_tr, y_tr, eval_set=[(X_v, y_v)], callbacks=[lgb.early_stopping(EARLY_STOPPING, verbose=False)])
    
    # XGBoost
    model_xgb = xgb.XGBRegressor(n_estimators=N_ESTIMATORS, early_stopping_rounds=EARLY_STOPPING, **XGB_PARAMS)
    model_xgb.fit(X_tr, y_tr, eval_set=[(X_v, y_v)], verbose=False)
    
    # CatBoost
    model_cat = CatBoostRegressor(**CATBOOST_PARAMS)
    model_cat.fit(X_tr, y_tr, eval_set=[(X_v, y_v)])

    def get_eval(models):
        preds = [m.predict(X_te) for m in models]
        avg_log = np.mean(preds, axis=0)
        final_pred = np.maximum(np.expm1(avg_log), 0)
        rmse = np.sqrt(mean_squared_error(y_te_true, final_pred))
        return rmse

    print("\n" + "=" * 60)
    print("🏆 RESULTS ON HOLDOUT TEST SET (2022) 🏆")
    print("=" * 60)
    
    combinations = {
        "1_Tree_(LightGBM)": [model_lgb],
        "2_Trees_(LightGBM + XGBoost)": [model_lgb, model_xgb],
        "3_Trees_(LightGBM + XGB + Cat)": [model_lgb, model_xgb, model_cat]
    }
    
    results = []
    for name, models in combinations.items():
        rmse = get_eval(models)
        results.append((name, rmse))
        
    df_res = pd.DataFrame(results, columns=["Scenario", "Test_RMSE"])
    print(df_res.to_string(index=False))

if __name__ == '__main__':
    main()
