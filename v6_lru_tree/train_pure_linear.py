"""
V6 Pure Linear: DLinear + NLinear ensemble WITHOUT Tree.
Directly uses averaged linear predictions as final revenue.
"""
import os
import pandas as pd
import numpy as np
import torch

from pipeline.config import *
from pipeline.feature_engineering import build_features, load_sales
from pipeline.lstf_linear import train_linear, build_linear_feature
from pipeline.evaluate import print_metrics

def main():
    set_seed(SEED)
    print("=" * 60)
    print("V6 PURE LINEAR: DLinear + NLinear (No Tree)")
    print("=" * 60)

    sales_df = load_sales()
    sub_raw = pd.read_csv(SUBMISSION_FILE, parse_dates=['Date'])
    sub_dummy = sub_raw.copy()
    sub_dummy['Revenue'] = np.nan
    full_df = pd.concat([sales_df, sub_dummy], ignore_index=True)

    internal_df = full_df[full_df['Revenue'].notna()].copy()
    internal_df['log_revenue'] = np.log1p(internal_df['Revenue'])

    # ==========================================================
    # PHASE 1: STRICT EVAL (Holdout 2022)
    # ==========================================================
    print("\n" + "=" * 60)
    print("PHASE 1: STRICT EVALUATION ON HOLDOUT 2022")
    print("=" * 60)

    eval_mask = internal_df['Date'] <= VAL_END
    y_eval_train = internal_df[eval_mask]['log_revenue'].values

    print("  >> Train DLinear...")
    model_d = train_linear(y_train=y_eval_train, seq_len=SEQ_LEN, pred_len=PRED_LEN,
                           model_name='DLinear', batch_size=BATCH_SIZE, epochs=EPOCHS,
                           lr=LEARNING_RATE, patience=PATIENCE)
    
    print("  >> Train NLinear...")
    model_n = train_linear(y_train=y_eval_train, seq_len=SEQ_LEN, pred_len=PRED_LEN,
                           model_name='NLinear', batch_size=BATCH_SIZE, epochs=EPOCHS,
                           lr=LEARNING_RATE, patience=PATIENCE)

    feat_d = build_linear_feature(model_d, y_eval_train, len(full_df), SEQ_LEN, PRED_LEN)
    feat_n = build_linear_feature(model_n, y_eval_train, len(full_df), SEQ_LEN, PRED_LEN)
    
    # Ensemble: average in log space, then convert
    ensemble_log = (feat_d + feat_n) / 2
    ensemble_revenue = np.maximum(np.expm1(ensemble_log), 0)

    # Eval on 2022 holdout
    te_mask = internal_df['Date'] > VAL_END
    te_true = internal_df[te_mask]['Revenue'].values
    te_idx = internal_df[te_mask].index.values
    te_pred = ensemble_revenue[te_idx]

    print("\n  📊 PURE LINEAR METRICS ON 2022:")
    print_metrics("DLinear+NLinear (No Tree)", te_true, te_pred)

    # ==========================================================
    # PHASE 2: PRODUCTION SUBMISSION
    # ==========================================================
    print("\n" + "=" * 60)
    print("PHASE 2: PRODUCTION SUBMISSION")
    print("=" * 60)

    y_full = internal_df['log_revenue'].values

    print("  >> Train Production DLinear...")
    model_d_prod = train_linear(y_train=y_full, seq_len=SEQ_LEN, pred_len=PRED_LEN,
                                model_name='DLinear', batch_size=BATCH_SIZE, epochs=EPOCHS,
                                lr=LEARNING_RATE, patience=PATIENCE)
    
    print("  >> Train Production NLinear...")
    model_n_prod = train_linear(y_train=y_full, seq_len=SEQ_LEN, pred_len=PRED_LEN,
                                model_name='NLinear', batch_size=BATCH_SIZE, epochs=EPOCHS,
                                lr=LEARNING_RATE, patience=PATIENCE)

    feat_d_prod = build_linear_feature(model_d_prod, y_full, len(full_df), SEQ_LEN, PRED_LEN)
    feat_n_prod = build_linear_feature(model_n_prod, y_full, len(full_df), SEQ_LEN, PRED_LEN)

    ensemble_log_prod = (feat_d_prod + feat_n_prod) / 2
    ensemble_rev_prod = np.maximum(np.expm1(ensemble_log_prod), 0)

    # Extract submission rows
    sub_mask = full_df['Revenue'].isna()
    sub_idx = full_df[sub_mask].index.values
    sub_pred = ensemble_rev_prod[sub_idx]

    final_submission = sub_raw.copy()
    final_submission['Revenue'] = np.round(sub_pred, 2)

    path = os.path.join(PROJECT_DIR, 'submission_v6_pure_linear.csv')
    final_submission.to_csv(path, index=False)

    print(f"\n  ✅ Saved: {path}")
    print(f"  Revenue mean 2023-2024: {sub_pred.mean():,.0f}")

    print("\n" + "=" * 60)
    print("🚀 DONE!")
    print("=" * 60)

if __name__ == '__main__':
    main()
