"""
V7 HYBRID PIPELINE
==================
DLinear-MV + NLinear-MV (Revenue × COGS, 2-channel) + COGS-Linear + Tree Ensemble

Thay đổi so với V6:
  - LSTF nhận 2 channel: log_revenue + log_cogs  → MultiVarDLinear / MultiVarNLinear
  - Thêm 'cogs_linear_pred' làm feature riêng cho Tree (COGS momentum)
  - get_feature_cols nhận diện đúng tất cả nhóm feature mới (trend, promo, interaction…)
  - log_cogs thay cho COGS raw để align scale với log_revenue
  - Impute COGS cho future dates bằng median seasonal (không cần thêm model)
"""

import os
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
import torch

from pipeline.config import *
from pipeline.feature_engineering import build_features, load_sales
from pipeline.evaluate import print_metrics, plot_predictions, plot_feature_importance
from pipeline.lstf_linear import (
    train_linear,
    build_linear_feature,
    build_cogs_feature,
)


# ─────────────────────────────────────────────
#  FEATURE SELECTION
# ─────────────────────────────────────────────

# Columns không phải feature
_EXCLUDE = {
    'Date', 'Revenue', 'COGS',
    'log_revenue', 'log_cogs',
    'Residual', 'linear_pred', 'cogs_linear_pred',
}

# Feature groups dựa theo prefix/pattern (để debug breakdown)
_FEATURE_GROUPS = {
    'time':        lambda c: any(x in c for x in [
                       'year','month','day','week','quarter','season',
                       'sin','cos','is_weekend','month_start','month_end']),
    'fourier':     lambda c: c.startswith('f_'),
    'events':      lambda c: any(x in c for x in [
                       'tet','hung','liberation','labor','national_day',
                       'new_year','xmas','double','payday','quarter_end',
                       'year_end','year_start','week_of_month','holiday',
                       'proximity', 'pre_']),
    'trend':       lambda c: any(x in c for x in [
                       'trend','covid','days_from','log_days','sqrt','pre_covid',
                       'post_covid','is_post','is_pre','is_covid']),
    'lags':        lambda c: any(x in c for x in [
                       'rev_lag','cogs_lag','rev_rmean','rev_rstd',
                       'cogs_rmean','yoy','margin','cogs_ratio','seasonality_idx']),
    'promo':       lambda c: c.startswith('promo_'),
    'auxiliary':   lambda c: any(c.startswith(p) for p in [
                       'orders_','web_','returns_','inv_','reviews_','ship_','pay_']),
    'interaction': lambda c: '_x_' in c,
    'lstf':        lambda c: c in ('linear_pred', 'cogs_linear_pred'),
}


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    """
    Trả về danh sách feature columns đúng kiểu dữ liệu, loại trừ target + meta cols.
    Luôn đặt 'linear_pred' và 'cogs_linear_pred' ở cuối (nếu có) để dễ ablation.
    """
    valid_dtypes = {np.float64, np.float32, np.int64, np.int32, np.bool_}

    base_cols = [
        c for c in df.columns
        if c not in _EXCLUDE
        and type(df[c].dtype) not in [object]
        and df[c].dtype.type in valid_dtypes
    ]

    # Tách LSTF features ra để đưa xuống cuối
    lstf_cols = [c for c in ['linear_pred', 'cogs_linear_pred'] if c in df.columns]
    other_cols = [c for c in base_cols if c not in lstf_cols]

    all_cols = other_cols + lstf_cols

    # Debug breakdown
    if len(all_cols) > 0:
        breakdown = {}
        for grp, fn in _FEATURE_GROUPS.items():
            cnt = sum(1 for c in all_cols if fn(c))
            if cnt > 0:
                breakdown[grp] = cnt
        print(f"  Feature breakdown: {breakdown}  →  total={len(all_cols)}")

    return all_cols


# ─────────────────────────────────────────────
#  TREE ENSEMBLE
# ─────────────────────────────────────────────

def fit_tree_ensemble(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_vl: np.ndarray, y_vl: np.ndarray,
) -> list:
    """Train LightGBM + XGBoost + CatBoost. Trả về list models."""
    print("    LightGBM...")
    lgb_model = lgb.LGBMRegressor(n_estimators=N_ESTIMATORS, **LGBM_PARAMS)
    lgb_model.fit(
        X_tr, y_tr,
        eval_set=[(X_vl, y_vl)],
        callbacks=[lgb.early_stopping(EARLY_STOPPING, verbose=False),
                   lgb.log_evaluation(period=200)],
    )

    print("    XGBoost...")
    xgb_model = xgb.XGBRegressor(
        n_estimators=N_ESTIMATORS,
        early_stopping_rounds=EARLY_STOPPING,
        **XGB_PARAMS,
    )
    xgb_model.fit(X_tr, y_tr, eval_set=[(X_vl, y_vl)], verbose=False)

    print("    CatBoost...")
    cat_model = CatBoostRegressor(**CATBOOST_PARAMS)
    cat_model.fit(X_tr, y_tr, eval_set=[(X_vl, y_vl)], verbose=200)

    return [lgb_model, xgb_model, cat_model]


def predict_ensemble(models: list, X: np.ndarray) -> np.ndarray:
    """
    Simple average trong log-space, convert về original scale.
    Clip tại 0 để tránh âm.
    """
    log_preds = np.stack([m.predict(X) for m in models], axis=0)
    return np.maximum(np.expm1(log_preds.mean(axis=0)), 0.0)


# ─────────────────────────────────────────────
#  LSTF FEATURE BUILDER (Revenue + COGS)
# ─────────────────────────────────────────────

def _build_lstf_features(
    df_full: pd.DataFrame,
    train_mask: pd.Series,
    model_dlinear_name: str = 'MultiVarDLinear',
    model_nlinear_name: str = 'MultiVarNLinear',
    tag: str = '',
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Train DLinear + NLinear (multivariate, Revenue+COGS) và
    COGS-Linear (univariate), trả về 3 feature arrays.

    Returns
    -------
    linear_pred      : np.ndarray shape [full_len]  (avg DLinear + NLinear)
    cogs_linear_pred : np.ndarray shape [full_len]
    """
    full_len = len(df_full)

    # ── Prepare sequences ──────────────────────────────────────
    train_rows = df_full[train_mask].copy()

    y_rev_train  = train_rows['log_revenue'].values
    y_cogs_train = train_rows['log_cogs'].values

    # Full COGS array: for future dates, no known value.
    # We fill with seasonal median (month × dow) computed from training data.
    full_cogs = np.zeros(full_len)
    full_cogs[:train_mask.sum()] = y_cogs_train

    if train_mask.sum() < full_len:
        # Build seasonal median lookup
        cogs_df = train_rows[['log_cogs']].copy()
        cogs_df['_month'] = df_full.loc[train_mask, 'Date'].dt.month.values
        cogs_df['_dow']   = df_full.loc[train_mask, 'Date'].dt.dayofweek.values
        seasonal = cogs_df.groupby(['_month', '_dow'])['log_cogs'].median()

        future_rows = df_full[~train_mask]
        for idx_pos, (_, row) in enumerate(future_rows.iterrows()):
            abs_pos = train_mask.sum() + idx_pos
            key = (row['Date'].month, row['Date'].dayofweek)
            full_cogs[abs_pos] = seasonal.get(key, y_cogs_train.mean())

    # ── DLinear (multivariate) ─────────────────────────────────
    print(f"  >> {tag} Train {model_dlinear_name}...")
    model_dl = train_linear(
        y_train=y_rev_train, y_aux=y_cogs_train,
        seq_len=SEQ_LEN, pred_len=PRED_LEN,
        model_name=model_dlinear_name,
        batch_size=BATCH_SIZE, epochs=EPOCHS,
        lr=LEARNING_RATE, patience=PATIENCE,
    )

    # ── NLinear (multivariate) ─────────────────────────────────
    print(f"  >> {tag} Train {model_nlinear_name}...")
    model_nl = train_linear(
        y_train=y_rev_train, y_aux=y_cogs_train,
        seq_len=SEQ_LEN, pred_len=PRED_LEN,
        model_name=model_nlinear_name,
        batch_size=BATCH_SIZE, epochs=EPOCHS,
        lr=LEARNING_RATE, patience=PATIENCE,
    )

    # ── Build linear_pred (avg of DLinear + NLinear) ───────────
    feat_dl = build_linear_feature(
        model=model_dl, y_train=y_rev_train, full_len=full_len,
        seq_len=SEQ_LEN, pred_len=PRED_LEN,
        y_aux_train=y_cogs_train, y_aux_full=full_cogs,
    )
    feat_nl = build_linear_feature(
        model=model_nl, y_train=y_rev_train, full_len=full_len,
        seq_len=SEQ_LEN, pred_len=PRED_LEN,
        y_aux_train=y_cogs_train, y_aux_full=full_cogs,
    )
    linear_pred = (feat_dl + feat_nl) / 2.0

    # ── Build cogs_linear_pred ─────────────────────────────────
    print(f"  >> {tag} Train COGS-Linear...")
    cogs_pred = build_cogs_feature(
        y_cogs_train=y_cogs_train, full_len=full_len,
        seq_len=SEQ_LEN, pred_len=PRED_LEN,
        model_name='DLinear',
        batch_size=BATCH_SIZE, epochs=max(EPOCHS // 2, 100),
        lr=LEARNING_RATE, patience=PATIENCE,
    )

    return linear_pred, cogs_pred


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

def main():
    set_seed(SEED)
    print("=" * 60)
    print("V7 HYBRID  |  MultiVar-LSTF + Tree Ensemble  |  Leakage-Free")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ──────────────────────────────────────────────────────────
    # DATA LOAD + FEATURE ENGINEERING
    # ──────────────────────────────────────────────────────────
    sales_df = load_sales()
    sub_raw  = pd.read_csv(SUBMISSION_FILE, parse_dates=['Date'])

    # Concat train + submission placeholder (Revenue=NaN for future)
    sub_dummy         = sub_raw.copy()
    sub_dummy['Revenue'] = np.nan
    sub_dummy['COGS']    = np.nan

    full_df = pd.concat([sales_df, sub_dummy], ignore_index=True)
    full_df = build_features(full_df)

    # Log-transform targets
    known_mask = full_df['Revenue'].notna()
    full_df.loc[known_mask, 'log_revenue'] = np.log1p(full_df.loc[known_mask, 'Revenue'])
    full_df.loc[known_mask, 'log_cogs']    = np.log1p(full_df.loc[known_mask, 'COGS'])

    # Impute log_cogs for rows where COGS=NaN (future) — needed for LSTF COGS channel
    # (will be overridden inside _build_lstf_features, but set here for safety)
    full_df['log_cogs'] = full_df['log_cogs'].fillna(
        full_df.loc[known_mask, 'log_cogs'].median()
    )
    full_df['log_revenue'] = full_df['log_revenue'].fillna(0.0)

    # ──────────────────────────────────────────────────────────
    # PHASE 1 — STRICT HOLDOUT EVALUATION ON 2022
    # ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PHASE 1: STRICT EVALUATION — Holdout 2022 (No Leakage)")
    print("=" * 60)

    # Eval LSTF trains on ≤ VAL_END only
    eval_train_mask = (full_df['Revenue'].notna()) & (full_df['Date'] <= VAL_END)

    linear_pred_eval, cogs_pred_eval = _build_lstf_features(
        df_full=full_df,
        train_mask=eval_train_mask,
        tag='Eval',
    )

    df_eval = full_df.copy()
    df_eval['linear_pred']      = linear_pred_eval
    df_eval['cogs_linear_pred'] = cogs_pred_eval

    # Subset to known rows only for tree training
    internal_eval = df_eval[df_eval['Revenue'].notna()].copy()

    feat_cols = get_feature_cols(internal_eval)

    # Train / Val / Test splits
    tr_df = internal_eval[internal_eval['Date'] <= TRAIN_END]
    vl_df = internal_eval[(internal_eval['Date'] > TRAIN_END) & (internal_eval['Date'] <= VAL_END)]
    te_df = internal_eval[internal_eval['Date'] > VAL_END]     # 2022 holdout

    print(f"\n  Split sizes — train={len(tr_df)} val={len(vl_df)} test={len(te_df)}")
    print("  >> Fitting eval tree ensemble...")
    eval_models = fit_tree_ensemble(
        tr_df[feat_cols].values, tr_df['log_revenue'].values,
        vl_df[feat_cols].values, vl_df['log_revenue'].values,
    )

    eval_pred = predict_ensemble(eval_models, te_df[feat_cols].values)

    print("\n  📊 STRICT METRICS ON 2022:")
    print_metrics("Phase 1 — 2022 Holdout", te_df['Revenue'].values, eval_pred)

    plot_feature_importance(eval_models[0], feat_cols, filename='eval_importance_v7.png')

    # ──────────────────────────────────────────────────────────
    # PHASE 2 — PRODUCTION SUBMISSION (2023–2024)
    # ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PHASE 2: PRODUCTION FORECAST — 2023–2024")
    print("=" * 60)

    prod_train_mask = full_df['Revenue'].notna()   # 100% of sales.csv

    linear_pred_prod, cogs_pred_prod = _build_lstf_features(
        df_full=full_df,
        train_mask=prod_train_mask,
        tag='Prod',
    )

    df_prod = full_df.copy()
    df_prod['linear_pred']      = linear_pred_prod
    df_prod['cogs_linear_pred'] = cogs_pred_prod

    prod_internal   = df_prod[df_prod['Revenue'].notna()].copy()
    prod_submission = df_prod[df_prod['Revenue'].isna()].copy()

    # Trong phase prod, dùng toàn bộ known data làm train,
    # một phần nhỏ cuối làm val cho early stopping
    prod_tr = prod_internal[prod_internal['Date'] <= VAL_END]
    prod_vl = prod_internal[prod_internal['Date'] > VAL_END]

    print(f"\n  Split sizes — prod_train={len(prod_tr)} prod_val={len(prod_vl)}")

    # Recompute feat_cols trên prod df (same columns, but sanity check)
    feat_cols_prod = get_feature_cols(prod_internal)
    assert set(feat_cols) == set(feat_cols_prod), \
        "Feature mismatch between eval and prod — check pipeline!"

    print("  >> Fitting production tree ensemble...")
    prod_models = fit_tree_ensemble(
        prod_tr[feat_cols_prod].values, prod_tr['log_revenue'].values,
        prod_vl[feat_cols_prod].values, prod_vl['log_revenue'].values,
    )

    # In-sample sanity check
    train_pred = predict_ensemble(prod_models, prod_tr[feat_cols_prod].values)
    print("\n  📊 In-sample check (should be near-perfect):")
    print_metrics("In-sample 2012–VAL_END", prod_tr['Revenue'].values, train_pred)

    # ── Final Submission ───────────────────────────────────────
    print("  >> Generating submission predictions...")
    sub_preds = predict_ensemble(prod_models, prod_submission[feat_cols_prod].values)
    
    # Chỉ dùng Date để map đúng dự báo Revenue, giữ nguyên COGS gốc từ sample_submission
    pred_df = prod_submission[['Date']].copy()
    pred_df['Revenue'] = np.round(sub_preds, 2)

    # Lấy Date và COGS gốc từ sample_submission để merge với dự báo
    final_submission = pd.merge(sub_raw[['Date', 'COGS']], pred_df, on='Date', how='left')
    
    # Đảm bảo thứ tự cột chuẩn
    final_submission = final_submission[['Date', 'Revenue', 'COGS']]

    submission_path = os.path.join(PROJECT_DIR, 'submission_v6_strict.csv')
    final_submission.to_csv(submission_path, index=False)

    print(f"\n  ✅ Submission saved → {submission_path}")
    print(f"  Revenue forecast stats:")
    print(f"    mean   = {sub_preds.mean():>14,.0f}")
    print(f"    median = {np.median(sub_preds):>14,.0f}")
    print(f"    min    = {sub_preds.min():>14,.0f}")
    print(f"    max    = {sub_preds.max():>14,.0f}")

    print("\n" + "=" * 60)
    print("V7 PIPELINE COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()