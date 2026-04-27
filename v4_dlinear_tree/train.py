"""
V4 HYBRID PIPELINE
==================
DLinear-MV (Revenue × COGS) + COGS-Linear + Tree Ensemble (Residual Learning)
Strict Expanding Window Cross Validation -> Phase B Full Fit
"""

import os
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.model_selection import TimeSeriesSplit
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

    # Loại bỏ các biến dễ gây overfit thời gian (cho phép tree dựa vào trend linear hơn)
    forbidden = {'year', 'day', 'trend_linear'}
    
    # Tự động loại bỏ các biến Trash từ kịch bản Null Importance (nếu có)
    null_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models/feature_null_scores.csv')
    if os.path.exists(null_csv):
        df_null = pd.read_csv(null_csv)
        # Nâng ngưỡng Signal_Score lên 2.0 (Aggressive Pruning)
        racia_feats = set(df_null[df_null['Signal_Score'] <= 2.0]['Feature'])
        forbidden.update(racia_feats)

    # Đảm bảo giữ lại các biến cốt lõi an toàn
    essential = {'linear_pred', 'cogs_linear_pred', 'promo_intensity', 'is_tet_month'}
    forbidden = forbidden - essential

    all_cols = [c for c in all_cols if c not in forbidden]

    if len(all_cols) > 0:
        breakdown = {}
        for grp, fn in _FEATURE_GROUPS.items():
            cnt = sum(1 for c in all_cols if fn(c))
            if cnt > 0:
                breakdown[grp] = cnt
        print(f"  Feature breakdown: {breakdown}  →  total={len(all_cols)}")

    return all_cols


# ─────────────────────────────────────────────
#  TREE ENSEMBLE (RESIDUAL LEARNING)
# ─────────────────────────────────────────────

def fit_tree_ensemble_cv(X_tr: np.ndarray, y_tr: np.ndarray, 
                         X_vl: np.ndarray, y_vl: np.ndarray,
                         w_tr: np.ndarray = None, w_vl: np.ndarray = None) -> tuple[list, list]:
    """Train Tree Ensemble với Early Stopping cho CV."""
    print("    LightGBM...")
    lgb_model = lgb.LGBMRegressor(n_estimators=5000, **LGBM_PARAMS)
    lgb_model.fit(
        X_tr, y_tr,
        sample_weight=w_tr,
        eval_set=[(X_vl, y_vl)],
        eval_sample_weight=[w_vl] if w_vl is not None else None,
        callbacks=[lgb.early_stopping(EARLY_STOPPING, verbose=False)],
    )

    print("    XGBoost...")
    xgb_model = xgb.XGBRegressor(
        n_estimators=5000,
        early_stopping_rounds=EARLY_STOPPING,
        **XGB_PARAMS,
    )
    xgb_model.fit(
        X_tr, y_tr, 
        sample_weight=w_tr,
        eval_set=[(X_vl, y_vl)],
        sample_weight_eval_set=[w_vl] if w_vl is not None else None,
        verbose=False
    )

    cat_params = CATBOOST_PARAMS.copy()
    cat_params['iterations'] = 5000
    cat_params['early_stopping_rounds'] = EARLY_STOPPING
    print("    CatBoost...")
    cat_model = CatBoostRegressor(**cat_params)
    cat_model.fit(
        X_tr, y_tr,
        sample_weight=w_tr,
        eval_set=[(X_vl, y_vl)],
        verbose=False
    )

    from sklearn.metrics import mean_squared_error
    lgb_rmse = np.sqrt(mean_squared_error(y_vl, lgb_model.predict(X_vl)))
    xgb_rmse = np.sqrt(mean_squared_error(y_vl, xgb_model.predict(X_vl)))
    cat_rmse = np.sqrt(mean_squared_error(y_vl, cat_model.predict(X_vl)))

    opt_iters = [lgb_model.best_iteration_, xgb_model.best_iteration, cat_model.get_best_iteration()]
    return [lgb_model, xgb_model, cat_model], opt_iters, [lgb_rmse, xgb_rmse, cat_rmse]

def fit_tree_ensemble_full(X_tr: np.ndarray, y_tr: np.ndarray, opt_iters: list, w_tr: np.ndarray = None) -> list:
    """Train Tree Ensemble KHÔNG Early Stopping, khóa mức trần lặp lại theo CV."""
    lgb_it, xgb_it, cat_it = [int(i) for i in opt_iters]
    
    print(f"    LightGBM (Khóa dừng ở {lgb_it} iterations)...")
    lgb_model = lgb.LGBMRegressor(n_estimators=lgb_it, **LGBM_PARAMS)
    lgb_model.fit(X_tr, y_tr, sample_weight=w_tr)

    print(f"    XGBoost (Khóa dừng ở {xgb_it} iterations)...")
    xgb_model = xgb.XGBRegressor(n_estimators=xgb_it, **XGB_PARAMS)
    xgb_model.fit(X_tr, y_tr, sample_weight=w_tr, verbose=False)

    cat_params = CATBOOST_PARAMS.copy()
    cat_params['iterations'] = cat_it
    # Xóa early_stopping_rounds trong full fit để ép cây chạy đủ số vòng
    if 'early_stopping_rounds' in cat_params:
        del cat_params['early_stopping_rounds']
        
    print(f"    CatBoost (Khóa dừng ở {cat_it} iterations)...")
    cat_model = CatBoostRegressor(**cat_params)
    cat_model.fit(X_tr, y_tr, sample_weight=w_tr, verbose=False)

    return [lgb_model, xgb_model, cat_model]

def print_top_features(models: list, feat_cols: list, top_n: int = 25):
    """Tính trung bình feature importance từ 3 model và in ra top N"""
    lgb_m, xgb_m, cat_m = models
    
    imp_lgb = lgb_m.feature_importances_
    imp_xgb = xgb_m.feature_importances_
    imp_cat = cat_m.feature_importances_
    
    # Chuẩn hóa về [0, 1] để trung bình công bằng
    imp_lgb = imp_lgb / (imp_lgb.sum() + 1e-9)
    imp_xgb = imp_xgb / (imp_xgb.sum() + 1e-9)
    imp_cat = imp_cat / (imp_cat.sum() + 1e-9)
    
    avg_imp = (imp_lgb + imp_xgb + imp_cat) / 3.0
    
    idx = np.argsort(avg_imp)[::-1][:top_n]
    
    print(f"\n  🌟 BẢNG VÀNG TOP {top_n} TRỌNG SỐ (Càng cao càng xịn):")
    for i, j in enumerate(idx):
        print(f"    {i+1:>2}. {feat_cols[j]:<30} | {avg_imp[j]:.4f}")

def predict_residual_ensemble(models: list, X: np.ndarray, base_pred: np.ndarray, weights=None) -> np.ndarray:
    """
    Dự báo Residual từ Tree, sau đó cộng gộp với baseline DLinear (base_pred).
    Đây là mấu chốt để giữ được Trend của DLinear cho tương lai!
    """
    res_preds = np.stack([m.predict(X) for m in models], axis=0) # [3, N]
    if weights is not None:
        avg_res = np.average(res_preds, axis=0, weights=weights)
    else:
        avg_res = res_preds.mean(axis=0)                           # [N]
    
    final_log = base_pred + avg_res
    return np.maximum(np.expm1(final_log), 0.0)


# ─────────────────────────────────────────────
#  LSTF FEATURE BUILDER (Revenue + COGS)
# ─────────────────────────────────────────────

def _build_lstf_features(
    df_full: pd.DataFrame,
    train_mask: pd.Series,
    model_dlinear_name: str = 'MultiVarDLinear',
    tag: str = '',
) -> tuple[np.ndarray, np.ndarray]:
    """
    Train DLinear (multivariate, Revenue+COGS) và COGS-Linear (univariate).
    """
    full_len = len(df_full)

    train_rows = df_full[train_mask].copy()

    y_rev_train  = train_rows['log_revenue'].values
    y_cogs_train = train_rows['log_cogs'].values

    # Full COGS array: for future dates, no known value.
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
    print(f"  >> [{tag}] Train {model_dlinear_name}...")
    model_dl = train_linear(
        y_train=y_rev_train, y_aux=y_cogs_train,
        seq_len=SEQ_LEN, pred_len=PRED_LEN,
        model_name=model_dlinear_name,
        batch_size=BATCH_SIZE, epochs=EPOCHS,
        lr=LEARNING_RATE, patience=PATIENCE,
    )

    linear_pred = build_linear_feature(
        model=model_dl, y_train=y_rev_train, full_len=full_len,
        seq_len=SEQ_LEN, pred_len=PRED_LEN,
        y_aux_train=y_cogs_train, y_aux_full=full_cogs,
    )

    # ── COGS-Linear ─────────────────────────────────
    print(f"  >> [{tag}] Train COGS-Linear...")
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
    print("V4 HYBRID | MultiVarDLinear + Tree Ensemble (Residual)")
    print("Kaggle Standard: Expanding Window CV (No Leakage) -> Full Fit")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ──────────────────────────────────────────────────────────
    # DATA LOAD + FEATURE ENGINEERING
    # ──────────────────────────────────────────────────────────
    sales_df = load_sales()
    sub_raw  = pd.read_csv(SUBMISSION_FILE, parse_dates=['Date'])

    sub_dummy = sub_raw.copy()
    sub_dummy['Revenue'] = np.nan
    sub_dummy['COGS']    = np.nan

    full_df = pd.concat([sales_df, sub_dummy], ignore_index=True)
    full_df = build_features(full_df)

    known_mask = full_df['Revenue'].notna()
    full_df.loc[known_mask, 'log_revenue'] = np.log1p(full_df.loc[known_mask, 'Revenue'])
    full_df.loc[known_mask, 'log_cogs']    = np.log1p(full_df.loc[known_mask, 'COGS'])

    full_df['log_cogs'] = full_df['log_cogs'].fillna(
        full_df.loc[known_mask, 'log_cogs'].median()
    )
    full_df['log_revenue'] = full_df['log_revenue'].fillna(0.0)

    # Lấy index
    internal_idx = full_df[full_df['Revenue'].notna()].index.values
    test_idx = full_df[full_df['Revenue'].isna()].index.values

    # ──────────────────────────────────────────────────────────
    # PHASE A: K-FOLD EXPANDING WINDOW CV
    # ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PHASE A: 4-FOLD EXPANDING WINDOW CROSS-VALIDATION")
    print("=" * 60)

    tscv = TimeSeriesSplit(n_splits=4, test_size=548)
    
    cv_scores = []
    optimal_iters_all = []
    rmse_all = []
    # Tạo dummy trước để get_feature_cols thu thập được
    full_df['linear_pred'] = 0.0
    full_df['cogs_linear_pred'] = 0.0
    feat_cols = get_feature_cols(full_df)

    # ──────────────────────────────────────────────────────────
    # CORRELATION PRUNING (>0.98)
    # ──────────────────────────────────────────────────────────
    print("\n  >> Lọc biến tương quan cao (> 0.98)...")
    corr_matrix = full_df.loc[known_mask, feat_cols].corr().abs()
    target_corr = full_df.loc[known_mask, feat_cols + ['log_revenue']].corr().abs()['log_revenue']
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    to_drop = set()
    for column in upper.columns:
        if column in to_drop: continue
        high_corr_peers = upper.index[upper[column] > 0.98].tolist()
        for peer in high_corr_peers:
            if peer in to_drop: continue
            corr_col = target_corr.get(column, 0)
            corr_peer = target_corr.get(peer, 0)
            if corr_col > corr_peer:
                to_drop.add(peer)
                print(f"     🗑️  Dropped: {peer:<25} (Corr with Revenue {corr_peer:.4f} < {column})")
            else:
                to_drop.add(column)
                print(f"     🗑️  Dropped: {column:<25} (Corr with Revenue {corr_col:.4f} <= {peer})")
                break
                
    feat_cols = [f for f in feat_cols if f not in to_drop]
    print(f"  ✅ Retained {len(feat_cols)} features sau khi Pruning.\n")

    # Khởi tạo mảng lưu OOF cho Phase B
    oof_linear_pred = np.zeros(len(internal_idx))
    oof_cogs_pred   = np.zeros(len(internal_idx))
    has_oof_mask    = np.zeros(len(internal_idx), dtype=bool)

    for fold, (trn_fold_idx, val_fold_idx) in enumerate(tscv.split(internal_idx)):
        print(f"\n[FOLD {fold+1}/4]")
        
        abs_trn_idx = internal_idx[trn_fold_idx]
        abs_val_idx = internal_idx[val_fold_idx]

        train_mask = pd.Series(False, index=full_df.index)
        train_mask.loc[abs_trn_idx] = True

        dl_pred, cogs_pred = _build_lstf_features(
            df_full=full_df,
            train_mask=train_mask,
            model_dlinear_name='MultiVarDLinear',
            tag=f'F{fold+1}'
        )

        # LƯU OOF TỪ TẬP VALIDATION SẠCH CỦA FOLD NÀY
        oof_linear_pred[val_fold_idx] = dl_pred[abs_val_idx]
        oof_cogs_pred[val_fold_idx]   = cogs_pred[abs_val_idx]
        has_oof_mask[val_fold_idx]    = True

        fold_df = full_df.copy()
        fold_df['linear_pred'] = dl_pred
        fold_df['cogs_linear_pred'] = cogs_pred

        tr_df = fold_df.loc[abs_trn_idx]
        vl_df = fold_df.loc[abs_val_idx]

        # Residuals
        tr_res = tr_df['log_revenue'].values - tr_df['linear_pred'].values
        vl_res = vl_df['log_revenue'].values - vl_df['linear_pred'].values

        # Bơm trọng số mạnh cho các đỉnh Peak (thay vì log bảo thủ)
        w_tr = (tr_df['Revenue'].values / 1e6) ** 1.5
        w_vl = (vl_df['Revenue'].values / 1e6) ** 1.5

        print("  >> Fitting tree ensemble (RESIDUAL LEARNING)...")
        models, opt_iters, rmse_list = fit_tree_ensemble_cv(
            tr_df[feat_cols].values, tr_res,
            vl_df[feat_cols].values, vl_res,
            w_tr=w_tr, w_vl=w_vl
        )
        optimal_iters_all.append(opt_iters)
        rmse_all.append(rmse_list)
        
        # Calculate weights for THIS fold
        fold_model_inv_sq = 1.0 / (np.array(rmse_list) ** 2)
        fold_model_weights = fold_model_inv_sq / fold_model_inv_sq.sum()

        val_preds = predict_residual_ensemble(
            models, 
            vl_df[feat_cols].values,
            base_pred=vl_df['linear_pred'].values,
            weights=fold_model_weights
        )

        rmse = np.sqrt(np.mean((vl_df['Revenue'].values - val_preds)**2))
        cv_scores.append(rmse)
        print(f"  --> Fold {fold+1} Validation RMSE: {rmse:>13,.0f}")

    print("\n  📊 AVERAGE CV OUT-OF-FOLD SCORES:")
    print(f"    RMSE = {np.mean(cv_scores):,.0f} (+/- {np.std(cv_scores):,.0f})")
    
    # ── FOLD WEIGHTING (Exponential Decay for iters) ──
    n_f = len(optimal_iters_all)
    fold_w = np.arange(1, n_f + 1)
    fold_w = fold_w / fold_w.sum()
    avg_iters = np.average(optimal_iters_all, axis=0, weights=fold_w).astype(int)
    
    # ── INVERSE RMSE WEIGHTING (P=2) cho 3 Model ──
    avg_rmse_models = np.mean(rmse_all, axis=0)
    inv_rmse_sq = 1.0 / (avg_rmse_models ** 2)
    model_weights = inv_rmse_sq / inv_rmse_sq.sum()
    
    print(f"    Weighted Optimal Iterations: {avg_iters.tolist()}")
    print(f"    Ensemble Weights (LGB, XGB, CAT): [{model_weights[0]:.1%}, {model_weights[1]:.1%}, {model_weights[2]:.1%}]")

    # ──────────────────────────────────────────────────────────
    # PHASE B: FULL FIT & INFERENCE
    # ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PHASE B: FULL FIT & KAGGLE SUBMISSION (2023-2024)")
    print("=" * 60)

    full_train_mask = pd.Series(False, index=full_df.index)
    full_train_mask.loc[internal_idx] = True

    dl_pred_full, cogs_pred_full = _build_lstf_features(
        df_full=full_df,
        train_mask=full_train_mask,
        model_dlinear_name='MultiVarDLinear',
        tag='PROD'
    )

    final_df = full_df.copy()

    # THAY ĐỔI Ở ĐÂY: Tạo dataframe sạch cho Production Train bằng OOF
    prod_tr = final_df.loc[internal_idx].copy()
    prod_tr['linear_pred']      = oof_linear_pred
    prod_tr['cogs_linear_pred'] = oof_cogs_pred

    # Chỉ train Tree trên những dòng có dự báo OOF (loại bỏ đoạn đầu tiên)
    clean_prod_tr = prod_tr[has_oof_mask]
    tr_res_prod = clean_prod_tr['log_revenue'].values - clean_prod_tr['linear_pred'].values
    
    # Bơm trọng số mạnh cho Prod Train
    w_tr_prod = (clean_prod_tr['Revenue'].values / 1e6) ** 1.5

    print("  >> Fitting production tree ensemble with OOF features...")
    final_models = fit_tree_ensemble_full(
        clean_prod_tr[feat_cols].values, tr_res_prod, opt_iters=avg_iters, w_tr=w_tr_prod
    )

    # In ra top Features và vẽ biểu đồ PNG từ model LightGBM
    print_top_features(final_models, feat_cols)
    plot_feature_importance(final_models[0], feat_cols, filename='plot_importance_v4.png')

    in_sample_pred = predict_residual_ensemble(
        final_models, 
        clean_prod_tr[feat_cols].values,
        base_pred=clean_prod_tr['linear_pred'].values,
        weights=model_weights
    )
    print("\n  📊 OOF Train check (Clean Full Data):")
    print_metrics("OOF 2013–2022", clean_prod_tr['Revenue'].values, in_sample_pred)

    print("  >> Generating submission predictions (Inference 2023-2024)...")
    
    prod_sub = final_df.loc[test_idx].copy()
    # KHI INFERENCE: Dùng model PROD đã train trên full data (dl_pred_full)
    prod_sub['linear_pred']      = dl_pred_full[test_idx]
    prod_sub['cogs_linear_pred'] = cogs_pred_full[test_idx]

    sub_preds = predict_residual_ensemble(
        final_models, 
        prod_sub[feat_cols].values,
        base_pred=prod_sub['linear_pred'].values,
        weights=model_weights
    )

    pred_df = prod_sub[['Date']].copy()
    pred_df['Revenue'] = np.round(sub_preds, 2)

    final_submission = pd.merge(sub_raw[['Date', 'COGS']], pred_df, on='Date', how='left')
    final_submission = final_submission[['Date', 'Revenue', 'COGS']]

    submission_path = os.path.join(PROJECT_DIR, 'submission_v4.csv')
    final_submission.to_csv(submission_path, index=False)

    print(f"\n  ✅ Submission saved → {submission_path}")
    print(f"  Revenue forecast stats:")
    print(f"    mean   = {sub_preds.mean():>14,.0f}")
    print(f"    median = {np.median(sub_preds):>14,.0f}")
    print(f"    min    = {sub_preds.min():>14,.0f}")
    print(f"    max    = {sub_preds.max():>14,.0f}")

    print("\n" + "=" * 60)
    print("V4 KAGGLE PIPELINE COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()
