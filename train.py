import os
import argparse
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

# Pipeline modules
from pipeline.config import *
from pipeline.feature_engineering import build_features, load_sales
from pipeline.patchtst import (train_patchtst, predict_patchtst_recursive,
                               predict_patchtst_teacherforced)
from pipeline.evaluate import (
    print_metrics, plot_predictions, plot_training_history,
    plot_feature_importance, plot_residuals
)

def get_feature_cols(df):
    """Get numeric feature column names, excluding target/date/id columns."""
    exclude = {'Date', 'Revenue', 'COGS', 'log_revenue', 'Residual', 'PatchTST_pred'}
    cols = [c for c in df.columns if c not in exclude]
    return [c for c in cols if df[c].dtype in [np.float64, np.float32, np.int64, np.int32]]

def main():
    set_seed(SEED)
    print("=" * 60)
    print("REVENUE FORECASTING — HYBRID RESIDUAL PIPELINE v2")
    print("=" * 60)

    # ==========================================================
    # STEP 1: Feature Engineering (Tabular & Time)
    # ==========================================================
    sales_df = load_sales()
    sub_raw = pd.read_csv(SUBMISSION_FILE, parse_dates=['Date'])
    
    # We append submission so calendar/lag features can compute naturally
    sub_dummy = sub_raw.copy()
    sub_dummy['Revenue'] = np.nan
    full_df = pd.concat([sales_df, sub_dummy], ignore_index=True)
    full_df = build_features(full_df)

    # Split back
    internal_df = full_df[full_df['Revenue'].notna()].copy()
    submission_df = full_df[full_df['Revenue'].isna()].copy()
    
    # Prepare PatchTST Features
    print("\n" + "=" * 60)
    print("STEP 2: Prepare PatchTST (Trend Backbone)")
    print("=" * 60)
    internal_df['log_revenue'] = np.log1p(internal_df['Revenue'])
    pt_features_all = internal_df[PATCHTST_FEATURES].values

    train_mask = internal_df['Date'] <= TRAIN_END
    val_mask = (internal_df['Date'] > TRAIN_END) & (internal_df['Date'] <= VAL_END)
    test_mask = internal_df['Date'] > VAL_END

    train_df = internal_df[train_mask]
    val_df = internal_df[val_mask]
    test_df = internal_df[test_mask]

    print(f"  Train: {train_df['Date'].min().date()} → {train_df['Date'].max().date()} ({len(train_df)} days)")
    print(f"  Val:   {val_df['Date'].min().date()} → {val_df['Date'].max().date()} ({len(val_df)} days)")
    print(f"  Test:  {test_df['Date'].min().date()} → {test_df['Date'].max().date()} ({len(test_df)} days)")

    # ==========================================================
    # STEP 3: Train PatchTST
    # ==========================================================
    pt_features_train = pt_features_all[:len(train_df)]
    patchtst_model, pt_history = train_patchtst(pt_features_train)
    plot_training_history(pt_history)

    # ==========================================================
    # STEP 4: PatchTST Predict & Embeddings
    # ==========================================================
    print("\n  Predicting PatchTST trend for all internal data...")
    val_start_idx = len(train_df)
    test_start_idx = len(train_df) + len(val_df)

    # We need predictions for entire internal_df to calculate true Residuals
    # Train set pred: use recursive or teacher-forced? 
    # For training LightGBM on train residuals, it's safer to use teacher-forcing
    # so we align with how validation gets teacher-forced.
    pt_pred_internal, pt_emb_internal = predict_patchtst_teacherforced(
        patchtst_model, pt_features_all, 
        start_idx=PATCHTST_CONTEXT_LEN, 
        n_steps=len(internal_df) - PATCHTST_CONTEXT_LEN,
        return_embeds=True
    )
    
    # Pad the beginning (first 'context_len' days) where PatchTST cannot predict yet
    pad_len = PATCHTST_CONTEXT_LEN
    pt_pred_padded = np.concatenate([internal_df['Revenue'].values[:pad_len], pt_pred_internal])
    
    # Default embedding 0s for padding
    d_model = pt_emb_internal.shape[1]
    emb_pad = np.zeros((pad_len, d_model))
    pt_emb_padded = np.vstack([emb_pad, pt_emb_internal])

    internal_df['PatchTST_pred'] = pt_pred_padded
    internal_df['Residual'] = internal_df['Revenue'] - internal_df['PatchTST_pred']
    
    # Store embeddings as features
    emb_cols = [f'pt_emb_{i}' for i in range(d_model)]
    emb_df = pd.DataFrame(pt_emb_padded, columns=emb_cols, index=internal_df.index)
    internal_df = pd.concat([internal_df, emb_df], axis=1)

    # ==========================================================
    # STEP 5: Train LightGBM (Residual Predictor) — K-Fold TS Split
    # ==========================================================
    print("\n" + "=" * 60)
    print("STEP 5: Train LightGBM on Residuals (TimeSeriesSplit)")
    print("=" * 60)
    
    train_df = internal_df[train_mask].reset_index(drop=True)
    val_df = internal_df[val_mask].reset_index(drop=True)
    test_df = internal_df[test_mask].reset_index(drop=True)

    top_features = get_feature_cols(internal_df)
    print(f"  LightGBM Features: {len(top_features)} (including {d_model} PatchTST embeddings)")

    tscv = TimeSeriesSplit(n_splits=N_FOLDS)
    lgb_models = []
    best_iters = []

    X_train = train_df[top_features].values
    y_train = train_df['Residual'].values

    for fold, (trn_idx, val_idx) in enumerate(tscv.split(X_train)):
        print(f"\n  [Fold {fold+1}/{N_FOLDS}]")
        X_t, y_t = X_train[trn_idx], y_train[trn_idx]
        X_v, y_v = X_train[val_idx], y_train[val_idx]

        model = lgb.LGBMRegressor(n_estimators=LGBM_N_ESTIMATORS, **LGBM_PARAMS)
        model.fit(
            X_t, y_t,
            eval_set=[(X_v, y_v)],
            callbacks=[lgb.early_stopping(LGBM_EARLY_STOPPING, verbose=False),
                       lgb.log_evaluation(500)],
        )
        lgb_models.append(model)
        best_iters.append(model.best_iteration_)
    
    avg_best_iter = int(np.mean(best_iters))
    print(f"\n  Avg Best Iteration: {avg_best_iter}")

    # ==========================================================
    # STEP 6: Combine Predictions (Trend + Residual)
    # ==========================================================
    print("\n" + "=" * 60)
    print("STEP 6: Evaluate Hybrid Pipeline")
    print("=" * 60)

    # Predict residuals using ensemble of K-Fold models
    lgb_val_res_pred = np.mean([m.predict(val_df[top_features]) for m in lgb_models], axis=0)
    lgb_test_res_pred = np.mean([m.predict(test_df[top_features]) for m in lgb_models], axis=0)

    # Combine: Final = PatchTST Trend + LGB Residual
    val_final_pred = np.maximum(val_df['PatchTST_pred'] + lgb_val_res_pred, 0)
    test_final_pred = np.maximum(test_df['PatchTST_pred'] + lgb_test_res_pred, 0)
    
    plot_feature_importance(lgb_models[-1], top_features)

    print("\n  📊 Ensemble Metrics (Stacked Output):")
    print_metrics("Hybrid on Val  (in-sample)", val_df['Revenue'].values, val_final_pred)
    print_metrics("Hybrid on Test (holdout)", test_df['Revenue'].values, test_final_pred)

    test_rmse_hybrid = np.sqrt(mean_squared_error(test_df['Revenue'].values, test_final_pred))
    test_rmse_pt_only = np.sqrt(mean_squared_error(test_df['Revenue'].values, test_df['PatchTST_pred']))
    
    print(f"\n  📈 RMSE Comparison (Test):")
    print(f"    PatchTST Only (Trend): {test_rmse_pt_only:>12,.0f}")
    print(f"    Hybrid Pipeline:       {test_rmse_hybrid:>12,.0f}")

    # ==========================================================
    # STEP 7: Visualizations
    # ==========================================================
    # Visualizations
    plot_predictions(
        test_df['Date'].values, test_df['Revenue'].values, test_final_pred,
        'Internal Test (2022) — Actual vs Hybrid Prediction',
        'test_predictions.png'
    )
    plot_predictions(
        val_df['Date'].values, val_df['Revenue'].values, val_final_pred,
        'Validation (2020-2021) — Actual vs Hybrid Prediction',
        'val_predictions.png'
    )
    plot_residuals(test_df['Revenue'].values, test_final_pred, test_df['Date'].values)

    # ==========================================================
    # STEP 8: Generate Validation Artifacts & Checkpoints
    # ==========================================================
    print("\n" + "=" * 60)
    print("STEP 8: Generate Kaggle Submission")
    print("=" * 60)

    # --- Retrain LightGBM on FULL internal residuals ---
    print("  Retraining LightGBM on FULL internal residuals...")
    lgb_full = lgb.LGBMRegressor(
        n_estimators=avg_best_iter,
        **{k: v for k, v in LGBM_PARAMS.items()}
    )
    lgb_full.fit(internal_df[top_features], internal_df['Residual'])

    # --- Predict PatchTST recursively for submission ---
    print("  PatchTST Recursive Prediction & Embedding extraction...")
    submission_df['log_revenue'] = 0  # Placeholder
    sub_pt_features = submission_df[PATCHTST_FEATURES].values
    
    pt_sub_pred, pt_sub_emb = predict_patchtst_recursive(
        patchtst_model, pt_features_all, sub_pt_features, return_embeds=True
    )
    
    # Attach embeddings to submission_df
    sub_emb_df = pd.DataFrame(pt_sub_emb, columns=emb_cols, index=submission_df.index)
    submission_df = pd.concat([submission_df.drop(columns=emb_cols, errors='ignore'), sub_emb_df], axis=1)

    # Predict LightGBM residuals
    sub_res_pred = lgb_full.predict(submission_df[top_features])

    # Final combined
    final_revenue = np.maximum(pt_sub_pred + sub_res_pred, 0)
    final_submission = sub_raw.copy()
    final_submission['Revenue'] = np.round(final_revenue, 2)

    submission_path = os.path.join(ROOT_DIR, 'submission.csv')
    final_submission.to_csv(submission_path, index=False)

    print(f"\n  ✅ Submission saved: {submission_path}")
    print(f"  Rows: {len(final_submission)}")
    print(f"  Date range: {final_submission['Date'].min()} → {final_submission['Date'].max()}")
    print(f"  Revenue range: {final_revenue.min():,.0f} → {final_revenue.max():,.0f}")
    print(f"  Revenue mean:  {final_revenue.mean():,.0f}")

    print("\n  Preview:")
    print(final_submission.head(10).to_string(index=False))

    print("\n" + "=" * 60)
    print("🚀 Hybrid Pipeline Complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
