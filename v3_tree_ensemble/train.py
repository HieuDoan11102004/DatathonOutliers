import os
import argparse
import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

# Pipeline modules
from pipeline.config import *
from pipeline.feature_engineering import build_features, load_sales
from pipeline.evaluate import (
    print_metrics, plot_predictions, plot_feature_importance, plot_residuals
)

def get_feature_cols(df):
    """Get numeric feature column names, excluding target/date/id columns."""
    exclude = {'Date', 'Revenue', 'COGS', 'log_revenue', 'Residual', 'PatchTST_pred'}
    cols = [c for c in df.columns if c not in exclude]
    return [c for c in cols if df[c].dtype in [np.float64, np.float32, np.int64, np.int32]]

def main():
    set_seed(SEED)
    print("=" * 60)
    print("REVENUE FORECASTING — GOD-TIER TREE ENSEMBLE v3")
    print("=" * 60)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ==========================================================
    # STEP 1: Feature Engineering (Tabular & Time from 10 tables)
    # ==========================================================
    sales_df = load_sales()
    sub_raw = pd.read_csv(SUBMISSION_FILE, parse_dates=['Date'])
    
    # Append submission so calendar/lag features can compute naturally
    sub_dummy = sub_raw.copy()
    sub_dummy['Revenue'] = np.nan
    full_df = pd.concat([sales_df, sub_dummy], ignore_index=True)
    full_df = build_features(full_df)

    # Split internal vs submission
    internal_df = full_df[full_df['Revenue'].notna()].copy()
    submission_df = full_df[full_df['Revenue'].isna()].copy()
    
    # Target is log1p(Revenue)
    internal_df['log_revenue'] = np.log1p(internal_df['Revenue'])

    train_mask = internal_df['Date'] <= TRAIN_END
    val_mask = (internal_df['Date'] > TRAIN_END) & (internal_df['Date'] <= VAL_END)
    test_mask = internal_df['Date'] > VAL_END

    train_df = internal_df[train_mask].reset_index(drop=True)
    val_df = internal_df[val_mask].reset_index(drop=True)
    test_df = internal_df[test_mask].reset_index(drop=True)

    print(f"  Train: {train_df['Date'].min().date()} → {train_df['Date'].max().date()} ({len(train_df)} days)")
    print(f"  Val:   {val_df['Date'].min().date()} → {val_df['Date'].max().date()} ({len(val_df)} days)")
    print(f"  Test:  {test_df['Date'].min().date()} → {test_df['Date'].max().date()} ({len(test_df)} days)")

    # ==========================================================
    # STEP 2: Train Ensemble Models (TimeSeriesSplit)
    # ==========================================================
    print("\n" + "=" * 60)
    print("STEP 2: Train 15-Model Ensemble (3 Models × 5 Folds)")
    print("=" * 60)
    
    top_features = get_feature_cols(internal_df)
    print(f"  Total Feature Columns: {len(top_features)}")

    tscv = TimeSeriesSplit(n_splits=N_FOLDS)
    
    lgb_models = []
    xgb_models = []
    cat_models = []

    X_train = train_df[top_features].values
    y_train = train_df['log_revenue'].values

    for fold, (trn_idx, val_idx) in enumerate(tscv.split(X_train)):
        print(f"\n  ► Fold {fold+1}/{N_FOLDS} -----------------------------------")
        X_t, y_t = X_train[trn_idx], y_train[trn_idx]
        X_v, y_v = X_train[val_idx], y_train[val_idx]

        # 1. LightGBM
        print("    Training LightGBM...")
        model_lgb = lgb.LGBMRegressor(n_estimators=N_ESTIMATORS, **LGBM_PARAMS)
        model_lgb.fit(
            X_t, y_t,
            eval_set=[(X_v, y_v)],
            callbacks=[lgb.early_stopping(EARLY_STOPPING, verbose=False)],
        )
        lgb_models.append(model_lgb)
        print(f"      Best Iteration: {model_lgb.best_iteration_}")

        # 2. XGBoost
        print("    Training XGBoost...")
        model_xgb = xgb.XGBRegressor(n_estimators=N_ESTIMATORS, early_stopping_rounds=EARLY_STOPPING, **XGB_PARAMS)
        model_xgb.fit(
            X_t, y_t,
            eval_set=[(X_v, y_v)],
            verbose=False
        )
        xgb_models.append(model_xgb)
        print(f"      Best Iteration: {model_xgb.best_iteration}")

        # 3. CatBoost
        print("    Training CatBoost...")
        model_cat = CatBoostRegressor(**CATBOOST_PARAMS)
        model_cat.fit(
            X_t, y_t,
            eval_set=[(X_v, y_v)],
        )
        cat_models.append(model_cat)
        print(f"      Best Iteration: {model_cat.get_best_iteration()}")

    # ==========================================================
    # STEP 3: Ensemble Evaluation (In-Sample & Out-Of-Sample)
    # ==========================================================
    print("\n" + "=" * 60)
    print("STEP 3: Evaluate Ensemble Predictions")
    print("=" * 60)

    def predict_ensemble(X_data):
        """Predict using the 15-model average."""
        preds = []
        for m in lgb_models: preds.append(m.predict(X_data))
        for m in xgb_models: preds.append(m.predict(X_data))
        for m in cat_models: preds.append(m.predict(X_data))
        
        # Take geometric mean or simple average of log predictions, then expm1
        avg_log = np.mean(preds, axis=0)
        return np.maximum(np.expm1(avg_log), 0)

    val_final_pred = predict_ensemble(val_df[top_features].values)
    test_final_pred = predict_ensemble(test_df[top_features].values)
    
    # Plot feature importance for one of the LightGBM models
    plot_feature_importance(lgb_models[-1], top_features, filename='lgbm_importance.png')

    print("\n  📊 Ensemble Metrics:")
    print_metrics("Ensemble on Val  (in-sample)", val_df['Revenue'].values, val_final_pred)
    print_metrics("Ensemble on Test (holdout)", test_df['Revenue'].values, test_final_pred)

    test_rmse = np.sqrt(mean_squared_error(test_df['Revenue'].values, test_final_pred))
    print(f"\n  📈 Final Holdout RMSE: {test_rmse:>12,.0f}")

    # ==========================================================
    # STEP 4: Visualizations
    # ==========================================================
    plot_predictions(
        test_df['Date'].values, test_df['Revenue'].values, test_final_pred,
        'Internal Test (2022) — Actual vs Ensemble Prediction',
        'test_predictions.png'
    )
    plot_predictions(
        val_df['Date'].values, val_df['Revenue'].values, val_final_pred,
        'Validation (2020-2021) — Actual vs Ensemble Prediction',
        'val_predictions.png'
    )
    plot_residuals(test_df['Revenue'].values, test_final_pred, test_df['Date'].values, filename='test_residuals.png')

    # ==========================================================
    # STEP 5: Generate Kaggle Submission
    # ==========================================================
    print("\n" + "=" * 60)
    print("STEP 5: Generate Kaggle Submission")
    print("=" * 60)

    # We use the cross-validated 15 models to predict the submission data
    print(f"  Predicting future constraints using {len(lgb_models) + len(xgb_models) + len(cat_models)} trees...")
    
    sub_final_pred = predict_ensemble(submission_df[top_features].values)

    final_submission = sub_raw.copy()
    final_submission['Revenue'] = np.round(sub_final_pred, 2)

    submission_path = os.path.join(PROJECT_DIR, 'submission.csv')
    final_submission.to_csv(submission_path, index=False)

    print(f"\n  ✅ Submission saved: {submission_path}")
    print(f"  Rows: {len(final_submission)}")
    print(f"  Date range: {final_submission['Date'].min().date()} → {final_submission['Date'].max().date()}")
    print(f"  Revenue range: {sub_final_pred.min():,.0f} → {sub_final_pred.max():,.0f}")
    print(f"  Revenue mean:  {sub_final_pred.mean():,.0f}")

    print("\n  Preview:")
    print(final_submission.head(10).to_string(index=False))

    print("\n" + "=" * 60)
    print("🚀 Ensemble Pipeline Complete!")
    print("=" * 60)

if __name__ == '__main__':
    main()
