"""
Ablation Study Script for V6 (LRU + Tree).
Evaluates the contribution of different feature modules by removing them one by one
and measuring the impact on the internal Holdout Test Set (2022).
Uses LightGBM for fast iteration.
"""
import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error

from pipeline.config import *
from pipeline.feature_engineering import build_features, load_sales
from pipeline.lru_model import train_lru, build_lru_feature
from pipeline.evaluate import print_metrics

def get_base_features(df):
    exclude = {'Date', 'Revenue', 'COGS', 'log_revenue', 'Residual'}
    cols = [c for c in df.columns if c not in exclude]
    return [c for c in cols if df[c].dtype in [np.float64, np.float32, np.int64, np.int32]]

def run_ablation():
    set_seed(SEED)
    print("=" * 60)
    print("🚀 V6 ABLATION STUDY: FEATURE MODULE IMPORTANCE")
    print("=" * 60)
    print("  Prepping base dataset...")

    sales_df = load_sales()
    sub_raw = pd.read_csv(SUBMISSION_FILE, parse_dates=['Date'])
    
    sub_dummy = sub_raw.copy()
    sub_dummy['Revenue'] = np.nan
    
    full_df = pd.concat([sales_df, sub_dummy], ignore_index=True)
    full_df = build_features(full_df)

    internal_df = full_df[full_df['Revenue'].notna()].copy()
    internal_df['log_revenue'] = np.log1p(internal_df['Revenue'])

    # 1. Generate LRU Pred strictly for Train+Val to prevent leakage into the 2022 Test Ablation Set
    print("\n  >> Running LRU (State-Space) to generate Sequence Output (Strict Validation Mode)...")
    train_val_mask = internal_df['Date'] <= VAL_END
    y_train_val = internal_df[train_val_mask]['log_revenue'].values
    
    model_lru = train_lru(
        y_train=y_train_val,
        seq_len=SEQ_LEN,
        pred_len=PRED_LEN,
        hidden_size=HIDDEN_UNITS,
        num_blocks=NUM_BLOCKS,
        attn_dropout=ATTN_DROPOUT,
        ff_dropout=FF_DROPOUT,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,     
        lr=LEARNING_RATE,
        patience=PATIENCE
    )
    
    lru_feat = build_lru_feature(
        model=model_lru,
        y_train=y_train_val,
        full_len=len(full_df),
        seq_len=SEQ_LEN,
        pred_len=PRED_LEN
    )
    
    full_df['lru_pred'] = lru_feat
    
    # Re-extract and recalculate
    internal_df = full_df[full_df['Revenue'].notna()].copy()
    internal_df['log_revenue'] = np.log1p(internal_df['Revenue'])

    # 2. Setup Splits (Using identical logic as train.py)
    train_mask = internal_df['Date'] <= TRAIN_END
    val_mask = (internal_df['Date'] > TRAIN_END) & (internal_df['Date'] <= VAL_END)
    test_mask = internal_df['Date'] > VAL_END

    train_df = internal_df[train_mask]
    val_df = internal_df[val_mask]
    test_df = internal_df[test_mask]

    base_features = get_base_features(internal_df)
    
    # 3. Define Ablation Scenarios
    scenarios = {
        "1_All_Features (Baseline)": [],
        "2_No_LRU_Output": ['lru_pred'],
        "3_No_Auxiliary_Macros": ['season_avg_orders', 'season_avg_traffic', 'season_avg_bounce'],
        "4_No_Time_Encodings": ['month_sin', 'month_cos', 'dow_sin', 'dow_cos', 'season'],
        "5_No_Behavioral_Tet_Payday": ['days_to_tet', 'days_since_tet', 'is_tet_week', 'days_since_payday'],
        "6_No_Deep_Lags": [c for c in base_features if c.startswith('rev_lag_') or c.startswith('rev_rmean_') or c.startswith('rev_rstd_')]
    }

    results = []

    print("\n" + "=" * 60)
    print("🧪 BEGINNING ABLATION TESTS")
    print("=" * 60)

    for scenario_name, features_to_remove in scenarios.items():
        print(f"\n► Testing: {scenario_name}")
        
        # Filter available features
        test_features = [f for f in base_features if f not in features_to_remove]
        
        # We ensure target is not in features
        test_features = [f for f in test_features if f not in ['log_revenue', 'Revenue', 'Date', 'COGS']]

        X_t, y_t = train_df[test_features].values, train_df['log_revenue'].values
        X_v, y_v = val_df[test_features].values, val_df['log_revenue'].values
        X_test, y_true_test = test_df[test_features].values, test_df['Revenue'].values

        # Train Fast Regressor
        model_lgb = lgb.LGBMRegressor(n_estimators=1000, **LGBM_PARAMS)
        model_lgb.fit(X_t, y_t, eval_set=[(X_v, y_v)], callbacks=[lgb.early_stopping(50, verbose=False)])

        # Predict
        test_pred_log = model_lgb.predict(X_test)
        test_pred = np.maximum(np.expm1(test_pred_log), 0)

        # RMSE
        rmse = np.sqrt(mean_squared_error(y_true_test, test_pred))
        print(f"  Test RMSE: {rmse:,.0f}")
        
        results.append({
            'Scenario': scenario_name,
            'Features_Used': len(test_features),
            'Test_RMSE': rmse
        })

    # Summary
    print("\n" + "=" * 60)
    print("🏆 ABLATION STUDY RESULTS 🏆")
    print("=" * 60)
    
    results_df = pd.DataFrame(results).sort_values('Scenario')
    baseline_rmse = results_df[results_df['Scenario'] == '1_All_Features (Baseline)']['Test_RMSE'].values[0]
    
    results_df['RMSE_Difference'] = results_df['Test_RMSE'] - baseline_rmse
    results_df['Impact'] = results_df['RMSE_Difference'].apply(
        lambda x: "Baseline" if x == 0 else (f"+{x:,.0f} (Worse)" if x > 0 else f"{x:,.0f} (Better)")
    )
    
    print(results_df[['Scenario', 'Features_Used', 'Test_RMSE', 'Impact']].to_string(index=False))
    
    output_path = os.path.join(PROJECT_DIR, 'ablation_results_v6.csv')
    results_df.to_csv(output_path, index=False)
    print(f"\n  ✅ Results saved to {output_path}")

if __name__ == "__main__":
    run_ablation()
