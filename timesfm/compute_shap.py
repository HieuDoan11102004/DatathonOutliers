"""
SHAP & Feature Importance Analysis Script
Computes SHAP values and XGBoost feature importance for Revenue & COGS/Revenue Ratio models.
"""
import sys, os
sys.path.insert(0, os.getcwd())

import numpy as np
import pandas as pd
import shap
import json
from pathlib import Path

from ensemble_forecast import TARGETS, fit_seasonal_baseline, predict_seasonal
from ensemble_forecast_v2 import _feature_frame
from ratio_tuned_selection_v2 import _fit_ratio_models

OUT = Path("visual_outputs")
OUT.mkdir(exist_ok=True)

# 1. Load data
print("Loading data...", flush=True)
sales = pd.read_csv("sales.csv", parse_dates=["Date"])[["Date", *TARGETS]]
sales = sales.sort_values("Date").reset_index(drop=True)

# 2. Fit models (full training set)
print("Fitting seasonal baseline...", flush=True)
baseline = fit_seasonal_baseline(sales)

print("Fitting ratio models (feature_group=all, ratio_variant=base)...", flush=True)
models = _fit_ratio_models(sales, baseline, feature_group="all", ratio_variant="base")

revenue_model = models.revenue_model
ratio_model = models.ratio_model
feature_columns = models.columns

# 3. Build feature matrix
print("Building feature matrix...", flush=True)
seasonal = predict_seasonal(baseline, sales["Date"])
features = _feature_frame(seasonal, sales["Date"].min())[feature_columns]
print(f"Feature matrix shape: {features.shape}, columns: {len(feature_columns)}", flush=True)

# 4. XGBoost native feature importance (gain)
print("\n=== XGBoost Feature Importance (Gain) ===", flush=True)
for name, model in [("Revenue_Residual", revenue_model), ("COGS_Revenue_Ratio", ratio_model)]:
    imp = model.get_booster().get_score(importance_type="gain")
    imp_named = {}
    for k, v in imp.items():
        if k.startswith("f") and k[1:].isdigit():
            idx = int(k[1:])
            imp_named[feature_columns[idx]] = v
        else:
            imp_named[k] = v
    
    imp_sorted = sorted(imp_named.items(), key=lambda x: x[1], reverse=True)
    imp_df = pd.DataFrame(imp_sorted, columns=["feature", "importance_gain"])
    imp_df["importance_pct"] = (imp_df["importance_gain"] / imp_df["importance_gain"].sum() * 100).round(2)
    imp_df.to_csv(OUT / f"feature_importance_{name}.csv", index=False)
    
    print(f"\n{name} — Top 15 features:")
    for i, (feat, gain) in enumerate(imp_sorted[:15], 1):
        pct = gain / sum(v for _, v in imp_sorted) * 100
        print(f"  {i:2d}. {feat:35s} gain={gain:12.1f} ({pct:5.1f}%)")

# 5. SHAP values
print("\n=== SHAP Analysis ===", flush=True)
shap_results = {}

for name, model, target_label in [
    ("Revenue_Residual", revenue_model, "Revenue Residual (VND)"),
    ("COGS_Revenue_Ratio", ratio_model, "COGS/Revenue Ratio Residual"),
]:
    print(f"\nComputing SHAP for {name}...", flush=True)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(features)
    
    # Mean absolute SHAP values
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    shap_df = pd.DataFrame({
        "feature": feature_columns,
        "mean_abs_shap": mean_abs_shap,
        "mean_shap": shap_values.mean(axis=0),  # direction of effect
        "std_shap": shap_values.std(axis=0),
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    shap_df["importance_pct"] = (shap_df["mean_abs_shap"] / shap_df["mean_abs_shap"].sum() * 100).round(2)
    shap_df.to_csv(OUT / f"shap_values_{name}.csv", index=False)
    
    print(f"\n{name} — Top 15 SHAP features:")
    for i, row in shap_df.head(15).iterrows():
        direction = "↑" if row["mean_shap"] > 0 else "↓"
        print(f"  {i+1:2d}. {row['feature']:35s} |SHAP|={row['mean_abs_shap']:12.1f} ({row['importance_pct']:5.1f}%) {direction}")
    
    shap_results[name] = {
        "features": feature_columns,
        "shap_values": shap_values,
        "shap_df": shap_df,
    }

# 6. Feature correlation with targets
print("\n=== Feature-Target Correlation ===", flush=True)
corr_data = features.copy()
corr_data["Revenue"] = sales["Revenue"].values
corr_data["COGS"] = sales["COGS"].values
corr_data["COGS_Revenue_Ratio"] = sales["COGS"].values / np.maximum(sales["Revenue"].values, 1.0)

corr_rev = corr_data[feature_columns].corrwith(corr_data["Revenue"]).abs().sort_values(ascending=False)
corr_cogs = corr_data[feature_columns].corrwith(corr_data["COGS_Revenue_Ratio"]).abs().sort_values(ascending=False)

corr_df = pd.DataFrame({
    "feature": feature_columns,
    "corr_with_revenue": corr_data[feature_columns].corrwith(corr_data["Revenue"]).values,
    "abs_corr_revenue": corr_data[feature_columns].corrwith(corr_data["Revenue"]).abs().values,
    "corr_with_ratio": corr_data[feature_columns].corrwith(corr_data["COGS_Revenue_Ratio"]).values,
    "abs_corr_ratio": corr_data[feature_columns].corrwith(corr_data["COGS_Revenue_Ratio"]).abs().values,
}).sort_values("abs_corr_revenue", ascending=False)
corr_df.to_csv(OUT / f"feature_target_correlations.csv", index=False)

print("\nTop 10 features correlated with Revenue:")
for i, (feat, corr) in enumerate(corr_rev.head(10).items(), 1):
    print(f"  {i:2d}. {feat:35s} |r|={corr:.3f}")

# 7. Summary JSON for report
summary = {}
for name in ["Revenue_Residual", "COGS_Revenue_Ratio"]:
    shap_df = shap_results[name]["shap_df"]
    summary[name] = {
        "top_features": [
            {
                "rank": i+1,
                "feature": row["feature"],
                "importance_pct": row["importance_pct"],
                "direction": "positive" if row["mean_shap"] > 0 else "negative",
                "mean_abs_shap": round(row["mean_abs_shap"], 2),
            }
            for i, row in shap_df.head(20).iterrows()
        ]
    }

with open(OUT / "shap_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"\nAll outputs saved to {OUT}/")
print("Done!", flush=True)
