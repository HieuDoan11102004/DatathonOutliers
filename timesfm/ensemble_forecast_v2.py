from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from ensemble_forecast import TARGETS, _mae, _mape, fit_seasonal_baseline, predict_seasonal
from external_features import FEATURE_COLUMNS, build_external_features

DEFAULT_RESIDUAL_WEIGHT = 0.4


def _feature_frame(seasonal_pred: pd.DataFrame, start_date: pd.Timestamp) -> pd.DataFrame:
    external = build_external_features(seasonal_pred["Date"])
    out = seasonal_pred.merge(external, on="Date", how="left")
    out["days_since_start"] = (out["Date"] - start_date).dt.days
    out["years_since_start"] = out["days_since_start"] / 365.25

    cols = [
        "year",
        "quarter",
        "month",
        "week_of_year",
        "day_of_year",
        "day_of_week",
        "is_weekend",
        "is_month_start",
        "is_month_end",
        "month_sin",
        "month_cos",
        "dow_sin",
        "dow_cos",
        "doy_sin",
        "doy_cos",
        "years_since_start",
        "Revenue_seasonal",
        "COGS_seasonal",
        *FEATURE_COLUMNS,
    ]
    return out[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(float)


def fit_residual_models_v2(train: pd.DataFrame, baseline) -> dict[str, XGBRegressor]:
    train_sorted = train.sort_values("Date").reset_index(drop=True)
    seasonal = predict_seasonal(baseline, train_sorted["Date"])
    features = _feature_frame(seasonal, train_sorted["Date"].min())

    models: dict[str, XGBRegressor] = {}
    for target in TARGETS:
        residual = train_sorted[target].to_numpy(dtype=float) - seasonal[
            f"{target}_seasonal"
        ].to_numpy(dtype=float)
        model = XGBRegressor(
            objective="reg:squarederror",
            n_estimators=450,
            learning_rate=0.025,
            max_depth=3,
            min_child_weight=10,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.3,
            reg_lambda=5.0,
            tree_method="hist",
            random_state=42,
            n_jobs=4,
        )
        model.fit(features, residual)
        models[target] = model
    return models


def predict_ensemble_v2(
    baseline,
    residual_models: dict[str, XGBRegressor],
    train_start: pd.Timestamp,
    dates: pd.Series,
    residual_weight: float = DEFAULT_RESIDUAL_WEIGHT,
) -> pd.DataFrame:
    seasonal = predict_seasonal(baseline, dates)
    features = _feature_frame(seasonal, train_start)

    out = seasonal[["Date"]].copy()
    for target in TARGETS:
        residual = residual_models[target].predict(features)
        pred = seasonal[f"{target}_seasonal"].to_numpy(dtype=float) + (
            residual_weight * residual
        )
        out[target] = np.maximum(pred, 0.0).round(2)
    return out


def walk_forward_validate_v2(
    sales: pd.DataFrame,
    n_splits: int = 3,
    valid_days: int = 90,
    residual_weight: float = DEFAULT_RESIDUAL_WEIGHT,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    sales = sales.sort_values("Date").reset_index(drop=True)
    n_rows = len(sales)

    for fold in range(n_splits):
        valid_end = n_rows - (n_splits - fold - 1) * valid_days
        valid_start = valid_end - valid_days
        train = sales.iloc[:valid_start].copy()
        valid = sales.iloc[valid_start:valid_end].copy()

        baseline = fit_seasonal_baseline(train)
        residual_models = fit_residual_models_v2(train, baseline)
        seasonal_pred = predict_seasonal(baseline, valid["Date"])
        v2_pred = predict_ensemble_v2(
            baseline=baseline,
            residual_models=residual_models,
            train_start=train["Date"].min(),
            dates=valid["Date"],
            residual_weight=residual_weight,
        )

        row: dict[str, object] = {
            "fold": fold + 1,
            "train_end": train["Date"].max().date(),
            "valid_start": valid["Date"].min().date(),
            "valid_end": valid["Date"].max().date(),
        }
        for target in TARGETS:
            row[f"{target}_seasonal_mae"] = _mae(
                valid[target], seasonal_pred[f"{target}_seasonal"]
            )
            row[f"{target}_seasonal_mape"] = _mape(
                valid[target], seasonal_pred[f"{target}_seasonal"]
            )
            row[f"{target}_v2_mae"] = _mae(valid[target], v2_pred[target])
            row[f"{target}_v2_mape"] = _mape(valid[target], v2_pred[target])
        rows.append(row)

    return pd.DataFrame(rows)


def build_submission_v2(
    output_file: str = "submission_ensemble_v2.csv",
    residual_weight: float = DEFAULT_RESIDUAL_WEIGHT,
) -> pd.DataFrame:
    sales = pd.read_csv("sales.csv", parse_dates=["Date"])[["Date", *TARGETS]]
    sample = pd.read_csv("sample_submission.csv", parse_dates=["Date"])[["Date"]]

    baseline = fit_seasonal_baseline(sales)
    residual_models = fit_residual_models_v2(sales, baseline)
    submission = predict_ensemble_v2(
        baseline=baseline,
        residual_models=residual_models,
        train_start=sales["Date"].min(),
        dates=sample["Date"],
        residual_weight=residual_weight,
    )
    submission["Date"] = submission["Date"].dt.strftime("%Y-%m-%d")
    submission = submission[["Date", "Revenue", "COGS"]]
    submission = submission.fillna(0.0)
    submission.to_csv(output_file, index=False)
    return submission


def main() -> None:
    residual_weight = float(os.getenv("RESIDUAL_WEIGHT", str(DEFAULT_RESIDUAL_WEIGHT)))
    sales = pd.read_csv("sales.csv", parse_dates=["Date"])[["Date", *TARGETS]]

    cv = walk_forward_validate_v2(
        sales,
        n_splits=3,
        valid_days=90,
        residual_weight=residual_weight,
    )
    print(f"residual_weight={residual_weight}")
    print("walk-forward validation (seasonal vs v2)")
    print(cv.round(4).to_string(index=False))

    metric_cols = [col for col in cv.columns if col.endswith(("mae", "mape"))]
    print("\nmean metrics")
    print(cv[metric_cols].mean().round(4).to_string())

    output_path = Path("submission_ensemble_v2.csv")
    submission = build_submission_v2(
        output_file=str(output_path),
        residual_weight=residual_weight,
    )
    print(f"\nsaved {len(submission)} rows to {output_path}")
    print(submission.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
