"""
ensemble_forecast_v3.py — Seasonal baseline v3
================================================
Cải tiến so với v2 (ensemble_forecast.py hiện tại):
  1. COVID exclusion: loại bỏ hoàn toàn 2020-2021 khỏi seasonal factor
  2. Grid-searchable decay: thêm DECAY_GRID để search tối ưu
  3. Recency-weighted seasonal profile (kế thừa từ v2)
  4. Growth rate tính từ non-COVID years only

Cách dùng:
  - Thay `from ensemble_forecast import ...` thành
    `from ensemble_forecast_v3 import ...` trong các file khác.
  - Hoặc chạy trực tiếp: `python ensemble_forecast_v3.py`
    để so sánh decay values trên walk-forward validation.
"""
from __future__ import annotations

import calendar
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

TARGETS = ("Revenue", "COGS")
DEFAULT_RESIDUAL_WEIGHT = 0.4

# ── Seasonal baseline config ────────────────────────────────────
SEASONAL_DECAY = 0.92
COVID_EXCLUDE_YEARS = {2020, 2021}   # Bỏ hẳn khỏi seasonal factors
GROWTH_EXCLUDE_YEARS = {2020, 2021}  # Bỏ hẳn khỏi growth estimation

# Grid search cho decay (dùng trong main/walk-forward)
DECAY_GRID = (0.85, 0.88, 0.90, 0.92, 0.95, 1.0)


@dataclass(frozen=True)
class SeasonalBaseline:
    base_year: int
    base_daily_mean: dict[str, float]
    growth: dict[str, float]
    seasonal_profile: pd.DataFrame


def _days_in_year(year: int) -> int:
    return 366 if calendar.isleap(int(year)) else 365


def _geometric_growth(series: pd.Series) -> float:
    yoy = series.pct_change().dropna()
    if yoy.empty:
        return 1.0
    return float((1.0 + yoy).prod() ** (1.0 / len(yoy)))


def _add_date_parts(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    d = out["Date"]
    out["year"] = d.dt.year
    out["quarter"] = d.dt.quarter
    out["month"] = d.dt.month
    out["day"] = d.dt.day
    out["week_of_year"] = d.dt.isocalendar().week.astype("int16")
    out["day_of_year"] = d.dt.dayofyear
    out["day_of_week"] = d.dt.dayofweek
    out["is_weekend"] = (out["day_of_week"] >= 5).astype("int8")
    out["is_month_start"] = d.dt.is_month_start.astype("int8")
    out["is_month_end"] = d.dt.is_month_end.astype("int8")
    out["month_sin"] = np.sin(2.0 * np.pi * (out["month"] - 1) / 12.0)
    out["month_cos"] = np.cos(2.0 * np.pi * (out["month"] - 1) / 12.0)
    out["dow_sin"] = np.sin(2.0 * np.pi * out["day_of_week"] / 7.0)
    out["dow_cos"] = np.cos(2.0 * np.pi * out["day_of_week"] / 7.0)
    out["doy_sin"] = np.sin(2.0 * np.pi * out["day_of_year"] / 365.25)
    out["doy_cos"] = np.cos(2.0 * np.pi * out["day_of_year"] / 365.25)
    return out


def fit_seasonal_baseline(
    train: pd.DataFrame,
    decay: float = SEASONAL_DECAY,
    covid_exclude: set[int] | None = None,
    growth_exclude: set[int] | None = None,
) -> SeasonalBaseline:
    """
    Fit seasonal baseline với 2 cải tiến:
    1. COVID exclusion: loại bỏ covid_exclude years khỏi seasonal factors
    2. Recency weighting: năm gần hơn có weight lớn hơn (decay)

    Args:
        train: DataFrame với columns [Date, Revenue, COGS]
        decay: weight multiplier mỗi năm lùi lại (1.0 = equal weight)
        covid_exclude: set các năm bỏ khỏi seasonal factor (default: {2020, 2021})
        growth_exclude: set các năm bỏ khỏi growth estimation (default: {2020, 2021})
    """
    if covid_exclude is None:
        covid_exclude = COVID_EXCLUDE_YEARS
    if growth_exclude is None:
        growth_exclude = GROWTH_EXCLUDE_YEARS

    data = _add_date_parts(train.sort_values("Date").reset_index(drop=True))
    counts = data.groupby("year").size()
    full_years = [int(year) for year, n_days in counts.items()
                  if n_days == _days_in_year(int(year))]
    if not full_years:
        raise ValueError("No complete year found for seasonal baseline")

    annual = data.groupby("year")[list(TARGETS)].sum()
    annual_full = annual.loc[full_years]
    base_year = max(full_years)

    # ── Growth rate: loại bỏ COVID years ─────────────────────────
    non_covid_years = [y for y in full_years if y not in growth_exclude]
    if len(non_covid_years) >= 2:
        growth = {
            target: _geometric_growth(annual_full.loc[non_covid_years, target])
            for target in TARGETS
        }
    else:
        # Fallback: dùng tất cả
        growth = {target: _geometric_growth(annual_full[target]) for target in TARGETS}

    base_daily_mean = {
        target: float(annual.loc[base_year, target]) / _days_in_year(base_year)
        for target in TARGETS
    }

    # ── Compute per-row seasonal factors ─────────────────────────
    annual_means = data.groupby("year")[list(TARGETS)].transform("mean")
    for target in TARGETS:
        data[f"{target}_seasonal_factor"] = data[target] / annual_means[target]

    # ── Recency-weighted seasonal profile (exclude COVID) ────────
    factor_cols = [f"{target}_seasonal_factor" for target in TARGETS]

    # Loại bỏ COVID years khỏi seasonal factor computation
    seasonal_data = data[~data["year"].isin(covid_exclude)].copy()

    if seasonal_data.empty:
        # Fallback nếu loại hết
        seasonal_data = data.copy()

    seasonal_data["_year_weight"] = decay ** (base_year - seasonal_data["year"])

    records = []
    for (m, d), group in seasonal_data.groupby(["month", "day"]):
        weights = group["_year_weight"].values
        record = {"month": m, "day": d}
        for col in factor_cols:
            vals = group[col].values
            mask = np.isfinite(vals) & np.isfinite(weights)
            if mask.any():
                record[col] = float(np.average(vals[mask], weights=weights[mask]))
            else:
                record[col] = 1.0
        records.append(record)
    seasonal_profile = pd.DataFrame(records)

    return SeasonalBaseline(
        base_year=base_year,
        base_daily_mean=base_daily_mean,
        growth=growth,
        seasonal_profile=seasonal_profile,
    )


def predict_seasonal(model: SeasonalBaseline, dates: pd.Series) -> pd.DataFrame:
    pred = pd.DataFrame({"Date": pd.to_datetime(dates)})
    pred = _add_date_parts(pred)
    pred["years_ahead"] = pred["year"] - model.base_year
    pred = pred.merge(model.seasonal_profile, on=["month", "day"], how="left")

    for target in TARGETS:
        factor_col = f"{target}_seasonal_factor"
        pred[factor_col] = pred[factor_col].fillna(1.0)
        pred[f"{target}_seasonal"] = (
            model.base_daily_mean[target]
            * (model.growth[target] ** pred["years_ahead"])
            * pred[factor_col]
        )

    return pred


def _residual_features(seasonal_pred: pd.DataFrame, start_date: pd.Timestamp) -> pd.DataFrame:
    out = seasonal_pred.copy()
    out["days_since_start"] = (out["Date"] - start_date).dt.days
    out["years_since_start"] = out["days_since_start"] / 365.25
    feature_cols = [
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
    ]
    return out[feature_cols].astype(float)


def fit_residual_models(train: pd.DataFrame, baseline: SeasonalBaseline) -> dict[str, XGBRegressor]:
    train_sorted = train.sort_values("Date").reset_index(drop=True)
    seasonal_in_sample = predict_seasonal(baseline, train_sorted["Date"])
    X = _residual_features(seasonal_in_sample, train_sorted["Date"].min())

    models: dict[str, XGBRegressor] = {}
    for target in TARGETS:
        residual = train_sorted[target].to_numpy(dtype=float) - seasonal_in_sample[
            f"{target}_seasonal"
        ].to_numpy(dtype=float)
        model = XGBRegressor(
            objective="reg:squarederror",
            n_estimators=350,
            learning_rate=0.03,
            max_depth=3,
            min_child_weight=8,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.2,
            reg_lambda=4.0,
            tree_method="hist",
            random_state=42,
            n_jobs=4,
        )
        model.fit(X, residual)
        models[target] = model
    return models


def predict_ensemble(
    baseline: SeasonalBaseline,
    residual_models: dict[str, XGBRegressor],
    train_start: pd.Timestamp,
    dates: pd.Series,
    residual_weight: float = DEFAULT_RESIDUAL_WEIGHT,
) -> pd.DataFrame:
    seasonal = predict_seasonal(baseline, dates)
    X = _residual_features(seasonal, train_start)

    out = seasonal[["Date"]].copy()
    for target in TARGETS:
        residual = residual_models[target].predict(X)
        pred = seasonal[f"{target}_seasonal"].to_numpy(dtype=float) + (
            residual_weight * residual
        )
        out[target] = np.maximum(pred, 0.0).round(2)
    return out


def _mae(actual: pd.Series, pred: pd.Series) -> float:
    return float(np.mean(np.abs(actual.to_numpy(dtype=float) - pred.to_numpy(dtype=float))))


def _mape(actual: pd.Series, pred: pd.Series) -> float:
    actual_arr = actual.to_numpy(dtype=float)
    pred_arr = pred.to_numpy(dtype=float)
    return float(np.mean(np.abs((actual_arr - pred_arr) / actual_arr)) * 100.0)


def walk_forward_validate(
    sales: pd.DataFrame,
    n_splits: int = 3,
    valid_days: int = 90,
    residual_weight: float = DEFAULT_RESIDUAL_WEIGHT,
    decay: float = SEASONAL_DECAY,
    covid_exclude: set[int] | None = None,
    growth_exclude: set[int] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    sales = sales.sort_values("Date").reset_index(drop=True)
    n_rows = len(sales)

    for fold in range(n_splits):
        valid_end = n_rows - (n_splits - fold - 1) * valid_days
        valid_start = valid_end - valid_days
        train = sales.iloc[:valid_start].copy()
        valid = sales.iloc[valid_start:valid_end].copy()

        baseline = fit_seasonal_baseline(train, decay=decay,
                                          covid_exclude=covid_exclude,
                                          growth_exclude=growth_exclude)
        residual_models = fit_residual_models(train, baseline)
        seasonal_pred = predict_seasonal(baseline, valid["Date"])
        ensemble_pred = predict_ensemble(
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
            row[f"{target}_ensemble_mae"] = _mae(valid[target], ensemble_pred[target])
            row[f"{target}_ensemble_mape"] = _mape(valid[target], ensemble_pred[target])
        rows.append(row)

    return pd.DataFrame(rows)


def build_ensemble_submission(
    train_file: str = "sales.csv",
    sample_file: str = "sample_submission.csv",
    output_file: str = "submission_ensemble.csv",
    residual_weight: float = DEFAULT_RESIDUAL_WEIGHT,
) -> pd.DataFrame:
    sales = pd.read_csv(train_file, parse_dates=["Date"])[["Date", *TARGETS]]
    sample = pd.read_csv(sample_file, parse_dates=["Date"])[["Date"]]

    baseline = fit_seasonal_baseline(sales)
    residual_models = fit_residual_models(sales, baseline)
    submission = predict_ensemble(
        baseline=baseline,
        residual_models=residual_models,
        train_start=sales["Date"].min(),
        dates=sample["Date"],
        residual_weight=residual_weight,
    )
    submission["Date"] = submission["Date"].dt.strftime("%Y-%m-%d")
    submission.to_csv(output_file, index=False)
    return submission


# ─────────────────────────────────────────────────────────────────
#  MAIN: Grid search decay + COVID exclusion trên walk-forward CV
# ─────────────────────────────────────────────────────────────────

def main() -> None:
    residual_weight = float(os.getenv("RESIDUAL_WEIGHT", str(DEFAULT_RESIDUAL_WEIGHT)))
    sales = pd.read_csv("sales.csv", parse_dates=["Date"])[["Date", *TARGETS]]

    print("=" * 70)
    print("SEASONAL BASELINE V3 — Grid Search: decay × COVID exclusion")
    print("=" * 70)
    print(f"residual_weight={residual_weight}")
    print(f"DECAY_GRID={DECAY_GRID}")
    print(f"COVID_EXCLUDE_YEARS={COVID_EXCLUDE_YEARS}")
    print()

    results = []

    # ── Config 1: Decay grid WITH COVID exclusion ────────────────
    for decay in DECAY_GRID:
        label = f"decay={decay:.2f}_no_covid"
        print(f"  Testing {label} ...", end="", flush=True)
        cv = walk_forward_validate(
            sales,
            residual_weight=residual_weight,
            decay=decay,
            covid_exclude=COVID_EXCLUDE_YEARS,
            growth_exclude=GROWTH_EXCLUDE_YEARS,
        )
        metric_cols = [c for c in cv.columns if c.endswith("mape")]
        means = cv[metric_cols].mean()
        avg_mape = (means.get("Revenue_ensemble_mape", 99) + means.get("COGS_ensemble_mape", 99)) / 2
        results.append({
            "config": label,
            "decay": decay,
            "covid_excluded": True,
            "Revenue_mape": means.get("Revenue_ensemble_mape", 99),
            "COGS_mape": means.get("COGS_ensemble_mape", 99),
            "avg_mape": avg_mape,
        })
        print(f"  avg_mape={avg_mape:.4f}")

    # ── Config 2: Decay grid WITHOUT COVID exclusion (baseline) ──
    for decay in DECAY_GRID:
        label = f"decay={decay:.2f}_all_years"
        print(f"  Testing {label} ...", end="", flush=True)
        cv = walk_forward_validate(
            sales,
            residual_weight=residual_weight,
            decay=decay,
            covid_exclude=set(),       # No exclusion
            growth_exclude=set(),
        )
        metric_cols = [c for c in cv.columns if c.endswith("mape")]
        means = cv[metric_cols].mean()
        avg_mape = (means.get("Revenue_ensemble_mape", 99) + means.get("COGS_ensemble_mape", 99)) / 2
        results.append({
            "config": label,
            "decay": decay,
            "covid_excluded": False,
            "Revenue_mape": means.get("Revenue_ensemble_mape", 99),
            "COGS_mape": means.get("COGS_ensemble_mape", 99),
            "avg_mape": avg_mape,
        })
        print(f"  avg_mape={avg_mape:.4f}")

    # ── Leaderboard ──────────────────────────────────────────────
    results_df = pd.DataFrame(results).sort_values("avg_mape")
    print("\n" + "=" * 70)
    print("LEADERBOARD (sorted by avg_mape)")
    print("=" * 70)
    print(results_df.round(4).to_string(index=False))

    best = results_df.iloc[0]
    print(f"\n✅ BEST: {best['config']}  avg_mape={best['avg_mape']:.4f}")

    Path("visual_outputs").mkdir(exist_ok=True)
    results_df.to_csv("visual_outputs/seasonal_v3_grid_search.csv", index=False)
    print("Saved to visual_outputs/seasonal_v3_grid_search.csv")


if __name__ == "__main__":
    main()
