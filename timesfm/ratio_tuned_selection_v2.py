# from __future__ import annotations

# from dataclasses import dataclass
# from pathlib import Path
# from time import perf_counter
# from typing import Iterable

# import numpy as np
# import pandas as pd
# from xgboost import XGBRegressor

# from ensemble_forecast import TARGETS, _mae, _mape, fit_seasonal_baseline, predict_seasonal
# from ensemble_forecast_v2 import _feature_frame

# REVENUE_WEIGHT_GRID = (1.15, 1.20, 1.25, 1.30, 1.35, 1.40, 1.45, 1.50, 1.55)
# RATIO_WEIGHT_GRID = (0.95, 1.00, 1.05, 1.10, 1.15, 1.20, 1.25, 1.30)
# SCENARIOS = (
#     ("90d_3fold", 3, 90, 0.10),
#     ("180d_3fold", 3, 180, 0.20),
#     ("365d_1fold", 1, 365, 0.25),
#     ("548d_1fold", 1, 548, 0.45),
# )
# FEATURE_GROUPS = ("all", "no_web")
# CLIP_RANGES = {
#     "wide": (0.0, 1.2),
#     "business": (0.60, 1.20),
#     "mid": (0.70, 1.10),
# }
# RATIO_PARAM_VARIANTS = {
#     "base": {
#         "n_estimators": 240,
#         "learning_rate": 0.03,
#         "max_depth": 3,
#         "min_child_weight": 10,
#         "reg_alpha": 0.3,
#         "reg_lambda": 5.0,
#     },
#     "conservative": {
#         "n_estimators": 240,
#         "learning_rate": 0.025,
#         "max_depth": 2,
#         "min_child_weight": 20,
#         "reg_alpha": 0.8,
#         "reg_lambda": 12.0,
#     },
#     "flex": {
#         "n_estimators": 260,
#         "learning_rate": 0.025,
#         "max_depth": 3,
#         "min_child_weight": 6,
#         "reg_alpha": 0.2,
#         "reg_lambda": 4.0,
#     },
# }

# WEB_COLUMNS = {
#     "sessions",
#     "unique_visitors",
#     "page_views",
#     "bounce_rate",
#     "avg_session_duration_sec",
# }
# FEATURE_ABLATIONS = {
#     "all": set(),
#     "no_web": WEB_COLUMNS,
# }


# @dataclass(frozen=True)
# class RatioModels:
#     feature_group: str
#     ratio_variant: str
#     columns: list[str]
#     revenue_model: XGBRegressor
#     ratio_model: XGBRegressor


# @dataclass(frozen=True)
# class PredictionCache:
#     dates: pd.Series
#     revenue_base: np.ndarray
#     cogs_base: np.ndarray
#     revenue_residual: np.ndarray
#     ratio_residual: np.ndarray


# def _make_revenue_xgb() -> XGBRegressor:
#     return XGBRegressor(
#         objective="reg:squarederror",
#         n_estimators=240,
#         learning_rate=0.03,
#         max_depth=2,
#         min_child_weight=10,
#         subsample=0.85,
#         colsample_bytree=0.85,
#         reg_alpha=0.3,
#         reg_lambda=5.0,
#         tree_method="hist",
#         random_state=43,
#         n_jobs=4,
#     )


# def _make_ratio_xgb(variant: str) -> XGBRegressor:
#     return XGBRegressor(
#         objective="reg:squarederror",
#         subsample=0.85,
#         colsample_bytree=0.85,
#         tree_method="hist",
#         random_state=44,
#         n_jobs=4,
#         **RATIO_PARAM_VARIANTS[variant],
#     )


# def _iter_folds(sales: pd.DataFrame) -> Iterable[tuple[str, int, int, float, pd.DataFrame, pd.DataFrame]]:
#     sales = sales.sort_values("Date").reset_index(drop=True)
#     n_rows = len(sales)
#     for scenario, n_splits, valid_days, scenario_weight in SCENARIOS:
#         for fold in range(n_splits):
#             valid_end = n_rows - (n_splits - fold - 1) * valid_days
#             valid_start = valid_end - valid_days
#             if valid_start <= 365 or valid_end > n_rows or valid_start >= valid_end:
#                 continue
#             yield (
#                 scenario,
#                 fold + 1,
#                 valid_days,
#                 scenario_weight,
#                 sales.iloc[:valid_start].copy(),
#                 sales.iloc[valid_start:valid_end].copy(),
#             )


# def _select_columns(features: pd.DataFrame, feature_group: str) -> list[str]:
#     removed = FEATURE_ABLATIONS[feature_group]
#     return [col for col in features.columns if col not in removed]


# def _fit_ratio_models(
#     train: pd.DataFrame,
#     baseline,
#     feature_group: str,
#     ratio_variant: str,
# ) -> RatioModels:
#     train_sorted = train.sort_values("Date").reset_index(drop=True)
#     seasonal = predict_seasonal(baseline, train_sorted["Date"])
#     features_all = _feature_frame(seasonal, train_sorted["Date"].min())
#     columns = _select_columns(features_all, feature_group)
#     features = features_all[columns]

#     revenue_residual = train_sorted["Revenue"].to_numpy(dtype=float) - seasonal[
#         "Revenue_seasonal"
#     ].to_numpy(dtype=float)
#     revenue_model = _make_revenue_xgb()
#     revenue_model.fit(features, revenue_residual)

#     actual_ratio = train_sorted["COGS"].to_numpy(dtype=float) / np.maximum(
#         train_sorted["Revenue"].to_numpy(dtype=float), 1.0
#     )
#     seasonal_ratio = seasonal["COGS_seasonal"].to_numpy(dtype=float) / np.maximum(
#         seasonal["Revenue_seasonal"].to_numpy(dtype=float), 1.0
#     )
#     ratio_residual = actual_ratio - seasonal_ratio
#     ratio_model = _make_ratio_xgb(ratio_variant)
#     ratio_model.fit(features, ratio_residual)

#     return RatioModels(
#         feature_group=feature_group,
#         ratio_variant=ratio_variant,
#         columns=columns,
#         revenue_model=revenue_model,
#         ratio_model=ratio_model,
#     )


# def _prediction_cache(
#     baseline,
#     models: RatioModels,
#     train_start: pd.Timestamp,
#     dates: pd.Series,
# ) -> PredictionCache:
#     seasonal = predict_seasonal(baseline, dates)
#     features = _feature_frame(seasonal, train_start)[models.columns]
#     return PredictionCache(
#         dates=seasonal["Date"],
#         revenue_base=seasonal["Revenue_seasonal"].to_numpy(dtype=float),
#         cogs_base=seasonal["COGS_seasonal"].to_numpy(dtype=float),
#         revenue_residual=models.revenue_model.predict(features),
#         ratio_residual=models.ratio_model.predict(features),
#     )


# def _predict_from_cache(
#     cache: PredictionCache,
#     revenue_weight: float,
#     ratio_weight: float,
#     clip_low: float,
#     clip_high: float,
# ) -> pd.DataFrame:
#     revenue = np.maximum(cache.revenue_base + revenue_weight * cache.revenue_residual, 0.0)
#     seasonal_ratio = cache.cogs_base / np.maximum(cache.revenue_base, 1.0)
#     ratio = np.clip(seasonal_ratio + ratio_weight * cache.ratio_residual, clip_low, clip_high)
#     out = pd.DataFrame({"Date": cache.dates})
#     out["Revenue"] = np.round(revenue, 2)
#     out["COGS"] = np.round(np.maximum(revenue * ratio, 0.0), 2)
#     return out


# def _score(
#     pred: pd.DataFrame,
#     valid: pd.DataFrame,
#     scenario: str,
#     fold: int,
#     valid_days: int,
#     scenario_weight: float,
#     candidate: str,
#     feature_group: str,
#     ratio_variant: str,
#     clip_name: str,
#     revenue_weight: float,
#     ratio_weight: float,
# ) -> dict[str, float | str | int]:
#     row: dict[str, float | str | int] = {
#         "scenario": scenario,
#         "fold": fold,
#         "valid_days": valid_days,
#         "scenario_weight": scenario_weight,
#         "valid_start": valid["Date"].min().date(),
#         "valid_end": valid["Date"].max().date(),
#         "candidate": candidate,
#         "feature_group": feature_group,
#         "ratio_variant": ratio_variant,
#         "clip_name": clip_name,
#         "revenue_weight": revenue_weight,
#         "ratio_weight": ratio_weight,
#     }
#     for target in TARGETS:
#         row[f"{target}_mae"] = _mae(valid[target], pred[target])
#         row[f"{target}_mape"] = _mape(valid[target], pred[target])
#     row["avg_mape"] = (row["Revenue_mape"] + row["COGS_mape"]) / 2.0  # type: ignore[operator]
#     row["avg_mae"] = (row["Revenue_mae"] + row["COGS_mae"]) / 2.0  # type: ignore[operator]
#     return row


# def _fold_count(sales: pd.DataFrame) -> int:
#     return sum(1 for _ in _iter_folds(sales))


# def _candidate_count_per_model() -> int:
#     return len(CLIP_RANGES) * len(REVENUE_WEIGHT_GRID) * len(RATIO_WEIGHT_GRID)


# def evaluate_ratio_tuned_v2(sales: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
#     rows: list[dict[str, float | str | int]] = []
#     started_at = perf_counter()
#     total_folds = _fold_count(sales)
#     candidates_per_model = _candidate_count_per_model()
#     models_per_fold = len(FEATURE_GROUPS) * len(RATIO_PARAM_VARIANTS)
#     total_candidates = total_folds * models_per_fold * candidates_per_model
#     completed = 0
#     print(
#         f"ratio_tuned_v2: folds={total_folds} models_per_fold={models_per_fold} "
#         f"candidates_per_model={candidates_per_model} total_candidates={total_candidates}",
#         flush=True,
#     )

#     for fold_index, (scenario, fold, valid_days, scenario_weight, train, valid) in enumerate(
#         _iter_folds(sales), start=1
#     ):
#         fold_start = perf_counter()
#         print(
#             f"[fold {fold_index}/{total_folds}] start scenario={scenario} fold={fold} "
#             f"train_end={train['Date'].max().date()} valid={valid['Date'].min().date()}..{valid['Date'].max().date()}",
#             flush=True,
#         )
#         baseline = fit_seasonal_baseline(train)
#         train_start = train["Date"].min()
#         valid_dates = valid["Date"]

#         for feature_group in FEATURE_GROUPS:
#             for ratio_variant in RATIO_PARAM_VARIANTS:
#                 fit_start = perf_counter()
#                 print(
#                     f"[fold {fold_index}/{total_folds}] fitting feature_group={feature_group} "
#                     f"ratio_variant={ratio_variant}",
#                     flush=True,
#                 )
#                 models = _fit_ratio_models(train, baseline, feature_group, ratio_variant)
#                 cache = _prediction_cache(baseline, models, train_start, valid_dates)
#                 print(
#                     f"[fold {fold_index}/{total_folds}] fitted+cached feature_group={feature_group} "
#                     f"ratio_variant={ratio_variant} elapsed={perf_counter() - fit_start:.1f}s",
#                     flush=True,
#                 )

#                 model_done = 0
#                 for clip_name, (clip_low, clip_high) in CLIP_RANGES.items():
#                     for rw in REVENUE_WEIGHT_GRID:
#                         for ratio_w in RATIO_WEIGHT_GRID:
#                             pred = _predict_from_cache(cache, rw, ratio_w, clip_low, clip_high)
#                             candidate = (
#                                 f"ratio_{feature_group}_{ratio_variant}_{clip_name}_"
#                                 f"rw{rw:.2f}_cw{ratio_w:.2f}"
#                             )
#                             rows.append(
#                                 _score(
#                                     pred,
#                                     valid,
#                                     scenario,
#                                     fold,
#                                     valid_days,
#                                     scenario_weight,
#                                     candidate,
#                                     feature_group,
#                                     ratio_variant,
#                                     clip_name,
#                                     rw,
#                                     ratio_w,
#                                 )
#                             )
#                             completed += 1
#                             model_done += 1
#                 elapsed = perf_counter() - started_at
#                 rate = completed / elapsed if elapsed > 0 else 0.0
#                 eta = (total_candidates - completed) / rate if rate > 0 else 0.0
#                 print(
#                     f"[progress] fold={fold_index}/{total_folds} model_candidates={model_done}/{candidates_per_model} "
#                     f"total={completed}/{total_candidates} elapsed={elapsed:.1f}s eta={eta:.1f}s",
#                     flush=True,
#                 )
#         print(
#             f"[fold {fold_index}/{total_folds}] done elapsed={perf_counter() - fold_start:.1f}s",
#             flush=True,
#         )

#     folds = pd.DataFrame(rows)
#     group_cols = [
#         "candidate",
#         "feature_group",
#         "ratio_variant",
#         "clip_name",
#         "revenue_weight",
#         "ratio_weight",
#     ]
#     metric_cols = ["Revenue_mae", "Revenue_mape", "COGS_mae", "COGS_mape", "avg_mae", "avg_mape"]
#     metric_summary = folds.groupby(group_cols, as_index=False)[metric_cols].mean()
#     scenario_summary = (
#         folds.groupby([*group_cols, "scenario"], as_index=False)
#         .agg(avg_mape=("avg_mape", "mean"), scenario_weight=("scenario_weight", "first"))
#     )
#     scenario_summary["weighted_component"] = (
#         scenario_summary["avg_mape"] * scenario_summary["scenario_weight"]
#     )
#     weighted = (
#         scenario_summary.groupby(group_cols, as_index=False)
#         .agg(
#             weighted_avg_mape=("weighted_component", "sum"),
#             scenario_weight_total=("scenario_weight", "sum"),
#             scenarios_covered=("scenario", "nunique"),
#         )
#     )
#     weighted["weighted_avg_mape"] = weighted["weighted_avg_mape"] / weighted[
#         "scenario_weight_total"
#     ]
#     coverage = folds.groupby(group_cols, as_index=False).agg(folds_evaluated=("fold", "size"))
#     leaderboard = (
#         metric_summary.merge(weighted, on=group_cols, how="left")
#         .merge(coverage, on=group_cols, how="left")
#         .sort_values(["weighted_avg_mape", "avg_mape", "avg_mae"], kind="stable")
#         .reset_index(drop=True)
#     )
#     return folds, leaderboard


# def build_best_submission(
#     best: pd.Series,
#     train_file: str = "sales.csv",
#     sample_file: str = "sample_submission.csv",
#     output_file: str = "submission_best_ratio_tuned_v2.csv",
# ) -> pd.DataFrame:
#     sales = pd.read_csv(train_file, parse_dates=["Date"])[["Date", *TARGETS]]
#     sample = pd.read_csv(sample_file, parse_dates=["Date"])[["Date"]]
#     baseline = fit_seasonal_baseline(sales)
#     models = _fit_ratio_models(
#         sales,
#         baseline,
#         feature_group=str(best["feature_group"]),
#         ratio_variant=str(best["ratio_variant"]),
#     )
#     cache = _prediction_cache(baseline, models, sales["Date"].min(), sample["Date"])
#     clip_low, clip_high = CLIP_RANGES[str(best["clip_name"])]
#     pred = _predict_from_cache(
#         cache,
#         revenue_weight=float(best["revenue_weight"]),
#         ratio_weight=float(best["ratio_weight"]),
#         clip_low=clip_low,
#         clip_high=clip_high,
#     )
#     pred["Date"] = pred["Date"].dt.strftime("%Y-%m-%d")
#     pred = pred[["Date", "Revenue", "COGS"]].round({"Revenue": 2, "COGS": 2})
#     pred.to_csv(output_file, index=False)
#     return pred


# def verify_submission(output_file: str = "submission_best_ratio_tuned_v2.csv") -> None:
#     sample = pd.read_csv("sample_submission.csv")
#     submission = pd.read_csv(output_file)
#     if list(submission.columns) != ["Date", "Revenue", "COGS"]:
#         raise AssertionError(f"Unexpected columns: {list(submission.columns)}")
#     if not submission["Date"].equals(sample["Date"]):
#         raise AssertionError("Submission Date values do not match sample_submission.csv")
#     if submission[["Revenue", "COGS"]].isna().any().any():
#         raise AssertionError("Submission contains missing target values")
#     if (submission[["Revenue", "COGS"]] < 0).any().any():
#         raise AssertionError("Submission contains negative target values")


# def main() -> None:
#     started_at = perf_counter()
#     print("ratio_tuned_v2: loading data", flush=True)
#     sales = pd.read_csv("sales.csv", parse_dates=["Date"])[["Date", *TARGETS]]
#     folds, leaderboard = evaluate_ratio_tuned_v2(sales)

#     output_dir = Path("visual_outputs")
#     output_dir.mkdir(exist_ok=True)
#     folds_path = output_dir / "ratio_tuned_selection_v2_folds.csv"
#     leaderboard_path = output_dir / "ratio_tuned_selection_v2_leaderboard.csv"
#     folds.to_csv(folds_path, index=False)
#     leaderboard.to_csv(leaderboard_path, index=False)

#     best = leaderboard.iloc[0]
#     submission = build_best_submission(best)
#     verify_submission()

#     print("\nratio tuned v2 leaderboard top 20")
#     print(leaderboard.head(20).round(4).to_string(index=False))
#     print(
#         f"\nbest={best['candidate']} weighted_avg_mape={best['weighted_avg_mape']:.4f} "
#         f"avg_mape={best['avg_mape']:.4f}",
#         flush=True,
#     )
#     print(f"saved {len(folds)} fold rows to {folds_path}")
#     print(f"saved {len(leaderboard)} candidates to {leaderboard_path}")
#     print(f"saved {len(submission)} rows to submission_best_ratio_tuned_v2.csv")
#     print(submission.head().to_string(index=False))
#     print(f"ratio_tuned_v2: total_elapsed={perf_counter() - started_at:.1f}s", flush=True)


# if __name__ == "__main__":
#     main()

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Iterable

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from ensemble_forecast import TARGETS, _mae, _mape, fit_seasonal_baseline, predict_seasonal
from ensemble_forecast_v2 import _feature_frame

REVENUE_WEIGHT_GRID = (1.15, 1.20, 1.25, 1.30, 1.35, 1.40, 1.45, 1.50, 1.55)
RATIO_WEIGHT_GRID = (0.95, 1.00, 1.05, 1.10, 1.15, 1.20, 1.25, 1.30)
SCENARIOS = (
    ("90d_3fold", 3, 90, 0.10),
    ("180d_3fold", 3, 180, 0.20),
    ("365d_1fold", 1, 365, 0.25),
    ("548d_1fold", 1, 548, 0.45),
)
FEATURE_GROUPS = ("all", "no_web")
CLIP_RANGES = {
    "wide": (0.0, 1.2),
    "business": (0.60, 1.20),
    "mid": (0.70, 1.10),
}
RATIO_PARAM_VARIANTS = {
    "base": {
        "n_estimators": 240,
        "learning_rate": 0.03,
        "max_depth": 3,
        "min_child_weight": 10,
        "reg_alpha": 0.3,
        "reg_lambda": 5.0,
    },
    "conservative": {
        "n_estimators": 240,
        "learning_rate": 0.025,
        "max_depth": 2,
        "min_child_weight": 20,
        "reg_alpha": 0.8,
        "reg_lambda": 12.0,
    },
    "flex": {
        "n_estimators": 260,
        "learning_rate": 0.025,
        "max_depth": 3,
        "min_child_weight": 6,
        "reg_alpha": 0.2,
        "reg_lambda": 4.0,
    },
}

WEB_COLUMNS = {
    "sessions",
    "unique_visitors",
    "page_views",
    "bounce_rate",
    "avg_session_duration_sec",
}
FEATURE_ABLATIONS = {
    "all": set(),
    "no_web": WEB_COLUMNS,
}


@dataclass(frozen=True)
class RatioModels:
    feature_group: str
    ratio_variant: str
    columns: list[str]
    revenue_model: XGBRegressor
    ratio_model: XGBRegressor


@dataclass(frozen=True)
class PredictionCache:
    dates: pd.Series
    revenue_base: np.ndarray
    cogs_base: np.ndarray
    revenue_residual: np.ndarray
    ratio_residual: np.ndarray


def _make_revenue_xgb() -> XGBRegressor:
    return XGBRegressor(
        objective="reg:squarederror",
        n_estimators=240,
        learning_rate=0.03,
        max_depth=3,
        min_child_weight=10,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.3,
        reg_lambda=5.0,
        tree_method="hist",
        random_state=43,
        n_jobs=4,
    )


def _make_ratio_xgb(variant: str) -> XGBRegressor:
    return XGBRegressor(
        objective="reg:squarederror",
        subsample=0.85,
        colsample_bytree=0.85,
        tree_method="hist",
        random_state=44,
        n_jobs=4,
        **RATIO_PARAM_VARIANTS[variant],
    )


def _iter_folds(sales: pd.DataFrame) -> Iterable[tuple[str, int, int, float, pd.DataFrame, pd.DataFrame]]:
    sales = sales.sort_values("Date").reset_index(drop=True)
    n_rows = len(sales)
    for scenario, n_splits, valid_days, scenario_weight in SCENARIOS:
        for fold in range(n_splits):
            valid_end = n_rows - (n_splits - fold - 1) * valid_days
            valid_start = valid_end - valid_days
            if valid_start <= 365 or valid_end > n_rows or valid_start >= valid_end:
                continue
            yield (
                scenario,
                fold + 1,
                valid_days,
                scenario_weight,
                sales.iloc[:valid_start].copy(),
                sales.iloc[valid_start:valid_end].copy(),
            )


def _select_columns(features: pd.DataFrame, feature_group: str) -> list[str]:
    removed = FEATURE_ABLATIONS[feature_group]
    return [col for col in features.columns if col not in removed]


def _fit_ratio_models(
    train: pd.DataFrame,
    baseline,
    feature_group: str,
    ratio_variant: str,
) -> RatioModels:
    train_sorted = train.sort_values("Date").reset_index(drop=True)
    seasonal = predict_seasonal(baseline, train_sorted["Date"])
    features_all = _feature_frame(seasonal, train_sorted["Date"].min())
    columns = _select_columns(features_all, feature_group)
    features = features_all[columns]

    revenue_residual = train_sorted["Revenue"].to_numpy(dtype=float) - seasonal[
        "Revenue_seasonal"
    ].to_numpy(dtype=float)
    # MAPE-aware sample weights: ngày doanh thu thấp được ưu tiên hơn
    revenue_values = np.maximum(train_sorted["Revenue"].to_numpy(dtype=float), 1.0)
    sample_w = 1.0 / revenue_values
    sample_w = sample_w / sample_w.mean()  # normalize để tránh gradient quá nhỏ
    revenue_model = _make_revenue_xgb()
    revenue_model.fit(features, revenue_residual, sample_weight=sample_w)

    actual_ratio = train_sorted["COGS"].to_numpy(dtype=float) / np.maximum(
        train_sorted["Revenue"].to_numpy(dtype=float), 1.0
    )
    seasonal_ratio = seasonal["COGS_seasonal"].to_numpy(dtype=float) / np.maximum(
        seasonal["Revenue_seasonal"].to_numpy(dtype=float), 1.0
    )
    ratio_residual = actual_ratio - seasonal_ratio
    ratio_model = _make_ratio_xgb(ratio_variant)
    ratio_model.fit(features, ratio_residual, sample_weight=sample_w)

    return RatioModels(
        feature_group=feature_group,
        ratio_variant=ratio_variant,
        columns=columns,
        revenue_model=revenue_model,
        ratio_model=ratio_model,
    )


def _prediction_cache(
    baseline,
    models: RatioModels,
    train_start: pd.Timestamp,
    dates: pd.Series,
) -> PredictionCache:
    seasonal = predict_seasonal(baseline, dates)
    features = _feature_frame(seasonal, train_start)[models.columns]
    return PredictionCache(
        dates=seasonal["Date"],
        revenue_base=seasonal["Revenue_seasonal"].to_numpy(dtype=float),
        cogs_base=seasonal["COGS_seasonal"].to_numpy(dtype=float),
        revenue_residual=models.revenue_model.predict(features),
        ratio_residual=models.ratio_model.predict(features),
    )


def _predict_from_cache(
    cache: PredictionCache,
    revenue_weight: float,
    ratio_weight: float,
    clip_low: float,
    clip_high: float,
) -> pd.DataFrame:
    revenue = np.maximum(cache.revenue_base + revenue_weight * cache.revenue_residual, 0.0)
    seasonal_ratio = cache.cogs_base / np.maximum(cache.revenue_base, 1.0)
    ratio = np.clip(seasonal_ratio + ratio_weight * cache.ratio_residual, clip_low, clip_high)
    out = pd.DataFrame({"Date": cache.dates})
    out["Revenue"] = np.round(revenue, 2)
    out["COGS"] = np.round(np.maximum(revenue * ratio, 0.0), 2)
    return out


def _score(
    pred: pd.DataFrame,
    valid: pd.DataFrame,
    scenario: str,
    fold: int,
    valid_days: int,
    scenario_weight: float,
    candidate: str,
    feature_group: str,
    ratio_variant: str,
    clip_name: str,
    revenue_weight: float,
    ratio_weight: float,
) -> dict[str, float | str | int]:
    row: dict[str, float | str | int] = {
        "scenario": scenario,
        "fold": fold,
        "valid_days": valid_days,
        "scenario_weight": scenario_weight,
        "valid_start": valid["Date"].min().date(),
        "valid_end": valid["Date"].max().date(),
        "candidate": candidate,
        "feature_group": feature_group,
        "ratio_variant": ratio_variant,
        "clip_name": clip_name,
        "revenue_weight": revenue_weight,
        "ratio_weight": ratio_weight,
    }
    for target in TARGETS:
        row[f"{target}_mae"] = _mae(valid[target], pred[target])
        row[f"{target}_mape"] = _mape(valid[target], pred[target])
    row["avg_mape"] = (row["Revenue_mape"] + row["COGS_mape"]) / 2.0  # type: ignore[operator]
    row["avg_mae"] = (row["Revenue_mae"] + row["COGS_mae"]) / 2.0  # type: ignore[operator]
    return row


def _fold_count(sales: pd.DataFrame) -> int:
    return sum(1 for _ in _iter_folds(sales))


def _candidate_count_per_model() -> int:
    return len(CLIP_RANGES) * len(REVENUE_WEIGHT_GRID) * len(RATIO_WEIGHT_GRID)


def evaluate_ratio_tuned_v2(sales: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, float | str | int]] = []
    started_at = perf_counter()
    total_folds = _fold_count(sales)
    candidates_per_model = _candidate_count_per_model()
    models_per_fold = len(FEATURE_GROUPS) * len(RATIO_PARAM_VARIANTS)
    total_candidates = total_folds * models_per_fold * candidates_per_model
    completed = 0
    print(
        f"ratio_tuned_v2: folds={total_folds} models_per_fold={models_per_fold} "
        f"candidates_per_model={candidates_per_model} total_candidates={total_candidates}",
        flush=True,
    )

    for fold_index, (scenario, fold, valid_days, scenario_weight, train, valid) in enumerate(
        _iter_folds(sales), start=1
    ):
        fold_start = perf_counter()
        print(
            f"[fold {fold_index}/{total_folds}] start scenario={scenario} fold={fold} "
            f"train_end={train['Date'].max().date()} valid={valid['Date'].min().date()}..{valid['Date'].max().date()}",
            flush=True,
        )
        baseline = fit_seasonal_baseline(train)
        train_start = train["Date"].min()
        valid_dates = valid["Date"]

        for feature_group in FEATURE_GROUPS:
            for ratio_variant in RATIO_PARAM_VARIANTS:
                fit_start = perf_counter()
                print(
                    f"[fold {fold_index}/{total_folds}] fitting feature_group={feature_group} "
                    f"ratio_variant={ratio_variant}",
                    flush=True,
                )
                models = _fit_ratio_models(train, baseline, feature_group, ratio_variant)
                cache = _prediction_cache(baseline, models, train_start, valid_dates)
                print(
                    f"[fold {fold_index}/{total_folds}] fitted+cached feature_group={feature_group} "
                    f"ratio_variant={ratio_variant} elapsed={perf_counter() - fit_start:.1f}s",
                    flush=True,
                )

                model_done = 0
                for clip_name, (clip_low, clip_high) in CLIP_RANGES.items():
                    for rw in REVENUE_WEIGHT_GRID:
                        for ratio_w in RATIO_WEIGHT_GRID:
                            pred = _predict_from_cache(cache, rw, ratio_w, clip_low, clip_high)
                            candidate = (
                                f"ratio_{feature_group}_{ratio_variant}_{clip_name}_"
                                f"rw{rw:.2f}_cw{ratio_w:.2f}"
                            )
                            rows.append(
                                _score(
                                    pred,
                                    valid,
                                    scenario,
                                    fold,
                                    valid_days,
                                    scenario_weight,
                                    candidate,
                                    feature_group,
                                    ratio_variant,
                                    clip_name,
                                    rw,
                                    ratio_w,
                                )
                            )
                            completed += 1
                            model_done += 1
                elapsed = perf_counter() - started_at
                rate = completed / elapsed if elapsed > 0 else 0.0
                eta = (total_candidates - completed) / rate if rate > 0 else 0.0
                print(
                    f"[progress] fold={fold_index}/{total_folds} model_candidates={model_done}/{candidates_per_model} "
                    f"total={completed}/{total_candidates} elapsed={elapsed:.1f}s eta={eta:.1f}s",
                    flush=True,
                )
        print(
            f"[fold {fold_index}/{total_folds}] done elapsed={perf_counter() - fold_start:.1f}s",
            flush=True,
        )

    folds = pd.DataFrame(rows)
    group_cols = [
        "candidate",
        "feature_group",
        "ratio_variant",
        "clip_name",
        "revenue_weight",
        "ratio_weight",
    ]
    metric_cols = ["Revenue_mae", "Revenue_mape", "COGS_mae", "COGS_mape", "avg_mae", "avg_mape"]
    metric_summary = folds.groupby(group_cols, as_index=False)[metric_cols].mean()
    scenario_summary = (
        folds.groupby([*group_cols, "scenario"], as_index=False)
        .agg(avg_mape=("avg_mape", "mean"), scenario_weight=("scenario_weight", "first"))
    )
    scenario_summary["weighted_component"] = (
        scenario_summary["avg_mape"] * scenario_summary["scenario_weight"]
    )
    weighted = (
        scenario_summary.groupby(group_cols, as_index=False)
        .agg(
            weighted_avg_mape=("weighted_component", "sum"),
            scenario_weight_total=("scenario_weight", "sum"),
            scenarios_covered=("scenario", "nunique"),
        )
    )
    weighted["weighted_avg_mape"] = weighted["weighted_avg_mape"] / weighted[
        "scenario_weight_total"
    ]
    coverage = folds.groupby(group_cols, as_index=False).agg(folds_evaluated=("fold", "size"))
    leaderboard = (
        metric_summary.merge(weighted, on=group_cols, how="left")
        .merge(coverage, on=group_cols, how="left")
        .sort_values(["weighted_avg_mape", "avg_mape", "avg_mae"], kind="stable")
        .reset_index(drop=True)
    )
    return folds, leaderboard


def build_best_submission(
    best: pd.Series,
    train_file: str = "sales.csv",
    sample_file: str = "sample_submission.csv",
    output_file: str = "submission_best_ratio_tuned_v2.csv",
) -> pd.DataFrame:
    sales = pd.read_csv(train_file, parse_dates=["Date"])[["Date", *TARGETS]]
    sample = pd.read_csv(sample_file, parse_dates=["Date"])[["Date"]]
    baseline = fit_seasonal_baseline(sales)
    models = _fit_ratio_models(
        sales,
        baseline,
        feature_group=str(best["feature_group"]),
        ratio_variant=str(best["ratio_variant"]),
    )
    cache = _prediction_cache(baseline, models, sales["Date"].min(), sample["Date"])
    clip_low, clip_high = CLIP_RANGES[str(best["clip_name"])]
    pred = _predict_from_cache(
        cache,
        revenue_weight=float(best["revenue_weight"]),
        ratio_weight=float(best["ratio_weight"]),
        clip_low=clip_low,
        clip_high=clip_high,
    )
    pred["Date"] = pred["Date"].dt.strftime("%Y-%m-%d")
    pred = pred[["Date", "Revenue", "COGS"]].round({"Revenue": 2, "COGS": 2})
    pred.to_csv(output_file, index=False)
    return pred


def verify_submission(output_file: str = "submission_best_ratio_tuned_v2.csv") -> None:
    sample = pd.read_csv("sample_submission.csv")
    submission = pd.read_csv(output_file)
    if list(submission.columns) != ["Date", "Revenue", "COGS"]:
        raise AssertionError(f"Unexpected columns: {list(submission.columns)}")
    if not submission["Date"].equals(sample["Date"]):
        raise AssertionError("Submission Date values do not match sample_submission.csv")
    if submission[["Revenue", "COGS"]].isna().any().any():
        raise AssertionError("Submission contains missing target values")
    if (submission[["Revenue", "COGS"]] < 0).any().any():
        raise AssertionError("Submission contains negative target values")


def main() -> None:
    started_at = perf_counter()
    print("ratio_tuned_v2: loading data", flush=True)
    sales = pd.read_csv("sales.csv", parse_dates=["Date"])[["Date", *TARGETS]]
    folds, leaderboard = evaluate_ratio_tuned_v2(sales)

    output_dir = Path("visual_outputs")
    output_dir.mkdir(exist_ok=True)
    folds_path = output_dir / "ratio_tuned_selection_v2_folds.csv"
    leaderboard_path = output_dir / "ratio_tuned_selection_v2_leaderboard.csv"
    folds.to_csv(folds_path, index=False)
    leaderboard.to_csv(leaderboard_path, index=False)

    best = leaderboard.iloc[0]
    submission = build_best_submission(best)
    verify_submission()

    print("\nratio tuned v2 leaderboard top 20")
    print(leaderboard.head(20).round(4).to_string(index=False))
    print(
        f"\nbest={best['candidate']} weighted_avg_mape={best['weighted_avg_mape']:.4f} "
        f"avg_mape={best['avg_mape']:.4f}",
        flush=True,
    )
    print(f"saved {len(folds)} fold rows to {folds_path}")
    print(f"saved {len(leaderboard)} candidates to {leaderboard_path}")
    print(f"saved {len(submission)} rows to submission_best_ratio_tuned_v2.csv")
    print(submission.head().to_string(index=False))
    print(f"ratio_tuned_v2: total_elapsed={perf_counter() - started_at:.1f}s", flush=True)


if __name__ == "__main__":
    main()
