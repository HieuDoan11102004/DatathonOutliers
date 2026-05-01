from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Iterable

import numpy as np
import pandas as pd
import torch

from chronos import ChronosBoltPipeline
from ensemble_forecast import TARGETS, _mae, _mape, fit_seasonal_baseline
from ratio_tuned_selection_v2 import (
    CLIP_RANGES as V2_CLIP_RANGES,
    _fit_ratio_models as fit_v2_ratio_models,
    _prediction_cache as v2_prediction_cache,
    _predict_from_cache as predict_v2_from_cache,
)

MODEL_ID = "amazon/chronos-bolt-tiny"
TRANSFORM = "raw"
CONTEXT_LENGTH = 2048
SCENARIOS = (
    ("90d_3fold", 3, 90, 0.10),
    ("180d_3fold", 3, 180, 0.20),
    ("365d_1fold", 1, 365, 0.25),
    ("548d_1fold", 1, 548, 0.45),
)
BLEND_WEIGHTS = (0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12)
V2_BEST = {
    "feature_group": "all",
    "ratio_variant": "flex",
    "clip_name": "business",
    "revenue_weight": 1.30,
    "ratio_weight": 1.10,
}


def _iter_folds(sales: pd.DataFrame) -> Iterable[tuple[str, int, int, float, pd.DataFrame, pd.DataFrame]]:
    sales = sales.sort_values("Date").reset_index(drop=True)
    n_rows = len(sales)
    for scenario, n_splits, valid_days, scenario_weight in SCENARIOS:
        for fold in range(n_splits):
            valid_end = n_rows - (n_splits - fold - 1) * valid_days
            valid_start = valid_end - valid_days
            if valid_start <= 365 or valid_end > n_rows or valid_start >= valid_end:
                continue
            yield scenario, fold + 1, valid_days, scenario_weight, sales.iloc[:valid_start].copy(), sales.iloc[valid_start:valid_end].copy()


def _chronos_target(pipeline: ChronosBoltPipeline, values: np.ndarray, horizon: int) -> np.ndarray:
    context = torch.tensor(values[-CONTEXT_LENGTH:].astype(np.float32), dtype=torch.float32)
    quantiles, _ = pipeline.predict_quantiles(context, prediction_length=horizon, quantile_levels=[0.5])
    arr = np.squeeze(quantiles.detach().cpu().numpy())
    if arr.ndim > 1:
        arr = arr.reshape(arr.shape[0], -1)[:, 0]
    return np.maximum(arr[:horizon].astype(float), 0.0)


def _chronos_predict(pipeline: ChronosBoltPipeline, train: pd.DataFrame, dates: pd.Series) -> pd.DataFrame:
    out = pd.DataFrame({"Date": pd.to_datetime(dates).reset_index(drop=True)})
    for target in TARGETS:
        out[target] = np.round(_chronos_target(pipeline, train[target].to_numpy(float), len(out)), 2)
    return out


def _predict_v2(train: pd.DataFrame, dates: pd.Series) -> pd.DataFrame:
    baseline = fit_seasonal_baseline(train)
    models = fit_v2_ratio_models(train, baseline, V2_BEST["feature_group"], V2_BEST["ratio_variant"])
    cache = v2_prediction_cache(baseline, models, train["Date"].min(), dates)
    clip_low, clip_high = V2_CLIP_RANGES[V2_BEST["clip_name"]]
    return predict_v2_from_cache(cache, V2_BEST["revenue_weight"], V2_BEST["ratio_weight"], clip_low, clip_high)


def _blend(base: pd.DataFrame, extra: pd.DataFrame, weight: float) -> pd.DataFrame:
    base = base.reset_index(drop=True)
    extra = extra.reset_index(drop=True)
    out = base[["Date"]].copy()
    for target in TARGETS:
        out[target] = ((1.0 - weight) * base[target].to_numpy(float) + weight * extra[target].to_numpy(float)).round(2)
    return out


def _score(pred: pd.DataFrame, actual: pd.DataFrame, scenario: str, fold: int, valid_days: int, scenario_weight: float, blend_weight: float) -> dict[str, float | str | int]:
    row: dict[str, float | str | int] = {
        "scenario": scenario,
        "fold": fold,
        "valid_days": valid_days,
        "scenario_weight": scenario_weight,
        "blend_weight_chronos": blend_weight,
        "candidate": f"blend_v2_chronos_{blend_weight:.2f}",
    }
    for target in TARGETS:
        row[f"{target}_mae"] = _mae(actual[target], pred[target])
        row[f"{target}_mape"] = _mape(actual[target], pred[target])
    row["avg_mape"] = (row["Revenue_mape"] + row["COGS_mape"]) / 2.0  # type: ignore[operator]
    row["avg_mae"] = (row["Revenue_mae"] + row["COGS_mae"]) / 2.0  # type: ignore[operator]
    return row


def _leaderboard(folds: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["candidate", "blend_weight_chronos"]
    metric_cols = ["Revenue_mae", "Revenue_mape", "COGS_mae", "COGS_mape", "avg_mae", "avg_mape"]
    metric_summary = folds.groupby(group_cols, as_index=False)[metric_cols].mean()
    scenario_summary = folds.groupby([*group_cols, "scenario"], as_index=False).agg(
        avg_mape=("avg_mape", "mean"), scenario_weight=("scenario_weight", "first")
    )
    scenario_summary["weighted_component"] = scenario_summary["avg_mape"] * scenario_summary["scenario_weight"]
    weighted = scenario_summary.groupby(group_cols, as_index=False).agg(
        weighted_avg_mape=("weighted_component", "sum"),
        scenario_weight_total=("scenario_weight", "sum"),
        scenarios_covered=("scenario", "nunique"),
    )
    weighted["weighted_avg_mape"] = weighted["weighted_avg_mape"] / weighted["scenario_weight_total"]
    coverage = folds.groupby(group_cols, as_index=False).agg(folds_evaluated=("fold", "size"))
    return (
        metric_summary.merge(weighted, on=group_cols, how="left")
        .merge(coverage, on=group_cols, how="left")
        .sort_values(["weighted_avg_mape", "avg_mape", "avg_mae"], kind="stable")
        .reset_index(drop=True)
    )


def verify_submission(path: str) -> None:
    sample = pd.read_csv("sample_submission.csv")
    sub = pd.read_csv(path)
    if list(sub.columns) != ["Date", "Revenue", "COGS"]:
        raise AssertionError(f"{path}: bad columns")
    if not sub["Date"].equals(sample["Date"]):
        raise AssertionError(f"{path}: dates mismatch")
    if sub[["Revenue", "COGS"]].isna().any().any():
        raise AssertionError(f"{path}: nulls")
    if (sub[["Revenue", "COGS"]] < 0).any().any():
        raise AssertionError(f"{path}: negatives")


def main() -> None:
    started = perf_counter()
    sales = pd.read_csv("sales.csv", parse_dates=["Date"])[["Date", *TARGETS]]
    sample = pd.read_csv("sample_submission.csv", parse_dates=["Date"])[["Date"]]
    kwargs = {"device_map": "cuda"} if torch.cuda.is_available() else {}
    print(f"chronos_full: loading {MODEL_ID} kwargs={kwargs}", flush=True)
    pipeline = ChronosBoltPipeline.from_pretrained(MODEL_ID, **kwargs)
    folds = list(_iter_folds(sales))
    rows: list[dict[str, float | str | int]] = []
    for idx, (scenario, fold, valid_days, scenario_weight, train, valid) in enumerate(folds, start=1):
        fold_start = perf_counter()
        print(f"[fold {idx}/{len(folds)}] {scenario} valid={valid['Date'].min().date()}..{valid['Date'].max().date()}", flush=True)
        v2 = _predict_v2(train, valid["Date"])
        chronos = _chronos_predict(pipeline, train, valid["Date"])
        for weight in BLEND_WEIGHTS:
            rows.append(_score(_blend(v2, chronos, weight), valid, scenario, fold, valid_days, scenario_weight, weight))
        print(f"[fold {idx}/{len(folds)}] done elapsed={perf_counter()-fold_start:.1f}s", flush=True)
    fold_scores = pd.DataFrame(rows)
    leaderboard = _leaderboard(fold_scores)
    Path("visual_outputs").mkdir(exist_ok=True)
    fold_scores.to_csv("visual_outputs/chronos_full_validation_folds.csv", index=False)
    leaderboard.to_csv("visual_outputs/chronos_full_validation_scores.csv", index=False)

    best = leaderboard.iloc[0]
    chronos_future = _chronos_predict(pipeline, sales, sample["Date"])
    v2_future = pd.read_csv("submission_best_ratio_tuned_v2.csv", parse_dates=["Date"])
    best_sub = _blend(v2_future, chronos_future, float(best["blend_weight_chronos"]))
    best_sub["Date"] = best_sub["Date"].dt.strftime("%Y-%m-%d")
    best_sub.to_csv("submission_best_chronos_full.csv", index=False)
    verify_submission("submission_best_chronos_full.csv")
    print("\nchronos full validation")
    print(leaderboard.round(4).to_string(index=False))
    print(f"saved submission_best_chronos_full.csv weight={float(best['blend_weight_chronos']):.2f}")
    print(f"chronos_full elapsed={perf_counter()-started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
