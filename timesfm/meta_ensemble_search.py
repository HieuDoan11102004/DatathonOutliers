from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
from time import perf_counter
from typing import Iterable

import numpy as np
import pandas as pd
import torch

from dlinear_forecast import dlinear_predict
from ttm_forecast import ttm_predict
from chronos import ChronosBoltPipeline
from chronos_full_validation import _chronos_predict, _iter_folds, verify_submission
from ensemble_forecast import TARGETS, _mae, _mape, fit_seasonal_baseline
from ratio_tuned_selection_v2 import (
    CLIP_RANGES as V2_CLIP_RANGES,
    _fit_ratio_models as fit_v2_ratio_models,
    _prediction_cache as v2_prediction_cache,
    _predict_from_cache as predict_v2_from_cache,
)

COMPONENTS = ("v2", "chronos_tiny", "chronos_mini", "chronos_small", "chronos_base", "ttm_512", "ttm_1024", "ttm_1536", "dlinear")
CHRONOS_MODELS = {
    "chronos_tiny": "amazon/chronos-bolt-tiny",
    "chronos_mini": "amazon/chronos-bolt-mini",
    "chronos_small": "amazon/chronos-bolt-small",
    "chronos_base": "amazon/chronos-bolt-base",
}
BUCKETS = (
    ("short", 0, 120),
    ("mid", 120, 365),
    ("long", 365, 10_000),
)
GRID_STEP = 0.05
# V2_BEST (Old config without external features)
# V2_BEST = {
#     "feature_group": "all",
#     "ratio_variant": "flex",
#     "clip_name": "business",
#     "revenue_weight": 1.30,
#     "ratio_weight": 1.10,
# }

V2_BEST = {
    "feature_group": "all",
    "ratio_variant": "base",
    "clip_name": "business",
    "revenue_weight": 1.20,
    "ratio_weight": 1.10,
}


@dataclass(frozen=True)
class WeightChoice:
    target: str
    bucket: str
    weights: dict[str, float]
    validation_mape: float


def _load_chronos_pipelines() -> dict[str, ChronosBoltPipeline]:
    kwargs = {"device_map": "cuda"} if torch.cuda.is_available() else {}
    pipelines = {}
    for name, model_id in CHRONOS_MODELS.items():
        print(f"meta: loading {name} {model_id} kwargs={kwargs}", flush=True)
        pipelines[name] = ChronosBoltPipeline.from_pretrained(model_id, **kwargs)
    return pipelines


def _bucket_for_index(i: int) -> str:
    for name, lo, hi in BUCKETS:
        if lo <= i < hi:
            return name
    return BUCKETS[-1][0]


def _predict_v2(train: pd.DataFrame, dates: pd.Series) -> pd.DataFrame:
    baseline = fit_seasonal_baseline(train)
    models = fit_v2_ratio_models(train, baseline, V2_BEST["feature_group"], V2_BEST["ratio_variant"])
    cache = v2_prediction_cache(baseline, models, train["Date"].min(), dates)
    clip_low, clip_high = V2_CLIP_RANGES[V2_BEST["clip_name"]]
    return predict_v2_from_cache(cache, V2_BEST["revenue_weight"], V2_BEST["ratio_weight"], clip_low, clip_high)


def _predict_ttm(train: pd.DataFrame, dates: pd.Series) -> dict[str, pd.DataFrame]:
    """Zero-shot forecast using 3 pretrained TTM revisions."""
    return ttm_predict(train, dates)


def _fold_prediction_rows(
    scenario: str,
    fold: int,
    valid_days: int,
    scenario_weight: float,
    valid: pd.DataFrame,
    preds: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows = []
    valid = valid.reset_index(drop=True)
    pred_reset = {name: pred.reset_index(drop=True) for name, pred in preds.items()}
    for i, actual in valid.iterrows():
        row: dict[str, float | str | int] = {
            "scenario": scenario,
            "fold": fold,
            "valid_days": valid_days,
            "scenario_weight": scenario_weight,
            "horizon_index": i,
            "bucket": _bucket_for_index(i),
            "Date": actual["Date"],
            "Revenue_actual": float(actual["Revenue"]),
            "COGS_actual": float(actual["COGS"]),
        }
        for comp, pred in pred_reset.items():
            row[f"Revenue_{comp}"] = float(pred.loc[i, "Revenue"])
            row[f"COGS_{comp}"] = float(pred.loc[i, "COGS"])
        rows.append(row)
    return pd.DataFrame(rows)


def build_validation_predictions() -> pd.DataFrame:
    sales = pd.read_csv("sales.csv", parse_dates=["Date"])[["Date", *TARGETS]]
    folds = list(_iter_folds(sales))
    pipelines = _load_chronos_pipelines()
    pieces = []
    started = perf_counter()
    for idx, (scenario, fold, valid_days, scenario_weight, train, valid) in enumerate(folds, start=1):
        fold_start = perf_counter()
        print(f"[fold {idx}/{len(folds)}] {scenario} valid={valid['Date'].min().date()}..{valid['Date'].max().date()}", flush=True)
        preds = {"v2": _predict_v2(train, valid["Date"])}
        
        # Chronos predictions
        for comp, pipe in pipelines.items():
            preds[comp] = _chronos_predict(pipe, train, valid["Date"])

        # TTM predictions (3 revisions)
        try:
            ttm_preds = _predict_ttm(train, valid["Date"])
            preds.update(ttm_preds)
        except Exception as e:
            print(f"  [warn] ttm failed: {e}, using v2 fallback", flush=True)
            preds["ttm_512"] = preds["v2"].copy()
            preds["ttm_1024"] = preds["v2"].copy()
            preds["ttm_1536"] = preds["v2"].copy()
        # DLinear prediction
        try:
            preds["dlinear"] = dlinear_predict(train, valid["Date"])
        except Exception as e:
            print(f"  [warn] dlinear failed: {e}, using v2 fallback", flush=True)
            preds["dlinear"] = preds["v2"].copy()
        pieces.append(_fold_prediction_rows(scenario, fold, valid_days, scenario_weight, valid, preds))
        print(f"[fold {idx}/{len(folds)}] done elapsed={perf_counter()-fold_start:.1f}s total={perf_counter()-started:.1f}s", flush=True)
    out = pd.concat(pieces, ignore_index=True)
    Path("visual_outputs").mkdir(exist_ok=True)
    out.to_csv("visual_outputs/meta_ensemble_validation_predictions.csv", index=False)
    return out


def _weighted_mape(y: np.ndarray, pred: np.ndarray, weights: np.ndarray) -> float:
    denom = np.maximum(np.abs(y), 1e-6)
    return float(np.average(np.abs((y - pred) / denom) * 100.0, weights=weights))


from scipy.optimize import minimize

def choose_weights(preds: pd.DataFrame) -> list[WeightChoice]:
    choices: list[WeightChoice] = []
    n_comps = len(COMPONENTS)
    
    # Objective function for SLSQP
    def objective(w, y, pred_matrix, row_weights):
        pred = pred_matrix @ w
        return _weighted_mape(y, pred, row_weights)

    bounds = [(0.0, 1.0) for _ in range(n_comps)]
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

    for target in TARGETS:
        actual_col = f"{target}_actual"
        pred_cols = [f"{target}_{comp}" for comp in COMPONENTS]
        for bucket, _, _ in BUCKETS:
            subset = preds[preds["bucket"].eq(bucket)].copy()
            if subset.empty:
                continue
            y = subset[actual_col].to_numpy(float)
            pred_matrix = subset[pred_cols].to_numpy(float)
            row_weights = subset["scenario_weight"].to_numpy(float)
            
            # Start with equal weights
            w0 = np.ones(n_comps) / n_comps
            res = minimize(
                objective, w0, args=(y, pred_matrix, row_weights),
                method="SLSQP", bounds=bounds, constraints=constraints,
                options={"maxiter": 1000, "ftol": 1e-6}
            )
            
            best_combo = res.x
            # Clean up small numerical noise
            best_combo[best_combo < 1e-3] = 0.0
            best_combo = best_combo / np.sum(best_combo)
            
            choices.append(
                WeightChoice(
                    target=target,
                    bucket=bucket,
                    weights=dict(zip(COMPONENTS, best_combo, strict=True)),
                    validation_mape=float(res.fun),
                )
            )
    return choices


def _choice_frame(choices: list[WeightChoice]) -> pd.DataFrame:
    rows = []
    for c in choices:
        row = {"target": c.target, "bucket": c.bucket, "validation_mape": c.validation_mape}
        row.update({f"w_{k}": v for k, v in c.weights.items()})
        rows.append(row)
    return pd.DataFrame(rows)


def _weights_lookup(choices: list[WeightChoice]) -> dict[tuple[str, str], dict[str, float]]:
    return {(c.target, c.bucket): c.weights for c in choices}


def apply_meta(preds: pd.DataFrame, choices: list[WeightChoice]) -> pd.DataFrame:
    lookup = _weights_lookup(choices)
    out = preds[["scenario", "fold", "valid_days", "scenario_weight", "horizon_index", "bucket", "Date", "Revenue_actual", "COGS_actual"]].copy()
    for target in TARGETS:
        vals = []
        for _, row in preds.iterrows():
            weights = lookup[(target, str(row["bucket"]))]
            vals.append(sum(weights[comp] * float(row[f"{target}_{comp}"]) for comp in COMPONENTS))
        out[f"{target}_pred"] = np.round(vals, 2)
    return out


def score_meta(meta: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for (scenario, fold), group in meta.groupby(["scenario", "fold"], sort=False):
        row: dict[str, float | str | int] = {
            "scenario": scenario,
            "fold": int(fold),
            "valid_days": int(group["valid_days"].iloc[0]),
            "scenario_weight": float(group["scenario_weight"].iloc[0]),
            "candidate": "meta_bucket_target_ensemble",
        }
        for target in TARGETS:
            row[f"{target}_mae"] = _mae(group[f"{target}_actual"], group[f"{target}_pred"])
            row[f"{target}_mape"] = _mape(group[f"{target}_actual"], group[f"{target}_pred"])
        row["avg_mape"] = (row["Revenue_mape"] + row["COGS_mape"]) / 2.0  # type: ignore[operator]
        row["avg_mae"] = (row["Revenue_mae"] + row["COGS_mae"]) / 2.0  # type: ignore[operator]
        rows.append(row)
    folds = pd.DataFrame(rows)
    metric_cols = ["Revenue_mae", "Revenue_mape", "COGS_mae", "COGS_mape", "avg_mae", "avg_mape"]
    means = folds[metric_cols].mean().to_dict()
    scenario_summary = folds.groupby("scenario", as_index=False).agg(avg_mape=("avg_mape", "mean"), scenario_weight=("scenario_weight", "first"))
    weighted_avg_mape = float((scenario_summary["avg_mape"] * scenario_summary["scenario_weight"]).sum() / scenario_summary["scenario_weight"].sum())
    leaderboard = pd.DataFrame([{**{"candidate": "meta_bucket_target_ensemble", "weighted_avg_mape": weighted_avg_mape, "folds_evaluated": len(folds), "scenarios_covered": folds["scenario"].nunique()}, **means}])
    return folds, leaderboard


def _future_component_predictions() -> pd.DataFrame:
    sales = pd.read_csv("sales.csv", parse_dates=["Date"])[["Date", *TARGETS]]
    sample = pd.read_csv("sample_submission.csv", parse_dates=["Date"])[["Date"]]
    pipelines = _load_chronos_pipelines()
    preds = {"v2": pd.read_csv("submission_best_ratio_tuned_v2.csv", parse_dates=["Date"])}
    
    # Chronos future prediction
    for comp, pipe in pipelines.items():
        preds[comp] = _chronos_predict(pipe, sales, sample["Date"])

    # TTM future prediction (3 revisions zero-shot)
    try:
        ttm_preds = _predict_ttm(sales, sample["Date"])
        preds.update(ttm_preds)
    except Exception as e:
        print(f"  [warn] ttm future failed: {e}, using v2 fallback", flush=True)
        preds["ttm_512"] = preds["v2"].copy()
        preds["ttm_1024"] = preds["v2"].copy()
        preds["ttm_1536"] = preds["v2"].copy()
    # DLinear future prediction
    try:
        preds["dlinear"] = dlinear_predict(sales, sample["Date"])
    except Exception as e:
        print(f"  [warn] dlinear future failed: {e}, using v2 fallback", flush=True)
        preds["dlinear"] = preds["v2"].copy()
    rows = []
    pred_reset = {name: pred.reset_index(drop=True) for name, pred in preds.items()}
    for i, date in enumerate(sample["Date"]):
        row: dict[str, float | str | int] = {"Date": date, "horizon_index": i, "bucket": _bucket_for_index(i)}
        for comp, pred in pred_reset.items():
            for target in TARGETS:
                row[f"{target}_{comp}"] = float(pred.loc[i, target])
        rows.append(row)
    return pd.DataFrame(rows)


def build_submission(choices: list[WeightChoice]) -> pd.DataFrame:
    future = _future_component_predictions()
    lookup = _weights_lookup(choices)
    out = future[["Date"]].copy()
    for target in TARGETS:
        vals = []
        for _, row in future.iterrows():
            weights = lookup[(target, str(row["bucket"]))]
            vals.append(sum(weights[comp] * float(row[f"{target}_{comp}"]) for comp in COMPONENTS))
        out[target] = np.round(vals, 2)
    out["Date"] = pd.to_datetime(out["Date"]).dt.strftime("%Y-%m-%d")
    out = out[["Date", "Revenue", "COGS"]]
    out.to_csv("submission_best_meta_ensemble.csv", index=False)
    return out


def verify_submission(path: str) -> None:
    sample = pd.read_csv("sample_submission.csv")
    sub = pd.read_csv(path)
    if list(sub.columns) != ["Date", "Revenue", "COGS"]:
        raise AssertionError(f"bad columns: {path}")
    if not sub["Date"].equals(sample["Date"]):
        raise AssertionError(f"date mismatch: {path}")
    if sub[["Revenue", "COGS"]].isna().any().any():
        raise AssertionError(f"nulls: {path}")
    if (sub[["Revenue", "COGS"]] < 0).any().any():
        raise AssertionError(f"negatives: {path}")


def main() -> None:
    started = perf_counter()
    preds = build_validation_predictions()
    choices = choose_weights(preds)
    choice_df = _choice_frame(choices)
    choice_df.to_csv("visual_outputs/meta_ensemble_weights.csv", index=False)
    meta_preds = apply_meta(preds, choices)
    meta_preds.to_csv("visual_outputs/meta_ensemble_fold_predictions.csv", index=False)
    fold_scores, leaderboard = score_meta(meta_preds)
    fold_scores.to_csv("visual_outputs/meta_ensemble_folds.csv", index=False)
    leaderboard.to_csv("visual_outputs/meta_ensemble_scores.csv", index=False)
    submission = build_submission(choices)
    verify_submission("submission_best_meta_ensemble.csv")
    print("\nmeta ensemble weights")
    print(choice_df.round(4).to_string(index=False))
    print("\nmeta ensemble scores")
    print(leaderboard.round(4).to_string(index=False))
    print(f"saved {len(submission)} rows to submission_best_meta_ensemble.csv")
    print(f"meta elapsed={perf_counter()-started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
