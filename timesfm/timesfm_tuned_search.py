from __future__ import annotations

import os
import traceback
import gc
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from ensemble_forecast import TARGETS, _mae, _mape

try:
    from timesfm_forecast import _load_model as _shared_load_model
except Exception:  # pragma: no cover - fallback path for import/runtime edge cases
    _shared_load_model = None

WEIGHTS = (0.00, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20)
FAMILIES = ("direct_raw", "direct_log1p", "recursive64_raw", "recursive64_log1p")
CHUNK_LEN = 64

SALES_PATH = Path("sales.csv")
SAMPLE_PATH = Path("sample_submission.csv")
META_VALIDATION_PATH = Path("visual_outputs/meta_ensemble_validation_predictions.csv")
META_SUBMISSION_PATH = Path("submission_best_meta_ensemble.csv")
META_WEIGHTS_PATH = Path("visual_outputs/meta_ensemble_weights.csv")

OUT_DIR = Path("visual_outputs")
FOLDS_OUT = OUT_DIR / "timesfm_tuned_folds.csv"
SCORES_OUT = OUT_DIR / "timesfm_tuned_scores.csv"
VALIDATION_PREDS_OUT = OUT_DIR / "timesfm_tuned_validation_predictions.csv"
ERROR_OUT = OUT_DIR / "timesfm_tuned_error.txt"
BEST_SUBMISSION_OUT = Path("submission_best_timesfm_tuned.csv")

_MODEL_HORIZON: int | None = None
_MODEL_INSTANCE: object | None = None


def _release_model_cache() -> None:
    global _MODEL_HORIZON, _MODEL_INSTANCE
    _MODEL_HORIZON = None
    _MODEL_INSTANCE = None
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _load_model_local(horizon: int):
    import timesfm

    repo_id = os.getenv("TIMESFM_REPO", "google/timesfm-1.0-200m-pytorch")
    local_dir = os.getenv("TIMESFM_LOCAL_DIR", ".cache/timesfm")
    context_len = int(os.getenv("TIMESFM_CONTEXT_LEN", "512"))
    backend = os.getenv("TIMESFM_BACKEND", "gpu")

    return timesfm.TimesFm(
        hparams=timesfm.TimesFmHparams(
            context_len=context_len,
            horizon_len=horizon,
            input_patch_len=32,
            output_patch_len=128,
            num_layers=20,
            num_heads=16,
            model_dims=1280,
            per_core_batch_size=1,
            backend=backend,
            point_forecast_mode="median",
        ),
        checkpoint=timesfm.TimesFmCheckpoint(
            version="torch",
            huggingface_repo_id=repo_id,
            local_dir=local_dir,
        ),
    )


def _model_for_horizon(horizon: int):
    global _MODEL_HORIZON, _MODEL_INSTANCE
    if horizon <= 0:
        raise ValueError(f"horizon must be positive, got {horizon}")
    if _MODEL_INSTANCE is not None and _MODEL_HORIZON == horizon:
        return _MODEL_INSTANCE

    _release_model_cache()

    loader = _shared_load_model if _shared_load_model is not None else _load_model_local
    backend = os.getenv("TIMESFM_BACKEND", "gpu")
    print(f"timesfm: loading model horizon={horizon} backend={backend}", flush=True)
    try:
        model = loader(horizon=horizon)
    except Exception as exc:
        msg = str(exc).lower()
        if backend != "cpu" and ("outofmemory" in msg or "out of memory" in msg):
            print("timesfm: GPU OOM detected, retrying with TIMESFM_BACKEND=cpu", flush=True)
            os.environ["TIMESFM_BACKEND"] = "cpu"
            _release_model_cache()
            model = loader(horizon=horizon)
        else:
            raise

    _MODEL_HORIZON = horizon
    _MODEL_INSTANCE = model
    return model


def _timesfm_segment(history: np.ndarray, horizon: int) -> np.ndarray:
    model = _model_for_horizon(horizon)
    point_forecast, _ = model.forecast(
        inputs=[history.astype(float)],
        freq=[0],
        normalize=True,
    )
    arr = np.asarray(point_forecast, dtype=float)[0, :horizon]
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return np.maximum(arr, 0.0)


def _forecast_family(history_raw: np.ndarray, horizon: int, family: str) -> np.ndarray:
    use_log = "log1p" in family
    recursive = family.startswith("recursive64")

    base = np.maximum(np.asarray(history_raw, dtype=float), 0.0)
    work_hist = np.log1p(base) if use_log else base

    if recursive:
        pieces: list[np.ndarray] = []
        remaining = horizon
        while remaining > 0:
            step = min(CHUNK_LEN, remaining)
            pred_work = _timesfm_segment(work_hist, step)
            pieces.append(pred_work)
            work_hist = np.concatenate([work_hist, pred_work])
            remaining -= step
        pred = np.concatenate(pieces)[:horizon]
    else:
        pred = _timesfm_segment(work_hist, horizon)

    if use_log:
        pred = np.expm1(pred)
    pred = np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
    return np.maximum(pred, 0.0)


def _scenario_filters() -> set[str]:
    raw = os.getenv("TIMESFM_TUNED_SCENARIOS", "").strip()
    if not raw:
        return set()
    return {part.strip() for part in raw.split(",") if part.strip()}


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
    if value < 0:
        raise ValueError(f"{name} must be >= 0, got {value}")
    return value


def _derive_meta_anchor(meta_validation: pd.DataFrame) -> pd.DataFrame:
    out = meta_validation.copy()
    if {"Revenue_pred", "COGS_pred"}.issubset(out.columns):
        out["Revenue_anchor_meta"] = out["Revenue_pred"].astype(float)
        out["COGS_anchor_meta"] = out["COGS_pred"].astype(float)
        return out

    if not META_WEIGHTS_PATH.exists():
        raise FileNotFoundError(
            "Missing meta anchor columns (Revenue_pred/COGS_pred) and missing "
            f"weights file: {META_WEIGHTS_PATH}"
        )

    weights = pd.read_csv(META_WEIGHTS_PATH)
    weight_cols = [c for c in weights.columns if c.startswith("w_")]
    if not weight_cols:
        raise ValueError(f"{META_WEIGHTS_PATH} has no weight columns")

    components = [c.replace("w_", "", 1) for c in weight_cols]
    needed_cols = {f"{target}_{comp}" for target in TARGETS for comp in components}
    missing = sorted(needed_cols - set(out.columns))
    if missing:
        raise ValueError(
            "Cannot reconstruct meta anchor from validation predictions; missing columns: "
            + ", ".join(missing)
        )

    lookup: dict[tuple[str, str], dict[str, float]] = {}
    for _, row in weights.iterrows():
        target = str(row["target"])
        bucket = str(row["bucket"])
        lookup[(target, bucket)] = {comp: float(row[f"w_{comp}"]) for comp in components}

    for target in TARGETS:
        vals: list[float] = []
        for _, row in out.iterrows():
            bucket = str(row["bucket"])
            key = (target, bucket)
            if key not in lookup:
                raise ValueError(f"Missing meta weight for target={target} bucket={bucket}")
            w = lookup[key]
            vals.append(sum(w[comp] * float(row[f"{target}_{comp}"]) for comp in components))
        out[f"{target}_anchor_meta"] = np.array(vals, dtype=float)

    return out


def _fold_table(meta_validation: pd.DataFrame) -> pd.DataFrame:
    table = (
        meta_validation.groupby(["scenario", "fold", "valid_days"], as_index=False)
        .agg(
            valid_start=("Date", "min"),
            valid_end=("Date", "max"),
            scenario_weight=("scenario_weight", "first"),
            horizon=("Date", "size"),
        )
        .sort_values(["valid_start", "valid_end", "scenario", "fold"], kind="stable")
        .reset_index(drop=True)
    )
    return table


def _weighted_leaderboard(folds: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["candidate", "submission_file"]
    metric_cols = ["Revenue_mae", "Revenue_mape", "COGS_mae", "COGS_mape", "avg_mae", "avg_mape"]

    metric_summary = folds.groupby(group_cols, as_index=False)[metric_cols].mean()
    scenario_summary = folds.groupby([*group_cols, "scenario"], as_index=False).agg(
        avg_mape=("avg_mape", "mean"),
        scenario_weight=("scenario_weight", "first"),
    )
    scenario_summary["weighted_component"] = scenario_summary["avg_mape"] * scenario_summary["scenario_weight"]

    weighted = scenario_summary.groupby(group_cols, as_index=False).agg(
        weighted_avg_mape=("weighted_component", "sum"),
        scenario_weight_total=("scenario_weight", "sum"),
        scenarios_covered=("scenario", "nunique"),
    )
    weighted["weighted_avg_mape"] = weighted["weighted_avg_mape"] / weighted["scenario_weight_total"]

    coverage = folds.groupby(group_cols, as_index=False).agg(folds_evaluated=("fold", "size"))
    out = (
        metric_summary.merge(weighted, on=group_cols, how="left")
        .merge(coverage, on=group_cols, how="left")
        .sort_values(["weighted_avg_mape", "avg_mape", "avg_mae"], kind="stable")
        .reset_index(drop=True)
    )

    ordered = [
        "candidate",
        "submission_file",
        "weighted_avg_mape",
        "folds_evaluated",
        "scenarios_covered",
        "Revenue_mae",
        "Revenue_mape",
        "COGS_mae",
        "COGS_mape",
        "avg_mae",
        "avg_mape",
    ]
    return out[ordered]


def _validate_submission(path: Path, sample: pd.DataFrame) -> None:
    sub = pd.read_csv(path)
    if list(sub.columns) != ["Date", "Revenue", "COGS"]:
        raise AssertionError(f"{path}: bad columns")
    if not sub["Date"].equals(sample["Date"]):
        raise AssertionError(f"{path}: dates mismatch")
    if sub[["Revenue", "COGS"]].isna().any().any():
        raise AssertionError(f"{path}: null values")
    if (sub[["Revenue", "COGS"]] < 0).any().any():
        raise AssertionError(f"{path}: negative values")


def _parse_candidate(candidate: str) -> tuple[str, float]:
    prefix = "timesfm_"
    if not candidate.startswith(prefix) or "_w" not in candidate:
        raise ValueError(f"Unexpected candidate format: {candidate}")
    body = candidate[len(prefix) :]
    family, weight_str = body.rsplit("_w", maxsplit=1)
    return family, float(weight_str)


def run() -> None:
    started = perf_counter()
    OUT_DIR.mkdir(exist_ok=True)

    sales = pd.read_csv(SALES_PATH, parse_dates=["Date"])[["Date", *TARGETS]].sort_values("Date")
    sample = pd.read_csv(SAMPLE_PATH)
    meta_validation = pd.read_csv(META_VALIDATION_PATH, parse_dates=["Date"])
    meta_submission = pd.read_csv(META_SUBMISSION_PATH)

    required_validation_cols = {
        "scenario",
        "fold",
        "valid_days",
        "scenario_weight",
        "horizon_index",
        "bucket",
        "Date",
        "Revenue_actual",
        "COGS_actual",
    }
    missing_required = sorted(required_validation_cols - set(meta_validation.columns))
    if missing_required:
        raise ValueError(f"{META_VALIDATION_PATH} missing required columns: {missing_required}")

    if list(meta_submission.columns) != ["Date", "Revenue", "COGS"]:
        raise ValueError(f"{META_SUBMISSION_PATH} must have columns Date, Revenue, COGS")
    if not meta_submission["Date"].equals(sample["Date"]):
        raise ValueError(f"{META_SUBMISSION_PATH}: date order does not match {SAMPLE_PATH}")

    meta_validation = _derive_meta_anchor(meta_validation)
    folds = _fold_table(meta_validation)

    scenario_allow = _scenario_filters()
    max_folds = _int_env("TIMESFM_TUNED_MAX_FOLDS", 0)
    max_horizon = _int_env("TIMESFM_TUNED_MAX_HORIZON", 0)

    if scenario_allow:
        folds = folds[folds["scenario"].isin(scenario_allow)].reset_index(drop=True)
    if max_horizon > 0:
        folds = folds[folds["horizon"] <= max_horizon].reset_index(drop=True)
    if max_folds > 0:
        folds = folds.head(max_folds).reset_index(drop=True)

    if folds.empty:
        raise ValueError("No folds selected for evaluation after applying filters")

    print(
        "timesfm_tuned: folds="
        f"{len(folds)} scenarios={folds['scenario'].nunique()} "
        f"max_horizon={int(folds['horizon'].max())}"
        + (
            " [bounded mode]"
            if (scenario_allow or max_folds > 0 or max_horizon > 0)
            else ""
        ),
        flush=True,
    )

    fold_rows: list[dict[str, float | int | str]] = []
    validation_rows: list[dict[str, float | int | str]] = []

    for idx, fold_info in folds.iterrows():
        scenario = str(fold_info["scenario"])
        fold_id = int(fold_info["fold"])
        valid_days = int(fold_info["valid_days"])
        scenario_weight = float(fold_info["scenario_weight"])
        valid_start = pd.Timestamp(fold_info["valid_start"])

        mask = (
            meta_validation["scenario"].eq(scenario)
            & meta_validation["fold"].eq(fold_id)
            & meta_validation["valid_days"].eq(valid_days)
        )
        fold_frame = meta_validation.loc[mask].sort_values("horizon_index").reset_index(drop=True)
        if fold_frame.empty:
            raise ValueError(f"No validation rows for scenario={scenario} fold={fold_id} valid_days={valid_days}")

        horizon = len(fold_frame)
        train = sales[sales["Date"] < valid_start]
        if train.empty:
            raise ValueError(f"No train rows before fold start {valid_start.date()} for scenario={scenario} fold={fold_id}")

        fold_started = perf_counter()
        print(
            f"[fold {idx + 1}/{len(folds)}] {scenario}#{fold_id} horizon={horizon} "
            f"train_rows={len(train)} valid_start={valid_start.date()}",
            flush=True,
        )

        family_target_preds: dict[str, dict[str, np.ndarray]] = {target: {} for target in TARGETS}
        for target in TARGETS:
            history = train[target].to_numpy(float)
            for family in FAMILIES:
                family_target_preds[target][family] = _forecast_family(history, horizon, family)

        for family in FAMILIES:
            anchor_arrays = {
                target: fold_frame[f"{target}_anchor_meta"].to_numpy(float)
                for target in TARGETS
            }
            actual_arrays = {
                target: fold_frame[f"{target}_actual"].to_numpy(float)
                for target in TARGETS
            }
            timesfm_arrays = {
                target: family_target_preds[target][family]
                for target in TARGETS
            }

            for weight in WEIGHTS:
                candidate = f"timesfm_{family}_w{weight:.2f}"
                submission_file = f"submission_best_timesfm_tuned_{family}_w{weight:.2f}.csv"

                blended = {
                    target: np.round((1.0 - weight) * anchor_arrays[target] + weight * timesfm_arrays[target], 2)
                    for target in TARGETS
                }

                row: dict[str, float | int | str] = {
                    "candidate": candidate,
                    "submission_file": submission_file,
                    "scenario": scenario,
                    "fold": fold_id,
                    "valid_days": valid_days,
                    "scenario_weight": scenario_weight,
                    "valid_start": valid_start,
                    "valid_end": pd.Timestamp(fold_info["valid_end"]),
                    "timesfm_family": family,
                    "timesfm_weight": weight,
                }
                for target in TARGETS:
                    row[f"{target}_mae"] = _mae(pd.Series(actual_arrays[target]), pd.Series(blended[target]))
                    row[f"{target}_mape"] = _mape(pd.Series(actual_arrays[target]), pd.Series(blended[target]))
                row["avg_mae"] = (float(row["Revenue_mae"]) + float(row["COGS_mae"])) / 2.0
                row["avg_mape"] = (float(row["Revenue_mape"]) + float(row["COGS_mape"])) / 2.0
                fold_rows.append(row)

                for i in range(horizon):
                    validation_rows.append(
                        {
                            "candidate": candidate,
                            "scenario": scenario,
                            "fold": fold_id,
                            "valid_days": valid_days,
                            "scenario_weight": scenario_weight,
                            "horizon_index": int(fold_frame.loc[i, "horizon_index"]),
                            "Date": fold_frame.loc[i, "Date"],
                            "Revenue_actual": actual_arrays["Revenue"][i],
                            "COGS_actual": actual_arrays["COGS"][i],
                            "Revenue_meta_anchor": anchor_arrays["Revenue"][i],
                            "COGS_meta_anchor": anchor_arrays["COGS"][i],
                            "Revenue_timesfm": timesfm_arrays["Revenue"][i],
                            "COGS_timesfm": timesfm_arrays["COGS"][i],
                            "Revenue_pred": blended["Revenue"][i],
                            "COGS_pred": blended["COGS"][i],
                            "timesfm_family": family,
                            "timesfm_weight": weight,
                        }
                    )

        print(
            f"[fold {idx + 1}/{len(folds)}] complete elapsed={perf_counter() - fold_started:.1f}s",
            flush=True,
        )

    folds_df = pd.DataFrame(fold_rows)
    if folds_df.empty:
        raise ValueError("No fold metrics were produced")
    scores_df = _weighted_leaderboard(folds_df)
    validation_df = pd.DataFrame(validation_rows)

    folds_df.to_csv(FOLDS_OUT, index=False)
    scores_df.to_csv(SCORES_OUT, index=False)
    if not validation_df.empty:
        validation_df.to_csv(VALIDATION_PREDS_OUT, index=False)

    best = scores_df.iloc[0]
    best_candidate = str(best["candidate"])
    best_family, best_weight = _parse_candidate(best_candidate)

    print(
        f"timesfm_tuned: best_candidate={best_candidate} "
        f"weighted_avg_mape={float(best['weighted_avg_mape']):.4f}",
        flush=True,
    )

    sample_dates = pd.to_datetime(sample["Date"])
    horizon_future = len(sample_dates)
    best_submission = pd.DataFrame({"Date": sample["Date"].copy()})

    for target in TARGETS:
        history = sales[target].to_numpy(float)
        timesfm_future = _forecast_family(history, horizon_future, best_family)
        anchor_future = meta_submission[target].to_numpy(float)
        best_submission[target] = np.round((1.0 - best_weight) * anchor_future + best_weight * timesfm_future, 2)

    best_submission.to_csv(BEST_SUBMISSION_OUT, index=False)
    _validate_submission(BEST_SUBMISSION_OUT, sample)

    print(f"saved {FOLDS_OUT}", flush=True)
    print(f"saved {SCORES_OUT}", flush=True)
    if not validation_df.empty:
        print(f"saved {VALIDATION_PREDS_OUT}", flush=True)
    print(f"saved {BEST_SUBMISSION_OUT}", flush=True)
    print(f"timesfm_tuned elapsed={perf_counter() - started:.1f}s", flush=True)


def main() -> None:
    try:
        run()
    except Exception:
        OUT_DIR.mkdir(exist_ok=True)
        ERROR_OUT.write_text(traceback.format_exc(), encoding="utf-8")
        print(f"timesfm_tuned failed, details written to {ERROR_OUT}", flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
