"""
TTM (TinyTimeMixer) forecasting module.
Uses pretrained ibm-granite/granite-timeseries-ttm-r2 for zero-shot forecasting.

TTM is a lightweight (<1M params) foundation model for time series,
based on the mixer architecture. Zero-shot: no training needed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

TARGETS = ("Revenue", "COGS")

# ── Model config ─────────────────────────────────────────────────────────
TTM_MODEL_ID = "ibm-granite/granite-timeseries-ttm-r2"
TTM_REVISIONS = {
    "ttm_512": {"revision": "512-720-r2", "context": 512, "pred": 720},
    "ttm_1024": {"revision": "1024-720-r2", "context": 1024, "pred": 720},
    "ttm_1536": {"revision": "1536-720-r2", "context": 1536, "pred": 720},
}

_ttm_models = {}

def _load_ttm(name: str):
    """Load TTM model once and cache it."""
    global _ttm_models
    if name in _ttm_models:
        return _ttm_models[name]

    from tsfm_public.models.tinytimemixer import TinyTimeMixerForPrediction

    rev_info = TTM_REVISIONS[name]
    print(f"ttm: loading {TTM_MODEL_ID} revision={rev_info['revision']}", flush=True)
    model = TinyTimeMixerForPrediction.from_pretrained(
        TTM_MODEL_ID,
        revision=rev_info['revision'],
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    _ttm_models[name] = model
    print(f"ttm: loaded {name} on {device}", flush=True)
    return model


def _ttm_forecast_single(series: np.ndarray, horizon: int, name: str) -> np.ndarray:
    """Zero-shot forecast using a specific TTM revision."""
    model = _load_ttm(name)
    device = next(model.parameters()).device
    rev_info = TTM_REVISIONS[name]
    ctx_len_max = rev_info["context"]
    pred_len_max = rev_info["pred"]

    log_series = np.log1p(np.maximum(series, 0.0)).astype(np.float32)

    pieces: list[np.ndarray] = []
    remaining = horizon
    current_series = log_series.copy()

    while remaining > 0:
        ctx_len = min(len(current_series), ctx_len_max)
        context = current_series[-ctx_len:]

        if len(context) < ctx_len_max:
            pad = np.zeros(ctx_len_max - len(context), dtype=np.float32)
            context = np.concatenate([pad, context])

        x = torch.tensor(context, dtype=torch.float32, device=device).reshape(1, ctx_len_max, 1)

        with torch.no_grad():
            output = model(x)
            pred = output.prediction_outputs[0, :, 0].cpu().numpy()

        steps = min(pred_len_max, remaining)
        pieces.append(pred[:steps])
        current_series = np.concatenate([current_series, pred[:steps]])
        remaining -= steps

    forecast_log = np.concatenate(pieces)[:horizon]
    return np.maximum(np.expm1(forecast_log), 0.0)


def ttm_predict(
    train: pd.DataFrame,
    future_dates: pd.Series,
) -> dict[str, pd.DataFrame]:
    """
    Zero-shot forecast using 3 TTM pretrained models.
    Returns a dict mapping component name to DataFrame.
    """
    train_sorted = train.sort_values("Date").reset_index(drop=True)
    horizon = len(future_dates)
    
    results = {}
    for name in TTM_REVISIONS:
        result = pd.DataFrame({"Date": future_dates.values})
        for target in TARGETS:
            series = train_sorted[target].to_numpy(dtype=float)
            preds = _ttm_forecast_single(series, horizon, name)
            result[target] = np.round(preds[:horizon], 2)
        results[name] = result

    return results
