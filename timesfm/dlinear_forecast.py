"""
DLinear forecasting model (LTSF-Linear family).
Paper: "Are Transformers Effective for Time Series Forecasting?" (Zeng et al., 2023)

DLinear decomposes time series into Trend + Seasonal via moving average,
then applies separate linear layers to each component.
This gives diversity to the ensemble (linear vs tree-based vs foundation models).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

TARGETS = ("Revenue", "COGS")

# ── Hyperparameters ──────────────────────────────────────────────────────
LOOKBACK = 90        # Input window (days of history the model sees)
KERNEL_SIZE = 25     # Moving average kernel for trend/seasonal decomposition
EPOCHS = 300
BATCH_SIZE = 32
LR = 1e-3
WEIGHT_DECAY = 1e-4


class MovingAvg(nn.Module):
    """Moving average block to extract trend."""
    def __init__(self, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len)
        # Pad front and back
        front = x[:, :1].repeat(1, (self.kernel_size - 1) // 2)
        end = x[:, -1:].repeat(1, (self.kernel_size - 1) // 2)
        x_pad = torch.cat([front, x, end], dim=1)
        return self.avg(x_pad.unsqueeze(1)).squeeze(1)


class DLinearModel(nn.Module):
    """
    DLinear: Decomposition + dual Linear layers.
    - Trend component  → Linear(lookback, horizon)
    - Seasonal component → Linear(lookback, horizon)
    - Output = trend_pred + seasonal_pred
    """
    def __init__(self, lookback: int, horizon: int, kernel_size: int = 25):
        super().__init__()
        self.lookback = lookback
        self.horizon = horizon
        self.decomp = MovingAvg(kernel_size)
        self.linear_trend = nn.Linear(lookback, horizon)
        self.linear_seasonal = nn.Linear(lookback, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, lookback)
        trend = self.decomp(x)
        seasonal = x - trend
        trend_pred = self.linear_trend(trend)
        seasonal_pred = self.linear_seasonal(seasonal)
        return trend_pred + seasonal_pred


def _create_sliding_windows(
    series: np.ndarray, lookback: int, horizon: int
) -> tuple[np.ndarray, np.ndarray]:
    """Create (X, y) pairs from a 1D time series using sliding windows."""
    X_list, y_list = [], []
    for i in range(len(series) - lookback - horizon + 1):
        X_list.append(series[i : i + lookback])
        y_list.append(series[i + lookback : i + lookback + horizon])
    if not X_list:
        return np.empty((0, lookback)), np.empty((0, horizon))
    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)


def _train_dlinear(
    series: np.ndarray,
    horizon: int,
    lookback: int = LOOKBACK,
    kernel_size: int = KERNEL_SIZE,
    epochs: int = EPOCHS,
    lr: float = LR,
) -> DLinearModel:
    """Train a DLinear model on a single univariate series."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Normalize (log1p to stabilize revenue/cogs magnitudes)
    log_series = np.log1p(np.maximum(series, 0.0))

    X, y = _create_sliding_windows(log_series, lookback, horizon)
    if len(X) < 10:
        raise ValueError(f"Not enough data: {len(X)} windows (need >= 10)")

    # Last-value normalization per window (NLinear-style, helps DLinear too)
    last_vals = X[:, -1:]
    X_norm = X - last_vals
    y_norm = y - last_vals

    dataset = TensorDataset(
        torch.tensor(X_norm, device=device),
        torch.tensor(y_norm, device=device),
    )
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

    model = DLinearModel(lookback, horizon, kernel_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    model.train()
    for epoch in range(epochs):
        for xb, yb in loader:
            pred = model(xb)
            loss = nn.functional.huber_loss(pred, yb, delta=1.0)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

    return model


def _predict_dlinear(
    model: DLinearModel,
    series: np.ndarray,
    lookback: int = LOOKBACK,
) -> np.ndarray:
    """Generate forecast from a trained DLinear model."""
    device = next(model.parameters()).device
    log_series = np.log1p(np.maximum(series, 0.0))

    # Take the last `lookback` values
    x = log_series[-lookback:]
    last_val = x[-1]
    x_norm = x - last_val

    model.eval()
    with torch.no_grad():
        inp = torch.tensor(x_norm, dtype=torch.float32, device=device).unsqueeze(0)
        pred_norm = model(inp).squeeze(0).cpu().numpy()

    # De-normalize
    pred_log = pred_norm + last_val
    return np.expm1(pred_log)  # inverse of log1p


def dlinear_predict(
    train: pd.DataFrame,
    future_dates: pd.Series,
    lookback: int = LOOKBACK,
) -> pd.DataFrame:
    """
    Train DLinear on historical sales and forecast future dates.
    Compatible with the meta_ensemble_search.py interface.

    Parameters
    ----------
    train : DataFrame with columns [Date, Revenue, COGS]
    future_dates : Series of future dates to predict
    lookback : lookback window size

    Returns
    -------
    DataFrame with columns [Date, Revenue, COGS]
    """
    train_sorted = train.sort_values("Date").reset_index(drop=True)
    horizon = len(future_dates)
    result = pd.DataFrame({"Date": future_dates.values})

    for target in TARGETS:
        series = train_sorted[target].to_numpy(dtype=float)
        model = _train_dlinear(series, horizon, lookback=lookback)
        preds = _predict_dlinear(model, series, lookback=lookback)
        # Clip negatives
        preds = np.maximum(preds, 0.0)
        result[target] = preds[:horizon]

    return result
