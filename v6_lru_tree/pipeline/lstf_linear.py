"""
LSTF-Linear Family: DLinear, NLinear, Linear
From "Are Transformers Effective for Time Series Forecasting?" (Zeng et al., 2023)

Cập nhật so với version cũ:
  - MultiVar variants (DLinear / NLinear): nhận N input channels, predict 1 output channel.
    Dùng Revenue + COGS cùng lúc → model thấy được cost structure khi forecast.
  - build_linear_feature:   tạo linear_pred cho Revenue (có thể multivariate input).
  - build_cogs_feature:     tạo cogs_linear_pred (COGS prediction làm feature phụ cho Trees).
  - train_linear:           unified entry — univariate hoặc multivariate tuỳ tham số.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


# ─────────────────────────────────────────────
#  DECOMPOSITION PRIMITIVES
# ─────────────────────────────────────────────

class _MovingAvg(nn.Module):
    """Causal-padded moving average để giữ time alignment."""
    def __init__(self, kernel_size: int, stride: int = 1):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, C]
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end   = x[:, -1:, :].repeat(1, self.kernel_size // 2, 1)
        x = torch.cat([front, x, end], dim=1)          # pad
        x = self.avg(x.permute(0, 2, 1)).permute(0, 2, 1)  # [B, L, C]
        return x


class _SeriesDecomp(nn.Module):
    def __init__(self, kernel_size: int):
        super().__init__()
        self.moving_avg = _MovingAvg(kernel_size)

    def forward(self, x: torch.Tensor):
        trend    = self.moving_avg(x)
        seasonal = x - trend
        return seasonal, trend


# ─────────────────────────────────────────────
#  UNIVARIATE MODELS (backward compatible)
# ─────────────────────────────────────────────

class NLinear(nn.Module):
    """
    NLinear univariate.
    Subtract last value → linear → add back.
    Mạnh khi series có strong local level (cần giữ magnitude).
    """
    def __init__(self, seq_len: int, pred_len: int, channels: int = 1):
        super().__init__()
        self.seq_len  = seq_len
        self.pred_len = pred_len
        self.linear   = nn.Linear(seq_len, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, 1]
        last = x[:, -1:, :].detach()
        x    = (x - last).permute(0, 2, 1)   # [B, 1, L]
        x    = self.linear(x).permute(0, 2, 1)  # [B, P, 1]
        return x + last


class DLinear(nn.Module):
    """
    DLinear univariate.
    Separate linear heads cho Trend và Seasonal component.
    """
    def __init__(self, seq_len: int, pred_len: int, channels: int = 1,
                 kernel_size: int = 25):
        super().__init__()
        self.decomp          = _SeriesDecomp(kernel_size)
        self.linear_seasonal = nn.Linear(seq_len, pred_len)
        self.linear_trend    = nn.Linear(seq_len, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seasonal, trend = self.decomp(x)
        s = self.linear_seasonal(seasonal.permute(0, 2, 1))  # [B, 1, P]
        t = self.linear_trend(trend.permute(0, 2, 1))
        return (s + t).permute(0, 2, 1)   # [B, P, 1]


class PlainLinear(nn.Module):
    """Linear: direct seq_len → pred_len mapping."""
    def __init__(self, seq_len: int, pred_len: int, channels: int = 1):
        super().__init__()
        self.linear = nn.Linear(seq_len, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x.permute(0, 2, 1)).permute(0, 2, 1)


# ─────────────────────────────────────────────
#  MULTIVARIATE INPUT → UNIVARIATE OUTPUT
# ─────────────────────────────────────────────

class MultiVarNLinear(nn.Module):
    """
    NLinear multivariate input, univariate output.
    Input : [B, L, in_channels]  (e.g. Revenue + COGS → in_channels=2)
    Output: [B, P, 1]            (Revenue prediction only)

    Mỗi input channel có linear head riêng, sau đó được học weight mix
    qua một linear layer nhỏ (channel fusion).
    """
    def __init__(self, seq_len: int, pred_len: int, in_channels: int = 2):
        super().__init__()
        self.seq_len    = seq_len
        self.pred_len   = pred_len
        self.in_channels = in_channels

        # Một linear head riêng mỗi channel
        self.channel_linears = nn.ModuleList([
            nn.Linear(seq_len, pred_len) for _ in range(in_channels)
        ])
        # Channel fusion: [B, P, in_channels] → [B, P, 1]
        self.fusion = nn.Linear(in_channels, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, C]
        last = x[:, -1:, :].detach()   # [B, 1, C]
        x_norm = x - last              # [B, L, C]

        channel_preds = []
        for i, lin in enumerate(self.channel_linears):
            ch = x_norm[:, :, i]                  # [B, L]
            ch_pred = lin(ch).unsqueeze(-1)        # [B, P, 1]
            channel_preds.append(ch_pred)

        stacked = torch.cat(channel_preds, dim=-1)  # [B, P, C]
        fused   = self.fusion(stacked)              # [B, P, 1]

        # Denormalize: add back only Revenue (channel 0) last value
        revenue_last = last[:, :, 0:1]             # [B, 1, 1]
        return fused + revenue_last


class MultiVarDLinear(nn.Module):
    """
    DLinear multivariate input, univariate output.
    Decompose mỗi channel, project riêng, fusion thành Revenue prediction.
    """
    def __init__(self, seq_len: int, pred_len: int, in_channels: int = 2,
                 kernel_size: int = 25):
        super().__init__()
        self.in_channels = in_channels
        self.decomp      = _SeriesDecomp(kernel_size)

        self.seasonal_heads = nn.ModuleList([
            nn.Linear(seq_len, pred_len) for _ in range(in_channels)
        ])
        self.trend_heads = nn.ModuleList([
            nn.Linear(seq_len, pred_len) for _ in range(in_channels)
        ])
        self.fusion = nn.Linear(in_channels, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, C]
        seasonal, trend = self.decomp(x)   # both [B, L, C]

        channel_preds = []
        for i in range(self.in_channels):
            s = self.seasonal_heads[i](seasonal[:, :, i])  # [B, P]
            t = self.trend_heads[i](trend[:, :, i])
            channel_preds.append((s + t).unsqueeze(-1))    # [B, P, 1]

        stacked = torch.cat(channel_preds, dim=-1)   # [B, P, C]
        return self.fusion(stacked)                  # [B, P, 1]


# ─────────────────────────────────────────────
#  MODEL FACTORY
# ─────────────────────────────────────────────

_UNIVAR_MODELS = {
    'NLinear': NLinear,
    'DLinear': DLinear,
    'Linear':  PlainLinear,
}
_MULTIVAR_MODELS = {
    'MultiVarNLinear': MultiVarNLinear,
    'MultiVarDLinear': MultiVarDLinear,
}
MODELS = {**_UNIVAR_MODELS, **_MULTIVAR_MODELS}


def get_model(model_name: str, seq_len: int, pred_len: int,
              in_channels: int = 1) -> nn.Module:
    if model_name not in MODELS:
        raise ValueError(f"Unknown model '{model_name}'. Choose from {list(MODELS.keys())}")
    if model_name in _MULTIVAR_MODELS:
        return MODELS[model_name](seq_len=seq_len, pred_len=pred_len, in_channels=in_channels)
    return MODELS[model_name](seq_len=seq_len, pred_len=pred_len)


# ─────────────────────────────────────────────
#  DATASET BUILDER
# ─────────────────────────────────────────────

def _make_dataset(y_sequences: np.ndarray, pred_len: int, seq_len: int):
    """
    Tạo sliding window dataset.
    y_sequences: [N, in_channels] hoặc [N] (univariate)
    Returns TensorDataset X [n, L, C], Y [n, P, 1]
    """
    univariate = y_sequences.ndim == 1
    if univariate:
        y_sequences = y_sequences[:, np.newaxis]   # [N, 1]

    n_channels = y_sequences.shape[1]
    X_list, Y_list = [], []

    n = len(y_sequences)
    for i in range(n - seq_len - pred_len):
        X_list.append(y_sequences[i: i + seq_len])           # [L, C]
        Y_list.append(y_sequences[i + seq_len: i + seq_len + pred_len, 0:1])  # [P, 1] revenue only

    if len(X_list) == 0:
        raise ValueError(f"Series too short for seq_len={seq_len}, pred_len={pred_len}")

    X = torch.tensor(np.array(X_list), dtype=torch.float32)   # [n, L, C]
    Y = torch.tensor(np.array(Y_list), dtype=torch.float32)   # [n, P, 1]
    return torch.utils.data.TensorDataset(X, Y), n_channels


# ─────────────────────────────────────────────
#  TRAINING ENGINE
# ─────────────────────────────────────────────

def train_linear(
    y_train: np.ndarray,
    seq_len: int,
    pred_len: int,
    model_name: str = 'NLinear',
    y_aux: np.ndarray = None,       # optional second channel (COGS)
    batch_size: int = 32,
    epochs: int = 100,
    lr: float = 1e-3,
    patience: int = 10,
) -> nn.Module:
    """
    Train một LSTF-Linear model.

    Params
    ------
    y_train : 1-D array, log_revenue values (training set only).
    y_aux   : 1-D array, cùng length với y_train. Nếu truyền vào thì
              model_name nên là 'MultiVarDLinear' / 'MultiVarNLinear'.
              y_aux thường là log(COGS+1).
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    multivar = (y_aux is not None)

    if multivar:
        assert len(y_aux) == len(y_train), "y_aux must be same length as y_train"
        sequences = np.stack([y_train, y_aux], axis=1)   # [N, 2]
        in_channels = 2
        # Auto-upgrade to multivar model if caller passed univar model name
        if model_name in _UNIVAR_MODELS:
            upgrade_map = {'DLinear': 'MultiVarDLinear', 'NLinear': 'MultiVarNLinear'}
            model_name  = upgrade_map.get(model_name, 'MultiVarDLinear')
            print(f"    [auto-upgrade] → {model_name} (multivariate input)")
    else:
        sequences   = y_train
        in_channels = 1

    print(f"    {model_name} | seq={seq_len} pred={pred_len} ch={in_channels} | device={device}")

    dataset, n_ch = _make_dataset(sequences, pred_len, seq_len)
    train_n  = int(0.9 * len(dataset))
    val_n    = len(dataset) - train_n
    tr_ds, vl_ds = torch.utils.data.random_split(dataset, [train_n, val_n])

    tr_loader = torch.utils.data.DataLoader(tr_ds, batch_size=batch_size, shuffle=True)
    vl_loader = torch.utils.data.DataLoader(vl_ds, batch_size=batch_size, shuffle=False)

    model     = get_model(model_name, seq_len, pred_len, in_channels=n_ch).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.1)
    criterion = nn.HuberLoss(delta=1.0)   # robust to Revenue spikes

    best_val   = float('inf')
    best_state = None
    no_improve = 0

    for epoch in range(epochs):
        # ── Train ──
        model.train()
        tr_loss = 0.0
        for bx, by in tr_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            tr_loss += loss.item() * bx.size(0)
        tr_loss /= len(tr_loader.dataset)

        # ── Validate ──
        model.eval()
        vl_loss = 0.0
        with torch.no_grad():
            for bx, by in vl_loader:
                bx, by = bx.to(device), by.to(device)
                vl_loss += criterion(model(bx), by).item() * bx.size(0)
        vl_loss /= len(vl_loader.dataset)
        scheduler.step()

        improved = vl_loss < best_val
        if improved:
            best_val   = vl_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
            tag = " *"
        else:
            no_improve += 1
            tag = ""

        if (epoch + 1) % 10 == 0 or improved:
            print(f"      Ep [{epoch+1:3d}/{epochs}]  tr={tr_loss:.4f}  vl={vl_loss:.4f}"
                  f"  vRMSE={np.sqrt(vl_loss):.4f}{tag}")

        if no_improve >= patience:
            print(f"      Early stop @ ep {epoch+1}. Best vl={best_val:.4f} "
                  f"(RMSE={np.sqrt(best_val):.4f})")
            break

    model.load_state_dict(best_state)
    return model


# ─────────────────────────────────────────────
#  FEATURE BUILDER — Revenue
# ─────────────────────────────────────────────

def build_linear_feature(
    model: nn.Module,
    y_train: np.ndarray,
    full_len: int,
    seq_len: int,
    pred_len: int,
    y_aux_train: np.ndarray = None,   # COGS aligned to y_train (train portion only)
    y_aux_full: np.ndarray  = None,   # COGS for the full range (train + test)
) -> np.ndarray:
    """
    Tạo 'linear_pred' feature cho toàn bộ timeline (train + test).

    Multivariate mode: truyền y_aux_train + y_aux_full.
      - Phase A (in-sample): cả hai channel đều known → dùng multivar input.
      - Phase B (out-of-sample): COGS không có → fill bằng last known COGS
        (COGS rất stable so với Revenue nên xấp xỉ này chấp nhận được).

    Phase A: sliding window sweep → multi-view average.
    Phase B: autoregressive unrolling cho out-of-sample dates.
    """
    device = next(model.parameters()).device
    model.eval()

    multivar = (y_aux_train is not None) and (y_aux_full is not None)

    # Determine in_channels from model
    in_ch = 1
    if hasattr(model, 'in_channels'):
        in_ch = model.in_channels
    elif hasattr(model, 'channel_linears'):
        in_ch = len(model.channel_linears)

    # Build full revenue array (test portion = 0, filled autoregressively)
    y_full = np.zeros(full_len)
    y_full[:len(y_train)] = y_train

    # Build full COGS array (test portion filled with last known value)
    if multivar:
        aux_full = np.zeros(full_len)
        aux_full[:len(y_aux_train)] = y_aux_train
        if y_aux_full is not None:
            aux_full = y_aux_full.copy()
        else:
            # Extrapolate: use mean of last 90 days cyclically
            cogs_base = y_aux_train[-365:] if len(y_aux_train) >= 365 else y_aux_train
            for i in range(len(y_aux_train), full_len):
                aux_full[i] = cogs_base[i % len(cogs_base)]

    linear_feat = np.zeros(full_len)
    counts      = np.zeros(full_len, dtype=int)

    def _make_x_tensor(rev_seq, cogs_seq=None):
        """Build input tensor, shape [1, L, C]."""
        if multivar and cogs_seq is not None and in_ch == 2:
            seq = np.stack([rev_seq, cogs_seq], axis=1)   # [L, 2]
        else:
            seq = rev_seq[:, np.newaxis]                   # [L, 1]
        return torch.tensor(seq[np.newaxis], dtype=torch.float32).to(device)

    def _extract_pred(tensor_out):
        """[1, P, 1] → np array [P]."""
        return tensor_out.squeeze().cpu().numpy()

    with torch.no_grad():
        # ── Phase A: In-sample sweep ──────────────────────────────
        max_start = len(y_train) - seq_len
        for i in range(max_start + 1):
            rev_seq  = y_full[i: i + seq_len]
            cogs_seq = aux_full[i: i + seq_len] if multivar else None

            x_t  = _make_x_tensor(rev_seq, cogs_seq)
            pred = _extract_pred(model(x_t))
            if pred.ndim == 0:
                pred = pred.reshape(1)

            t_s = i + seq_len
            t_e = min(t_s + pred_len, full_len)
            n   = t_e - t_s
            linear_feat[t_s:t_e] += pred[:n]
            counts[t_s:t_e]      += 1

        # ── Phase B: Autoregressive out-of-sample ────────────────
        cur = len(y_train)
        while cur < full_len:
            start    = cur - seq_len
            rev_seq  = y_full[start: cur]
            cogs_seq = aux_full[start: cur] if multivar else None

            x_t  = _make_x_tensor(rev_seq, cogs_seq)
            pred = _extract_pred(model(x_t))
            if pred.ndim == 0:
                pred = pred.reshape(1)

            t_e = min(cur + pred_len, full_len)
            n   = t_e - cur
            # Inject into y_full for next autoregressive step
            y_full[cur:t_e]          = pred[:n]
            linear_feat[cur:t_e]    += pred[:n]
            counts[cur:t_e]         += 1
            cur += pred_len

    # Average overlapping predictions (Phase A only; Phase B has count=1)
    mask = counts > 0
    linear_feat[mask]  /= counts[mask]
    linear_feat[~mask]  = np.mean(linear_feat[mask][:365]) if mask.sum() > 0 else 0.0

    return linear_feat


# ─────────────────────────────────────────────
#  FEATURE BUILDER — COGS (extra tree feature)
# ─────────────────────────────────────────────

def build_cogs_feature(
    y_cogs_train: np.ndarray,
    full_len: int,
    seq_len: int,
    pred_len: int,
    model_name: str = 'DLinear',
    batch_size: int = 32,
    epochs: int = 80,
    lr: float = 1e-3,
    patience: int = 10,
) -> np.ndarray:
    """
    Train một linear model riêng cho COGS, tạo 'cogs_linear_pred' feature.
    COGS ổn định hơn Revenue → model hội tụ nhanh hơn.
    """
    print("    Training COGS linear model...")
    cogs_model = train_linear(
        y_train=y_cogs_train,
        seq_len=seq_len, pred_len=pred_len,
        model_name=model_name,
        batch_size=batch_size, epochs=epochs, lr=lr, patience=patience,
    )
    cogs_feat = build_linear_feature(
        model=cogs_model,
        y_train=y_cogs_train,
        full_len=full_len,
        seq_len=seq_len, pred_len=pred_len,
    )
    print(f"    cogs_linear_pred generated — mean={cogs_feat.mean():.4f}")
    return cogs_feat