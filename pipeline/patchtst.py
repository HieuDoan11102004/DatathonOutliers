"""
PatchTST — Multivariate Patch Time Series Transformer (from scratch).

Input: (batch, context_len, n_features) — Revenue + time features
Output: (batch,) — predicted log1p(Revenue)

Key features vs original:
  - Multivariate: 10 channels (Revenue, cyclical time, holidays)
  - RevIN on Revenue channel only
  - Positional encoding + Dropout + LayerNorm
  - AdamW + CosineAnnealing + Early stopping
  - Gradient clipping
"""
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pipeline.config import *


class PatchTST(nn.Module):
    """
    Multivariate Patch Time Series Transformer.
    
    Input:  (batch, context_len, n_features)
            Channel 0 = log1p(Revenue), channels 1+ = time features
    Output: (batch,) — predicted log1p(Revenue)
    """

    def __init__(self, n_features, context_len=PATCHTST_CONTEXT_LEN,
                 patch_len=PATCHTST_PATCH_LEN, stride=PATCHTST_STRIDE,
                 d_model=PATCHTST_D_MODEL, n_heads=PATCHTST_N_HEADS,
                 n_layers=PATCHTST_N_LAYERS, d_ff=PATCHTST_D_FF,
                 dropout=PATCHTST_DROPOUT):
        super().__init__()
        self.context_len = context_len
        self.patch_len = patch_len
        self.stride = stride
        self.n_features = n_features
        self.n_patches = (context_len - patch_len) // stride + 1

        # Patch embedding: flatten (patch_len × n_features) → d_model
        self.patch_embed = nn.Linear(patch_len * n_features, d_model)

        # Learnable positional encoding
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.n_patches, d_model) * 0.02
        )
        self.embed_dropout = nn.Dropout(dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            activation='gelu',
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.encoder_norm = nn.LayerNorm(d_model)

        # Prediction head
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(d_model * self.n_patches),
            nn.Linear(d_model * self.n_patches, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, 1),
        )

    def forward(self, x):
        """
        x: (batch, context_len, n_features)
        Returns: (batch,) — predicted log1p(Revenue)
        """
        batch_size = x.shape[0]

        # -- RevIN on Revenue channel (channel 0) only --
        rev_mean = x[:, :, 0].mean(dim=1, keepdim=True)   # (batch, 1)
        rev_std = x[:, :, 0].std(dim=1, keepdim=True) + 1e-5

        x_norm = x.clone()
        x_norm[:, :, 0] = (x[:, :, 0] - rev_mean) / rev_std

        # -- Create patches along time axis --
        # unfold(1, patch_len, stride): (batch, n_patches, n_features, patch_len)
        patches = x_norm.unfold(1, self.patch_len, self.stride)
        # Rearrange to (batch, n_patches, patch_len * n_features)
        patches = patches.permute(0, 1, 3, 2).contiguous()
        patches = patches.reshape(batch_size, self.n_patches, -1)

        # -- Embed patches --
        z = self.patch_embed(patches) + self.pos_embed
        z = self.embed_dropout(z)

        # -- Transformer encoder --
        z = self.encoder(z)
        z = self.encoder_norm(z)

        # -- Prediction head --
        out = self.head(z).squeeze(-1)

        # -- RevIN denormalize --
        out = out * rev_std.squeeze(1) + rev_mean.squeeze(1)

        return out

    def extract_embeddings(self, x):
        """Extract hidden state embeddings for LightGBM (mean pooled over patches)."""
        batch_size = x.shape[0]

        # RevIN
        rev_mean = x[:, :, 0].mean(dim=1, keepdim=True)
        rev_std = x[:, :, 0].std(dim=1, keepdim=True) + 1e-5
        x_norm = x.clone()
        x_norm[:, :, 0] = (x[:, :, 0] - rev_mean) / rev_std

        # Patches
        patches = x_norm.unfold(1, self.patch_len, self.stride)
        patches = patches.permute(0, 1, 3, 2).contiguous()
        patches = patches.reshape(batch_size, self.n_patches, -1)

        # Embed + Positional
        z = self.patch_embed(patches) + self.pos_embed
        
        # Transformer
        z = self.encoder(z)
        z = self.encoder_norm(z)

        # Mean pooling to get 1 vector per window -> (batch, d_model)
        return z.mean(dim=1)



# ============================================================
# Dataset Builder
# ============================================================

def build_sliding_windows(features_array, context_len):
    """
    Build sliding window dataset from multivariate features.
    
    Args:
        features_array: (n_days, n_features) — channel 0 = log1p(Revenue)
        context_len: lookback window
    
    Returns:
        X: (N, context_len, n_features) — input windows
        y: (N,) — target = log1p(Revenue) at next step
    """
    X, y = [], []
    for i in range(context_len, len(features_array)):
        X.append(features_array[i - context_len:i])       # (context_len, n_features)
        y.append(features_array[i, 0])                     # log1p(Revenue) at time t
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


# ============================================================
# Training
# ============================================================

def train_patchtst(features_array, context_len=PATCHTST_CONTEXT_LEN,
                   epochs=PATCHTST_EPOCHS, patience=PATCHTST_PATIENCE,
                   batch_size=PATCHTST_BATCH_SIZE, val_split=0.85):
    """
    Train multivariate PatchTST.
    
    Args:
        features_array: (n_days, n_features)
                        Column 0 = log1p(Revenue), rest = time features
        context_len: lookback window
        val_split: chronological split for internal validation
    
    Returns:
        model: trained PatchTST
        history: dict with train/val loss per epoch
    """
    set_seed(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_features = features_array.shape[1]
    print(f"  PatchTST device: {device}")
    print(f"  Input: {n_features} features × {context_len} context")

    # Build sliding windows
    X, y = build_sliding_windows(features_array, context_len)
    print(f"  Sliding windows: {len(X)} samples")

    # Chronological split for internal validation  
    n_train = int(val_split * len(X))
    X_tr, y_tr = X[:n_train], y[:n_train]
    X_va, y_va = X[n_train:], y[n_train:]
    print(f"  PT-Train: {len(X_tr)}, PT-Val: {len(X_va)}")

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)),
        batch_size=batch_size, shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_va), torch.from_numpy(y_va)),
        batch_size=batch_size, shuffle=False,
    )

    # Model
    model = PatchTST(n_features=n_features, context_len=context_len).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {n_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=PATCHTST_LR, weight_decay=PATCHTST_WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.SmoothL1Loss()

    best_val_rmse = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'val_rmse': []}
    best_path = os.path.join(OUTPUT_DIR, 'best_patchtst.pth')

    for epoch in range(epochs):
        # -- Train --
        model.train()
        train_loss = 0.0
        n_train_samples = 0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            pred = model(bx)
            loss = criterion(pred, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * len(bx)
            n_train_samples += len(bx)

        train_loss /= n_train_samples

        # -- Validate --
        model.eval()
        val_loss = 0.0
        val_preds_list = []
        val_targets_list = []
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                pred = model(bx)
                val_loss += criterion(pred, by).item() * len(bx)
                val_preds_list.append(pred.cpu())
                val_targets_list.append(by.cpu())

        val_loss /= len(X_va)
        val_preds_cat = torch.cat(val_preds_list)
        val_targets_cat = torch.cat(val_targets_list)
        val_rmse = torch.sqrt(torch.mean(
            (torch.expm1(val_preds_cat) - torch.expm1(val_targets_cat)) ** 2
        )).item()

        scheduler.step()
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_rmse'].append(val_rmse)

        # -- Early Stopping (based on val RMSE) --
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            patience_counter = 0
            torch.save(model.state_dict(), best_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  ⏹ Early stopping at epoch {epoch + 1}")
                break

        if (epoch + 1) % 10 == 0:
            lr = optimizer.param_groups[0]['lr']
            print(f"  Epoch {epoch+1:3d} | Train Loss: {train_loss:.6f} | "
                  f"Val Loss: {val_loss:.6f} | Val RMSE: {val_rmse:,.0f} | LR: {lr:.2e}")

    # Load best checkpoint
    model.load_state_dict(torch.load(best_path, weights_only=True))
    model.eval()
    print(f"  ✅ PatchTST best checkpoint — Val RMSE: {best_val_rmse:,.0f}")
    return model, history


# ============================================================
# Prediction
# ============================================================

def predict_patchtst_teacherforced(model, features_array, start_idx, n_steps,
                                    context_len=PATCHTST_CONTEXT_LEN, return_embeds=False):
    """
    Teacher-forcing: uses TRUE preceding features as context.
    No error accumulation — each prediction uses actual historical values.
    
    Use for val/test where true Revenue is known.
    """
    device = next(model.parameters()).device
    model.eval()
    predictions = []
    embeddings = []

    with torch.no_grad():
        for i in range(n_steps):
            idx = start_idx + i
            context = features_array[idx - context_len:idx]  # (context_len, n_features)
            x = torch.FloatTensor(context).unsqueeze(0).to(device)
            pred_log = model(x).item()
            predictions.append(max(np.expm1(pred_log), 0))
            if return_embeds:
                emb = model.extract_embeddings(x).cpu().numpy()[0]
                embeddings.append(emb)

    if return_embeds:
        return np.array(predictions), np.array(embeddings)
    return np.array(predictions)


def predict_patchtst_recursive(model, known_features, future_features,
                                context_len=PATCHTST_CONTEXT_LEN, return_embeds=False):
    """
    Recursive prediction for submission: future Revenue is unknown.
    """
    device = next(model.parameters()).device
    model.eval()

    # Initialize context from last context_len days of known data
    context = list(known_features[-context_len:])
    predictions = []
    embeddings = []

    with torch.no_grad():
        for i in range(len(future_features)):
            ctx_array = np.array(context[-context_len:], dtype=np.float32)
            x = torch.FloatTensor(ctx_array).unsqueeze(0).to(device)
            pred_log = model(x).item()
            pred_revenue = max(np.expm1(pred_log), 0)
            predictions.append(pred_revenue)
            
            if return_embeds:
                emb = model.extract_embeddings(x).cpu().numpy()[0]
                embeddings.append(emb)

            # Build new context row: true time features + predicted Revenue
            new_row = future_features[i].copy().astype(np.float32)
            new_row[0] = np.log1p(pred_revenue)  # Fill Revenue channel
            context.append(new_row)

    if return_embeds:
        return np.array(predictions), np.array(embeddings)
    return np.array(predictions)
