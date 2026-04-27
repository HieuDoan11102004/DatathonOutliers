"""
DLinear PyTorch Implementation for Direct Multi-Step Forecasting.
Separates Time Series into Trend and Seasonal components natively.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size // 2), 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x

class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

class DLinear(nn.Module):
    """
    DLinear Model: Linear approach to Decomposition.
    """
    def __init__(self, seq_len, pred_len, channels=1, kernel_size=25):
        super(DLinear, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        # Decomposition block
        self.decompsition = series_decomp(kernel_size)
        
        # Individual layers for Seasonal and Trend
        self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
        self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)
        
    def forward(self, x):
        # x: [Batch, Seq_len, Channels]
        seasonal_init, trend_init = self.decompsition(x)
        
        # We need independent projections for each channel (or shared). 
        # Here we perform projection over the sequence dimension.
        seasonal_init = seasonal_init.permute(0, 2, 1)  # [Batch, Channels, Seq_len]
        trend_init = trend_init.permute(0, 2, 1)        # [Batch, Channels, Seq_len]
        
        seasonal_output = self.Linear_Seasonal(seasonal_init) # [Batch, Channels, Pred_len]
        trend_output = self.Linear_Trend(trend_init)          # [Batch, Channels, Pred_len]
        
        x = seasonal_output + trend_output
        return x.permute(0, 2, 1) # [Batch, Pred_len, Channels]

def train_dlinear(y_train, seq_len, pred_len, batch_size, epochs, lr, patience=10):
    """
    Train DLinear using continuous sliding windows over the training sequence.
    y_train: 1D numpy array of historical Revenue (or log Revenue).
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"    DLinear running on: {device}")
    
    # Create dataset
    X, Y = [], []
    for i in range(len(y_train) - seq_len - pred_len):
        X.append(y_train[i:i + seq_len])
        Y.append(y_train[i + seq_len:i + seq_len + pred_len])
    
    if len(X) == 0:
        raise ValueError(f"Training sequence too short for seq_len={seq_len} and pred_len={pred_len}")
        
    X = torch.tensor(np.array(X), dtype=torch.float32).unsqueeze(-1)
    Y = torch.tensor(np.array(Y), dtype=torch.float32).unsqueeze(-1)
    
    dataset = torch.utils.data.TensorDataset(X, Y)
    # Split 90/10 for internal validation
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    model = DLinear(seq_len=seq_len, pred_len=pred_len).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    best_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_x.size(0)
        train_loss /= len(train_loader.dataset)
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item() * batch_x.size(0)
        val_loss /= len(val_loader.dataset)
        
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_state = model.state_dict()
            patience_counter = 0
            improved = "*"
        else:
            patience_counter += 1
            improved = ""
            
        # Log every 5 epochs or if improved. Convert MSE loss to RMSE metric.
        if (epoch + 1) % 5 == 0 or improved == "*":
            train_rmse = np.sqrt(train_loss)
            val_rmse = np.sqrt(val_loss)
            print(f"      Epoch [{epoch+1:03d}/{epochs:03d}] | Train RMSE: {train_rmse:.4f} | Val RMSE: {val_rmse:.4f} {improved}")
            
        if patience_counter >= patience:
            best_rmse = np.sqrt(best_loss)
            print(f"      Early stopping at epoch {epoch+1}. Best Val RMSE: {best_rmse:.4f}")
            break
            
    model.load_state_dict(best_model_state)
    return model

def build_dlinear_feature(model, y_train_pure, full_len, seq_len, pred_len):
    """
    Generate 'dlinear_pred' using strict autoregressive unrolling.
    """
    device = next(model.parameters()).device
    model.eval()
    
    y_full = np.zeros(full_len)
    y_full[:len(y_train_pure)] = y_train_pure
    
    dlinear_feat = np.zeros(full_len)
    counts = np.zeros(full_len)
    
    with torch.no_grad():
        # Sweep through pure train to get multi-view average for training set feature extraction
        max_train_start = len(y_train_pure) - seq_len
        for i in range(max_train_start + 1):
            x = y_full[i:i + seq_len]
            x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)
            pred = model(x_tensor).squeeze().cpu().numpy()
            
            target_start = i + seq_len
            target_end = min(target_start + pred_len, full_len)
            
            pred_valid = pred[:target_end - target_start]
            dlinear_feat[target_start:target_end] += pred_valid
            counts[target_start:target_end] += 1
                
        # Now we need to autoregressively unroll for the remainder of full_len out-of-sample
        current_idx = len(y_train_pure) 
        while current_idx < full_len:
            start_x = current_idx - seq_len
            x = y_full[start_x:current_idx]
            
            x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)
            pred = model(x_tensor).squeeze().cpu().numpy()
            
            target_end = min(current_idx + pred_len, full_len)
            pred_valid = pred[:target_end - current_idx]
            
            # Autoregressively inject predictions into the sequence so we can slide the window across them
            y_full[current_idx:target_end] = pred_valid
            
            # Also record it to dlinear_feat
            dlinear_feat[current_idx:target_end] += pred_valid
            counts[current_idx:target_end] += 1
            
            current_idx += pred_len
        
    mask = counts > 0
    dlinear_feat[mask] = dlinear_feat[mask] / counts[mask]
    
    if len(dlinear_feat[mask]) > 0:
        dlinear_feat[~mask] = np.mean(dlinear_feat[mask][:365])

    return dlinear_feat
