"""
Evaluation: metrics, visualizations, feature importance.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pipeline.config import PLOTS_DIR


def print_metrics(label, y_true, y_pred):
    """Print MAE, RMSE, R² for a given set of predictions."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"  {label:40s} | MAE: {mae:>12,.0f} | RMSE: {rmse:>12,.0f} | R²: {r2:.4f}")
    return {'label': label, 'mae': mae, 'rmse': rmse, 'r2': r2}


def plot_predictions(dates, actual, predicted, title, filename='predictions.png'):
    """Plot actual vs predicted Revenue."""
    os.makedirs(PLOTS_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(dates, actual, lw=0.9, label='Actual Revenue', alpha=0.8)
    ax.plot(dates, predicted, lw=0.9, label='Predicted Revenue', alpha=0.8, linestyle='--')
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Date')
    ax.set_ylabel('Revenue')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, filename), dpi=150)
    plt.close()
    print(f"  📊 Saved: {filename}")


def plot_training_history(history, filename='patchtst_loss.png'):
    """Plot PatchTST training history: loss + RMSE."""
    os.makedirs(PLOTS_DIR, exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(12, 5))

    # Left axis: Loss (SmoothL1)
    ax1.plot(history['train_loss'], label='Train Loss', lw=1.2, color='steelblue')
    ax1.plot(history['val_loss'], label='Val Loss', lw=1.2, color='coral')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss (SmoothL1)')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # Right axis: Val RMSE (Revenue scale)
    ax2 = ax1.twinx()
    ax2.plot(history['val_rmse'], label='Val RMSE', lw=1.2, color='green', linestyle='--')
    ax2.set_ylabel('Val RMSE (Revenue)')
    ax2.legend(loc='upper right')

    # Mark best RMSE
    best_epoch = np.argmin(history['val_rmse'])
    best_rmse = history['val_rmse'][best_epoch]
    ax2.axvline(best_epoch, color='green', linestyle=':', alpha=0.5)
    ax2.annotate(f'Best: {best_rmse:,.0f}\n(epoch {best_epoch+1})',
                 xy=(best_epoch, best_rmse), fontsize=9,
                 arrowprops=dict(arrowstyle='->', color='green'),
                 xytext=(best_epoch + 5, best_rmse * 1.1))

    fig.suptitle('PatchTST Training History', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, filename), dpi=150)
    plt.close()
    print(f"  📊 Saved: {filename}")


def plot_feature_importance(model, feature_names, top_n=30,
                            filename='feature_importance.png'):
    """Plot LightGBM feature importance (gain)."""
    os.makedirs(PLOTS_DIR, exist_ok=True)
    importances = pd.Series(model.feature_importances_, index=feature_names)
    top = importances.nlargest(top_n)

    fig, ax = plt.subplots(figsize=(10, 8))
    top.sort_values().plot(kind='barh', ax=ax, color='steelblue')
    ax.set_title(f'Top {top_n} Feature Importance (Gain)', fontsize=14)
    ax.set_xlabel('Importance')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, filename), dpi=150)
    plt.close()
    print(f"  📊 Saved: {filename}")


def plot_residuals(y_true, y_pred, dates=None, filename='residuals.png'):
    """Plot residual analysis."""
    os.makedirs(PLOTS_DIR, exist_ok=True)
    residuals = y_true - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Residual distribution
    axes[0].hist(residuals, bins=50, color='steelblue', edgecolor='white', alpha=0.7)
    axes[0].set_title('Residual Distribution')
    axes[0].set_xlabel('Residual (Actual - Predicted)')
    axes[0].axvline(0, color='red', linestyle='--', lw=1)

    # Residuals over time
    if dates is not None:
        axes[1].scatter(dates, residuals, s=5, alpha=0.5, color='steelblue')
        axes[1].set_title('Residuals Over Time')
        axes[1].set_xlabel('Date')
    else:
        axes[1].scatter(range(len(residuals)), residuals, s=5, alpha=0.5, color='steelblue')
        axes[1].set_title('Residuals Over Index')

    axes[1].axhline(0, color='red', linestyle='--', lw=1)
    axes[1].set_ylabel('Residual')

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, filename), dpi=150)
    plt.close()
    print(f"  📊 Saved: {filename}")


def print_fold_summary(fold_metrics):
    """Print K-Fold CV metrics summary table."""
    print("\n  ┌────────┬──────────────┬──────────────┬──────────┐")
    print("  │  Fold  │     MAE      │     RMSE     │    R²    │")
    print("  ├────────┼──────────────┼──────────────┼──────────┤")
    for m in fold_metrics:
        print(f"  │ {m['fold']:>4d}   │ {m['mae']:>12,.0f} │ {m['rmse']:>12,.0f} │ {m['r2']:>8.4f} │")
    print("  ├────────┼──────────────┼──────────────┼──────────┤")
    avg_mae = np.mean([m['mae'] for m in fold_metrics])
    avg_rmse = np.mean([m['rmse'] for m in fold_metrics])
    avg_r2 = np.mean([m['r2'] for m in fold_metrics])
    print(f"  │  Avg   │ {avg_mae:>12,.0f} │ {avg_rmse:>12,.0f} │ {avg_r2:>8.4f} │")
    print("  └────────┴──────────────┴──────────────┴──────────┘")
