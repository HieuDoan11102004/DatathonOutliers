from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import timesfm

TARGETS = ("Revenue", "COGS")


def _backend() -> str:
    return os.getenv("TIMESFM_BACKEND", "gpu")


def _load_model(horizon: int) -> timesfm.TimesFm:
    repo_id = os.getenv("TIMESFM_REPO", "google/timesfm-1.0-200m-pytorch")
    local_dir = os.getenv("TIMESFM_LOCAL_DIR", ".cache/timesfm")
    context_len = int(os.getenv("TIMESFM_CONTEXT_LEN", "512"))

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
            backend=_backend(),
            point_forecast_mode="median",
        ),
        checkpoint=timesfm.TimesFmCheckpoint(
            version="torch",
            huggingface_repo_id=repo_id,
            local_dir=local_dir,
        ),
    )


def build_timesfm_submission(
    train_file: str = "sales.csv",
    sample_file: str = "sample_submission.csv",
    output_file: str = "submission_timesfm.csv",
) -> pd.DataFrame:
    train = pd.read_csv(train_file, parse_dates=["Date"]).sort_values("Date")
    sample = pd.read_csv(sample_file, parse_dates=["Date"])[["Date"]]
    horizon = len(sample)

    print(f"loading TimesFM for horizon={horizon}")
    model = _load_model(horizon=horizon)

    submission = sample.copy()
    for target in TARGETS:
        history = train[target].astype(float).to_numpy()
        point_forecast, _ = model.forecast(
            inputs=[history],
            freq=[0],  # Daily frequency is high-frequency in TimesFM.
            normalize=True,
        )
        preds = np.maximum(point_forecast[0, :horizon], 0.0)
        submission[target] = np.round(preds, 2)

    submission["Date"] = submission["Date"].dt.strftime("%Y-%m-%d")
    submission.to_csv(output_file, index=False)
    return submission


def main() -> None:
    output_path = Path(os.getenv("TIMESFM_OUTPUT", "submission_timesfm.csv"))
    submission = build_timesfm_submission(output_file=str(output_path))
    print(f"saved {len(submission)} rows to {output_path}")
    print(submission.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
