"""
Feature Engineering for Revenue Forecasting.
Loads sales.csv + auxiliary tables, creates all features.
"""
import pandas as pd
import numpy as np
from pipeline.config import *


def load_sales():
    """Load and sort sales data."""
    df = pd.read_csv(SALES_FILE, parse_dates=['Date'])
    return df.sort_values('Date').reset_index(drop=True)


def add_time_features(df):
    """Calendar and cyclical time features from Date."""
    d = df['Date']
    df['year'] = d.dt.year
    df['month'] = d.dt.month
    df['day'] = d.dt.day
    df['day_of_week'] = d.dt.dayofweek
    df['day_of_year'] = d.dt.dayofyear
    df['week_of_year'] = d.dt.isocalendar().week.astype(int)
    df['quarter'] = d.dt.quarter
    
    # Season (1=Spring, 2=Summer, 3=Autumn, 4=Winter) based on Vietnam typical seasons
    # Using simple month mapping
    season_map = {1: 4, 2: 1, 3: 1, 4: 1, 5: 2, 6: 2, 7: 2, 8: 3, 9: 3, 10: 3, 11: 4, 12: 4}
    df['season'] = df['month'].map(season_map)

    # Boolean flags
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['is_month_start'] = d.dt.is_month_start.astype(int)
    df['is_month_end'] = d.dt.is_month_end.astype(int)
    df['is_quarter_start'] = d.dt.is_quarter_start.astype(int)
    df['is_quarter_end'] = d.dt.is_quarter_end.astype(int)

    # Cyclical encoding (captures circular nature of calendar)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['doy_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
    df['doy_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)

    # Day position within month (0→1)
    df['day_of_month_ratio'] = df['day'] / d.dt.days_in_month

    return df


def add_holiday_features(df):
    """Vietnamese holidays and e-commerce events."""
    d = df['Date']

    # -- Tet Nguyen Dan (approximate: 7-day period) --
    tet_starts = {
        2013: '2013-02-09', 2014: '2014-01-30', 2015: '2015-02-17',
        2016: '2016-02-07', 2017: '2017-01-27', 2018: '2018-02-14',
        2019: '2019-02-02', 2020: '2020-01-23', 2021: '2021-02-10',
        2022: '2022-01-31', 2023: '2023-01-20', 2024: '2024-02-08',
    }
    df['is_tet'] = 0
    for year, start in tet_starts.items():
        start_dt = pd.Timestamp(start)
        mask = (d >= start_dt - pd.Timedelta(days=3)) & (d <= start_dt + pd.Timedelta(days=7))
        df.loc[mask, 'is_tet'] = 1

    # -- Fixed national holidays --
    df['is_national_holiday'] = 0
    for md in ['01-01', '04-30', '05-01', '09-02']:
        df.loc[d.dt.strftime('%m-%d') == md, 'is_national_holiday'] = 1

    # -- E-commerce mega-sale events --
    df['is_ecommerce_event'] = 0
    for md in ['11-11', '12-12']:
        mask = d.dt.strftime('%m-%d') == md
        df.loc[mask, 'is_ecommerce_event'] = 1
    # Black Friday: 4th Friday of November
    nov_mask = d.dt.month == 11
    fri_mask = d.dt.dayofweek == 4
    day_mask = d.dt.day.between(22, 28)
    df.loc[nov_mask & fri_mask & day_mask, 'is_ecommerce_event'] = 1

    # -- Payday effect (1st, 15th of month) --
    df['is_payday'] = ((d.dt.day == 1) | (d.dt.day == 15)).astype(int)

    # Combined holiday flag
    df['is_any_holiday'] = ((df['is_tet'] == 1) |
                            (df['is_national_holiday'] == 1) |
                            (df['is_ecommerce_event'] == 1)).astype(int)

    # -- Distance to holidays --
    # Default to a large number if no holiday found
    holiday_dates = df.loc[df['is_any_holiday'] == 1, 'Date'].values
    if len(holiday_dates) > 0:
        # Avoid apply for speed, use merge_asof or numpy searchsorted
        dates = df['Date'].values
        
        # Days since last holiday
        idx = np.searchsorted(holiday_dates, dates, side='right') - 1
        idx_safe = np.clip(idx, 0, len(holiday_dates) - 1)
        last_holidays = np.where(idx >= 0, holiday_dates[idx_safe], pd.NaT)
        df['days_since_last_holiday'] = (d - pd.to_datetime(last_holidays)).dt.days
        
        # Days to next holiday
        idx_next = np.searchsorted(holiday_dates, dates, side='left')
        idx_next_safe = np.clip(idx_next, 0, len(holiday_dates) - 1)
        next_holidays = np.where(idx_next < len(holiday_dates), holiday_dates[idx_next_safe], pd.NaT)
        df['days_to_next_holiday'] = (pd.to_datetime(next_holidays) - d).dt.days

        df['days_since_last_holiday'] = df['days_since_last_holiday'].fillna(365).clip(0, 365)
        df['days_to_next_holiday'] = df['days_to_next_holiday'].fillna(365).clip(0, 365)
    else:
        df['days_since_last_holiday'] = 365
        df['days_to_next_holiday'] = 365

    return df


def add_revenue_lag_features(df):
    """Lag, rolling, EWM, diff features from Revenue."""
    rev = df['Revenue']

    # Lag features (MUST be > 548 days because test set spans 1.5 years!)
    # 728 days = exactly 104 weeks (2 years ago, same day of week)
    # 730 days = exactly 2 years ago
    # 1092 days = exactly 156 weeks (3 years ago)
    for lag in [548, 728, 730, 1092]:
        df[f'rev_lag_{lag}'] = rev.shift(lag)

    # Rolling statistics based on the SAFE lag (728)
    shifted_safe = rev.shift(728)
    for w in [7, 30, 90]:
        df[f'rev_rmean_from728_{w}'] = shifted_safe.rolling(w, min_periods=1).mean()
        df[f'rev_rstd_from728_{w}'] = shifted_safe.rolling(w, min_periods=1).std()

    return df


def add_cogs_features(df):
    """
    KAGGLE RULE COMPLIANCE:
    'Sử dụng Revenue/COGS từ tập test làm đặc trưng sẽ bị LOẠI BÀI'
    We MUST NOT extract any features from COGS.
    """
    return df

def add_auxiliary_features(df):
    """
    KAGGLE LONG-HORIZON COMPLIANCE:
    Test set is 2023-01-01 to 2024-07-01 (548 days).
    Auxiliary tables (orders, web_traffic, payments, etc.) ONLY have data up to 2022.
    If we map them to Date directly, they will all be NaN for the test set.
    Filling them with 0 will cause the Model to output 0 for Revenue.
    
    Therefore, auxiliary tables cannot be used for DIRECT multi-step forecasting 
    unless heavily aggregated as static mapping. For now, we drop them to guarantee
    robustness of the Tree Ensemble.
    """
    return df


def add_seasonal_features(df):
    """Historical seasonal averages (computed only from known Revenue)."""
    # These features capture "what's typical for this time of year"
    known = df[df['Revenue'].notna()].copy()

    # Average Revenue by month (across all years)
    month_avg = known.groupby('month')['Revenue'].mean().to_dict()
    df['rev_month_avg'] = df['month'].map(month_avg)

    # Average Revenue by day_of_week
    dow_avg = known.groupby('day_of_week')['Revenue'].mean().to_dict()
    df['rev_dow_avg'] = df['day_of_week'].map(dow_avg)

    # Average Revenue by week_of_year
    woy_avg = known.groupby('week_of_year')['Revenue'].mean().to_dict()
    df['rev_woy_avg'] = df['week_of_year'].map(woy_avg)

    # Average Revenue by quarter
    q_avg = known.groupby('quarter')['Revenue'].mean().to_dict()
    df['rev_quarter_avg'] = df['quarter'].map(q_avg)

    # Average Revenue by season
    s_avg = known.groupby('season')['Revenue'].mean().to_dict()
    df['rev_season_avg'] = df['season'].map(s_avg)

    return df


def build_features(sales_df=None):
    """
    Main entry point: build all features.
    
    If sales_df is None, loads from sales.csv.
    Can also accept a DataFrame with submission dates appended
    (Revenue=NaN for those dates).
    """
    if sales_df is None:
        df = load_sales()
    else:
        df = sales_df.copy().sort_values('Date').reset_index(drop=True)

    print("=" * 60)
    print("FEATURE ENGINEERING")
    print("=" * 60)

    print("  Adding time features...")
    df = add_time_features(df)

    print("  Adding holiday features...")
    df = add_holiday_features(df)

    print("  Adding Revenue lag/rolling features...")
    df = add_revenue_lag_features(df)

    print("  Adding COGS features...")
    df = add_cogs_features(df)

    print("  Adding seasonal averages...")
    df = add_seasonal_features(df)

    print("  Adding auxiliary features...")
    df = add_auxiliary_features(df)

    # Fill remaining NaN in auxiliary columns with 0
    exclude = ['Date', 'Revenue', 'COGS']
    for col in df.columns:
        if col not in exclude and df[col].dtype in [np.float64, np.float32]:
            df[col] = df[col].fillna(0)

    n_features = len([c for c in df.columns if c not in exclude])
    print(f"\n  ✅ Total features created: {n_features}")
    return df
